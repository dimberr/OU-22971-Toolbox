"""Model registry helpers for champion loading, bootstrap, and promotion.

Responsibilities:
- Load the @champion model version from the MLflow registry.
- Build the standard sklearn Pipeline used for all training runs.
- Bootstrap: train on reference, log, register, and set @champion.
- Register a logged model as a new version and set the @champion alias.
- Tag old champion as previous_champion when a new one is promoted.
"""

from __future__ import annotations

import datetime
import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.sklearn import load_model as load_sklearn_model
from mlflow.sklearn import log_model as log_sklearn_model
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from typing import cast

from lib.features import FeatureSpec


FEATURE_SPEC_ARTIFACT = "feature_spec.json"


MODEL_ARTIFACT_NAME = "model"
CHAMPION_ALIAS = "champion"

_BOOTSTRAP_MAX_DEPTH = 8
_BOOTSTRAP_MIN_SAMPLES_LEAF = 200
_BOOTSTRAP_RANDOM_STATE = 0
_BOOTSTRAP_CCP_ALPHA = 0.0  # 0.0 = no cost-complexity pruning


def load_champion_model(model_name: str) -> tuple[Pipeline, str] | tuple[None, None]:
    """Try to load the @champion model version from the MLflow registry.

    Returns (model, version_number) if the alias exists, or (None, None) if not.
    """
    client = mlflow.MlflowClient()
    try:
        alias_mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        loaded = load_sklearn_model(f"models:/{model_name}@{CHAMPION_ALIAS}")
        return cast(Pipeline, loaded), alias_mv.version
    except MlflowException:
        return None, None


def champion_version(model_name: str) -> str | None:
    """Return the @champion alias version, or None if no champion is set.

    Lighter than `load_champion_model` because it skips loading the model
    artifacts; intended for UI status checks.
    """
    client = mlflow.MlflowClient()
    try:
        return client.get_model_version_by_alias(model_name, CHAMPION_ALIAS).version
    except MlflowException:
        return None


def get_champion_feature_spec(model_name: str) -> FeatureSpec:
    """Download the FeatureSpec that was logged alongside the @champion's training run.

    This is the inference-time counterpart to `mlflow.log_dict(spec.to_dict(),
    feature_spec.json)` performed at training. Loading the spec from MLflow
    (instead of refitting it from a reference parquet) guarantees the
    inference pipeline uses *exactly* the same clip bounds and feature
    columns the model was trained on -- no training/serving skew possible
    even if the on-disk reference data drifts.

    Raises RuntimeError with an actionable message if the alias has no
    source run or the artifact is missing (older models pre-dating this
    convention).
    """
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    if mv.run_id is None:
        raise RuntimeError(
            f"Champion {model_name} v{mv.version} has no source run; "
            "cannot load feature_spec.json."
        )
    try:
        spec_dict = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/{FEATURE_SPEC_ARTIFACT}")
    except (MlflowException, OSError) as exc:
        raise RuntimeError(
            f"Champion {model_name} v{mv.version} (run {mv.run_id}) has no "
            f"{FEATURE_SPEC_ARTIFACT} artifact. Re-bootstrap or retrain so the "
            "spec is logged alongside the model."
        ) from exc
    return FeatureSpec.from_dict(spec_dict)


def bootstrap_champion(
    *,
    X_ref: pd.DataFrame,
    y_ref: np.ndarray,
    model_name: str,
    run_id: str,
    reference_path: str,
) -> tuple[Pipeline, str]:
    """Train on reference, log to the active run, register, and set @champion.

    The caller is responsible for being inside an active mlflow run whose
    id matches `run_id`. Returns (fitted_model, version_number).
    """
    model = build_model()
    model.fit(X_ref, y_ref)

    rmse_ref = float(np.sqrt(mean_squared_error(y_ref, model.predict(X_ref))))

    log_sklearn_model(sk_model=model, name=MODEL_ARTIFACT_NAME, input_example=X_ref.head(5))
    mlflow.log_params({
        "bootstrap_max_depth": _BOOTSTRAP_MAX_DEPTH,
        "bootstrap_min_samples_leaf": _BOOTSTRAP_MIN_SAMPLES_LEAF,
        "bootstrap_random_state": _BOOTSTRAP_RANDOM_STATE,
        "bootstrap_ccp_alpha": _BOOTSTRAP_CCP_ALPHA,
    })
    mlflow.log_metric("rmse_ref", rmse_ref)
    mlflow.set_tag("bootstrap", "true")

    version = register_as_champion(
        model_name=model_name,
        run_id=run_id,
        version_tags={
            "role": "champion",
            "promotion_reason": "bootstrap",
            "decision_reason": "bootstrap",
            "trained_on": reference_path,
        },
    )

    mlflow.log_dict(
        {
            "action": "bootstrap",
            "reason": "No champion found in registry",
            "model_name": model_name,
            "version": version,
            "promotion_reason": "bootstrap",
        },
        artifact_file="load_champion/decision.json",
    )

    return model, version


def build_model(
    random_state: int = _BOOTSTRAP_RANDOM_STATE,
    max_depth: int = _BOOTSTRAP_MAX_DEPTH,
    min_samples_leaf: int = _BOOTSTRAP_MIN_SAMPLES_LEAF,
    ccp_alpha: float = _BOOTSTRAP_CCP_ALPHA,
) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeRegressor(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
        )),
    ])


def register_as_champion(
    *,
    model_name: str,
    run_id: str,
    version_tags: dict[str, str],
) -> str:
    """Register the model logged in run_id and flip the @champion alias to it.

    If a previous champion version exists, it is demoted (role=previous_champion).
    Returns the new version number as a string.
    """
    client = mlflow.MlflowClient()

    old_version = _current_champion_version(client, model_name)

    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    new_version = registered.version

    for key, value in version_tags.items():
        client.set_model_version_tag(model_name, new_version, key, value)

    if old_version is not None:
        _demote_previous_champion(client, model_name, old_version)

    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, new_version)
    return str(new_version)


def register_candidate(
    *,
    model_name: str,
    run_id: str,
    version_tags: dict[str, str],
    artifact_name: str = MODEL_ARTIFACT_NAME,
) -> str:
    """Register the candidate logged in run_id as a new model version.

    Does NOT flip @champion. Tags are applied to the new version.
    """
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_name}"
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    new_version = registered.version

    for key, value in version_tags.items():
        client.set_model_version_tag(model_name, new_version, key, value)

    return str(new_version)


def promote_to_champion(
    *,
    model_name: str,
    candidate_version: str,
    promotion_reason: str,
) -> str | None:
    """Flip @champion to candidate_version. Demote previous, retag new.

    Returns the demoted version (None if no previous champion existed).
    """
    client = mlflow.MlflowClient()
    old_version = _current_champion_version(client, model_name)

    if old_version is not None:
        _demote_previous_champion(client, model_name, old_version)

    promoted_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    new_tags = {
        "role": "champion",
        "promoted_at": promoted_at,
        "promotion_reason": promotion_reason,
        "decision_reason": promotion_reason,
        "validation_status": "approved",
    }
    for key, value in new_tags.items():
        client.set_model_version_tag(model_name, candidate_version, key, value)

    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, candidate_version)
    return old_version


def mark_candidate_rejected(
    *,
    model_name: str,
    candidate_version: str,
    decision_reason: str,
) -> None:
    """Tag a registered candidate version as rejected with the reason."""
    client = mlflow.MlflowClient()
    rejected_tags = {
        "validation_status": "rejected",
        "decision_reason": decision_reason,
    }
    for key, value in rejected_tags.items():
        client.set_model_version_tag(model_name, candidate_version, key, value)


def _current_champion_version(client: mlflow.MlflowClient, model_name: str) -> str | None:
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        return mv.version
    except MlflowException:
        return None


def _demote_previous_champion(
    client: mlflow.MlflowClient,
    model_name: str,
    version: str,
) -> None:
    client.set_model_version_tag(model_name, version, "role", "previous_champion")
    client.set_model_version_tag(
        model_name,
        version,
        "demoted_at",
        datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
    )
