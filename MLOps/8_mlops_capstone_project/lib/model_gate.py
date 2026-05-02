"""Model gate (Step E) — evaluate champion on the batch and decide whether to retrain.

Decision rule:
- Run NannyML PerformanceCalculator with RMSE on labeled reference + batch.
- Default StandardDeviationThreshold (3-sigma vs reference per-chunk RMSE) flags
  any analysis chunk whose RMSE drifts beyond the noise band.
- retrain_needed = at least one analysis chunk alerted.

Also computes rmse_champion / rmse_baseline / rmse_increase_pct for visibility.
The baseline (rmse_ref) is read from the champion's MLflow training run.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import nannyml as nml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

RMSE_REF_METRIC = "rmse_ref"

_Y_TRUE = "y_true"
_Y_PRED = "y_pred"
_RMSE_ALERT_COL = ("rmse", "alert")


@dataclass
class ModelGateResult:
    rmse_champion: float
    rmse_baseline: float
    rmse_increase_pct: float
    alert_chunks: int
    total_chunks: int
    retrain_needed: bool
    retrain_reason: str
    per_chunk: pd.DataFrame


def evaluate_champion(
    *,
    model: Pipeline,
    X_ref: pd.DataFrame,
    y_ref: np.ndarray,
    X_batch: pd.DataFrame,
    y_batch: np.ndarray,
    rmse_baseline: float,
) -> ModelGateResult:
    y_pred_batch = model.predict(X_batch)
    rmse_champion = float(np.sqrt(mean_squared_error(y_batch, y_pred_batch)))
    rmse_increase_pct = (rmse_champion - rmse_baseline) / rmse_baseline * 100

    per_chunk, alert_chunks, total_chunks = _run_perf_calculator(
        model=model,
        X_ref=X_ref,
        y_ref=y_ref,
        X_batch=X_batch,
        y_batch=y_batch,
        y_pred_batch=y_pred_batch,
    )

    retrain_needed = alert_chunks > 0
    retrain_reason = _build_retrain_reason(
        alert_chunks=alert_chunks,
        total_chunks=total_chunks,
        rmse_champion=rmse_champion,
        rmse_baseline=rmse_baseline,
    )

    return ModelGateResult(
        rmse_champion=rmse_champion,
        rmse_baseline=rmse_baseline,
        rmse_increase_pct=rmse_increase_pct,
        alert_chunks=alert_chunks,
        total_chunks=total_chunks,
        retrain_needed=retrain_needed,
        retrain_reason=retrain_reason,
        per_chunk=per_chunk,
    )


def get_champion_baseline_rmse(model_name: str, version: str) -> float | None:
    """Retrieve rmse_ref logged during the champion's training run from MLflow."""
    client = mlflow.MlflowClient()
    mv = client.get_model_version(model_name, version)
    if mv.run_id is None:
        return None
    run = mlflow.get_run(mv.run_id)
    value = run.data.metrics.get(RMSE_REF_METRIC)
    return float(value) if value is not None else None


def _run_perf_calculator(
    *,
    model: Pipeline,
    X_ref: pd.DataFrame,
    y_ref: np.ndarray,
    X_batch: pd.DataFrame,
    y_batch: np.ndarray,
    y_pred_batch: np.ndarray,
) -> tuple[pd.DataFrame, int, int]:
    ref_df = _attach_targets_and_preds(X_ref, y_true=y_ref, y_pred=model.predict(X_ref))
    analysis_df = _attach_targets_and_preds(X_batch, y_true=y_batch, y_pred=y_pred_batch)

    calc = nml.PerformanceCalculator(
        metrics=["rmse"],
        y_true=_Y_TRUE,
        y_pred=_Y_PRED,
        problem_type="regression",
    )
    calc.fit(ref_df)
    result = calc.calculate(analysis_df).filter(period="analysis").to_df()

    alerts = result[_RMSE_ALERT_COL]
    alert_chunks = int(alerts.sum())
    total_chunks = int(len(alerts))
    return _flatten_columns(result), alert_chunks, total_chunks


def _attach_targets_and_preds(
    features: pd.DataFrame,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    out = features.copy()
    out[_Y_TRUE] = y_true
    out[_Y_PRED] = y_pred
    return out


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat = df.copy()
    flat.columns = pd.Index(
        ["__".join(str(p) for p in c) if isinstance(c, tuple) else str(c) for c in flat.columns]
    )
    return flat.reset_index(drop=True)


def _build_retrain_reason(
    *,
    alert_chunks: int,
    total_chunks: int,
    rmse_champion: float,
    rmse_baseline: float,
) -> str:
    summary = (
        f"rmse_champion={rmse_champion:.4f}, rmse_baseline={rmse_baseline:.4f}"
    )
    if alert_chunks > 0:
        return (
            f"NannyML PerformanceCalculator alerted {alert_chunks}/{total_chunks} batch chunks "
            f"({summary})"
        )
    return f"No chunks alerted out of {total_chunks} ({summary})"
