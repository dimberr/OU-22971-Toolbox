"""MLflow logging helpers — one function per flow gate/step.

Each helper receives the analysis result for its gate and handles all
artifact, metric, and decision logging. No analysis logic lives here.
"""

from __future__ import annotations

import mlflow
import pandas as pd
from mlflow.sklearn import log_model as log_sklearn_model
from sklearn.pipeline import Pipeline

from .integrity import CheckResult, hard_failure_reasons
from .model_gate import ModelGateResult
from .model_registry import MODEL_ARTIFACT_NAME
from .promotion import PromotionResult
from .retrain import CandidateResult


_HARD_DASHBOARD_METRICS = [
    "schema_missing_cols",
    "missing_frac_max",
    "range_worst_bad_frac",
    "domain_worst_bad_frac",
    "duration_neg_frac",
]

_SOFT_DASHBOARD_METRICS = [
    "soft_missing_alert_chunks",
    "soft_missing_alert_cols",
    "soft_unseen_alert_chunks",
    "soft_unseen_alert_cols",
    "soft_drift_alert_chunks",
    "soft_drift_alert_cols",
]


def _hard_decision(result: CheckResult) -> dict[str, object]:
    reasons = hard_failure_reasons(result)
    return {
        "action": "reject_batch" if reasons else "proceed",
        "reasons": reasons,
    }


def _soft_decision(result: CheckResult) -> dict[str, object]:
    return {
        "action": "proceed",
        "integrity_warn": bool(result.warnings),
        "warnings": result.warnings,
    }


def log_integrity_result(result: CheckResult, check: str) -> None:
    for name, table in result.tables.items():
        mlflow.log_table(table, artifact_file=f"integrity/{check}/{name}.json")

    dashboard_keys = _HARD_DASHBOARD_METRICS if check == "hard" else _SOFT_DASHBOARD_METRICS
    mlflow.log_metrics(
        {f"integrity_{check}_{k}": result.metrics.get(k, float("nan")) for k in dashboard_keys}
    )

    decision = _hard_decision(result) if check == "hard" else _soft_decision(result)
    mlflow.log_dict(decision, artifact_file=f"integrity/{check}/decision.json")

    if check == "soft":
        mlflow.set_tag("integrity_warn", "true" if result.warnings else "false")


def log_candidate_result(
    result: CandidateResult,
    model: Pipeline,
    x_sample: pd.DataFrame,
) -> None:
    log_sklearn_model(
        sk_model=model,
        name=MODEL_ARTIFACT_NAME,
        input_example=x_sample.head(5),
    )
    mlflow.log_metrics({
        "rmse_candidate": result.rmse_candidate,
        "rmse_candidate_vs_champion_pct": result.rmse_delta_pct,
        "candidate_train_rows": float(result.train_rows),
        "candidate_window_months": float(result.window_months),
    })
    mlflow.log_params({
        f"candidate_{k}": v for k, v in result.hyperparams.items()
    })
    mlflow.set_tag("candidate_logged", "true")
    mlflow.log_dict(
        {
            "action": "train_candidate",
            "rmse_candidate": result.rmse_candidate,
            "rmse_champion_eval": result.rmse_champion_eval,
            "rmse_delta_pct": result.rmse_delta_pct,
            "train_window_months": result.window_months,
            "train_window_start": result.train_window_start,
            "train_window_end": result.train_window_end,
            "train_rows": result.train_rows,
            "train_files": result.train_files,
        },
        artifact_file="retrain/decision.json",
    )


def log_model_gate_result(result: ModelGateResult) -> None:
    mlflow.log_metrics({
        "rmse_champion": result.rmse_champion,
        "rmse_baseline": result.rmse_baseline,
        "rmse_increase_pct": result.rmse_increase_pct,
        "model_gate_alert_chunks": float(result.alert_chunks),
        "model_gate_total_chunks": float(result.total_chunks),
    })
    mlflow.set_tag("retrain_recommended", "true" if result.retrain_needed else "false")
    mlflow.log_table(result.per_chunk, artifact_file="model_gate/perf_per_chunk.json")
    mlflow.log_dict(
        {
            "rule": "NannyML PerformanceCalculator (RMSE) with default 3-sigma threshold",
            "rmse_champion": result.rmse_champion,
            "rmse_baseline": result.rmse_baseline,
            "rmse_increase_pct": result.rmse_increase_pct,
            "alert_chunks": result.alert_chunks,
            "total_chunks": result.total_chunks,
            "retrain_needed": result.retrain_needed,
            "retrain_reason": result.retrain_reason,
        },
        artifact_file="model_gate/decision.json",
    )


def log_promotion_result(
    result: PromotionResult,
    *,
    candidate_version: str,
    demoted_version: str | None,
) -> None:
    mlflow.log_metrics({
        # `rmse_ref` mirrors `rmse_candidate_ref` on purpose: if this candidate
        # is promoted, its source MLflow run becomes the new champion's source
        # run, and the next flow's `get_champion_baseline_rmse` reads `rmse_ref`
        # from it (same name `bootstrap_champion` uses). Logging unconditionally
        # keeps every model version's source run uniform.
        "rmse_ref": result.rmse_candidate_ref,
        "rmse_candidate_ref": result.rmse_candidate_ref,
        "rmse_champion_ref": result.rmse_champion_ref,
        "rmse_ref_delta_pct": result.rmse_ref_delta_pct,
        "promotion_min_improvement_pct": result.min_improvement_pct,
        "promotion_max_ref_regression_pct": result.max_ref_regression_pct,
    })
    mlflow.set_tags({
        "promotion_recommended": "true" if result.promoted else "false",
        "candidate_version": candidate_version,
    })
    mlflow.log_dict(
        {
            "rule": "Promote iff P1 (eval valid) & P2 (beats champion) & P3 (no ref regression) & P4 (integrity sanity)",
            "promoted": result.promoted,
            "candidate_version": candidate_version,
            "demoted_version": demoted_version,
            "rmse_candidate_eval": result.rmse_candidate_eval,
            "rmse_champion_eval": result.rmse_champion_eval,
            "rmse_candidate_ref": result.rmse_candidate_ref,
            "rmse_champion_ref": result.rmse_champion_ref,
            "rmse_ref_delta_pct": result.rmse_ref_delta_pct,
            "min_improvement_pct": result.min_improvement_pct,
            "max_ref_regression_pct": result.max_ref_regression_pct,
            "integrity_warn": result.integrity_warn,
            "criteria": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in result.criteria
            ],
            "decision_reason": result.decision_reason,
        },
        artifact_file="promotion/decision.json",
    )
