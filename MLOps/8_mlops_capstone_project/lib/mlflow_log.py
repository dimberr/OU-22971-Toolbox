"""MLflow logging helpers — one function per flow gate/step.

Each helper receives the analysis result for its gate and handles all
artifact, metric, and decision logging. No analysis logic lives here.
"""

from __future__ import annotations

import mlflow

from .integrity import CheckResult, hard_failure_reasons


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
