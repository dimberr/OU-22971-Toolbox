"""Soft (NannyML-based) integrity checks for the green taxi raw batch.

Two NannyML calculators run side-by-side:
- MissingValuesCalculator: detects missingness spike vs reference.
- UnseenValuesCalculator: detects categorical values not seen in reference.

Output is packed into a CheckResult identical in shape to the hard gate:
- metrics: alert summary counts.
- tables: per-chunk results + alerting-column summary (for MLflow logging).
- warnings: human-readable strings, one per alerting column per calculator.
"""

from __future__ import annotations

import nannyml as nml
import pandas as pd

from ..green_taxi_schema import PICKUP_COL
from ._base import CheckResult


# Columns NannyML monitors for missingness spike vs reference.
# ehail_fee is intentionally excluded (historically junk in TLC data).
_SOFT_MISSING_COLS: tuple[str, ...] = (
    "VendorID", "passenger_count", "trip_distance",
    "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "total_amount", "congestion_surcharge",
    "RatecodeID", "PULocationID", "DOLocationID",
    "payment_type", "trip_type", "store_and_fwd_flag",
)

# Columns NannyML monitors for unseen categorical values vs reference.
# Numeric "code" columns count as categorical here.
_SOFT_UNSEEN_COLS: tuple[str, ...] = (
    "RatecodeID", "payment_type", "trip_type",
    "store_and_fwd_flag",
    "PULocationID", "DOLocationID",
)


def _filter_present(cols: tuple[str, ...], *frames: pd.DataFrame) -> list[str]:
    return [c for c in cols if all(c in f.columns for f in frames)]


def _cast_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out[col].astype("category")
    return out


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat = df.copy()
    flat.columns = pd.Index(
        ["__".join(str(p) for p in c) if isinstance(c, tuple) else str(c) for c in flat.columns]
    )
    return flat.reset_index(drop=True)


def _summarize_alerts(result_df: pd.DataFrame) -> tuple[int, list[str]]:
    if not isinstance(result_df.columns, pd.MultiIndex):
        return 0, []
    alerts = result_df.xs("alert", axis=1, level=1)
    alerting = sorted([str(c) for c in alerts.columns if bool(alerts[c].any())])
    total = int(alerts.sum().sum())
    return total, alerting


def _run_missing_values_calc(
    *,
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    timestamp_col: str,
    chunk_period: str,
) -> tuple[dict[str, float], dict[str, pd.DataFrame], list[str]]:
    cols = _filter_present(_SOFT_MISSING_COLS, reference, batch)
    if not cols:
        return {}, {}, []

    calc = nml.MissingValuesCalculator(
        column_names=cols,
        timestamp_column_name=timestamp_col,
        chunk_period=chunk_period,
        normalize=True,
    )
    calc.fit(reference)
    result = calc.calculate(batch).filter(period="analysis").to_df()

    total_alerts, alerting_cols = _summarize_alerts(result)
    metrics = {
        "soft_missing_alert_chunks": float(total_alerts),
        "soft_missing_alert_cols": float(len(alerting_cols)),
    }
    tables = {
        "missing_per_chunk": _flatten_columns(result),
        "missing_alert_columns": pd.DataFrame({"column": alerting_cols}),
    }
    warnings = [f"missing_values: column '{c}' alerted vs reference" for c in alerting_cols]
    return metrics, tables, warnings


def _run_unseen_values_calc(
    *,
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    timestamp_col: str,
    chunk_period: str,
) -> tuple[dict[str, float], dict[str, pd.DataFrame], list[str]]:
    cols = _filter_present(_SOFT_UNSEEN_COLS, reference, batch)
    if not cols:
        return {}, {}, []

    ref_cat = _cast_categorical(reference, cols)
    batch_cat = _cast_categorical(batch, cols)

    calc = nml.UnseenValuesCalculator(
        column_names=cols,
        timestamp_column_name=timestamp_col,
        chunk_period=chunk_period,
    )
    calc.fit(ref_cat)
    result = calc.calculate(batch_cat).filter(period="analysis").to_df()

    total_alerts, alerting_cols = _summarize_alerts(result)
    metrics = {
        "soft_unseen_alert_chunks": float(total_alerts),
        "soft_unseen_alert_cols": float(len(alerting_cols)),
    }
    tables = {
        "unseen_per_chunk": _flatten_columns(result),
        "unseen_alert_columns": pd.DataFrame({"column": alerting_cols}),
    }
    warnings = [f"unseen_values: column '{c}' alerted vs reference" for c in alerting_cols]
    return metrics, tables, warnings


def run_soft_integrity_checks(
    *,
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    timestamp_col: str = PICKUP_COL,
    chunk_period: str = "W",
) -> CheckResult:
    metrics: dict[str, float] = {}
    tables: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []

    for run in (_run_missing_values_calc, _run_unseen_values_calc):
        m, t, w = run(
            reference=reference,
            batch=batch,
            timestamp_col=timestamp_col,
            chunk_period=chunk_period,
        )
        metrics.update(m)
        tables.update(t)
        warnings.extend(w)

    return CheckResult(metrics=metrics, tables=tables, warnings=warnings)
