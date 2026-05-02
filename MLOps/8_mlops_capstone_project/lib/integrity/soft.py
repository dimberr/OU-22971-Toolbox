"""Soft (NannyML-based) integrity checks for the green taxi raw batch.

Three NannyML calculators run side-by-side:
- MissingValuesCalculator: detects missingness spike vs reference.
- UnseenValuesCalculator: detects categorical values not seen in reference.
- UnivariateDriftCalculator: detects per-column distribution drift vs reference
  (Jensen-Shannon distance for both continuous and categorical columns).

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

# Continuous numeric columns NannyML monitors for distribution drift.
# Categorical drift uses _SOFT_UNSEEN_COLS (same list, cast to category dtype).
_SOFT_DRIFT_CONT_COLS: tuple[str, ...] = (
    "VendorID", "passenger_count", "trip_distance",
    "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "total_amount", "congestion_surcharge",
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


def _summarize_drift_alerts(result_df: pd.DataFrame) -> tuple[int, list[str]]:
    # UnivariateDriftCalculator's columns are a 3-level MultiIndex:
    # (column_name, method, attribute). Alerts live on the last level.
    if not isinstance(result_df.columns, pd.MultiIndex):
        return 0, []
    alerts = result_df.xs("alert", axis=1, level=-1)
    alerting = sorted({
        str(col) for col, method in alerts.columns
        if bool(alerts[(col, method)].any())
    })
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


def _run_univariate_drift_calc(
    *,
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    timestamp_col: str,
    chunk_period: str,
) -> tuple[dict[str, float], dict[str, pd.DataFrame], list[str]]:
    cont_cols = _filter_present(_SOFT_DRIFT_CONT_COLS, reference, batch)
    cat_cols = _filter_present(_SOFT_UNSEEN_COLS, reference, batch)
    column_names = cont_cols + cat_cols
    if not column_names:
        return {}, {}, []

    ref_cat = _cast_categorical(reference, cat_cols)
    batch_cat = _cast_categorical(batch, cat_cols)

    calc = nml.UnivariateDriftCalculator(
        column_names=column_names,
        timestamp_column_name=timestamp_col,
        chunk_period=chunk_period,
        continuous_methods=["jensen_shannon"],
        categorical_methods=["jensen_shannon"],
    )
    calc.fit(ref_cat)
    result = calc.calculate(batch_cat).filter(period="analysis").to_df()

    total_alerts, alerting_cols = _summarize_drift_alerts(result)
    metrics = {
        "soft_drift_alert_chunks": float(total_alerts),
        "soft_drift_alert_cols": float(len(alerting_cols)),
    }
    tables = {
        "drift_per_chunk": _flatten_columns(result),
        "drift_alert_columns": pd.DataFrame({"column": alerting_cols}),
    }
    warnings = [f"univariate_drift: column '{c}' alerted vs reference" for c in alerting_cols]
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

    calculators = (
        _run_missing_values_calc,
        _run_unseen_values_calc,
        _run_univariate_drift_calc,
    )
    for run in calculators:
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
