"""Hard-rule integrity checks for the green taxi raw batch.

Soft-return semantics: every check returns metrics + tables instead of raising.
The caller decides pass/fail policy via `hard_failure_reasons` / `hard_is_ok`.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from ..green_taxi_schema import (
    DROPOFF_COL,
    GREEN_TAXI_SCHEMA,
    PICKUP_COL,
    TARGET_COL,
    TRIP_DISTANCE_COL,
    ColumnSpec,
    family_ok,
)
from ._base import CheckResult


_NEG_DURATION_TOLERANCE = 0.05    # >5% rows with dropoff < pickup -> hard fail
_IMPOSSIBLE_VALUE_TOLERANCE = 0.01  # >1% rows with negative trip_distance -> hard fail
_DATETIME_NAN_TOLERANCE = 0.01    # >1% rows with unparseable pickup/dropoff -> hard fail


# ---------------------------------------------------------------------------
# Failure-reason helpers (hard gate only)
# ---------------------------------------------------------------------------

def _reason_missing_cols(metrics: dict[str, float], tables: dict[str, pd.DataFrame]) -> str | None:
    if int(metrics.get("schema_missing_cols", 0)) == 0:
        return None
    tbl = tables.get("schema_missing", pd.DataFrame())
    cols = tbl["column"].tolist() if "column" in tbl.columns else []
    return f"required columns missing: {cols}"


def _reason_target_fully_missing(tables: dict[str, pd.DataFrame]) -> str | None:
    tbl = tables.get("missingness", pd.DataFrame())
    if tbl.empty or "column" not in tbl.columns:
        return None
    for _, row in tbl.iterrows():
        if row["column"] == TARGET_COL and float(row["missing_frac"]) >= 1.0:
            return f"{TARGET_COL} is fully missing - cannot evaluate model"
    return None


def _reason_impossible_trip_distance(tables: dict[str, pd.DataFrame]) -> str | None:
    tbl = tables.get("range_checks", pd.DataFrame())
    if tbl.empty or "column" not in tbl.columns:
        return None
    for _, row in tbl.iterrows():
        if row["column"] == TRIP_DISTANCE_COL:
            bad_frac = float(row["bad_frac"])
            if bad_frac > _IMPOSSIBLE_VALUE_TOLERANCE:
                return (
                    f"{TRIP_DISTANCE_COL} has {bad_frac:.3%} out-of-range values"
                    f" (tolerance={_IMPOSSIBLE_VALUE_TOLERANCE:.0%})"
                )
    return None


def _reason_negative_duration(metrics: dict[str, float]) -> str | None:
    neg_dur = metrics.get("duration_neg_frac", 0.0)
    if neg_dur <= _NEG_DURATION_TOLERANCE:
        return None
    return f"duration_neg_frac={neg_dur:.3%} exceeds tolerance={_NEG_DURATION_TOLERANCE:.0%}"


def _reason_unparseable_datetimes(metrics: dict[str, float]) -> str | None:
    nan_frac = metrics.get("duration_nan_frac", 0.0)
    if nan_frac <= _DATETIME_NAN_TOLERANCE:
        return None
    return f"duration_nan_frac={nan_frac:.3%} exceeds tolerance={_DATETIME_NAN_TOLERANCE:.0%}"


def hard_failure_reasons(result: CheckResult) -> list[str]:
    candidates = [
        _reason_missing_cols(result.metrics, result.tables),
        _reason_target_fully_missing(result.tables),
        _reason_impossible_trip_distance(result.tables),
        _reason_negative_duration(result.metrics),
        _reason_unparseable_datetimes(result.metrics),
    ]
    return [r for r in candidates if r is not None]


def hard_is_ok(result: CheckResult) -> bool:
    return len(hard_failure_reasons(result)) == 0


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------

def _check_schema(
    df: pd.DataFrame,
    schema: tuple[ColumnSpec, ...],
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    schema_cols = {c.name for c in schema}
    required_cols = {c.name for c in schema if c.required}
    present_cols = set(df.columns)

    missing = sorted(required_cols - present_cols)
    extra = sorted(present_cols - schema_cols)

    dtype_rows: list[dict[str, object]] = []
    bad_family = 0
    for spec in schema:
        if spec.name not in df.columns:
            continue
        actual_dtype = df[spec.name].dtype
        ok = family_ok(actual_dtype=actual_dtype, family=spec.dtype)
        if not ok:
            bad_family += 1
        dtype_rows.append(
            {
                "column": spec.name,
                "expected_family": spec.dtype.value,
                "actual_dtype": str(actual_dtype),
                "family_ok": bool(ok),
            }
        )

    tables = {
        "schema_missing": pd.DataFrame({"column": missing}),
        "schema_extra": pd.DataFrame({"column": extra}),
        "schema_dtypes": pd.DataFrame(
            dtype_rows,
            columns=pd.Index(["column", "expected_family", "actual_dtype", "family_ok"]),
        ),
    }
    metrics = {
        "schema_missing_cols": float(len(missing)),
        "schema_extra_cols": float(len(extra)),
        "schema_bad_family_dtypes": float(bad_family),
    }
    return metrics, tables


def _check_missingness(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    if df.shape[1] == 0:
        tables = {
            "missingness": pd.DataFrame(
                columns=pd.Index(["column", "dtype", "missing_frac", "missing_count", "n_unique"])
            )
        }
        metrics = {"missing_frac_mean": float("nan"), "missing_frac_max": float("nan")}
        return metrics, tables

    miss_frac = cast(pd.Series, df.isna().mean(axis=0))
    miss_count = cast(pd.Series, df.isna().sum(axis=0))
    nunique: pd.Series = df.nunique(dropna=False)

    miss = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_frac": miss_frac.values,
            "missing_count": miss_count.values,
            "n_unique": nunique.values,
        }
    ).sort_values(by="missing_frac", ascending=False, kind="stable")

    metrics = {
        "missing_frac_mean": float(np.nanmean(miss_frac.to_numpy(dtype=float))),
        "missing_frac_max": float(np.nanmax(miss_frac.to_numpy(dtype=float))),
    }
    return metrics, {"missingness": miss}


def _check_duplicates(df: pd.DataFrame) -> dict[str, float]:
    dup = int(df.duplicated().sum()) if len(df) else 0
    return {
        "duplicate_rows": float(dup),
        "duplicate_rows_frac": float(dup / max(len(df), 1)),
    }


def _check_ranges(
    df: pd.DataFrame,
    schema: tuple[ColumnSpec, ...],
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    for spec in schema:
        if spec.value_range is None or spec.name not in df.columns:
            continue
        lo, hi = spec.value_range
        x: pd.Series = pd.to_numeric(df[spec.name], errors="coerce")  # type: ignore[assignment]
        valid: pd.Series = x.dropna()
        if valid.empty:
            rows.append({"column": spec.name, "lo": lo, "hi": hi, "bad_frac": 1.0, "min": np.nan, "max": np.nan})
            continue
        bad = (valid < lo) | (valid > hi)
        rows.append(
            {
                "column": spec.name,
                "lo": lo,
                "hi": hi,
                "bad_frac": float(bad.mean()),
                "min": float(valid.min()),
                "max": float(valid.max()),
            }
        )

    if not rows:
        return {}, {}

    rng = pd.DataFrame(rows).sort_values(by="bad_frac", ascending=False)
    metrics = {
        "range_worst_bad_frac": float(rng["bad_frac"].max()),
        "range_any_bad_cols": float((rng["bad_frac"] > 0).sum()),
    }
    return metrics, {"range_checks": rng}


def _check_domains(
    df: pd.DataFrame,
    schema: tuple[ColumnSpec, ...],
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    for spec in schema:
        if spec.allowed_values is None or spec.name not in df.columns:
            continue
        s = df[spec.name]
        bad = ~s.isna() & ~s.isin(list(spec.allowed_values))
        rows.append(
            {
                "column": spec.name,
                "bad_frac": float(bad.mean()) if len(s) else 0.0,
                "bad_count": int(bad.sum()) if len(s) else 0,
                "n_unique": int(s.nunique(dropna=True)) if len(s) else 0,
            }
        )

    if not rows:
        return {}, {}

    dom = pd.DataFrame(rows).sort_values(by="bad_frac", ascending=False)
    metrics = {
        "domain_worst_bad_frac": float(dom["bad_frac"].max()),
        "domain_any_bad_cols": float((dom["bad_count"] > 0).sum()),
    }
    return metrics, {"domain_checks": dom}


def _check_datetime_sanity(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    if PICKUP_COL not in df.columns or DROPOFF_COL not in df.columns:
        return {}, {}

    pickup = pd.to_datetime(df[PICKUP_COL], errors="coerce")
    dropoff = pd.to_datetime(df[DROPOFF_COL], errors="coerce")
    dur = (dropoff - pickup).dt.total_seconds() / 60.0
    n = len(dur)

    metrics = {
        "duration_neg_frac": float((dur < 0).mean()) if n else 0.0,
        "duration_over_6h_frac": float((dur > 360).mean()) if n else 0.0,
        "duration_nan_frac": float(dur.isna().mean()) if n else 0.0,
    }
    tables = {
        "datetime_checks": pd.DataFrame(
            [
                {"column": "duration_min", "check": "duration_negative", "bad_frac": metrics["duration_neg_frac"]},
                {"column": "duration_min", "check": "duration_over_6h", "bad_frac": metrics["duration_over_6h_frac"]},
                {"column": "duration_min", "check": "duration_nan", "bad_frac": metrics["duration_nan_frac"]},
            ],
            columns=pd.Index(["column", "check", "bad_frac"]),
        )
    }
    return metrics, tables


def _check_zone_validity(
    df: pd.DataFrame,
    zone_lookup_path: Path,
) -> dict[str, float]:
    if not zone_lookup_path.exists():
        return {}

    zones = _load_zone_lookup(zone_lookup_path)
    if zones is None or "LocationID" not in zones.columns:
        return {}

    loc_numeric: pd.Series = pd.to_numeric(zones["LocationID"], errors="coerce")  # type: ignore[assignment]
    valid_ids = list(loc_numeric.dropna().astype(int))

    metrics: dict[str, float] = {}
    for col in ("PULocationID", "DOLocationID"):
        if col not in df.columns:
            continue
        col_numeric: pd.Series = pd.to_numeric(df[col], errors="coerce")  # type: ignore[assignment]
        s = col_numeric.dropna().astype(int)
        bad = ~s.isin(valid_ids)
        metrics[f"{col}_unknown_frac"] = float(bad.mean()) if len(s) else 0.0

    return metrics


def _load_zone_lookup(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".geojson", ".json"):
        import json
        data = json.loads(path.read_text())
        if data.get("type") != "FeatureCollection":
            return None
        rows = [feat.get("properties", {}) for feat in data.get("features", [])]
        return pd.DataFrame(rows)
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_hard_integrity_checks(
    *,
    df_raw: pd.DataFrame,
    schema: tuple[ColumnSpec, ...] = GREEN_TAXI_SCHEMA,
    zone_lookup_path: str | None = None,
) -> CheckResult:
    df = df_raw.copy()
    metrics: dict[str, float] = {}
    tables: dict[str, pd.DataFrame] = {}

    for check_metrics, check_tables in [
        _check_schema(df, schema),
        _check_missingness(df),
        (_check_duplicates(df), {}),
        _check_ranges(df, schema),
        _check_domains(df, schema),
        _check_datetime_sanity(df),
    ]:
        metrics.update(check_metrics)
        tables.update(check_tables)

    if zone_lookup_path is not None:
        metrics.update(_check_zone_validity(df, Path(zone_lookup_path)))

    return CheckResult(metrics=metrics, tables=tables)
