"""Feature engineering for Green Taxi tip prediction.

Pipeline (applied identically to reference and batch):
  1. Filter to credit-card trips (payment_type == 1).
  2. Add calendar features from pickup datetime + duration_min.
  3. Clip heavy-tailed numeric columns using bounds fit on reference.
  4. Apply log1p to those same columns after clipping.
  5. Select a fixed feature column list; extract tip_amount as target.

The clip bounds are always fit on the *reference* dataset, then applied
unchanged to the batch so the transform never leaks batch distribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from lib.green_taxi_schema import (
    CLIP_QUANTILE,
    CREDIT_CARD_PAYMENT_TYPE,
    DROPOFF_COL,
    HEAVY_TAIL_COLS,
    PAYMENT_TYPE_COL,
    PICKUP_COL,
    TARGET_COL,
)


def time_split_batch(
    df_raw: pd.DataFrame,
    *,
    eval_pct: float,
    date_col: str = PICKUP_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-order the batch by `date_col` and split off the most recent `eval_pct` as eval.

    Used by Step E (champion evaluation) and Step F (candidate evaluation) so both
    score on the same held-out tail and can be compared head-to-head fairly.
    """
    if not 0.0 < eval_pct < 1.0:
        raise ValueError(f"eval_pct must be in (0, 1); got {eval_pct}")

    sorted_df = cast(
        pd.DataFrame,
        df_raw.sort_values(date_col, kind="mergesort").reset_index(drop=True),
    )
    split_idx = int(len(sorted_df) * (1.0 - eval_pct))
    train = cast(pd.DataFrame, sorted_df.iloc[:split_idx].copy())
    eval_ = cast(pd.DataFrame, sorted_df.iloc[split_idx:].copy())
    return train, eval_


# Columns dropped from the feature matrix.
# total_amount leaks the target (it includes tip_amount).
# Dropoff datetime features are redundant once duration_min is present.
_DROP_COLS: set[str] = {
    TARGET_COL,
    "total_amount",
    "lpep_dropoff_year",
    "lpep_dropoff_month",
    "lpep_dropoff_weekday",
    "lpep_dropoff_hour",
    PAYMENT_TYPE_COL,
}


@dataclass
class FeatureSpec:
    feature_cols: list[str]
    clip_bounds: dict[str, tuple[float, float]]  # column -> (lo, hi) applied before log1p

    def to_dict(self) -> dict:
        return {
            "feature_cols": self.feature_cols,
            "clip_bounds": {col: list(bounds) for col, bounds in self.clip_bounds.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpec":
        return cls(
            feature_cols=list(data["feature_cols"]),
            clip_bounds={
                col: (float(bounds[0]), float(bounds[1]))
                for col, bounds in data["clip_bounds"].items()
            },
        )


def fit_feature_spec(df_ref: pd.DataFrame) -> FeatureSpec:
    """Fit clip bounds from the reference dataset and return a FeatureSpec.

    Call this once on reference; pass the result to engineer_features for
    both reference and batch.
    """
    df_cc = _filter_credit_card(df_ref)
    df_with_time = _add_datetime_features(df_cc)

    clip_bounds: dict[str, tuple[float, float]] = {}
    for col in HEAVY_TAIL_COLS:
        if col not in df_with_time.columns:
            continue
        series = _to_numeric_series(cast(pd.Series, df_with_time[col])).dropna()
        lo = float(max(0.0, series.quantile(0.01)))
        hi = float(series.quantile(CLIP_QUANTILE))
        clip_bounds[col] = (lo, hi)

    # Derive feature cols from a sample-engineered reference frame.
    x_sample, _ = _build_feature_matrix_and_target(df_with_time, clip_bounds)
    return FeatureSpec(feature_cols=list(x_sample.columns), clip_bounds=clip_bounds)


def engineer_features(
    df_raw: pd.DataFrame,
    spec: FeatureSpec,
    *,
    credit_card_only: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the full feature pipeline using a pre-fit FeatureSpec.

    Returns (X, y) where X has exactly spec.feature_cols as columns and
    y is the tip_amount target as a float array.
    """
    df = _filter_credit_card(df_raw) if credit_card_only else df_raw.copy()
    df = _add_datetime_features(df)
    x, y = _build_feature_matrix_and_target(df, spec.clip_bounds)

    missing = [c for c in spec.feature_cols if c not in x.columns]
    if missing:
        raise ValueError(f"Engineered frame is missing expected feature columns: {missing}")

    result = cast(pd.DataFrame, x[spec.feature_cols].copy())
    return result, y


def _to_numeric_series(s: pd.Series) -> pd.Series:
    # Coerces non-numeric values to NaN instead of raising.
    return pd.Series(pd.to_numeric(s, errors="coerce"))


def _filter_credit_card(df: pd.DataFrame) -> pd.DataFrame:
    # Keeps only rows where the customer paid by credit card.
    # If the column is absent (e.g. in a transformed frame), returns as-is.
    if PAYMENT_TYPE_COL not in df.columns:
        return df.copy()
    return cast(pd.DataFrame, df[df[PAYMENT_TYPE_COL] == CREDIT_CARD_PAYMENT_TYPE].copy())


def _add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    # Extracts year, month, weekday, and hour from pickup and dropoff datetimes,
    # and derives duration_min as dropoff minus pickup in minutes.
    out = df.copy()
    for col in (PICKUP_COL, DROPOFF_COL):
        if col not in out.columns:
            continue
        dt = pd.to_datetime(out[col], errors="coerce")
        prefix = col.replace("_datetime", "")
        out[f"{prefix}_year"] = dt.dt.year.astype("Int64")
        out[f"{prefix}_month"] = dt.dt.month.astype("Int64")
        out[f"{prefix}_weekday"] = dt.dt.dayofweek.astype("Int64")
        out[f"{prefix}_hour"] = dt.dt.hour.astype("Int64")

    if PICKUP_COL in out.columns and DROPOFF_COL in out.columns:
        pickup = pd.to_datetime(out[PICKUP_COL], errors="coerce")
        dropoff = pd.to_datetime(out[DROPOFF_COL], errors="coerce")
        out["duration_min"] = (dropoff - pickup).dt.total_seconds() / 60.0

    return out


def _apply_clip_and_log(df: pd.DataFrame, clip_bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    # For each heavy-tail column: clips to [lo, hi] then applies log1p to compress the tail.
    out = df.copy()
    for col, (lo, hi) in clip_bounds.items():
        if col not in out.columns:
            continue
        clipped = _to_numeric_series(cast(pd.Series, out[col])).clip(lower=lo, upper=hi)
        out[col] = np.log1p(clipped)
    return out


def _build_feature_matrix_and_target(
    df: pd.DataFrame,
    clip_bounds: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, np.ndarray]:
    # Extracts target, applies clip+log transforms, selects numeric columns,
    # and drops columns that are leakage or redundant (see _DROP_COLS).
    y = _to_numeric_series(cast(pd.Series, df[TARGET_COL])).fillna(0.0).to_numpy(dtype=float)

    df = _apply_clip_and_log(df, clip_bounds)

    numeric = df.select_dtypes(include=["number"]).copy()
    # Cast nullable Int64 to float64 to avoid issues with sklearn and MLflow.
    for c in numeric.columns:
        if pd.api.types.is_integer_dtype(numeric[c]):
            numeric[c] = numeric[c].astype("float64")

    x_df = cast(pd.DataFrame, numeric.drop(columns=[c for c in _DROP_COLS if c in numeric.columns]))
    return x_df, y
