"""Step F - retrain a candidate model on a rolling time window.

Rolling-window strategy (R2):
- Source: any `green_tripdata_*.parquet` file in `historical_dir`.
- The file matching the current `batch_path` is excluded so the eval slice
  isn't leaked back into training via the historical pool.
- The current batch's training slice (raw 80%) is appended to the pool.
- A cutoff filter (`pickup_datetime >= max - window_months`) keeps only the
  rolling window. With <12 months of accumulated data this is a no-op today.

Engineering uses the SAME `FeatureSpec` fit on reference (Step C) so the
candidate and champion live in identical feature space.

Evaluation uses the SAME held-out batch slice Step E used for the champion,
so `rmse_candidate` vs `rmse_champion` is a fair head-to-head.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from lib.features import FeatureSpec, engineer_features
from lib.green_taxi_schema import PICKUP_COL
from lib.model_registry import build_model

HISTORICAL_FILE_GLOB = "green_tripdata_*.parquet"


@dataclass
class CandidateResult:
    rmse_candidate: float
    rmse_champion_eval: float
    rmse_delta_pct: float
    train_rows: int
    train_window_start: str
    train_window_end: str
    train_files: list[str]
    window_months: int
    hyperparams: dict[str, float]


def build_rolling_training_set(
    *,
    historical_dir: Path,
    batch_path: Path,
    batch_train_raw: pd.DataFrame,
    window_months: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Concat historical files (minus the current batch file) + batch_train,
    then keep only rows inside the rolling window."""
    historical_paths = _discover_historical_paths(historical_dir, batch_path)
    historical_frames = [pd.read_parquet(p) for p in historical_paths]

    combined = pd.concat([*historical_frames, batch_train_raw], ignore_index=True)
    combined[PICKUP_COL] = pd.to_datetime(combined[PICKUP_COL], errors="coerce")
    combined = combined.dropna(subset=[PICKUP_COL])

    cutoff = combined[PICKUP_COL].max() - pd.DateOffset(months=window_months)
    windowed = cast(
        pd.DataFrame,
        combined[combined[PICKUP_COL] >= cutoff].reset_index(drop=True),
    )

    file_names = [p.name for p in historical_paths]
    return windowed, file_names


def train_and_evaluate_candidate(
    *,
    training_raw: pd.DataFrame,
    feature_spec: FeatureSpec,
    X_batch_eval: pd.DataFrame,
    y_batch_eval: np.ndarray,
    rmse_champion_eval: float,
    train_files: list[str],
    window_months: int,
) -> tuple[Pipeline, CandidateResult]:
    X_train, y_train = engineer_features(training_raw, feature_spec)

    model = build_model()
    model.fit(X_train, y_train)

    rmse_candidate = float(
        np.sqrt(mean_squared_error(y_batch_eval, model.predict(X_batch_eval)))
    )
    rmse_delta_pct = (rmse_candidate - rmse_champion_eval) / rmse_champion_eval * 100.0

    pickup_series = pd.to_datetime(training_raw[PICKUP_COL], errors="coerce")
    result = CandidateResult(
        rmse_candidate=rmse_candidate,
        rmse_champion_eval=rmse_champion_eval,
        rmse_delta_pct=rmse_delta_pct,
        train_rows=int(len(training_raw)),
        train_window_start=pickup_series.min().isoformat(),
        train_window_end=pickup_series.max().isoformat(),
        train_files=train_files,
        window_months=window_months,
        hyperparams=_extract_hyperparams(model),
    )
    return model, result


def _discover_historical_paths(historical_dir: Path, batch_path: Path) -> list[Path]:
    if not historical_dir.exists():
        return []
    batch_resolved = batch_path.resolve()
    return sorted(
        p for p in historical_dir.glob(HISTORICAL_FILE_GLOB) if p.resolve() != batch_resolved
    )


def _extract_hyperparams(model: Pipeline) -> dict[str, float]:
    tree = model.named_steps["tree"]
    return {
        "max_depth": float(tree.max_depth),
        "min_samples_leaf": float(tree.min_samples_leaf),
        "random_state": float(tree.random_state),
        "ccp_alpha": float(tree.ccp_alpha),
    }
