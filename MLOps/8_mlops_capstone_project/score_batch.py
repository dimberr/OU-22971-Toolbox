"""Batch inference: load @champion, score a batch, log predictions.parquet.

Standalone entry point separate from `flow_starter.py` because the goal is
production scoring (write predictions for downstream use), not the
gate/retrain decision pipeline.

Usage:
    python score_batch.py \\
        --reference-path TLC_Data/green_tripdata_2020-01.parquet \\
        --batch-path     TLC_Data/green_tripdata_2020-04.parquet \\
        --model-name     green_taxi_tip_model

Output:
    - local file: ui_state/predictions/<run_id>.parquet
    - MLflow artifact: predictions.parquet (under the inference run)
    - MLflow metrics: n_rows_input, n_rows_scored, mean_pred, p50_pred, p95_pred
    - MLflow tags: pipeline=capstone_inference, model_version, batch_path
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from lib.features import engineer_features, fit_feature_spec
from lib.green_taxi_schema import PICKUP_COL
from lib.helper import init_mlflow, load_batch, load_reference
from lib.model_registry import champion_version, load_champion_model


PREDICTIONS_ARTIFACT = "predictions.parquet"


def main() -> None:
    args = _parse_args()
    init_mlflow(args.model_name)

    version = champion_version(args.model_name)
    if version is None:
        raise SystemExit(
            f"No @champion alias for model '{args.model_name}'. "
            "Run bootstrap.py (or a flow that promotes a candidate) first."
        )
    model, _ = load_champion_model(args.model_name)
    assert model is not None  # champion_version above guarantees the alias exists

    ref_df = load_reference(args.reference_path)
    batch_df = load_batch(args.batch_path)
    spec = fit_feature_spec(ref_df)
    x_batch, y_batch = engineer_features(batch_df, spec)

    preds = model.predict(x_batch)

    predictions_df = _build_predictions_df(
        x_index=x_batch.index,
        batch_df=batch_df,
        preds=preds,
        actuals=y_batch,
        model_name=args.model_name,
        model_version=version,
        batch_path=args.batch_path,
    )

    with mlflow.start_run() as run:
        mlflow.set_tags({
            "pipeline": "capstone_inference",
            "model_name": args.model_name,
            "model_version": version,
            "batch_path": args.batch_path,
            "reference_path": args.reference_path,
        })
        mlflow.log_metrics({
            "n_rows_input": float(len(batch_df)),
            "n_rows_scored": float(len(predictions_df)),
            "mean_pred": float(np.mean(preds)),
            "p50_pred": float(np.percentile(preds, 50)),
            "p95_pred": float(np.percentile(preds, 95)),
        })
        _log_predictions(predictions_df, run_id=run.info.run_id)

    print(
        f"Scored {len(predictions_df)} / {len(batch_df)} rows with "
        f"{args.model_name} v{version}; logged predictions.parquet."
    )


def _build_predictions_df(
    *,
    x_index: pd.Index,
    batch_df: pd.DataFrame,
    preds: np.ndarray,
    actuals: np.ndarray,
    model_name: str,
    model_version: str,
    batch_path: str,
) -> pd.DataFrame:
    pickup_series = batch_df.loc[x_index, PICKUP_COL] if PICKUP_COL in batch_df.columns else None
    return pd.DataFrame({
        "row_index": x_index.to_numpy(),
        PICKUP_COL: pickup_series.to_numpy() if pickup_series is not None else pd.NaT,
        "predicted_tip_amount": preds,
        "actual_tip_amount": actuals,
        "model_name": model_name,
        "model_version": model_version,
        "batch_path": batch_path,
        "scored_at": pd.Timestamp.utcnow(),
    })


def _log_predictions(predictions_df: pd.DataFrame, *, run_id: str) -> None:
    persisted = _persist_local_copy(predictions_df, run_id=run_id)
    if persisted is not None:
        mlflow.log_artifact(str(persisted))
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / PREDICTIONS_ARTIFACT
        predictions_df.to_parquet(path, index=False)
        mlflow.log_artifact(str(path))


def _persist_local_copy(predictions_df: pd.DataFrame, *, run_id: str) -> Path | None:
    """Best-effort save under ui_state/predictions/<run_id>/predictions.parquet
    so the host can inspect it AND mlflow.log_artifact preserves the friendly
    filename `predictions.parquet`.

    Returns the saved path on success, or None if the directory isn't writable
    (in which case the caller falls back to a temp file).
    """
    out_dir = Path("ui_state/predictions") / run_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    out_path = out_dir / PREDICTIONS_ARTIFACT
    predictions_df.to_parquet(out_path, index=False)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a batch with the @champion model.")
    parser.add_argument("--reference-path", required=True)
    parser.add_argument("--batch-path", required=True)
    parser.add_argument("--model-name", default="green_taxi_tip_model")
    return parser.parse_args()


if __name__ == "__main__":
    main()
