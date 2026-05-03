"""One-time bootstrap: train the initial champion from the reference dataset.

Run this when no `@champion` alias exists for the registered model. After
this, batch runs of `flow_starter.py` will load the champion and skip the
bootstrap branch in `load_champion`.

Usage:
    python bootstrap.py --reference-path TLC_Data/green_tripdata_2020-01.parquet

Idempotent: if a champion already exists, this script exits without
training a second one.
"""

from __future__ import annotations

import argparse

import mlflow

from lib.features import engineer_features, fit_feature_spec
from lib.helper import init_mlflow, load_reference
from lib.model_registry import bootstrap_champion, load_champion_model


def main() -> None:
    args = _parse_args()
    init_mlflow(args.model_name)

    existing_model, existing_version = load_champion_model(args.model_name)
    if existing_model is not None:
        print(
            f"Champion already exists: {args.model_name} v{existing_version}. "
            "Nothing to do."
        )
        return

    with mlflow.start_run() as run:
        mlflow.set_tags({
            "pipeline": "capstone_bootstrap",
            "reference_path": args.reference_path,
            "model_name": args.model_name,
        })

        ref = load_reference(args.reference_path)
        spec = fit_feature_spec(ref)
        x_ref, y_ref = engineer_features(ref, spec)
        mlflow.log_dict(spec.to_dict(), "feature_spec.json")

        _, version = bootstrap_champion(
            X_ref=x_ref,
            y_ref=y_ref,
            model_name=args.model_name,
            run_id=run.info.run_id,
            reference_path=args.reference_path,
        )
        print(f"Bootstrapped champion: {args.model_name} v{version}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the initial champion model.")
    parser.add_argument("--reference-path", required=True)
    parser.add_argument("--model-name", default="green_taxi_tip_model")
    return parser.parse_args()


if __name__ == "__main__":
    main()
