from pathlib import Path
from typing import cast

import mlflow
import numpy as np
import pandas as pd
from metaflow.flowspec import FlowSpec
from metaflow.parameters import Parameter
from metaflow.decorators import step

from lib.integrity import hard_is_ok, run_hard_integrity_checks, run_soft_integrity_checks
from lib.mlflow_log import (
    log_candidate_result,
    log_integrity_result,
    log_model_gate_result,
    log_promotion_result,
)
from lib.green_taxi_schema import GREEN_TAXI_SCHEMA
from lib.helper import flow_run, init_mlflow, load_batch, load_reference
from lib.features import FeatureSpec, engineer_features, fit_feature_spec, time_split_batch
from lib.model_registry import (
    bootstrap_champion,
    load_champion_model,
    mark_candidate_rejected,
    promote_to_champion,
    register_candidate,
)
from lib.model_gate import RMSE_REF_METRIC, evaluate_champion, get_champion_baseline_rmse
from lib.promotion import evaluate_promotion
from lib.retrain import build_rolling_training_set, train_and_evaluate_candidate


class MLFlowCapstoneFlow(FlowSpec):
    reference_path = Parameter("reference-path")
    batch_path = Parameter("batch-path")
    taxi_zone_lookup_path = Parameter("taxi-zone-lookup-path", default="TLC_Data/NYC_Taxi_Zones.geojson", required=False)
    model_name = Parameter("model-name", default="green_taxi_tip_model")
    historical_dir = Parameter("historical-dir", default="TLC_Data", required=False)
    rolling_window_months = Parameter("rolling-window-months", default=12, required=False)
    batch_eval_pct = Parameter("batch-eval-pct", default=0.2, required=False)
    min_improvement_pct = Parameter("min-improvement-pct", default=0.01, required=False)
    max_ref_regression_pct = Parameter("max-ref-regression-pct", default=0.01, required=False)

    @step
    def start(self):
        init_mlflow(str(self.model_name))
        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            mlflow.set_tags({
                "pipeline": "capstone_monitoring",
                "reference_path": str(self.reference_path),
                "batch_path": str(self.batch_path),
                "model_name": str(self.model_name),
            })
        self.next(self.load_data)

    @step
    def load_data(self):
        self.ref, self.batch = load_reference(self.reference_path), load_batch(self.batch_path)
        self.next(self.hard_integrity_gate)

    @step
    def hard_integrity_gate(self):
        zone_lookup_path = str(self.taxi_zone_lookup_path) if self.taxi_zone_lookup_path else None

        hard_check_result = run_hard_integrity_checks(
            df_raw=self.batch,
            schema=GREEN_TAXI_SCHEMA,
            zone_lookup_path=zone_lookup_path,
        )

        with flow_run(model_name=str(self.model_name), run_id=self.run_id):
            log_integrity_result(hard_check_result, check="hard")

        ok = hard_is_ok(hard_check_result)
        self.batch_rejected = not ok
        self.integrity_decision = "pass" if ok else "fail"
        self.next(
            {"pass": self.soft_integrity_gate, "fail": self.end},  # pyright: ignore[reportArgumentType]
            condition="integrity_decision",
        )

    @step
    def soft_integrity_gate(self):
        soft_check_result = run_soft_integrity_checks(
            reference=self.ref,
            batch=self.batch,
        )

        with flow_run(model_name=str(self.model_name), run_id=self.run_id):
            log_integrity_result(soft_check_result, check="soft")

        self.integrity_warn = bool(soft_check_result.warnings)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        self.feature_spec: FeatureSpec = fit_feature_spec(self.ref)

        # Time-split the raw batch: last `batch_eval_pct` becomes the eval slice
        # used by both Step E (champion) and Step F (candidate). The train slice
        # is reused as the freshest contribution to the candidate's training set.
        self.batch_train_raw, batch_eval_raw = time_split_batch(
            self.batch, eval_pct=cast(float, self.batch_eval_pct),
        )

        self.X_ref, self.y_ref = engineer_features(self.ref, self.feature_spec)
        self.X_batch_train, self.y_batch_train = engineer_features(
            self.batch_train_raw, self.feature_spec,
        )
        self.X_batch_eval, self.y_batch_eval = engineer_features(
            batch_eval_raw, self.feature_spec,
        )

        with flow_run(model_name=str(self.model_name), run_id=self.run_id):
            mlflow.log_dict(self.feature_spec.to_dict(), "feature_spec.json")

        self.next(self.load_champion)

    @step
    def load_champion(self):
        model, version = load_champion_model(str(self.model_name))

        if model is None:
            with flow_run(model_name=str(self.model_name), run_id=self.run_id):
                model, version = bootstrap_champion(
                    X_ref=self.X_ref,
                    y_ref=self.y_ref,
                    model_name=str(self.model_name),
                    run_id=self.run_id,
                    reference_path=str(self.reference_path),
                )
            self.is_bootstrap = True
        else:
            with flow_run(model_name=str(self.model_name), run_id=self.run_id):
                mlflow.set_tag("champion_version", version)
                mlflow.log_dict(
                    {"action": "load_champion", "model_name": str(self.model_name), "version": version},
                    artifact_file="load_champion/decision.json",
                )
            self.is_bootstrap = False

        self.champion_model = model
        self.champion_version = version
        self.next(self.model_gate)

    @step
    def model_gate(self):
        rmse_baseline = get_champion_baseline_rmse(str(self.model_name), str(self.champion_version))

        if rmse_baseline is None:
            raise RuntimeError(
                f"Champion version {self.champion_version} has no '{RMSE_REF_METRIC}' "
                "metric in MLflow. Re-bootstrap or manually log it."
            )

        X_batch_full = pd.concat([self.X_batch_train, self.X_batch_eval], ignore_index=True)
        y_batch_full = np.concatenate([self.y_batch_train, self.y_batch_eval])

        result = evaluate_champion(
            model=self.champion_model,
            X_ref=self.X_ref,
            y_ref=self.y_ref,
            X_batch_full=X_batch_full,
            y_batch_full=y_batch_full,
            X_batch_eval=self.X_batch_eval,
            y_batch_eval=self.y_batch_eval,
            rmse_baseline=rmse_baseline,
        )

        with flow_run(model_name=str(self.model_name), run_id=self.run_id):
            log_model_gate_result(result)

        self.retrain_needed = result.retrain_needed
        self.rmse_champion_eval = result.rmse_champion
        self.retrain_decision = "retrain" if result.retrain_needed else "skip"
        self.next(
            {"retrain": self.retrain, "skip": self.end},  # pyright: ignore[reportArgumentType]
            condition="retrain_decision",
        )

    @step
    def retrain(self):
        window_months = cast(int, self.rolling_window_months)
        training_raw, train_files = build_rolling_training_set(
            historical_dir=Path(str(self.historical_dir)),
            batch_path=Path(str(self.batch_path)),
            batch_train_raw=self.batch_train_raw,
            window_months=window_months,
        )

        candidate_model, candidate_result = train_and_evaluate_candidate(
            training_raw=training_raw,
            feature_spec=self.feature_spec,
            X_batch_eval=self.X_batch_eval,
            y_batch_eval=self.y_batch_eval,
            rmse_champion_eval=self.rmse_champion_eval,
            train_files=train_files,
            window_months=window_months,
        )

        with flow_run(model_name=str(self.model_name), run_id=self.run_id):
            log_candidate_result(candidate_result, candidate_model, self.X_batch_eval)

        self.candidate_model = candidate_model
        self.candidate_rmse = candidate_result.rmse_candidate
        self.candidate_rmse_delta_pct = candidate_result.rmse_delta_pct
        self.next(self.promote)

    @step
    def promote(self):
        candidate_version = register_candidate(
            model_name=str(self.model_name),
            run_id=self.run_id,
            version_tags={
                "role": "candidate",
                "validation_status": "pending",
                "trained_on_batches": str(self.batch_path),
                "eval_batch_id": str(self.batch_path),
                "decision_reason": "pending",
            },
        )

        promotion = evaluate_promotion(
            champion_model=self.champion_model,
            candidate_model=self.candidate_model,
            X_ref=self.X_ref,
            y_ref=self.y_ref,
            rmse_champion_eval=self.rmse_champion_eval,
            rmse_candidate_eval=self.candidate_rmse,
            integrity_warn=bool(self.integrity_warn),
            min_improvement_pct=cast(float, self.min_improvement_pct),
            max_ref_regression_pct=cast(float, self.max_ref_regression_pct),
        )

        demoted_version: str | None = None
        if promotion.promoted:
            demoted_version = promote_to_champion(
                model_name=str(self.model_name),
                candidate_version=candidate_version,
                promotion_reason=promotion.decision_reason,
            )
        else:
            mark_candidate_rejected(
                model_name=str(self.model_name),
                candidate_version=candidate_version,
                decision_reason=promotion.decision_reason,
            )

        with flow_run(model_name=str(self.model_name), run_id=self.run_id):
            log_promotion_result(
                promotion,
                candidate_version=candidate_version,
                demoted_version=demoted_version,
            )

        self.candidate_version = candidate_version
        self.promoted = promotion.promoted
        self.next(self.end)

    @step
    def end(self):
        if self.batch_rejected:
            print(
                "Batch rejected by hard integrity checks. "
                "No evaluation, retrain, or promotion performed."
            )
            return

        if not self.retrain_needed:
            print(
                "Champion still healthy; no retrain triggered. "
                f"rmse_champion={self.rmse_champion_eval:.4f} on batch_eval "
                f"(integrity_warn={self.integrity_warn})."
            )
            return

        if self.promoted:
            print(
                f"Candidate v{self.candidate_version} PROMOTED to @champion. "
                f"rmse_candidate={self.candidate_rmse:.4f} vs "
                f"rmse_champion={self.rmse_champion_eval:.4f} on batch_eval "
                f"(delta={self.candidate_rmse_delta_pct:+.2f}%)."
            )
            return

        print(
            f"Candidate v{self.candidate_version} registered but REJECTED for promotion. "
            f"rmse_candidate={self.candidate_rmse:.4f} vs "
            f"rmse_champion={self.rmse_champion_eval:.4f} on batch_eval "
            f"(delta={self.candidate_rmse_delta_pct:+.2f}%). "
            "Champion alias unchanged."
        )


if __name__ == "__main__":
    MLFlowCapstoneFlow()