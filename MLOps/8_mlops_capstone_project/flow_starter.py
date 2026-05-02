import mlflow
from metaflow.flowspec import FlowSpec
from metaflow.parameters import Parameter
from metaflow.decorators import step

from lib.integrity import hard_is_ok, run_hard_integrity_checks, run_soft_integrity_checks
from lib.mlflow_log import log_integrity_result
from lib.green_taxi_schema import GREEN_TAXI_SCHEMA
from lib.helper import flow_run, init_mlflow, load_batch, load_reference
from lib.features import FeatureSpec, engineer_features, fit_feature_spec
from lib.model_registry import (
    bootstrap_champion,
    load_champion_model,
)


class MLFlowCapstoneFlow(FlowSpec):
    reference_path = Parameter("reference-path")
    batch_path = Parameter("batch-path")
    taxi_zone_lookup_path = Parameter("taxi-zone-lookup-path", default="TLC_Data/NYC_Taxi_Zones.geojson", required=False)
    model_name = Parameter("model-name", default="green_taxi_tip_model")

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
        self.next(self.soft_integrity_gate if ok else self.end)

    @step
    def soft_integrity_gate(self):
        soft_check_result = run_soft_integrity_checks(
            reference=self.ref,
            batch=self.batch,
        )

        with flow_run(model_name=str(self.model_name), run_id=self.run_id):
            log_integrity_result(soft_check_result, check="soft")

        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        self.feature_spec: FeatureSpec = fit_feature_spec(self.ref)

        self.X_ref, self.y_ref = engineer_features(self.ref, self.feature_spec)
        self.X_batch, self.y_batch = engineer_features(self.batch, self.feature_spec)

        with flow_run(model_name=str(self.model_name), run_id=self.run_id):
            mlflow.log_dict(self.feature_spec.to_dict(), "feature_spec.json")

        self.next(self.load_champion)

    @step
    def load_champion(self):
        model, version = load_champion_model(str(self.model_name))

        if model is None:
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
        # TODO: Step E - evaluate champion on batch, decide whether to retrain.
        self.next(self.end)

    @step
    def end(self):
        if self.batch_rejected:
            print("Batch data failed hard integrity checks. Flow ended without evaluation.")
        else:
            print("Flow complete.")


if __name__ == "__main__":
    MLFlowCapstoneFlow()