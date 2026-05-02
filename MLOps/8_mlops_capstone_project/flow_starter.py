import mlflow
from metaflow.flowspec import FlowSpec
from metaflow.parameters import Parameter
from metaflow.decorators import step

from lib.integrity import hard_is_ok, run_hard_integrity_checks, run_soft_integrity_checks
from lib.mlflow_log import log_integrity_result
from lib.green_taxi_schema import GREEN_TAXI_SCHEMA
from lib.helper import flow_run, init_mlflow, load_batch, load_reference


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

        self.next(self.soft_integrity_gate if hard_is_ok(hard_check_result) else self.end)

    @step
    def soft_integrity_gate(self):
        soft_check_result = run_soft_integrity_checks(
            reference=self.ref,
            batch=self.batch,
        )

        with flow_run(model_name=str(self.model_name), run_id=self.run_id):
            log_integrity_result(soft_check_result, check="soft")

        self.next(self.load_champion)

    @step
    def load_champion(self):
        # TODO: Add relevant steps and flow logic.
        self.next(self.end)
    
    @step
    def end(self):
        print("Batch data failed integrity checks. Ending flow.")


if __name__ == "__main__":
    MLFlowCapstoneFlow()