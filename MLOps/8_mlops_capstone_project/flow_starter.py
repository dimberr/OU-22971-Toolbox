from metaflow.flowspec import FlowSpec
from metaflow.parameters import Parameter
from metaflow.decorators import step
import mlflow
import pandas as pd

def init_mlflow(model_name):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(model_name)


def load_reference(reference_path):
    # Load reference dataset from the specified path
    ref_df = pd.read_parquet(reference_path, engine='pyarrow')
    return ref_df

def load_batch(batch_path):
    # Load batch dataset from the specified path
    batch_df = pd.read_parquet(batch_path, engine='pyarrow')
    return batch_df

def run_integrity_checks(ref, batch) -> tuple[bool, dict]:
    # Run data integrity checks using hard checks and NannyML
    # Return a boolean indicating if checks passed and a report
    return True, {"report": "All checks passed."}



class MLFlowCapstoneFlow(FlowSpec):
    reference_path = Parameter("reference-path")
    batch_path = Parameter("batch-path")
    model_name = Parameter("model-name", default="green_taxi_tip_model")

    @step
    def start(self):
        init_mlflow(self.model_name)
        self.next(self.load_data)

    @step
    def load_data(self):
        self.ref, self.batch = load_reference(self.reference_path), load_batch(self.batch_path)
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        ok, report = run_integrity_checks(self.ref, self.batch)  # hard + NannyML
        self.next(self.load_champion)# if ok else self.end)

    @step
    def load_champion(self):
        # TODO: Add relevant steps and flow logic.
        self.next(self.end)
    
    @step
    def end(self):
        print("Batch data failed integrity checks. Ending flow.")


if __name__ == "__main__":
    MLFlowCapstoneFlow()