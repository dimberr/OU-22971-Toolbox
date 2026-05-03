import contextlib
import os
from collections.abc import Iterator

import mlflow
import pandas as pd


DEFAULT_MLFLOW_TRACKING_URI = "http://localhost:5000"


def init_mlflow(model_name: str) -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(model_name)


@contextlib.contextmanager
def flow_run(model_name: str, run_id: str) -> Iterator[None]:
    """Re-init mlflow tracking and reattach to the parent flow run for this step.

    Each Metaflow step may execute in a fresh subprocess, so the tracking URI
    and experiment have to be re-set, and the parent run has to be reattached
    by id. All step-level logging then lands on the single flow-scoped run.

    On any exception inside the wrapped block, the run is tagged
    ``flow_status=interrupted`` before the exception propagates. The outer
    ``mlflow.start_run`` context will then mark the run lifecycle as FAILED.
    Together they give the UI an accurate "this run did not complete" signal
    even when a later step (e.g. ``end``) never gets a chance to run.
    """
    init_mlflow(model_name)
    with mlflow.start_run(run_id=run_id):
        try:
            yield
        except BaseException:
            try:
                mlflow.set_tag("flow_status", "interrupted")
            except Exception:
                pass
            raise


# Reasons to split this into a separate function:
# 1. Reference loading might require special handling: sampling a large historical dataset,
#    filtering to only the training window, applying specific column subset, etc.
# 2. Batch loading might require also filtering, deduplication, different validations, etc.
# 3. Failures in reference loading and in batch loading can be handled differently.
# 4. For now it's simple, but it's a good practice to separate the concerns.
def load_reference(reference_path):
    return pd.read_parquet(reference_path)


def load_batch(batch_path):
    return pd.read_parquet(batch_path)
