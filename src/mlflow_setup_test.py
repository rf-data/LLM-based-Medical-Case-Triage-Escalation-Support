# 
import os
from mlflow.tracking import MlflowClient
import mlflow

import src.utils.general_helper as gh
import src.utils.mlflow_helper as mh

# configuration
gh.load_env_vars()
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(mlflow_uri)
assert mlflow.get_tracking_uri().startswith("http")

client = MlflowClient()

# --- sanity check: existing experiments ---
experiments = client.search_experiments()
assert len(experiments) > 0, "MLflow reachable, but no experiments found"

print(f"[SANITY CHECK] --> SUCCESS: MLflow has {len(experiments)} experiments")

# --- write check ---
mlflow.set_experiment("restart_check")
with mlflow.start_run():
    mlflow.log_metric("ping", 1)

print("[WRITE TEST] --> SUCCESS")

# --- fingerprint check ---
exp_name = os.getenv("NAME_FINGERPRINT")

mh.mlflow_fingerprint_check(client, exp_name)
