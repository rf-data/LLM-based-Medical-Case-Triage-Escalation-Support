# 
import os
import mlflow

import src.utils.general_helper as gh

# configuration
gh.load_env_vars()
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(mlflow_uri)
assert mlflow.get_tracking_uri().startswith("http")

mlflow.set_experiment("restart_check")
with mlflow.start_run():
    mlflow.log_metric("ping", 1)
