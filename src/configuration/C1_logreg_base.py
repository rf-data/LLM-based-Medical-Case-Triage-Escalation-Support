import os
from utils.preprocess_helper import prepare_reports
import utils.general_helper as gh


config = {
    "tags": {
        "approach": "LogReg",
        "subapproach": "baseline",
        "vers_approach": "v1",
        "vers_data": "v2",
        "vers_logic": "llm v4",
        "vers_preprocess": "v1"
    },
    "parameter": {
        "git_commit": gh.get_git_commit(),
        "source_dataset": "synthetic",
        "class_weight": "balanced",
        "max_iter": int(1000),
        "solver": "liblinear",
        "random_state": 42,
        "model_name": "LogReg_OHE",
        "model_version": "v1", 
        "model_id": "m-206ce16a51214e629b162c488d9a1ed4"
    },
    "mode": "LogReg",
    "preprocess_function": prepare_reports,
    "experiment_name": "escalation_ML_models",
    "artifacts_location": os.getenv("MLFLOW_ARTIFACTS"),
}

num_feats = [
        "confidence",
        "n_risk_factors",
        "n_missing_information"
        ]

cat_feats = [
        "severity",
        "uncertainty_level",
        "clarity",
        "domain"
        ]