import os
from utils.preprocess_helper import prepare_reports
import utils.general_helper as gh


config = {
    "tags": {
        "approach": "LogReg",
        "subapproach": "HyperOpt",
        "vers_approach": "v4",
        "vers_data": "v2",
        "vers_logic": "llm v4",
        "vers_preprocess": "v1"
    },
    "parameter": {
        "git_commit": gh.get_git_commit(),
        "source_dataset": "synthetic",
        "class_weight": "n.a.",
        "max_iter": int(1000),
        "solver": "liblinear",
        "random_state": 42,
        "model_name": "LogReg_OHE",
        "model_version": "v1", 
        "model_id": "m-206ce16a51214e629b162c488d9a1ed4",
        "cross_validate": "Yes",
        "group_split": "Yes",
        "split_mode_hyperopt": "group_shuffle",
        "split_mode_validation": "group_kfold",
        "test_size": 0.2,
        "n_splits": 30,
        "n_folds": 5,
        "clf__class_weight": [None, "balanced"],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__l1_ratio": [0, 1],
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

clf__C = config["parameter"]["clf__C"]      # [0.01, 0.1, 1.0, 10.0]
clf__class_weight = config["parameter"]["clf__class_weight"]     # [None, "balanced"]
clf__l1_ratio = config["parameter"]["clf__l1_ratio"]     # ["l2", "l1"]

param_grid = {
        "clf__C": clf__C,
        "clf__l1_ratio": clf__l1_ratio,
        "clf__class_weight": clf__class_weight,
    }
