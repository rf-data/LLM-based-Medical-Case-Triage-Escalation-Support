# from dataclasses import dataclass, field
# from typing import Dict, List
import os
from pathlib import Path 
import mlflow
# import json
from datetime import datetime
from joblib import dump #, load
# import pandas as pd
# import logging
# import inspect
from mlflow.tracking import MlflowClient

import src.utils.path_helper as ph
import src.utils.general_helper as gh
# from core.session import session
from src.core.logger import create_logger
from core.mlflow_logger import get_experiment_logger


def save_model(model, model_name, folder=None):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    # 
    if folder is None:
        folder = os.getenv("PATH_MODEL")

    ph.ensure_dir(folder)
    model_path = Path(f"{folder}/{model_name}.joblib")
    dump(model, model_path)

    # logging
    event_logger.info("Model saved as '.../%s'", ph.shorten_path(model_path))

    return


def create_mlflow_client(initial=False):
    # configuration - initial setup
    gh.load_env_vars()
    
    database = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(database) 

    start_time = datetime.now()

    client = MlflowClient()

    root = ph.find_project_root()
    project_name = os.getenv("PROJECT_NAME", "default_project")
    folder = f"{root}/mlflow/logs"    
    mlflow_log = create_logger(name=f"mlflow_{project_name}", file_name="mlflow", folder=folder)
    
    if initial:
        start_text = f"""
{"=" * 80}
[START] {start_time.isoformat()} | Database '{database}'
{"=" * 80}
"""
        mlflow_log.info(start_text) 
    
    mlflow_log.info(f"Initial MLflow client setup completed.")

    return client, mlflow_log


def mlflow_fingerprint_check(
    client: MlflowClient,
    logger,
    experiment_name = None,
):
    if not experiment_name:
        gh.load_env_vars()
        experiment_name = os.getenv("FINGERPRINT_EXP")
        # raise ValueError(
        #     "mlflow_fingerprint_check(): experiment_name must be a non-empty string"
        # )
    
    exp = client.get_experiment_by_name(experiment_name)
    assert exp is not None, logger.error(f"Fingerprint experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id]
                    )

    assert len(runs) > 0, logger.error("Fingerprint run not found")

    logger.info(f"[FINGERPRINT CHECK] --> SUCCESS: MLflow fingerprint run has {len(runs)} runs.")



# def mlflow_logging(log_dict): #, mode="baseline"):
#     # configuration
#     exp_name = log_dict["experiment_name"]
#     arti_loc = log_dict["artifacts_location"]
#     logger = ExperimentLogger(
#                 experiment_name=exp_name,
#                 artifact_location=arti_loc
#                             )
    
#     # create experiment
#     logger.setup_experiment()

#     # logging general information
#     logger.set_tag("approach", log_dict["approach"])
#     logger.set_tag("version_approach", 
#                    log_dict["version_approach"])
#     logger.log_param("git commit", 
#                      log_dict["git_commit"])
    
#     # logging information on dataset
#     logger.log_artifact(log_dict["file_path_data"])
#     logger.log_artifact(
#             log_dict["file_name_data"]
#             )
#     logger.log_param("name_dataset", 
#                      log_dict["name_dataset"])
#     logger.log_param("source_dataset", 
#                      log_dict["source_dataset"])
#     logger.log_param("size_dataset", 
#                      log_dict["size_dataset"])
#     logger.log_param("version_dataset", 
#                      log_dict["version_dataset"])

#     # artifacts (file path, dataset, logic, red flags, ...) 
#     # fn = log_dict["df_fn"]
#     # fn_path = log_dict["file_name_df_fn"]
    
#     # gh.ensure_dir(fn_path)
#     # fn.to_csv(fn_path, index=False)

#     # logger.log_artifact(fn_path)
#     version = config["vers_logic"]
#     fn_path = f"logic/{version}"
#     logic = log_dict["logic"]

#     if isinstance(logic, list) and len(logic)==3:
#         complete, root, dependent = logic
#         _, snapshot_root = root
        
#         func_name, snapshot_dependent = dependent
#         code_hashed = snapshot_dependent["sha256"]
#         code = snapshot_dependent["source"]

#         logger.log_text(f"{fn_path}/logic_root.py",
#                     snapshot_root["source"]) 
    
#     elif isinstance(logic, dict) and len(logic)==1:
#         complete = logic
#         func_name = logic["name"] 
#         code_hashed = logic["sha256"]
#         code = logic["source"]

#     else: 
#         logger.logger.error(f"log_dict['logic'] must contain 1 dict or a list of 3 tuples -- here: {len(log_dict["logic"])} elements of type {[(i, type(e).__name__) for (i, e) in enumerate(log_dict["logic"])]}")
    
#     # escalation_rule = log_dict["escalation_rule"]

#     logger.set_tag("source", f"{func_name} ({version})")       # {fn_path}/
#     logger.set_tag("source_hashed", code_hashed)    # {fn_path}/code_hashed"
    
#     logger.log_text(f"{fn_path}/logic_escalation.py",
#                     code)  
#     logger.log_text(
#                 f"{fn_path}/logic_complete.json",
#                 json.dumps(complete,
#                         indent=2, ensure_ascii=False)
#                         )
    
#     rules = log_dict.get("escalation_rules", None)
#     if rules:
#         rule_name = rules["name"]
#         logger.set_tag("escalate_rule", rule_name)

#         rule_hashed = rules["sha256"]
#         logger.set_tag("escalate_rule_hashed", rule_hashed)

#         rules_code = rules["source"]
#         logger.log_text(f"{fn_path}/logic_rules.py",
#                     rules_code) 

#     if "red_flags" in log_dict.keys():
#         logger.log_text(
#                 log_dict["file_name_red_flags"],
#                 json.dumps(log_dict["red_flags"],
#                         indent=2, ensure_ascii=False)
#                         )
#     if "prompt" in log_dict.keys():
#         logger.log_text(
#                 log_dict["file_name_prompt"], 
#                 log_dict["prompt"]
#                 )
#     if "allowed_values" in log_dict.keys():
#         logger.log_text(
#                 log_dict["file_name_values"], 
#                 log_dict["allowed_values"]
#                 )
#     if "json_scheme" in log_dict.keys():
#         logger.log_text(
#                 log_dict["file_name_json_scheme"], 
#                 json.dumps(log_dict["json_scheme"], 
#                         indent=2, ensure_ascii=False)
#                         )
    
#     # logging metrics
#     logger.log_metric("runtime_total_sec", 
#                       log_dict["runtime_total_sec"])
#     logger.log_metric("runtime_per_sample_sec", 
#                       log_dict["runtime_per_sample_sec"])

#     report = log_dict["class_report"]
#     tn, fp, fn, tp = log_dict["confusion_matrix"]
#     metrics = log_dict["class_metrics"]
#     precision, recall, f1 = metrics[0], metrics[1], metrics[2]

#     logger.log_text(
#                 "ClassReport.json",
#                 json.dumps(report, indent=2)
#             )
#     logger.log_metric("precision", precision)
#     logger.log_metric("precision_escalation", 
#                       report["escalation"]["precision"])
#     logger.log_metric("recall", recall)
#     logger.log_metric("recall_escalation", 
#                       report["escalation"]["recall"])
#     logger.log_metric("f1", f1)
#     logger.log_metric("true_negatives", tn)
#     logger.log_metric("true_positives", tp)
#     logger.log_metric("false_negatives", fn)
#     logger.log_metric("false_positives", fp)

#     # logging additional infos
#     if "notes" in log_dict.keys():
#         logger.log_param("additional_infos", log_dict["notes"])
    
#     # execute logging
#     logger.flush(log_dict["run_name"])

"""
mlflow.set_tag("approach", "rule_based")
mlflow.set_tag("dataset", "ambiguous")
mlflow.set_tag("stage", "baseline")
mlflow.set_tag("owner", "robfra")
"""