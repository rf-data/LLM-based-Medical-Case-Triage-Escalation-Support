from dataclasses import dataclass, field
from typing import Dict, List
import os
from pathlib import Path 
import mlflow
import json
# import inspect
from mlflow.tracking import MlflowClient

import src.utils.general_helper as gh


def create_mlflow_client():
    # configuration - initial setup
    gh.load_env_vars()
    database = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(database) 

    client = MlflowClient()

    return client


def create_mlflow_fingerprint(client, 
                              exp_name=None, 
                              run_name=None,
                              artifact_location=None):
    gh.load_env_vars()
    if not artifact_location:
        artifact_location = os.getenv("MLFLOW_ARTIFACTS")

    if not exp_name:
        exp_name = os.getenv("")

    if not run_name:
        run_name = os.getenv("")

    client.create_experiment(
                    name=run_name,
                    artifact_location=str(artifact_location)
                                    )

    client.set_experiment(exp_name)

    print(f"Created fingerprint experiment ('{exp_name}') and run ('{run_name}')")
    print(f"Use {gh.shorten_path(artifact_location, 3)} as artifact location. ")


def mlflow_fingerprint_check(
    client: MlflowClient,
    experiment_name="_mlflow_fingerprint",
):
    exp = client.get_experiment_by_name(experiment_name)
    assert exp is not None, f"Fingerprint experiment '{experiment_name}' not found"

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.fingerprint = 'true'",
        max_results=1,
    )

    assert len(runs) > 0, "Fingerprint run not found"

    print(f"[FINGERPRINT CHECK] --> SUCCESS: MLflow fingerprint run has {len(runs)} runs.")


@dataclass
class ExperimentLogger:
    experiment_name: str
    artifact_location: Path | None = None   # field(default_factory=Path)
    
    tags: Dict[str, object] = field(default_factory=dict)
    params: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    texts: Dict[str, str] = field(default_factory=dict)
    # dicts: Dict[str, str] = field(default_factory=dict)

    def log_artifact(self, path: str):
        self.artifacts.append(path)

    def log_dict(self, key: str, value: str):
        self.dicts[key] = value

    def log_metric(self, key: str, value: float):
        self.metrics[key] = value

    def log_param(self, key: str, value: object):
        self.params[key] = value

    def log_text(self, name: str, text: str):
        self.texts[name] = text

    def set_tag(self, key: str, value: str):
        self.tags[key] = value

    def setup_experiment(self):
        gh.load_env_vars()
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(mlflow_uri)
        
        exp = mlflow.get_experiment_by_name(
            self.experiment_name
            )

        if exp is None:
            mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=str(self.artifact_location)
                                    )

        mlflow.set_experiment(self.experiment_name)

    # def setup_experiment(self):
    #     if self.artifact_location:
    #         mlflow.create_experiment(
    #             name=self.experiment_name,
    #             artifact_location=str(self.artifact_location)
    #         )
    #     mlflow.set_experiment(self.experiment_name)
    
    # def set_artifact_location(self, folder=None):
    #     if not folder:
    #         self.artifact_location = Path("home/robfra/0_Portfolio_Projekte/LLM/mlflow/artifacts")
    #     else:
    #         self.artifact_location = Path(folder)

    # def set_experiment(self, name=None): # , name_experiment):
    #     self.experiment_name = name

    def flush(self, run_name: str):
        with mlflow.start_run(run_name=run_name):
        # logger.flush()
        
            for k, v in self.params.items():
                mlflow.log_param(k, v)

            for k, v in self.metrics.items():
                mlflow.log_metric(k, v)

            for path in self.artifacts:
                mlflow.log_artifact(path)

            for name, text in self.texts.items():
                mlflow.log_text(text, name)

            for k, v in self.tags.items():
                mlflow.set_tag(k, v)

            print("MLflow logging successful")


def mlflow_logging(log_dict): #, mode="baseline"):
    # configuration
    exp_name = log_dict["experiment_name"]
    arti_loc = log_dict["artifacts_location"]
    logger = ExperimentLogger(
                experiment_name=exp_name,
                artifact_location=arti_loc
                            )
    
    # create experiment
    logger.setup_experiment()

    # logging general information
    logger.set_tag("approach", log_dict["approach"])
    logger.set_tag("version_approach", 
                   log_dict["version_approach"])
    logger.log_param("git commit", 
                     log_dict["git_commit"])
    
    # logging information on dataset
    logger.log_artifact(log_dict["file_path_data"])
    logger.log_artifact(
            log_dict["file_name_data"]
            )
    logger.log_param("name_dataset", 
                     log_dict["name_dataset"])
    logger.log_param("source_dataset", 
                     log_dict["source_dataset"])
    logger.log_param("size_dataset", 
                     log_dict["size_dataset"])
    logger.log_param("version_dataset", 
                     log_dict["version_dataset"])

    # artifacts (file path, dataset, logic, red flags, ...) 
    # fn = log_dict["df_fn"]
    # fn_path = log_dict["file_name_df_fn"]
    
    # gh.ensure_dir(fn_path)
    # fn.to_csv(fn_path, index=False)

    # logger.log_artifact(fn_path)
    fn_path = log_dict["file_name_logic"]
    logic = log_dict["logic"]

    if isinstance(logic, list) and len(logic)==3:
        complete, root, dependent = logic
        _, snapshot_root = root
        
        func_name, snapshot_dependent = dependent
        code_hashed = snapshot_dependent["sha256"]
        code = snapshot_dependent["source"]

        logger.log_text(f"{fn_path}/logic_root.py",
                    snapshot_root["source"]) 
    
    elif isinstance(logic, dict) and len(logic)==1:
        complete = logic
        func_name = logic["name"] 
        code_hashed = logic["sha256"]
        code = logic["source"]

    else: 
        print(f"[ERROR] log_dict['logic'] must contain 1 dict or a list of 3 tuples -- here: {len(log_dict["logic"])} elements of type {[(i, type(e).__name__) for (i, e) in enumerate(log_dict["logic"])]}")
    logger.set_tag(f"{fn_path}/func_name", func_name)
    logger.set_tag(f"{fn_path}/code_hashed", code_hashed)
    logger.log_text(f"{fn_path}/logic_escalation.py",
                    code)  
    logger.log_text(
                f"{fn_path}/logic_complete.json",
                json.dumps(complete,
                        indent=2, ensure_ascii=False)
                        )
    
    
    if "red_flags" in log_dict.keys():
        logger.log_text(
                log_dict["file_name_red_flags"],
                json.dumps(log_dict["red_flags"],
                        indent=2, ensure_ascii=False)
                        )
    if "prompt" in log_dict.keys():
        logger.log_text(
                log_dict["file_name_prompt"], 
                log_dict["prompt"]
                )
    if "allowed_values" in log_dict.keys():
        logger.log_text(
                log_dict["file_name_values"], 
                log_dict["allowed_values"]
                )
    if "json_scheme" in log_dict.keys():
        logger.log_text(
                log_dict["file_name_json_scheme"], 
                json.dumps(log_dict["json_scheme"], 
                        indent=2, ensure_ascii=False)
                        )
    
    # logging metrics
    logger.log_metric("runtime_total_sec", 
                      log_dict["runtime_total_sec"])
    logger.log_metric("runtime_per_sample_sec", 
                      log_dict["runtime_per_sample_sec"])

    report = log_dict["class_report"]
    tn, fp, fn, tp = log_dict["confusion_matrix"]
    metrics = log_dict["class_metrics"]
    precision, recall, f1 = metrics[0], metrics[1], metrics[2]

    logger.log_text(
                "ClassReport.json",
                json.dumps(report, indent=2)
            )
    logger.log_metric("precision", precision)
    logger.log_metric("precision_escalation", 
                      report["escalation"]["precision"])
    logger.log_metric("recall", recall)
    logger.log_metric("recall_escalation", 
                      report["escalation"]["recall"])
    logger.log_metric("f1", f1)
    logger.log_metric("true_negatives", tn)
    logger.log_metric("true_positives", tp)
    logger.log_metric("false_negatives", fn)
    logger.log_metric("false_positives", fp)

    # logging additional infos
    if "notes" in log_dict.keys():
        logger.log_param("additional_infos", log_dict["notes"])
    
    # execute logging
    logger.flush(log_dict["run_name"])

"""
mlflow.set_tag("approach", "rule_based")
mlflow.set_tag("dataset", "ambiguous")
mlflow.set_tag("stage", "baseline")
mlflow.set_tag("owner", "robfra")
"""