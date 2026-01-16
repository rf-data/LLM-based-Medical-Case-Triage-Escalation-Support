# src/core/mlflow_logger.py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
import os
import mlflow

# from src.utils.experiment_logger_impl import ExperimentLogger
import src.utils.general_helper as gh
import src.utils.logger as log 

# --------------
# MLflow Experiment Logger
# --------------

@dataclass
class ExperimentLogger:
    backup_dir: Path = Path(f"{gh.find_project_root()}/mlflow/backups")
    experiment_name: str
    artifact_location: Path | None = None   # field(default_factory=Path)
    
    tags: Dict[str, object] = field(default_factory=dict)
    params: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    texts: Dict[str, str] = field(default_factory=dict)
    # dicts: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        root = gh.find_project_root()
        project_name = os.getenv("PROJECT_NAME", "default_project")
        folder = f"{root}/mlflow/logs"    

        self.logger = log.create_logger(f"mlflow_{project_name}", 
                                        "mlflow", 
                                        folder=folder)

    def local_backup(self, folder=None):
        """Saves all logged data to local files for backup purposes."""
        if not folder:
            folder = Path(f"{self.backup_dir}/{self.experiment_name}")
            # gh.ensure_dir(folder)
        # backup_dir.mkdir(parents=True, exist_ok=True)

        # Save params
        params_path = folder / "params.json"
        gh.save_dict(params_path, self.params)

        # Save metrics
        metrics_path = folder / "metrics.json"
        gh.save_dict(metrics_path, self.metrics)

        # Save texts
        texts_path = folder / "texts.json"
        gh.save_dict(texts_path, self.texts)

        self.logger.info(f"Local backup saved to {folder}")

    def log_artifact(self, path: str):
        self.artifacts.append(path)

    # def log_dict(self, key: str, value: str):
    #     self.dicts[key] = value

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

        self.logger.info(
                f"Setting up MLflow experiment: {self.experiment_name}"
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
        self.logger.info(
                f"Starting MLflow run: {run_name} "
                f"(experiment={self.experiment_name})"
                        )
        
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            self.logger.info(f"MLflow run_id={run_id}")

            for k, v in self.params.items():
                mlflow.log_param(k, v)
                self.logger.info(f"[PARAM] {k}={v}")

            for k, v in self.metrics.items():
                mlflow.log_metric(k, v)
                self.logger.info(f"[METRIC] {k}={v}")

            for path in self.artifacts:
                mlflow.log_artifact(path)
                self.logger.info(f"[ARTIFACT] {path}")

            for name, text in self.texts.items():
                mlflow.log_text(text, name)
                self.logger.info(f"[TEXT] {name} (len={len(text)})")

            for k, v in self.tags.items():
                mlflow.set_tag(k, v)
                self.logger.info(f"[TAG] {k}={v}")

            self.logger.info(
                    f"MLflow run finished successfully: {run_name} ({run_id})"
                            )

# ---------------
# Main MLflow logging function
# ---------------

_experiment_logger: ExperimentLogger | None = None

def get_experiment_logger(
    experiment_name: str | None = None,
    artifact_location: Path | None = None,
) -> ExperimentLogger:
    global _experiment_logger

    if _experiment_logger is None:
        if experiment_name is None:
            raise RuntimeError(
                "ExperimentLogger not initialized yet. "
                "Call get_experiment_logger(experiment_name=...) once."
            )

        _experiment_logger = ExperimentLogger(
            experiment_name=experiment_name,
            artifact_location=artifact_location,
        )

    return _experiment_logger