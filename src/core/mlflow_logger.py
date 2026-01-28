# src/core/mlflow_logger.py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
import os
import mlflow

# from utils.experiment_logger_impl import ExperimentLogger
import utils.general_helper as gh
import core.logger as log 
from core.session import session
import utils.file_helper as fh
import utils.path_helper as ph

# --------------
# MLflow Experiment Logger
# --------------

@dataclass
class ExperimentLogger:
    experiment_name: str
    artifact_location: Path | None = None   # field(default_factory=Path)
    backup_dir: Path = Path(f"{ph.find_project_root()}/mlflow/backups")
    
    tags: Dict[str, object] = field(default_factory=dict)
    params: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    texts: Dict[str, str] = field(default_factory=dict)
    # dicts: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        root = ph.find_project_root()
        project_name = os.getenv("PROJECT_NAME", "default_project")
        folder = f"{root}/mlflow/logs"    

        self.logger = log.create_logger(f"mlflow_{project_name}", 
                                        "mlflow", 
                                        folder=folder)

    def load_latest_backup(self, folder: Path | None = None):
        folder = Path(folder or self.backup_dir / self.experiment_name)

        files = list(folder.glob("*_params.json"))
        if not files:
            raise FileNotFoundError(f"No backups found in {folder}")

        latest = max(files, key=lambda p: p.stem.split("_")[0])
        timestamp = latest.stem.replace("_params", "")

        self.load_logger(folder, timestamp)
        
        # folder = Path(folder) or Path(self.backup_dir / self.experiment_name)

        # backup = {}
        # for name in ["params", "metrics", "texts"]:
        #     files = list(folder.glob(f"*_{name}.json"))

        #     if not files: 
        #         raise FileNotFoundError(f"No backups of '{name}' found in {folder}")

            # backup[f"{name}"] = files 


    def load_logger(self, folder: Path, timestamp: str):
        """
        Load logger state (tags, params, metrics, texts) from a local backup.
        
        Parameters
        ----------
        folder : Path
            Backup folder (e.g. mlflow/backups/<experiment_name>)
        timestamp : str
            Timestamp prefix used in backup filenames
        """
        self.logger.info("Start loading logger: {folder} | {timestamp}")

        folder = Path(folder)
        missing = False
        attributes = ["params", "metrics", "texts", "tags"]

        for attr in attributes:  
            path = folder / f"{timestamp}_{attr}.json"  

            if path.exists():
                setattr(self, attr, fh.load_dict(path))

            else:
                self.logger.warning(f"No {attr} backup found at {path}")
                missing = True

        if missing:
            self.logger.warning(
                f"Not all backups found in {folder}")
            raise FileNotFoundError()

        self.logger.info(
            f"ExperimentLogger state restored from backup "
            f"(timestamp={timestamp})"
        )
        # if metrics_path.exists():
        #     self.metrics = fh.load_dict(metrics_path)
        # else:
        #     self.logger.warning(f"No metrics backup found at {metrics_path}")
        #     missing = True

        # if texts_path.exists():
        #     self.texts = fh.load_dict(texts_path)
        # else:
        #     self.logger.warning(f"No texts backup found at {texts_path}")
        #     missing = True
    
    def local_backup(self, folder=None):
        """Saves all logged data to local files for backup purposes."""
        if not folder:
            folder = Path(f"{self.backup_dir}/{self.experiment_name}")
            # gh.ensure_dir(folder)
        # backup_dir.mkdir(parents=True, exist_ok=True)

        now = session.now

        # Save tags
        tags_path = folder / f"{now}_tags.json"
        fh.save_dict(tags_path, self.tags)

        # Save params
        params_path = folder / f"{now}_params.json"
        fh.save_dict(params_path, self.params)

        # Save metrics
        metrics_path = folder / f"{now}_metrics.json"
        fh.save_dict(metrics_path, self.metrics)

        # Save texts
        texts_path = folder / f"{now}_texts.json"
        fh.save_dict(texts_path, self.texts)

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