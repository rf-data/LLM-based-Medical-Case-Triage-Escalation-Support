from pathlib import Path
import os
from src.core.mlflow_logger import get_experiment_logger
import src.utils.file_helper as fh
import src.utils.general_helper as gh

# exp_name = 
# path = Path(folder) / f"{exp_name}"

# _old = fh.load_dict(path)
# 

def manual_logging(exp_name, arti_path, folder, run_name):
    logger = get_experiment_logger(
                experiment_name=exp_name,
                artifact_location=arti_path
                            )

    logger.load_latest_backup(folder)
    logger.set_tag("recovered_from_backup", "true")

    logger.flush(run_name=run_name)


if __name__ == "__main__":
    gh.load_env_vars()

    snapshot_path = Path("/home/robfra/0_Portfolio_Projekte/LLM/mlflow/backups/escalation_clinical_reports/2026-01-21_11:12:05_session_snapshot.json")
    _old = fh.load_dict(snapshot_path)

    exp_name = _old["experiment_name"]
    arti_path = _old["artifacts_location"]
    approach = _old["tags"]["approach"]
    now = _old["now"]

    f_backup = os.getenv("MLFLOW_BACKUP")
    folder = Path(f"{f_backup}/{exp_name}")
    # run_name = os.getenv("PROJECTNAME") 

    run_name = f"{approach}_{now}_recov"
    manual_logging(exp_name, arti_path, folder, run_name)
