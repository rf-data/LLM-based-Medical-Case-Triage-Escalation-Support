from pathlib import Path
from src.utils.mlflow_helper import mlflow_logging
import src.utils.general_helper as gh

dict_path = Path("/home/robfra/0_Portfolio_Projekte/LLM/logs/back_up_logdict.json")

backup_dict = gh.load_dict(dict_path)

mlflow_logging(backup_dict)