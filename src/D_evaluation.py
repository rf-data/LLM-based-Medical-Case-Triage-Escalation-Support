
##
# import json
import pandas as pd
from pathlib import Path
import os

import utils.general_helper as gh
import utils.path_helper as ph
import utils.file_helper as fh
import utils.evaluation_helper as eval
# import utils.escalation_helper as esc
from core.session import session
from core.mlflow_logger import get_experiment_logger

def evaluate_escalation(df, pred_column):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    # encode ?
    y_true, y_pred = eval.encode_labels(df, pred_column)

    # create ClassReport, ConfMatrix, ClassMetrics
    report = eval.create_classification_report(y_true, y_pred)
    cm = eval.create_confusion_matrix(y_true, y_pred)
    metrics = eval.create_metrics(y_true, y_pred)

    # save files locally
    now = session.now
    mode = session.mode
    version_run = session.tags.get("vers_approach", "tba") 
    
    folder = os.getenv("PATH_EVALUATED", None)
    if folder is None:
        gh.load_env_vars()
        folder = os.getenv("PATH_EVALUATED", None)
    
    path_cm = Path(f"{folder}/ConfMatrix/{now}_{mode}_{version_run}_cm.json")
    path_cr = Path(f"{folder}/ClassReport/{now}_{mode}_{version_run}_cr.json")
    path_metrics = Path(f"{folder}/metrics/{now}_{mode}_{version_run}_metrics.json")
    # path_fn_fp = Path(f"{folder}/fn_fp") # /{now}_{mode}_{version_run}_metrics.json") 
    
    for path, file in zip([path_cm, path_cr, path_metrics],
                        [cm, report, metrics]):
        if not path.exists():
            ph.ensure_dir(path)
            fh.save_dict(path, file)
        else: 
            event_logger.error("File '%s' already exists. Hence, no overwrite", 
                               path)
    
if __name__ == "__main__":
    evaluate_escalation()