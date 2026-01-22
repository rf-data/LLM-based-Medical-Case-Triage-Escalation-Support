##
import json
import pandas as pd
from pathlib import Path
import os
from sklearn.metrics import (precision_recall_fscore_support,
                             classification_report, 
                             confusion_matrix)
import src.utils.general_helper as gh
import src.utils.path_helper as ph
import src.utils.file_helper as fh
from src.core.session import session
from src.core.mlflow_logger import get_experiment_logger


def encode_labels(df):
    LABEL_TO_INT = {
            "no_escalation": 0,
            "escalation": 1,
                }

    BOOL_TO_INT = {
                False: 0,
                True: 1,
                }

    mode = session.mode
    version = session.tags.get("vers_approach", "tba")

    y_true = df["expected_action"].map(LABEL_TO_INT)
    y_pred = df[f"pred_{mode}_{version}"].map(BOOL_TO_INT)
    
    # check for invalid values
    if y_true.isna().any():
        raise ValueError("Unknown label in expected_action")

    if y_pred.isna().any():
        raise ValueError("Invalid prediction values")

    return y_true, y_pred


def create_evaluation_metrics(y_true, y_pred): 
    # setup logger
    logger = get_experiment_logger()
    event_logger = logger.logger

    # creation of ClassReport
    report = classification_report(y_true, 
                                    y_pred, 
                                    labels=[0, 1],
                                    target_names=["no_escalation", "escalation"],
                                    zero_division=0,
                                    output_dict=True)

    # log ClassReport
    logger.log_text(
                "ClassReport.json",
                json.dumps(report, indent=2)
            )
    

    logger.log_metric("precision_escalation", 
                      report["escalation"]["precision"])
    
    logger.log_metric("recall_escalation", 
                      report["escalation"]["recall"])
    
    # create ConfMatrix
    tn, fp, fn, tp = confusion_matrix(                  # tn, fp, fn, tp
                        y_true, 
                        y_pred, 
                        labels=[False, True]
                        ).ravel()

    event_logger.info(f"false_negatives:\t{fn}")
    event_logger.info(f"false_positives:\t{fp}")

    cm = {
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }
    # log values from ConfMatrix
    logger.log_text("ConfMatrix.json", json.dumps(cm, indent=2))
    logger.log_metric("true_negatives", tn)
    logger.log_metric("true_positives", tp)
    logger.log_metric("false_negatives", fn)
    logger.log_metric("false_positives", fp)

    # compile Classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(          # precision, recall, f1, support
                                            y_true,
                                            y_pred,
                                            average="binary",
                                            pos_label=True,
                                            zero_division=0
                                            )

    metrics = {
        "precision": int(precision), 
        "recall": int(recall), 
        "f1": int(f1)
    }

    event_logger.info(f"precision: {precision}")
    event_logger.info(f"recall: {recall}")
    event_logger.info(f"f1: {f1}")

    # log ClassMetrics 
    logger.log_metric("precision", precision)
    logger.log_metric("recall", recall)
    logger.log_metric("f1", f1)


    # save files locally
    now = session.now
    mode = session.mode
    version_run = session.tags.get("vers_approach", "tba") 
    
    folder = os.getenv("PROCESSED", None)
    if folder is None:
        gh.load_env_vars()
        folder = os.getenv("PROCESSED", None)
    
    path_cm = Path(f"{folder}/ConfMatrix/{now}_{mode}_{version_run}_cm.json")
    path_cr = Path(f"{folder}/ClassReport/{now}_{mode}_{version_run}_cr.json")
    path_metrics = Path(f"{folder}/metrics/{now}_{mode}_{version_run}_metrics.json")
    
    for path, file in zip([path_cm, path_cr, path_metrics],
                        [cm, report, metrics]):
        if not path.exists():
            ph.ensure_dir(path)
            fh.save_dict(path, file)
        else: 
            event_logger.error(f"File '{path}' already exists. Hence, no overwrite")
    return 

def need_for_escalation(df_input): # : pd.DataFrame -> pd.DataFrame:
    """
    Determine if escalation is needed based on information in df.

    Args:
        df (pd.DataFrame): DataFrame containing level of 'expected_action', 'severity', 
        'uncertainty' and 'confidence'.
    """
    df = df_input.copy()

    def escalation_postprocess(
                        exp_action: bool, 
                        severity: str, 
                        confidence: float,
                        uncertainty: str) -> bool:
        # default: trust the LLM
        real_action = exp_action
        
        if exp_action == True:
            if (
            severity == "low"
            and confidence < 0.4
            and uncertainty == "high"
                ):
                real_action = False
            
        return real_action
    
 # if (
            # severity == "low"
            # or confidence < 0.4
            # or uncertainty == "high"
            #     ):
            #     real_action_2 = False

    REQUIRED_COLS = ["severity", "confidence", "uncertainty", "expected_action"]
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None

    df["expected_action"] = df.apply(
                            lambda row: escalation_postprocess(
                                row.get("expected_action", "unknown"),
                                row.get("severity", "unknown"), 
                                row.get("confidence", "unknown"), 
                                row.get("uncertainty", "unknown")), 
                                axis=1)
                                
    return df


def evaluate_escalation(df_input):
    # start logger
    logger = get_experiment_logger()

    version = session.tags.get("vers_logic", "tba")
    fn_path = f"logic/{version}"
    df = df_input.copy()

    df_esc = need_for_escalation(df)
    snapshot_dict = gh.snapshot_single_function(need_for_escalation)

    # log
    logger.set_tag("post_process_rule", snapshot_dict["name"])
    logger.set_tag("post_process_rule_hashed", snapshot_dict["sha256"])
    logger.log_text(f"{fn_path}/logic_rules.py",
                    snapshot_dict["source"])
    
    y_true, y_pred = encode_labels(df_esc)
    create_evaluation_metrics(y_true, y_pred)