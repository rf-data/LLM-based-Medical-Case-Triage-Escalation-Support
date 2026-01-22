##
import json
import pandas as pd
from sklearn.metrics import (precision_recall_fscore_support,
                             classification_report, 
                             confusion_matrix)
import src.utils.general_helper as gh
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

    print("false_negatives", fn)
    print("false_positives", fp)

    cm = {
        "true_negatives": tn,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
    }
    # logs values from ConfMatrix
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

    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")

    # log ClassMetrics 
    logger.log_metric("precision", precision)
    logger.log_metric("recall", recall)
    logger.log_metric("f1", f1)

    return 


def need_for_escalation(df_input): # : pd.DataFrame -> pd.DataFrame:
    """
    Determine if escalation is needed based on information in df.

    Args:
        df (pd.DataFrame): DataFrame containing level of 'severity', 
        'uncertainty' and 'confidence'.
    """
    df = df_input.copy()

    def escalation_logic(severity: str, 
                         confidence: float,
                         uncertainty: str) -> bool:
        # hard escalation logic
        if severity == "high" and confidence >= 0.5:
            return "escalation"
        
        # conditional escalation logic / medium severity
        elif (severity == "medium" 
              and confidence >= 0.7
              and uncertainty == "low"):
            return "escalation"
    
        else:
            return "no_escalation"

    REQUIRED_COLS = ["severity", "confidence", "uncertainty"]
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None

    df["expected_action"] = df.apply(
                            lambda row: escalation_logic(
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
    logger.set_tag("escalate_rule", snapshot_dict["name"])
    logger.set_tag("escalate_rule_hashed", snapshot_dict["sha256"])
    logger.log_text(f"{fn_path}/logic_rules.py",
                    snapshot_dict["source"])
    
    y_true, y_pred = encode_labels(df_esc)
    create_evaluation_metrics(y_true, y_pred)