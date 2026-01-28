##
import json
# import pandas as pd
# from pathlib import Path
# import os
from sklearn.metrics import (precision_recall_fscore_support,
                             classification_report, 
                             confusion_matrix)
# import utils.general_helper as gh
# import utils.path_helper as ph
# import utils.file_helper as fh
# import utils.escalation_helper as esc
# from core.session import session
from core.mlflow_logger import get_experiment_logger


def encode_labels(df, pred_column):
    LABEL_TO_INT = {
            "no_escalation": 0,
            "escalation": 1,
                }

    BOOL_TO_INT = {
                False: 0,
                True: 1,
                }

    # mode = session.mode
    # version = session.tags.get("vers_approach", "tba")

    y_true = df["expected_action"].map(LABEL_TO_INT)
    y_pred = df[pred_column].map(BOOL_TO_INT)
    
    # check for invalid values
    if y_true.isna().any():
        raise ValueError("Unknown label in column 'expected_action')")

    if y_pred.isna().any():
        raise ValueError(f"Invalid prediction values (col: {pred_column})")

    return y_true, y_pred


def create_classification_report(y_true, y_pred): 
    # setup logger
    exp_logger = get_experiment_logger()
    # event_logger = exp_logger.logger

    # creation of ClassReport
    report = classification_report(y_true, 
                                    y_pred, 
                                    labels=[0, 1],
                                    target_names=["no_escalation", "escalation"],
                                    zero_division=0,
                                    output_dict=True)

    # log ClassReport
    exp_logger.log_text(
                "ClassReport.json",
                json.dumps(report, indent=2)
            )

    exp_logger.log_metric("precision_escalation", 
                      report["escalation"]["precision"])
    
    exp_logger.log_metric("recall_escalation", 
                      report["escalation"]["recall"])
    
    return report

def create_confusion_matrix(y_true, y_pred): 
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    # create ConfMatrix
    tn, fp, fn, tp = confusion_matrix(                  # tn, fp, fn, tp
                        y_true, 
                        y_pred, 
                        labels=[False, True]
                        ).ravel()

    event_logger.info("false_negatives: %s", fn)
    event_logger.info("false_positives: %s", fp)

    cm = {
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }
    # log values from ConfMatrix
    exp_logger.log_text("ConfMatrix.json", json.dumps(cm, indent=2))
    exp_logger.log_metric("true_negatives", tn)
    exp_logger.log_metric("true_positives", tp)
    exp_logger.log_metric("false_negatives", fn)
    exp_logger.log_metric("false_positives", fp)

    return cm

def create_metrics(y_true, y_pred):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    # compile Classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(          # precision, recall, f1, support
                                            y_true,
                                            y_pred,
                                            average="binary",
                                            pos_label=True,
                                            zero_division=0
                                            )

    metrics = {
        "precision": float(precision), 
        "recall": float(recall), 
        "f1": float(f1)
    }

    event_logger.info("precision: %s", precision)
    event_logger.info("recall: %s", recall)
    event_logger.info("f1: %s", f1)

    # log ClassMetrics 
    exp_logger.log_metric("precision", precision)
    exp_logger.log_metric("recall", recall)
    exp_logger.log_metric("f1", f1)

    return  metrics
