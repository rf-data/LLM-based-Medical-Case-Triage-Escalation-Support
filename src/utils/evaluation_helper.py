##

import pandas as pd
from sklearn.metrics import (precision_recall_fscore_support,
                             classification_report, 
                             confusion_matrix)
import src.utils.general_helper as gh
  
def encode_labels(df, mode, config):
    LABEL_TO_INT = {
            "no_escalation": 0,
            "escalation": 1,
                }

    BOOL_TO_INT = {
                False: 0,
                True: 1,
                }

    y_true = df["expected_action"].map(LABEL_TO_INT)
    y_pred = df[f"pred_{mode}_{config["vers_run"]}"].map(BOOL_TO_INT)
    
    # check for invalid values
    if y_true.isna().any():
        raise ValueError("Unknown label in expected_action")

    if y_pred.isna().any():
        raise ValueError("Invalid prediction values")

    return y_true, y_pred


def create_evaluation_metrics(y_true, y_pred, log_dict): 
    # creation of ClassReport
    report = classification_report(y_true, 
                                    y_pred, 
                                    labels=[0, 1],
                                    target_names=["no_escalation", "escalation"],
                                    zero_division=0,
                                    output_dict=True)

    # create ConfMatrix + Classification metrics
    cm = confusion_matrix(                  # tn, fp, fn, tp
                        y_true, 
                        y_pred, 
                        labels=[False, True]
                        ).ravel()

    print("false_negatives", cm[2])
    print("false_positives", cm[1])

    metrics = precision_recall_fscore_support(          # precision, recall, f1, support
                                            y_true,
                                            y_pred,
                                            average="binary",
                                            pos_label=True,
                                            zero_division=0
                                            )

    print(f"precision: {metrics[0]}")
    print(f"recall: {metrics[1]}")
    print(f"f1: {metrics[2]}")

    #  "runtime_total_sec": elapsed,
    #     "runtime_per_sample_sec": (elapsed / len(df)),
    #     "class_report": report,
    #     "confusion_matrix": cm, 
    #     "class_metrics": metrics,
    #     "file_name_df_fn": f"mlflow/evaluation/false_negatives_{mode}_{vers_run}.csv",
    #     "df_fn": false_negatives, 

    log_dict["class_report"] = report
    log_dict["confusion_matrix"] = cm
    log_dict["class_metrics"] = metrics

    return log_dict


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
              and uncertainty=="low"):
            return "escalation"
    
        else:
            return "no_escalation"

    df["expected_action"] = df.apply(
                            lambda row: escalation_logic(
                                row["severity"], 
                                row["confidence"], 
                                row["uncertainty"]), 
                                axis=1)
    return df


def evaluate_escalation(df_input, mode, config, log_dict):
    df = df_input.copy()

    df_esc = need_for_escalation(df)
    snapshot_dict = gh.snapshot_single_function(need_for_escalation)
    log_dict["escalation_rule"] = snapshot_dict

    y_true, y_pred = encode_labels(df_esc, mode, config)
    _ = create_evaluation_metrics(y_true, y_pred, log_dict)