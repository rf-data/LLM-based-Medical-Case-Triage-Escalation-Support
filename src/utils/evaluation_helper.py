##
import json
import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import (train_test_split, 
                                     GroupKFold, 
                                     GroupShuffleSplit)
from sklearn.metrics import (precision_recall_fscore_support,
                             classification_report, 
                             confusion_matrix,
                             roc_auc_score, 
                             average_precision_score,
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             fbeta_score)
import utils.general_helper as gh
import utils.path_helper as ph
import utils.visualisation_helper as viz
# import utils.file_helper as fh
# import utils.escalation_helper as esc
# from core.session import session
from core.mlflow_logger import get_experiment_logger
from core.session import session

import numpy as np

def compile_roc_pr_auc(y_test, y_proba, data_viz=False):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    # logging
    exp_logger.log_metric("ROC-AUC", roc_auc)
    exp_logger.log_metric("Precision-Recall-AUC", pr_auc)

    event_logger.info("ROC-AUC: %.4f", roc_auc)
    event_logger.info("PR-AUC: %.4f", pr_auc)

    if data_viz == True:
        pr_path = ph.create_save_path("plots", "_pr_curve", ".png")
        roc_path = ph.create_save_path("plots", "_roc_auc", ".png")

        viz.create_roc_auc(y_test, y_proba, roc_path)
        viz.create_pr_curve(y_test, y_proba, pr_path)

        exp_logger.log_artifact(roc_path)
        exp_logger.log_artifact(pr_path)

    return {"ROC_AUC": roc_auc,
            "PR_AUC": pr_auc}


def save_coef_df(num_feats, cat_feats, 
                pipe=None, model=None):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    if pipe:
        ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
        coefs = pipe.named_steps["clf"].coef_[0]
    
    elif model:
        hi = ""
        coefs = ""

    else:
        event_logger.info("Please, provide a trained model or a pipeline containing a model training")  
        return
    
    cat_feat_names = ohe.get_feature_names_out(cat_feats)
    feat_names = (num_feats + 
                  list(cat_feat_names))
    
    coef_df = pd.DataFrame({
        "feature": feat_names,
        "coef": coefs
            })\
            .sort_values("coef", ascending=False)
    
    # save coefs as df
    now = session.now
    mode = session.mode # ", "tba")
    version_run = session.tags.get("vers_approach", "tba") 

    folder = os.getenv("PATH_EVALUATED")
    f_path = Path(f"{folder}/coef_df/{now}_{mode}_{version_run}_coefs.csv")
    coef_df.to_csv(f_path)
    
    return coef_df


def create_cv_metrics(df, cols=None, metrics=None):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    if cols is None:
        cols = ["n_test", "pos_rate", 
                "roc_auc", "pr_auc", 
                "precision", "recall", 
                "f1", "f2"]
    
    if metrics is None:
        metrics= ["mean", "std", "min", "max"]
    
    summary = df[cols].agg(metrics).copy()

    # log + save 'summary'
    f_path = ph.create_save_path("cv_metrics", 
                                 "_cv_metrics", 
                                 ".csv")
    summary.to_csv(f_path)
    exp_logger.log_artifact(f_path)

    return 

# def add_metrics_to_cv_dict(split_data, model):
    
#     for _, data in split_data.items():
#         X_train = data["X_train"]
#         y_train = data["y_train"]
#         X_test = data["X_test"]
#         y_test = data["y_test"]

#         model.fit(X_train, y_train)

#         y_proba = model.predict_proba(X_test)[:, 1]
#         y_pred = (y_proba >= 0.5).astype(int)
#             # "y_proba": y_proba,
#             # "y_pred": y_pred

#         data["roc_auc"] = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else None,
#         data["pr_auc"] = average_precision_score(y_test, y_proba)
#         data["precision"] = precision_score(y_test, y_pred, zero_division=0)
#         data["recall"] = recall_score(y_test, y_pred, zero_division=0)
#         data["f1"] = f1_score(y_test, y_pred, zero_division=0)

#     return pd.DataFrame(split_data.values())


# def evaluate_cv_metrics(df, metrics_dict):
#     # metrics = metrics_dict.keys()
#     # method = 

#     summary = df.agg(metrics_dict)
#     # ["mean", "std", "min", "max"])
#     print(summary)
#     return 
    



# def F2_sweep(model, X_train, y_train, val_split=True):
#     if val_split == True:
#         y_val, y_proba_val = validation_split(model, X_train, y_train)

#     elif val_split == False:
#         # CrossVal folgt
#         cv = ""
    
#     # sweep on val
#     thresh_df, best_row = threshold_sweep_analysis(y_val, y_proba_val)
#     best_t = best_row["threshold"]

#     # final eval on test (fixed threshold!)
#     test_proba = model.predict_proba(X_test)[:, 1]
#     y_test_hat = (test_proba >= best_t).astype(int)

#     return 

def threshold_sweep_analysis(model, X_train, y_train, metric):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    y_val, y_proba_val = eval.validation_split(model, X_train, y_train)
    thresh_df = eval.create_threshold_df(y_val, y_proba_val)
    best_row = eval.find_optimal_thresh(thresh_df, metric)
    best_t = best_row["threshold"]

    return best_t, thresh_df


def validation_split(model, X_train, y_train):
    random_state=session.parameters.get("random_state", 42)

    X_train2, X_val, y_train2, y_val = train_test_split(
                                            X_train, 
                                            y_train, 
                                            test_size=0.2, 
                                            random_state=random_state, 
                                            stratify=y_train
                                                )
    # train model on validation dataset
    model.fit(X_train2, y_train2)
    y_proba_val = model.predict_proba(X_val)[:, 1]

    return y_val, y_proba_val


def create_threshold_df(y_test, y_proba, thresholds=None):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    event_logger.info("Starting 'threshold sweep analysis'")

    # fallback for 'thresholds'
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    # 
    rows = []
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)

        rows.append({
            "threshold": float(t),
            "n_test": int(len(y_test)),
            "n_pos": int(y_test.sum()),
            "pos_rate": float(y_test.mean()),
            "precision": precision_score(y_test, y_hat, zero_division=0),
            "recall": recall_score(y_test, y_hat, zero_division=0),
            "f1": f1_score(y_test, y_hat, zero_division=0),
            "f2": fbeta_score(y_test, y_hat, beta=2, zero_division=0),
        })

    # create thresh_df
    thresh_df = pd.DataFrame(rows)

    return thresh_df

def find_optimal_thresh(thresh_df, metric="f2"):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    if metric == "f2":
        # best threshold by F2
        best_row = thresh_df.loc[thresh_df["f2"].idxmax()]
    
    # logging
    event_logger.info(
        "Best threshold by %s: %.3f (F2=%.4f, Recall=%.3f, Precision=%.3f)",
        metric,
        best_row["threshold"],
        best_row["f2"],
        best_row["recall"],
        best_row["precision"],
        )

    return best_row 

def find_min_thresh(target_metric, target_value, thresh_df):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    # filter df
    candidates = thresh_df[thresh_df[f"{target_metric}"] >= target_value]
    
    # find idx
    min_thresh = candidates["threshold"].idxmin()


    # logging
    event_logger.info("Lowest threshold for %s >= %s: %s", 
                      target_metric, 
                      target_value, 
                      min_thresh)
    
    return 

# print("Threshold for recall>=0.95:", choice)
    
    # [r for r in thresh_df.rows if r[f"{target_metric}"] >= target_value]


    # min_thresh = min(candidates, key=lambda r: r["threshold"]) if candidates else None

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

    f2 = fbeta_score(
                y_true,
                y_pred,
                beta=2,
                average="binary",
                pos_label=1,
                zero_division=0
                )

    metrics = {
        "precision": float(precision), 
        "recall": float(recall), 
        "f1": float(f1),
        "f2": float(f2)
    }

    event_logger.info("precision: %.4f", precision)
    event_logger.info("recall: %.4f", recall)
    event_logger.info("f1: %.4f", f1)
    event_logger.info("f2: %.4f", f2)

    # log ClassMetrics 
    exp_logger.log_metric("precision", precision)
    exp_logger.log_metric("recall", recall)
    exp_logger.log_metric("f1", f1)
    exp_logger.log_metric("f2", f2)

    return  metrics
