# imports
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
                            #  precision_recall_fscore_support,
                            #  classification_report, 
                            #  confusion_matrix,
                            #  roc_auc_score, 
                            #  average_precision_score,
                             precision_score, 
                             recall_score, 
                             f1_score, 
                             fbeta_score
                             )

from core.mlflow_logger import get_experiment_logger
import utils.evaluation_helper as eval
import utils.path_helper as ph
import utils.split_helper as split


#

def threshold_sweep_analysis(model, X_train, y_train, metric):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    y_val, y_proba_val = split.validation_split(model, X_train, y_train)
    thresh_df = create_threshold_df(y_val, y_proba_val)
    best_row = find_optimal_thresh(thresh_df, metric)
    best_t = float(best_row["threshold"])

    return best_t, thresh_df


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


def topk_threshold_sweep(topk_df, 
                         base_model, 
                         X_train, 
                         X_test, 
                         y_train,
                         y_test,
                         metric="f2",
                         fold_id=None):
    """
    For each of the top-k hyperparameter configurations:
    - fit model
    - perform threshold sweep on inner validation split
    - evaluate final metrics on test set
    """
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    all_results = []
    all_thresh = []

    for rank, row in topk_df.iterrows():
        event_logger.info(
            "Threshold analysis for rank %d (fold: %s) | C=%s | l1_ratio=%s | class_weight=%s",
            rank + 1,
            fold_id +1 if fold_id else "n.a.",
            row["C"],
            row["l1_ratio"],
            row["class_weight"],
        )

        # 1. configure model 
        model = clone(base_model)
        model.set_params(
            clf__C=row["C"],
            clf__l1_ratio=row["l1_ratio"],
            clf__class_weight=None if row["class_weight"] == "None" else row["class_weight"],
        )

        # 2. Threshold sweep (inner validation!)
        best_t, thresh_df = threshold_sweep_analysis(
                                                model, 
                                                X_train, 
                                                y_train, 
                                                metric
                                                )
        thresh_df["rank"] = rank + 1
        
        # 3. final training on full train set
        model.fit(X_train, y_train)

        y_proba_test = model.predict_proba(X_test)[:, 1]
        y_hat_test = (y_proba_test >= best_t).astype(int)

        img_id = f"r{rank+1}_{fold_id}" if fold_id else rank + 1
        roc_pr_auc = eval.compile_roc_pr_auc(
                                    y_test, 
                                    y_proba_test, 
                                    data_viz=True,
                                    split_id=img_id,
                                    ) 
        
        result = {
            "rank": rank + 1,
            "C": row["C"],
            "l1_ratio": row["l1_ratio"],
            "class_weight": row["class_weight"],
            "best_t": best_t,
            "roc_auc": roc_pr_auc["ROC_AUC"],
            "pr_auc": roc_pr_auc["PR_AUC"],
            "precision": precision_score(y_test, y_hat_test, zero_division=0),
            "recall": recall_score(y_test, y_hat_test, zero_division=0),
            "f1": f1_score(y_test, y_hat_test, zero_division=0),
            "f2": fbeta_score(y_test, y_hat_test, beta=2, zero_division=0),
        }

        if fold_id:
            thresh_df["fold"] = fold_id + 1
            result["fold"] = fold_id + 1

        all_thresh.append(thresh_df)
        all_results.append(result)

        # create # save coef_df
        coef_df = eval.create_coef_df(model)

        if fold_id is None:
            coef_path = ph.create_save_path(
                                        "coef_df",
                                        f"coefs_rank{rank + 1}",
                                        ".csv",
                                        )
        else:
            coef_path = ph.create_save_path(
                                        "coef_df",
                                        f"coefs_rank{rank + 1}_{fold_id}",
                                        ".csv",
                                        )
            
        coef_df.to_csv(coef_path)

    return all_results, all_thresh


        
        

        

    

"""
def threshold_sweep_analysis(model, X_train, y_train, metric):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    y_val, y_proba_val = split.validation_split(model, X_train, y_train)
    thresh_df = create_threshold_df(y_val, y_proba_val)
    best_row = find_optimal_thresh(thresh_df, metric)
    best_t = best_row["threshold"]

    return best_t, thresh_df
"""

        # # 2. 
        # y_val, y_proba_val = split.validation_split(
        #                         model_clone, X_train, y_train
        #                                     )

        # thresh_df = create_threshold_df(y_val, y_proba_val)
        # best_row = find_optimal_thresh(thresh_df, metric=metric)
        # best_t = float(best_row["threshold"])

        # thresh_df["rank"] = rank + 1
        # all_thresh.append(thresh_df)

        # 3. final training on full train set
        # model.fit(X_train, y_train)

        # y_proba_test = model.predict_proba(X_test)[:, 1]
        # y_hat_test = (y_proba_test >= best_t).astype(int)

        # best_t, thresh_data = threshold_sweep_analysis(
        #                                             model, 
        #                                             X_train, 
        #                                             y_train, 
        #                                             metric="f2"
        #                                             )

        # result["best_t"] = best_t
        # thresh_data["rank"] = rank +1 
        
        # 3. Final evaluation on test
        # model.fit(X_train, y_train)

        # y_proba_new= model.predict_proba(X_test)[:, 1]
        # y_hat_new = (y_proba_new >= best_t).astype(int)

        

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

