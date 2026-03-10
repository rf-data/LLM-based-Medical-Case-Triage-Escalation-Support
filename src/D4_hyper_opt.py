# # imports'
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, 
                             recall_score, 
                             average_precision_score,
                             fbeta_score)
from sklearn.base import clone
# from sklearn.model_selection import
# from datetime import datetime
# import pandas as pd

from core.mlflow_logger import get_experiment_logger
from core.session import session
import utils.thresh_sweep_helper as thresh
# import utils.preprocess_helper as pre
# import utils.evaluation_helper as eval


def run_HyperOpt(model, param_combinations, fn_group_split, metric):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    # 
    n_splits = session.parameters.get("n_splits", 30)
    test_size = session.parameters.get("test_size", 0.2)
    
    # -----------------------------
    # Grid loop
    # -----------------------------
    results = []
    skipped_splits = []

    for combi_id, params in enumerate(param_combinations):
        # extract hyperparameter
        C, l1_ratio, class_weight = params
        
        # inital logging
        event_logger.info(
                "--- Testing config # %s: C=%s | l1_ratio=%s | class_weight=%s --- ",
                combi_id+1, C, l1_ratio, class_weight,
            )

        # instantiate model
        model_clone = clone(model)
        model_clone.set_params(
                    clf__C=C,
                    clf__l1_ratio=l1_ratio,
                    clf__class_weight=class_weight,
                )
        
        f2_scores = []
        precision_scores  = []
        pr_auc_scores = []
        recall_scores = []
        best_t_values = []
        thresh_df_list = []
        group_split = fn_group_split()  

        for split_id, X_train, X_test, y_train, y_test in group_split:
            event_logger.info(
                "--- Testing config # %s --- split # %s ---", 
                combi_id+1, split_id+1
                ) 
            # 1. Train model    
            model_clone.fit(X_train, y_train)
        
            # 3. evaluate run
            y_pred = model_clone.predict(X_test)
            y_proba = model_clone.predict_proba(X_test)[:, 1]

            f2 = fbeta_score(
                    y_test,
                    y_pred,
                    beta=2,
                    pos_label=1,
                    zero_division=0,
                )

            precision = precision_score(y_test, 
                                        y_pred, 
                                        zero_division=0)
            
            pr_auc = average_precision_score(y_test, y_proba)

            recall = recall_score(y_test, 
                                  y_pred, 
                                  zero_division=0)

            # 
            best_t, thresh_df = thresh.threshold_sweep_analysis(model, X_train, y_train, metric)      

            thresh_df["split_id"] = split_id 
            thresh_df["combi_id"] = combi_id

            f2_scores.append(f2)
            precision_scores.append(precision)
            pr_auc_scores.append(pr_auc)
            recall_scores.append(recall)
            best_t_values.append(best_t)
            thresh_df_list.append(thresh_df)
            
        # aggregate metrics
        if len(f2_scores) == 0:
            event_logger.warning(
                "Skipping config C=%s | l1_ratio=%s | class_weight=%s ",
                "→ no valid splits",
                C, l1_ratio, class_weight)  
            
            info = (f"Skipping config C={C} | l1_ratio={l1_ratio} | class_weight={class_weight} ",
                "→ no valid splits\n")
            skipped_splits.append([info])
        
        else:   
            f2_mean = np.mean(f2_scores)
            f2_median = np.median(f2_scores)
            f2_std = np.std(f2_scores)
            f2_min = np.min(f2_scores)
            f2_q10 = np.quantile(f2_scores, 0.10)
            f2_q25 = np.quantile(f2_scores, 0.25)

            row = {
                "C": C,
                "l1_ratio": l1_ratio,
                "class_weight": str(class_weight),
                "n_splits": n_splits,
                "test_size": test_size,
                "mean_f2": f2_mean,
                "median_f2": f2_median,
                "std_f2": f2_std,
                "min_f2": f2_min,
                "q10_f2": f2_q10,
                "q25_f2": f2_q25,
                "mean_pr_auc": np.mean(pr_auc_scores),
                "median_pr_auc": np.median(pr_auc_scores),
                "std_pr_auc": np.std(pr_auc_scores),
                "mean_precision": np.mean(precision_scores),
                "median_precision": np.median(precision_scores),
                "std_precision": np.std(precision_scores),
                "mean_recall": np.mean(recall_scores),
                "median_recall": np.median(recall_scores),
                "std_recall": np.std(recall_scores),
                "mean_best_t": np.mean(best_t_values),
                "median_best_t": np.median(best_t_values),
                "std_best_t": np.std(best_t_values),
                "q10_best_t": np.quantile(best_t_values, 0.10),
                "q25_best_t": np.quantile(best_t_values, 0.25),
            }   

            results.append(row)    
    
    # result_df = pd.DataFrame(results)
    if len(thresh_df_list) >= 1:
        thresh_df = pd.concat(thresh_df_list, ignore_index=True)
    
    else:
        thresh_df = pd.DataFrame()
    
    return results, thresh_df, skipped_splits