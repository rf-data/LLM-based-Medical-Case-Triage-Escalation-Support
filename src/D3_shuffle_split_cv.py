# imports'
# from sklearn.metrics import (roc_auc_score, 
#                              average_precision_score,
#                              precision_score, 
#                              recall_score, 
#                              fbeta_score,
#                              f1_score)
# from sklearn.model_selection import GroupKFold
from datetime import datetime
import pandas as pd

from core.mlflow_logger import get_experiment_logger
from core.session import session
import utils.preprocess_helper as pre
import utils.path_helper as ph
import utils.evaluation_helper as eval

def shuffle_split_cv(X, y, pipe, as_group=True):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    if as_group:
        data = pre.group_shuffle_split(X, y)

    # 
    all_results = [] 
    all_thresh = []

    # start CV
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    event_logger.info("Start LogReg GroupSplit-CrossVal: %s", now)
    session.now = now

    for split_id, X_train, X_test, y_train, y_test in data:
        event_logger.info("=== Split %s ===", split_id)

        # 1. Train model
        model = clone(pipe)
        model.fit(X_train, y_train)

        # 2.


        # 3. Threshold sweep (inner validation!)
        best_t, thresh_data = eval.threshold_sweep_analysis(
                                                        model, 
                                                        X_train, 
                                                        y_train, 
                                                        metric="f2"
                                                        )

        thresh_data["split_id"] = split_id
        all_thresh.append(thresh_data)

        # 4. Final evaluation on test
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= best_t).astype(int)

        metrics = eval.create_metrics(y_test, y_pred)

        metrics["split_id"] = split_id
        metrics["best_threshold"] = best_t
        metrics["n_test"] = len(y_test)
        metrics["n_pos"] = int(y_test.sum())

        all_results.append(metrics)

    # create + save result_df
    result_df = pd.DataFrame(all_results)
    result_path = ph.create_save_path("cv_metrics", "_split_cv", ".csv")
    result_df.to_csv(result_path)

    # create + save thresh_df
    thresh_df = pd.DataFrame(all_thresh)
    thresh_path = ph.create_save_path("thresh_df", "_thresh_df", ".csv")
    thresh_df.to_csv(thresh_path)

    return 