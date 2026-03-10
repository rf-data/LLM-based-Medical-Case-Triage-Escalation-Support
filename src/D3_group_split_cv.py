# imports'
# from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from datetime import datetime
import pandas as pd

from core.mlflow_logger import get_experiment_logger
from core.session import session
import utils.preprocess_helper as pre
import utils.path_helper as ph
import utils.evaluation_helper as eval

def group_split_cv(X, y, model):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    split_mode = session.parameters.get("split_mode", "tba")
    group_split = pre.group_split(X, y, split_mode)
    
    # 
    all_results = [] 
    best_t_results = []
    all_thresh = []

    # start CV
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    event_logger.info("Start LogReg GroupSplit-CrossVal: %s", now)
    session.now = now

    for split_id, X_train, X_test, y_train, y_test in group_split:
        # 1. Train model
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)

        # 2. evaluate run
        split_result, thresh_data, best_t_metrics = eval.evaluate_run(
                                                                X_train, 
                                                                X_test, 
                                                                y_train, 
                                                                y_test,
                                                                model_clone, 
                                                                split_id
                                                                )

        # collect output
        all_thresh.append(thresh_data)
        all_results.append(split_result)
        best_t_results.append(best_t_metrics)

    # create + save result_df
    result_df = pd.DataFrame(all_results)
    result_path = ph.create_save_path(f"{split_mode}_cv", "results", ".csv")
    result_df.to_csv(result_path)
    exp_logger.log_artifact(result_path)

    # create + save eval_result_df
    eval_result_df = eval.evaluate_result_df(result_df)
    eval_result_path = ph.create_save_path(f"{split_mode}_cv", "results_eval", ".csv")
    eval_result_df.to_csv(eval_result_path)
    exp_logger.log_artifact(eval_result_path)

    # create + save thresh_df
    if len(all_thresh) >= 1:
        thresh_df = pd.concat(all_thresh, ignore_index=True)
        thresh_path = ph.create_save_path(f"{split_mode}_cv", "thresh_df", ".csv")
        thresh_df.to_csv(thresh_path)
    
    # create + save best_t_result_df
    best_t_result_df = pd.DataFrame(best_t_results)
    best_t_result_path = ph.create_save_path(f"{split_mode}_cv", "best_t_results_df", ".csv")
    best_t_result_df.to_csv(best_t_result_path)

    return 