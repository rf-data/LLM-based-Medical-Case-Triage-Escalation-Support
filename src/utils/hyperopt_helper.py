# imports 
import pandas as pd

from core.mlflow_logger import get_experiment_logger
import utils.thresh_sweep_helper as thresh


def validate_topk_model(hyperopt_df, model, fn_group_split, metric):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    thresh_list = []
    result_list = []
    group_split = fn_group_split()  

    for split_id, X_train, X_test, y_train, y_test in group_split:
        event_logger.info(
                "--- Validation fold # %s", 
                split_id
                ) 

        all_result, all_thresh = thresh.topk_threshold_sweep(
                                                hyperopt_df,
                                                model, 
                                                X_train, 
                                                X_test, 
                                                y_train,
                                                y_test,
                                                metric,
                                                split_id
                                                )

        thresh_list += all_thresh
        result_list += all_result

    result_df = pd.DataFrame(result_list)
    if len(thresh_list) >= 1:
        thresh_df = pd.concat(thresh_list, ignore_index=True)
    else:
        thresh_df = pd.DataFrame()

    return result_df, thresh_df