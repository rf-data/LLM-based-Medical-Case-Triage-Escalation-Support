## imports
import pandas as pd
import json
from sklearn.model_selection import GroupShuffleSplit
from core.mlflow_logger import get_experiment_logger
from core.session import session


def make_feature_signature(X):
    # setup logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    event_logger.info("Examine if group of duplicates are in data")
    X["feature_signature"] = X.astype(str).agg("|".join, axis=1)

    groups = X["feature_signature"]
    X_new = X.drop(columns=["feature_signature"]).copy()

     # logging
    n_unique = X["feature_signature"].nunique()
    group_stats = X["feature_signature"].value_counts().describe().to_dict()

    exp_logger.log_param("n_group", n_unique)
    exp_logger.log_text("group stats", 
                        json.dumps(group_stats, 
                                   indent=2, 
                                   ensure_ascii=False))

    event_logger.info("Unique feature signatures: %s", n_unique)
    event_logger.info("Distribution of 'Unique features' : \n%s", group_stats)
    
    return X_new, groups


def prepare_reports(f_path, 
                    col_to_drop):
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger
    
    # loading df + dropping columns unnecessary
    df = pd.read_csv(f_path)
    df = df.drop(columns=col_to_drop)
    
    # split dataset
    y = df["expected_action"].map({"no_escalation": 0, "escalation": 1}).copy()
    X = df.drop(columns=["expected_action"]).copy()

    event_logger.info("Head of X:\n%s", X.head(5))
    event_logger.info("\nHead of y:\n%s", y.head(5))

    # # debug prints
    # dup_rows = X.duplicated().mean()
    # print("-"*25, " [START DEBUGGING] ", "-"*25)
    # print("Duplicate share:", dup_rows)

    # # optional: check duplicates with target consistency
    # dups = df[df.duplicated(subset=X.columns.tolist(), keep=False)]
    # print("Duplicated rows:", len(dups))
    # print(dups["expected_action"].value_counts())
    # print("-"*25, " [END DEBUGGING] ", "-"*25)

    return X, y


def pretraining_checks(X_train, X_test,y_train, y_test):
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    event_logger.info("Start pretraining checks")
    event_logger.info("Train shape: %s", X_train.shape)
    event_logger.info("Test shape: %s", X_test.shape)
    event_logger.info("Class distribution (train): \n%s", 
                      y_train.value_counts(normalize=True))
    event_logger.info("Class distribution (test): \n%s", 
                      y_test.value_counts(normalize=True))
    
    return  


def group_shuffle_split(X, y):
    # apply 'group_split' or 'train_test_split'
    # split = session.tags.get("group_split", None)
    # random = session.tags.get("random_state", None)
    # cv = session.tags.get("cross_validate", None)
    # if split == True:
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger
    
    X, groups = make_feature_signature(X)
                                                        
    random = session.parameters.get("random_state", 42)
    n_splits = session.parameters.get("n_splits", 1)
    test_size = session.parameters.get("test_size", 0.2)

    event_logger.info("Apply GroupShuffleSplit(n_splits=%s, test_size=%s)", 
                      n_splits, test_size)

    gss = GroupShuffleSplit(n_splits=n_splits, 
                            test_size=test_size, 
                            random_state=random)
    
    for split_id, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # pre-training check
        assert list(X_train.columns) == list(X_test.columns)

        yield split_id, X_train, X_test,y_train, y_test 

    #     if cv == True:
    #         event_logger.info("Prepared data for group-aware cross-validation")
    #         result_dict = eval.prepare_group_aware_cv(X, 
    #                                                 y, 
    #                                                 groups)
    #         return result_dict

    #     elif cv == False:
            

    # elif split in (False, None):
    #     event_logger.info("Apply train_test_splitting on data")
    #     X_train, X_test,y_train, y_test = train_test_split(
    #                                                 X, y,
    #                                                 test_size=0.2, 
    #                                                 random_state=random, 
    #                                                 stratify=y
    #                                                 )

    # pre-training check
    # assert list(X_train.columns) == list(X_test.columns)

    # return X_train, X_test, y_train, y_test
