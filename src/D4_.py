# # imports'
# from sklearn.metrics import (roc_auc_score, 
#                              average_precision_score,
#                              precision_score, 
#                              recall_score, 
#                              fbeta_score,
#                              f1_score)
# from sklearn.model_selection import
# from datetime import datetime
# import pandas as pd

# from core.mlflow_logger import get_experiment_logger
# from core.session import session
# import utils.preprocess_helper as pre
# import utils.evaluation_helper as eval


# def group_aware_cv(X, y, model):
#     # setup logger
#     exp_logger = get_experiment_logger()
#     event_logger = exp_logger.logger

#     # 
#     X, groups = pre.make_feature_signature(X)

#     gkf = GroupKFold(n_splits=5)

#     cv_results = []
#     # start CrossVal
#     now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#     event_logger.info("Start LogReg CrossVal: %s", now)
#     session.now = now

#     for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]        

#         # pre-training check
#         assert list(X_train.columns) == list(X_test.columns)
#         pre.pretraining_checks(X_train, X_test,y_train, y_test)


#         model.fit(X_train, y_train)

#         y_proba = model.predict_proba(X_test)[:, 1]
#         y_hat = (y_proba >= 0.5).astype(int)

#         fold_result = {
#             "fold": fold,
#             "n_test": len(y_test),
#             "pos_rate": y_test.mean(),
#             "roc_auc": roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else None,
#             "pr_auc": average_precision_score(y_test, y_proba), 
#             "precision": precision_score(y_test, y_hat, zero_division=0), 
#             "recall": recall_score(y_test, y_hat, zero_division=0),
#             "f1": f1_score(y_test, y_hat, zero_division=0),
#             "f2": fbeta_score(y_test, y_hat, beta=2, zero_division=0),
#         }

#         cv_results.append(fold_result)
    
#     cv_df = pd.DataFrame(cv_results)
#     eval.create_cv_metrics(cv_df)

#     return 