
## imports
# import pandas as pd
import json
# import os
import click
from pathlib import Path
# from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from core.mlflow_logger import get_experiment_logger
from core.session import session
from configuration.C2_logreg_base_shuflle_split_cv import config, num_feats, cat_feats
import utils.preprocess_helper as pre
import utils.general_helper as gh
# import utils.evaluation_helper as eval
# import utils.mlflow_helper as mh
# import utils.path_helper as ph
# import utils.file_helper as fh
# from D1_evaluation import evaluate_escalation
from D2_single_run import single_run
from D3_shuffle_split_cv import shuffle_split_cv
from D4_group_aware_cv import group_aware_cv


# ----------------------------------------------------

# define paths and other arguments
f_path = "/home/robfra/0_Portfolio_Projekte/LLM/data/data_processed/reports/llm_resp_postprocessed.csv"
col_to_drop = ["expected_action_final", "confidence_derived", "expected_action_llm",
            "missing_information", "risk_factors", "Unnamed: 0"]


# ----------------------------------------------------

@click.command()
@click.option(
    "--logging/--no-logging",
    default=False,
    help="Enable MLflow logging"
)
def logreg_training(logging=False):
     # load env variable and configurations
    gh.load_env_vars()

    # setup session + load config-file   
    session.load_config(config)
    class_weight = session.parameters.get("class_weight")
    max_iter = session.parameters.get("max_iter")
    solver = session.parameters.get("solver")
    random_state=session.parameters.get("random_state", 42)
    # cross_validate=session.parameters.get("cross_validate", "tba")
    group_split=session.parameters.get("group_split", "tba")
    n_shuffle_splits=session.parameters.get("n_shuffle_splits")
    n_folds=session.parameters.get("n_folds")

    session.num_feats = num_feats
    session.cat_feats = cat_feats
    session.save_session()

    # setup logger
    exp_name = session.experiment_name
    artifact_path = session.artifacts_location
    exp_logger = get_experiment_logger(
                experiment_name=exp_name,
                artifact_location=artifact_path
                            )
    exp_logger.setup_experiment()
    event_logger = exp_logger.logger
    
    session.backup_dir = exp_logger.backup_dir
    
    
    # log parameters from config/session
    for param, value in session.parameters.items():
        exp_logger.log_param(param, value)

    # defining pipelines
    preprocess = ColumnTransformer(
                        transformers=[
                            ("num", StandardScaler(), num_feats),
                            ("cat", OneHotEncoder(
                                drop="first", 
                                handle_unknown="ignore", 
                                sparse_output=False), 
                                cat_feats)
                        ],
                        remainder="drop"
                    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            random_state=random_state,  
            max_iter=max_iter,
            solver=solver
        )),
    ])

    # prepare data 
    X, y = pre.prepare_reports(f_path, col_to_drop)
    
    # dtype casting to allow NaN
    X[num_feats] = X[num_feats].astype("float64")

    # print(f"[DEBUG] 'cross_validate' = {cross_validate}\t'group_split' = {group_split}")
    if group_split == "Yes":
        if n_shuffle_splits and n_shuffle_splits > 1:
            shuffle_split_cv(X, y, pipe, as_group=True)
        
        elif n_folds and n_folds > 1: 
            group_aware_cv(X, y, pipe)

        else: 
            event_logger.error("Invalid Conditions: group_split=%s\tn_shuffle_splits=%s\tn_folds=%s",
                               group_split,
                               n_shuffle_splits,
                               n_folds)

    else: 
        single_run(X, y, pipe)

    # add a comment
#     initial_notes = """
# ### Initial Observation
# - Logistic Regression achieves perfect metrics (P/R/F1 = 1.0)
# - Investigation shows ~89% feature-level duplicates
# - Model effectively performs deterministic mapping

# ### Interpretation
# This run represents LLM rule distillation, not generalizable ML.
# This run should be repeated after a group_split. 
# """

    # logging
    preprocess_func = session.preprocess_function
    version = session.tags.get("vers_preprocess", "tba")
    fn_path = Path(f"preprocessing/{version}")

    snapshots = gh.snapshot_single_function(preprocess_func)
    code = snapshots["source"]
    func_name = snapshots["name"]
    code_hashed = snapshots["sha256"]

    exp_logger.log_text(f"{fn_path}/preprocessing_complete.json",
                        json.dumps(snapshots, 
                                   indent=2, 
                                   ensure_ascii=False))

    exp_logger.log_text(f"{fn_path}/preprocessing.py",
                    code) 
    exp_logger.set_tag("source", f"{func_name} ({version})")
    exp_logger.set_tag("source_hashed", code_hashed)
#     exp_logger.log_text(
#         "analysis/01_initial_interpretation.md",
#         initial_notes
#             )
#     exp_logger.set_tag("model_nature", "llm_distillation")
#     exp_logger.set_tag("known_limitation", "feature-level duplicates")
#     exp_logger.set_tag("evaluation_validity", "group_split_required")

    if logging:
        approach = session.tags.get("approach")
        subapproach = session.tags.get("subapproach", "na")
        vers_approach = session.tags.get("vers_approach")

        run_name = f"{approach}_{vers_approach} ({subapproach} - {session.now})"
        exp_logger.flush(run_name)
        session.save_snapshot()
        exp_logger.local_backup() 

    else:
        session.save_snapshot()
        exp_logger.local_backup()

if __name__ == "__main__":
    logreg_training()

# evaluation
# def evaluate_reports(y_test, y_pred):
#     # start logger
#     exp_logger = get_experiment_logger()
#     event_logger = exp_logger.logger

#     # evaluation
#     report = eval.create_classification_report(y_test, y_pred)
#     cm = eval.create_confusion_matrix(y_test, y_pred)
#     metrics = eval.create_metrics(y_test, y_pred)

#     # save files
#     gh.load_env_vars()
#     folder = os.getenv("PATH_EVALUATED", None)

#     now = session.now
#     mode = session.mode
#     version_run = session.tags.get("vers_approach", "tba") 
    
#     path_cm = Path(f"{folder}/ConfMatrix/{now}_{mode}_{version_run}_cm.json")
#     path_cr = Path(f"{folder}/ClassReport/{now}_{mode}_{version_run}_cr.json")
#     path_metrics = Path(f"{folder}/metrics/{now}_{mode}_{version_run}_metrics.json")
#     # path_fn_fp = Path(f"{folder}/fn_fp") # /{now}_{mode}_{version_run}_metrics.json") 
    
#     for path, file in zip([path_cm, path_cr, path_metrics],
#                         [cm, report, metrics]):
#         if not path.exists():
#             ph.ensure_dir(path)
#             fh.save_dict(path, file)
#         else: 
#             event_logger.error("File '%s' already exists. Hence, no overwrite", 
#                                path)

#     return report, cm, metrics


    # ("evaluation", evaluate_reports(y_test, y_pred))

# log = LogisticRegression(
#         class_weight="balanced",
#         max_iter=1000,
#         solver="liblinear"
#         )

# # Training
# log.fit(X_train, y_train)



# save model (local + MLflow)


# severity,             --> str_tertiary
# uncertainty_level,    --> str_tertiary
# confidence,           --> float[0;1]
# clarity,              --> str_binary
# domain,               --> str_binary
# n_risk_factors,       --> int
# n_missing_information, --> int

# def enlabel_column(df_input, enlabel_dict):
    
#     df = df_input.copy()

#     for col in df.columns:
#         for key, mapping in enlabel_dict.items():
#             if col == key:
#                 df[col] = df[col].replace(mapping)

#     return df


# scaling?


# label encoding
# enlabel_dict = {
#     "expected_action": {
#             "escalation": int(1),
#             "no_escalation": int(0) 
#             }
# }
    # "severity": {
    #         "low": int(0),
    #         "medium": int(1),
    #         "high": int(2)
    #         }, 
    
    # "uncertainty_level":  {
    #         "low": int(0),
    #         "medium": int(1),
    #         "high": int(2)
    #         }, 

    # "clarity": {
    #         "clear": int(0),
    #         "ambiguous": int(1)
    #         },

    # "domain": {
    #         "psych": int(0),
    #         "somatic": int(1)
    #         }
    #         }

# for col in df_slim.columns:
#     for key, mapping in enlabel_dict.items():
#         if col == key:
#             df_slim[col] = df_slim[col].replace(mapping)
