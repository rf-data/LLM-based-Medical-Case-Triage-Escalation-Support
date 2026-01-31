from pathlib import Path
import os
import numpy as np
import pandas as pd

from core.session import session
from core.mlflow_logger import get_experiment_logger
# import utils.evaluation_helper as eval
# import utils.escalation_helper as esc
# import utils.path_helper as ph
import utils.file_helper as fh
# import utils.general_helper as gh


def need_for_escalation(df_input): # : pd.DataFrame -> pd.DataFrame:
    """
    Determine if escalation is needed based on information in df.

    Args:
        df (pd.DataFrame): DataFrame containing level of 'expected_action', 'severity', 
        'uncertainty' and 'confidence', as well as 'n_risk_factors' and 'n_missing_information'.
    """
    df = df_input.copy()

    def escalation_postprocess(
                        exp_action: bool, 
                        severity: str, 
                        confidence: float,
                        uncertainty_level: str,
                        n_risk_factors: int,
                        n_missing_information: int) -> bool:
        
        if exp_action is True:
            if severity == "low":
                return False

            if (severity == "medium" and
                n_risk_factors == 0 and
                n_missing_information == 0 and
                confidence >= 0.7):
                return False

        elif exp_action is False:
            if severity == "high":      # safety first
                return True
                
            if severity == "medium":
                if (confidence >= 0.8 or 
                    n_missing_information >= 2 or
                    n_risk_factors >= 2):
                    return True
                
                return False

        # event_logger.info(
        #         "[POST] exp=%r sev=%r rf=%r mi=%r unc=%r",
        #         exp_action,
        #         severity,
        #         n_risk_factors,
        #         n_missing_information,
        #         uncertainty_level,
        #         )

        return exp_action

    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    
    df["expected_action_final"] = df.apply(
                            lambda row: escalation_postprocess(
                                exp_action=row.get("expected_action_llm", "unknown"),
                                severity=row.get("severity", "unknown"), 
                                confidence=row.get("confidence", "unknown"), 
                                uncertainty_level=row.get("uncertainty_level", "unknown"), 
                                n_risk_factors=row.get("n_risk_factors", "unknown"),
                                n_missing_information=row.get("n_missing_information", "unknown")),
                                axis=1)

    event_logger.info("[CHECK] 'result_df (post-processed)' columns: \n%s\n", df.columns.tolist())

    return df


            
# LABEL_MAP = {
#     "escalation": True,
#     "no_escalation": False,
#     "true": True,
#     "false": False,
#     "True": True,
#     "False": False,
#     True: True,
#     False: False,
# }

# def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
#     for col in ["expected_action", "expected_action_llm", "expected_action_final"]:
#         if col in df.columns:
#             print(f"column '{col}' found")
#             df[col] = (
#                 df[col]
#                 .astype(str)
#                 .str.strip()
#                 .str.lower()
#                 .map(LABEL_MAP)
#             )

#     return df

# def normalize_bool_column(s: pd.Series) -> pd.Series:
#     return (
#         s.astype(str)
#          .str.strip()
#          .str.lower()
#          .map({
#              "true": True,
#              "false": False,
#              True: True,
#              False: False,
#          })
#          .astype("boolean")
#     )

def evaluate_rule_fn_fp(df, extract_cols=None, folder=None):

    if extract_cols is None:
        extract_cols = ["expected_action", "expected_action_rule"]
     
    if folder is None:
        folder = os.getenv("PATH_EVALUATED")

    now = session.now
    mode = session.mode
    version_run = session.tags.get("vers_approach", "tba") 

    enlabel_dict = {True: "escalation",
                False: "no_escalation"}
    df_dict = extract_fn_fp(df, extract_cols, enlabel_dict) 
    
    for name, df in df_dict.items():
        # save files locally
        f_path = Path(f"{folder}/fn_fp/{now}_{mode}_{version_run}_{name}.json")
        df.to_csv(f_path)
    
    return df_dict


def evaluate_llm_fn_fp(df, filter_cols=None, extract_cols=None, folder=None, split_value=None):
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    if filter_cols is None:
        filter_cols = ["domain", "clarity", "n_risk_factors", 
                       "n_missing_information", "severity", 
                        "confidence", "uncertainty_level", "confidence_derived"]
    
    subapproach = session.tags.get("subapproach", None)

    if extract_cols is None:
        extract_cols = ["expected_action"]
        if subapproach == "post_processing":
            extract_cols.append("expected_action_final")
        elif subapproach == "baseline":
            extract_cols.append("expected_action_llm")
        else:
            event_logger.error("Unknown argument for 'subapproach' in session.tags: %s", subapproach)

    if folder is None:
        folder = os.getenv("PATH_EVALUATED")
    
    now = session.now
    mode = session.mode
    version_run = session.tags.get("vers_approach", "tba") 
    
    enlabel_dict = {True: "escalation",
                False: "no_escalation"}
    
    df_dict = extract_fn_fp(df, extract_cols, 
                            enlabel_dict, split_value=split_value)

    for name, df in df_dict.items():
        # save files locally
        f_path = Path(f"{folder}/fn_fp/{now}_{mode}_{version_run}_{name}.csv")
        df.to_csv(f_path)

        if name != "all":
            df_grouped = df.groupby("severity")[filter_cols].value_counts()
            f_path_grouped = Path(f"{folder}/fn_fp/{now}_{mode}_{version_run}_{name}_grouped.csv")
            df_grouped.to_csv(f_path_grouped)

    return df_dict

def extract_fn_fp_logreg(y_test, y_pred):
    false_dict = {}
    false_dict["false_negative"] = np.where((y_test.values == 1) & 
                                            (y_pred == 0))[0]
    false_dict["false_postive"] = np.where((y_test.values == 0) & 
                                           (y_pred == 1))[0]

    # save idx
    fh.save_dict(false_dict)

    return false_dict

def extract_fn_fp(df, cols, enlabel_dict, split_value=None):
    if split_value:
        df = fh.col_name_correct(df, f"{split_value}") 

    df_lab = df.replace(enlabel_dict).copy()

    vals = list(enlabel_dict.values())

    df_fn = df_lab[(df_lab[cols[0]] == vals[0]) & 
                    (df_lab[cols[1]] == vals[1])]
    df_fp = df_lab[(df_lab[cols[0]] == vals[1]) & 
                    (df_lab[cols[1]] ==vals[0])]
        # df_3_rename["expected_action_llm"]]

    # for name, df in [("df_fn", df_fn), ("df_fp", df_fp)]: 
    #     print(f"[CHECK] df: {name.upper()}")
    #     df_quick_check(df)

    return {
        "all": df,
        "false_negative": df_fn,
        "false_postive": df_fp} 

