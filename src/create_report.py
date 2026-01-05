## 
# imports
from datetime import datetime
import time
import os
from pathlib import Path
import pandas as pd
import click
from sklearn.metrics import (precision_recall_fscore_support,
                             classification_report, 
                             confusion_matrix)

import src.utils.general_helper as gh
from src.utils.escalation_baseline import baseline_escalation
from src.utils.escalation_llm import llm_escalation
from src.utils.prompt import prompt_v1, allowed_values_v1
from src.utils.json_scheme import scheme_v1

from src.utils.mlflow_helper import mlflow_logging
from src.utils.red_flags import RED_FLAGS


@click.command()
@click.option(
    "--logging/--no-logging",
    default=False,
    help="Enable MLflow logging"
)
def create_report(data="ambiguous", mode="llm", logging=False):
    # configurations
    vers_run = "v1"
    vers_data = "v1"
    vers_logic = "v1"
    vers_flags = "v1"
    vers_prompt = "v1"
    vers_values = "v1"
    vers_json = "v1"

    # load env variable
    gh.load_env_vars()

    if data == "ambiguous":
        f_path = Path(os.getenv("AMBIGUOUS_DATA"))
        f_name = "data/data_generic/reports_ambiguous.csv" 

    elif data == "clear":
        f_path = Path(os.getenv("CLEAR_DATA"))
        f_name = "data/data_generic/reports_clear.csv"

    elif isinstance(data, Path):
        f_path = Path(data)

    # loading texts as df --> fct file load
    df = pd.read_csv(f_path, index_col="Unnamed: 0")
    
    if mode == "baseline":  # LLM
       fct_escalate = baseline_escalation
    
    elif mode == "llm":
        fct_escalate = llm_escalation

        UNCERTAINTY_TO_CONF = {
                        "low": 0.85,
                        "medium": 0.6,
                        "high": 0.3,
                    }

    else:
        print(f"Unknown mode: {mode}")
        return 

    y_true = df["escalation_required"]

    start = time.perf_counter()
    result = df["report_text"].apply(fct_escalate)
    elapsed = time.perf_counter() - start
    
    result_df = pd.DataFrame(result.tolist())

    if mode == "llm":
        df[f"pred_{mode}_{vers_run}"] = result_df["escalation_required"]
        df[f"confidence_llm_{vers_run}"] = result_df["confidence"]
        df[f"uncertainty_llm_{vers_run}"] = result_df["uncertainty_level"]
        df[f"confidence_derived_llm_{vers_run}"] = (
                                        result_df["uncertainty_level"]
                                        .map(UNCERTAINTY_TO_CONF)
                                                    )
    
    if mode == "baseline":
        df[f"pred_{mode}_{vers_run}"] = result

    # save df
    df.to_csv(f_path)

    # (1)standardisierten Evaluation-Pipeline (DataFrame → Metrics)
    # alle nachfolgenden SChritte pro Subgruppe dann psyche vs Körper, clear vs ambiguous 
    # (2) Error-Bucket-Analyse
    # false_negatives = df[
    #         (df["escalation_required"] == True) &
    #         (df[f"pred_{mode}_{vers_run}"] == False)
    #     ]
    
    # create ClassReport
    y_pred = df[f"pred_{mode}_{vers_run}"]

    report = classification_report(y_true, 
                                   y_pred, 
                                   zero_division=0,
                                   output_dict=True)

    # create ConfMatrix + Classification metrics
    cm = confusion_matrix(                  # tn, fp, fn, tp
                        y_true, 
                        y_pred, 
                        labels=[False, True]
                        ).ravel()

    print("false_negatives", cm[2])
    print("false_positives", cm[1])

    metrics = precision_recall_fscore_support(          # precision, recall, f1, support
                                            y_true,
                                            y_pred,
                                            average="binary",
                                            pos_label=True,
                                            zero_division=0
                                            )

    print(f"precision: {metrics[0]}")
    print(f"recall: {metrics[1]}")
    print(f"f1: {metrics[2]}")

    # store parameters to log in dict 
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    log_dict = {
        # general information
        "experiment_name": "escalation_clinical_reports",
        # "tracking_uri": os.getenv(""),
        "artifacts_location": os.getenv("MLFLOW_ARTIFACTS"),
        "git_commit": gh.get_git_commit(), 
        "run_name": f"{now}_{mode}",
        "approach": "{mode}",
        "version_approach": f"{vers_run}",

        # data
        "file_path_data": f_path, 
        "file_name_data": f_name, 
        "name_dataset": data,
        "source_dataset": "synthetic",
        "size_dataset": len(df), 
        "version_dataset": vers_data,

        # logic, red_flags,...
        "file_name_logic": f"logic/{vers_logic}/{mode}.txt",
        "logic": fct_escalate, 

        # metrics
        "runtime_total_sec": elapsed,
        "runtime_per_sample_sec": (elapsed / len(df)),
        "class_report": report,
        "confusion_matrix": cm, 
        "class_metrics": metrics,
        "file_name_df_fn": f"mlflow/evaluation/false_negatives_{mode}_{vers_run}.csv",
        "df_fn": false_negatives, 

        # comments and notes
        "notes": [
            "known_limitation -- Baseline matches synthetic dataset features; expected to fail on ambiguous cases",
            "Recall undefined for classes with no true samples; zero_division=0 applied"
                ]
                }

    if mode == "baseline":
        log_dict["file_name_red_flags"] = f"red_flags/{vers_flags}.json"
        log_dict["red_flags"] = RED_FLAGS
        
    if mode == "llm":
        log_dict["file_name_prompt"] = f"prompts/{vers_prompt}/system_prompt.txt"
        log_dict["prompt"] = prompt_v1.strip()
        log_dict["file_name_values"] = f"prompts/{vers_values}/allowed_values.txt"
        log_dict["allowed_values"] = allowed_values_v1.strip()
        log_dict["file_name_json_scheme"] = f"json_scheme/{vers_json}.json"
        log_dict["json_scheme"] = scheme_v1      

    if logging:
        mlflow_logging(log_dict)
        
if __name__ == "__main__":
    create_report()
   
