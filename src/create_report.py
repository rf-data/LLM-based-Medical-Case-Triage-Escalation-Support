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
def create_report(f_path=None, mode="llm", logging=False):
    # configurations
    version = "v1"

    if not f_path:
        gh.load_env_vars()
        f_path=Path(os.getenv("AMBIGUOUS_DATA"))

    # loading texts as df --> fct file load
    df = pd.read_csv(f_path)
    
    if mode == "baseline":  # LLM
       fct_escalate = baseline_escalation
    
    elif mode == "llm":
        fct_escalate = llm_escalation

    else:
        print(f"Unknown mode: {mode}")

    y_true = df["escalation_required"]

    start = time.perf_counter()
    result = df["report_text"].apply(fct_escalate)
    elapsed = time.perf_counter() - start

    df[f"pred_{mode}_{version}"] = result["escalation_required"]
    
    if mode == "llm":
        UNCERTAINTY_TO_CONF = {
                        "low": 0.85,
                        "medium": 0.6,
                        "high": 0.3,
                    }
        
        df[f"confidence_llm_{version}"] = result["confidence"]
        df[f"uncertainty_llm_{version}"] = result["uncertainty_level"]
        df[f"confidence_derived_llm_{version}"] = UNCERTAINTY_TO_CONF[result["uncertainty_level"]]
    
    # save df
    df.to_csv(f_path)

    false_negatives = df[
            (df["escalation_required"] == True) &
            (df[f"pred_{mode}_{version}"] == False)
        ]
    # false_negatives.to_csv(
    #         "analysis/false_negatives_llm_v1.csv",
    #         index=False
    #     )

    # logger.log_artifact("analysis/false_negatives_llm_v1.csv")

    
    

    # create ClassReport
    y_pred = df[f"pred_{mode}_{version}"]

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
        "run_name": f"{now}_LLM",
        "approach": "LLM (openAI)",
        "version_approach": f"{version}",

        # data
        "file_path_data": f_path, 
        "file_name_data": "data_generic/reports_ambiguous.csv", 
        "name_dataset": "ambiguous",
        "source_dataset": "synthetic",
        "size_dataset": len(df), 
        "version_dataset": "v1",

        # logic, red_flags,...
        "file_name_logic": f"logic/{version}/{mode}.txt",
        "logic": fct_escalate, 
        # "file_name_red_flags": "red_flags/version_1.json",
        # "red_flags": RED_FLAGS,
        "file_name_prompt": f"prompts/{version}/system_prompt.txt", 
        "prompt": prompt_v1.strip(),
        "file_name_values": f"prompts/{version}/allowed_values.txt",
        "allowed_values": allowed_values_v1.strip(),
        "file_name_json_scheme": f"json_scheme/{version}.json",
        "json_scheme": scheme_v1,
        
        # metrics
        "runtime_total_sec": elapsed,
        "runtime_per_sample_sec": (elapsed / len(df)),
        "class_report": report,
        "confusion_matrix": cm, 
        "class_metrics": metrics,
        "file_name_df_fn": f"analysis/false_negatives_{mode}_{version}.csv",
        "df_fn": false_negatives, 

        # comments and notes
        "notes": [
            # "known_limitation -- Baseline matches synthetic dataset features; expected to fail on ambiguous cases",
            # "Recall undefined for classes with no true samples; zero_division=0 applied"
                ]
                }

    if logging:
        mlflow_logging(log_dict)
        
if __name__ == "__main__":
    create_report()
   
