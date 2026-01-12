## 
# imports
from datetime import datetime
import time
import mlflow
import os
from pathlib import Path
import pandas as pd
import click
from sklearn.metrics import (precision_recall_fscore_support,
                             classification_report, 
                             confusion_matrix)

import src.utils.general_helper as gh
import src.utils.escalation_helper as eh
from src.utils.escalation_baseline import baseline_escalation
from src.utils.escalation_llm import llm_escalation_batch
from src.utils.prompt import prompt_v1, allowed_values_v1
from src.utils.json_scheme import scheme_v1

from src.utils.mlflow_helper import mlflow_logging
from src.utils.red_flags import RED_FLAGS


# -----------------------
# HELPER FUNCTIONS
# -----------------------

def get_data_df(data, log_dict):
    # load env variable
    gh.load_env_vars()

    if data == "ambiguous":
        f_path = Path(os.getenv("AMBIGUOUS_DATA"))
        f_name = "data/data_generic/reports_ambiguous.csv" 

    elif data == "clear":
        f_path = Path(os.getenv("CLEAR_DATA"))
        f_name = "data/data_generic/reports_clear.csv"

    elif data == "version_2":
        f_path = Path(os.getenv("DATA_V2"))
        f_name = "data/data_generic/escalation_dataset_v2.csv"

    elif isinstance(data, Path):
        f_path = Path(data)

    # loading texts as df --> fct file load
    df = pd.read_csv(f_path, 
                     # index_col="Unnamed: 0"
                     )
    print(f"df check -- head:\n{df.head(2)}\n")
    # print(f"")
    # logging
    log_dict["file_name_data"] = f_name
    log_dict["file_path_data"] = f_path
    log_dict["name_dataset"] = data
    log_dict["source_dataset"] = "synthetic" 
    log_dict["size_dataset"] = len(df) 
    
    return df, log_dict

def case_escalation(df_input, mode, config, log_dict, save_df=True):
    df = df_input.copy()

    if mode == "baseline":  # LLM
       fct_escalate = baseline_escalation
    
    elif mode == "llm":
        prompt = log_dict["prompt"]
        scheme = log_dict["json_scheme"]
        allowed_values = log_dict["allowed_values"]
        namespace = config["namespace"]

        fct_escalate = eh.make_cached_escalation(prompt=prompt,
                                                 scheme=scheme, 
                                                 allowed_values=allowed_values,
                                                 namespace=namespace)

        UNCERTAINTY_TO_CONF = {
                        "low": 0.85,
                        "medium": 0.6,
                        "high": 0.3,
                    }

    else:
        print(f"Unknown mode: {mode}")
        return 

    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(f"[INFO] Start processing reports: {now}")

    start = time.perf_counter()
    all_results = []
    
    for i, chunk in enumerate(gh.iter_chunks(df, chunk_size=25), start=1):
        print(f"[INFO] Processing chunk {i}")

        texts = chunk["report_text"].tolist()

        chunk_results = eh.batch_apply(
                                texts=texts,
                                fn=fct_escalate,
                                batch_size=5
                                    ) 
        
        assert isinstance(chunk_results, list)
        assert all(isinstance(r, dict) for r in chunk_results)
        # if isinstance(chunk_results, dict):
        #     chunk_results = [chunk_results]
            
        all_results.extend(chunk_results)

        # chuck_results = eh.batch_apply(
        #                           texts=texts,
        #                           fn=fct_escalate
        #                               )
        
#         results_cleaned = []
#         for i, r in enumerate(chunk_results):
            
#                 results_cleaned[i] = r

#                 for r in all_results):
#     raise RuntimeError(
#         f"Invalid result structure. Example element: {type(all_results[0])}"
#     )

# result_df = pd.DataFrame(all_results)

    # result = df["report_text"].apply(fct_escalate)
    elapsed = time.perf_counter() - start

    log_dict["run_name"] = f"{now}_{mode}"
    log_dict["runtime_total_sec"] = elapsed
    log_dict["runtime_per_sample_sec"] = (elapsed / len(df))

    if mode == "llm": 
        func_name, dep_func = config.get("dependent_function", (None, None))
        snapshots = gh.snapshot_dependent_functions(fct_escalate,
                                                dependencies=[dep_func])
        log_dict["logic"] = [snapshots, 
                             ("root", snapshots["root"]), 
                             (f"{func_name}", snapshots["dependencies"][f"{func_name}"])]

    else:
        snapshot_dict = gh.snapshot_single_function(fct_escalate)
        log_dict["logic"] = snapshot_dict
                            
                        
    result_df = pd.DataFrame(all_results)
    # print(f"[DEBUG] columns:\n{result_df.columns}")
    # print(f"[DEBUG] head:\n{result_df.head()}")

    if mode == "llm":
        df[f"pred_{mode}_{config["vers_run"]}"] = result_df["expected_action"]
        df[f"confidence_llm_{config["vers_run"]}"] = result_df["confidence"]
        df[f"uncertainty_llm_{config["vers_run"]}"] = result_df["uncertainty_level"]
        df[f"confidence_derived_llm_{config["vers_run"]}"] = (
                                        result_df["uncertainty_level"]
                                        .map(UNCERTAINTY_TO_CONF)
                                                    )
    
    if mode == "baseline":
        df[f"pred_{mode}_{config["vers_run"]}"] = result_df

    # save df
    if save_df == True:
        folder = Path(os.getenv("PROCESSED"))
        gh.ensure_dir(folder)
        f_path = folder / f"{now}_reports_{mode}_{config["vers_run"]}.csv"

        df.to_csv(f_path)

    if isinstance(save_df, Path):
        f_path = Path(save_df)
        gh.ensure_dir(f_path)

        df.to_csv(f_path)

    return df, log_dict


def evaluate_escalation(df_input, mode, config, log_dict):
    df = df_input.copy()

    LABEL_TO_INT = {
            "no_escalation": 0,
            "escalation": 1,
                }

    BOOL_TO_INT = {
                False: 0,
                True: 1,
                }

    y_true = df["expected_action"].map(LABEL_TO_INT)
    y_pred = df[f"pred_{mode}_{config["vers_run"]}"].map(BOOL_TO_INT)
    
    # check for invalid values
    if y_true.isna().any():
        raise ValueError("Unknown label in expected_action")

    if y_pred.isna().any():
        raise ValueError("Invalid prediction values")

    # creation of ClassReport
    report = classification_report(y_true, 
                                    y_pred, 
                                    labels=[0, 1],
                                    target_names=["no_escalation", "escalation"],
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

    #  "runtime_total_sec": elapsed,
    #     "runtime_per_sample_sec": (elapsed / len(df)),
    #     "class_report": report,
    #     "confusion_matrix": cm, 
    #     "class_metrics": metrics,
    #     "file_name_df_fn": f"mlflow/evaluation/false_negatives_{mode}_{vers_run}.csv",
    #     "df_fn": false_negatives, 

    log_dict["class_report"] = report
    log_dict["confusion_matrix"] = cm
    log_dict["class_metrics"] = metrics
    log_dict["class_report"] = report

    return log_dict

# --------------
# MAIN FUNCTION
# --------------

@click.command()
@click.option(
    "--logging/--no-logging",
    default=False,
    help="Enable MLflow logging"
)
def escalation_check(data="version_2", mode="llm", logging=False):
    # configurations
    config = {
        "namespace":"llm_batch_v1",  
        "vers_run": "v1",
        "vers_data": "v2",
        "vers_logic": "v1_5",
        "vers_flags": "v1",
        "vers_prompt": "v1",
        "vers_values": "v1",
        "vers_json": "v1",
        "dependent_function": ("llm_escalation_batch", llm_escalation_batch)
        }
    
    log_dict = {
        # general information
        "experiment_name": "escalation_clinical_reports",
        # "tracking_uri": os.getenv(""),
        "artifacts_location": os.getenv("MLFLOW_ARTIFACTS"),
        "git_commit": gh.get_git_commit(), 
        "approach": f"{mode} (batch_size=5)",
        "version_approach": f"{config["vers_run"]}",
        "version_dataset": config["vers_data"]
    }

    if mode == "baseline":
        log_dict["file_name_red_flags"] = f"red_flags/{config["vers_flags"]}.json"
        log_dict["red_flags"] = RED_FLAGS
        
    if mode == "llm":
        log_dict["file_name_prompt"] = f"prompts/{config["vers_prompt"]}/system_prompt.txt"
        log_dict["prompt"] = prompt_v1.strip()
        log_dict["file_name_values"] = f"prompts/{config["vers_values"]}/allowed_values.txt"
        log_dict["allowed_values"] = allowed_values_v1.strip()
        log_dict["file_name_json_scheme"] = f"json_scheme/{config["vers_json"]}.json"
        log_dict["json_scheme"] = scheme_v1      


    df, log_dict = get_data_df(data, log_dict)  # --> funktion definieren

    df, log_dict = case_escalation(df, mode, config, log_dict)
    log_dict["file_name_logic"] = f"logic/{config["vers_logic"]}"

    log_dict = evaluate_escalation(df, mode, config, log_dict)

    # log_dict["notes"] = (
    #         "known_limitation -- Baseline matches synthetic dataset features; expected to fail on ambiguous cases",
    #         "Recall undefined for classes with no true samples; zero_division=0 applied"
    #             )
    
    # run_id = "fa5f4e86a14f47d581d0c9c314f6829a"
    # code = log_dict["logic"]["code"]
    # fn_path = log_dict["file_name_logic"]

    # with mlflow.start_run(run_id=run_id):
    #     mlflow.log_text(code, f"{fn_path}/logic_snapshot.py")
    #     print("SUCCESS")

    if logging:
        log_path = Path(os.getenv("PATH_LOGDICT"))
        gh.save_dict(log_path, log_dict)

        mlflow_logging(log_dict)



if __name__ == "__main__":
    escalation_check()

    
#####

    # (1) standardisierten Evaluation-Pipeline (DataFrame → Metrics)
    # alle nachfolgenden SChritte pro Subgruppe dann psyche vs Körper, clear vs ambiguous 
    
    # df aufteilen
        # df_clear = df[df["clarity"] == "clear"].copy()
        # df_ambig = df[df["clarity"] == "ambiguous"].copy()
        # df_som = df[df["domain"] == "somatic"].copy()
        # df_psych = df[df["domain"] == "psych"].copy()
        # 
        # for df, name in zip([df, df_clear, df_ambig, df_body, df_psych],
        #               ["", "clear", "ambig", "soma", "psych"]): 

    # (2) Error-Bucket-Analyse
    # false_negatives = df[
    #         (df["escalation_required"] == True) &
    #         (df[f"pred_{mode}_{vers_run}"] == False)
    #     ]
    
    # create ClassReport
    # y_pred = df[f"pred_{mode}_{vers_run}"]

    # report = classification_report(y_true, 
    #                                y_pred, 
    #                                zero_division=0,
    #                                output_dict=True)

    
    # store parameters to log in dict 
    

    
        # metrics
        

        # comments and notes
        
            
   
