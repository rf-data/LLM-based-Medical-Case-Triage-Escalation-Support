## 
# imports

import os
from pathlib import Path
import json
import click

import src.utils.general_helper as gh
import src.utils.escalation_helper as esc
import src.utils.evaluation_helper as eval
import src.utils.path_helper as ph
import src.utils.file_helper as fh

import src.utils.escalation_llm as llm
import src.utils.escalation_baseline as base
# import src.utils.mlflow_helper as mh 

from src.utils.escalation_llm import llm_escalation_batch
from src.configuration.prompt import prompt_v1, allowed_values_v1
from src.configuration.json_scheme import scheme_v1

# from src.utils.mlflow_helper import mlflow_logging
from src.configuration.red_flags import RED_FLAGS

from src.core.session import session
from src.core.mlflow_logger import get_experiment_logger
# -----------------------
# HELPER FUNCTIONS
# -----------------------


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
    gh.load_env_vars()
    folder = os.getenv("PROCESSED")
    
    config = {
        "tags": {
            "approach": f"{mode}",
            "as batch": True, 
            "post-processing": True, 
            "vers_approach": "v3",
            "vers_data": "v2",
            "vers_logic": "v3",
            "vers_flags": "v1",
            "vers_prompt": "v1",
            "vers_values": "v1",
            "vers_json": "v1",
                },
        "parameter": {
            "git_commit": gh.get_git_commit(),
            "source_dataset": "synthetic",
            },
        # "artifact": {
        #     "filename_data": ,
        #     "filepath_data": ,

        #     },
        "mode": mode,
        "experiment_name": "escalation_clinical_reports", 
        "artifacts_location": os.getenv("MLFLOW_ARTIFACTS"),
        "namespace": "escalation_check",

        "prompt": prompt_v1.strip(),
        "allowed_values": allowed_values_v1.strip(),
        "json_scheme": scheme_v1,
        "dep_function": llm_escalation_batch,
        "dep_function_name": "llm_escalation_batch",
        # "escalation_rules": 
        }
    
    # setup session 
    session.load_config(config)
    session.save_session()
    # session.save_snapshot()

    # setup logger
    exp_name = config["experiment_name"]
    arti_loc = config["artifacts_location"]
    exp_logger = get_experiment_logger(
                experiment_name=exp_name,
                artifact_location=arti_loc
                            )
    exp_logger.setup_experiment()
    event_logger = exp_logger.logger

    session.backup_dir = exp_logger.backup_dir

    if mode == "baseline":
        vers_flags = session.tags.get("vers_flags", "tba")

        exp_logger.log_text(
                "red_flags.json",
                json.dumps(RED_FLAGS,
                        indent=2, ensure_ascii=False)
                        )

        path_flag = Path(f"{folder}/flag/{vers_flags}/red_flags.json")
        if not path_flag.exists():
            ph.ensure_dir(path_flag)
            fh.save_dict(path_flag, RED_FLAGS)            
        else:
            event_logger.error(f"File '{path_flag}' already exists. Hence, no overwrite")

    if mode == "llm":
        exp_logger.log_text(
                "system_prompt.txt", 
                config["prompt"]
                )

        exp_logger.log_text(
                "allowed_values.txt", 
                config["allowed_values"]
                )

        exp_logger.log_text(
                "json_scheme.json", 
                json.dumps(config["json_scheme"], 
                        indent=2, ensure_ascii=False)
                        )
        
        # save texts local
        vers_prompt = session.tags.get("vers_prompt", "tba")
        vers_values = session.tags.get("vers_values", "tba")
        vers_json = session.tags.get("vers_json", "tba")

        path_prompt = Path(f"{folder}/prompt/{vers_prompt}/system_prompt.txt")
        path_values = Path(f"{folder}/allowed_values/{vers_values}/allowed_values.txt")
        path_json = Path(f"{folder}/json_scheme/{vers_json}/json_scheme.json")

        for path, file in zip([path_prompt, path_values, path_json],
                            [config["prompt"], config["allowed_values"], config["json_scheme"]]):
            if not path.exists():
                if path != path_json:
                    ph.ensure_dir(path)
                    fh.save_text(path, file)
                    
                else:
                    fh.save_dict(path, file)
            else:
                event_logger.error(f"File '{path}' already exists. Hence, no overwrite")
    
    # get data
    df = esc.get_data_df(data)

    if mode == "baseline":
        df_esc = base.baseline_escalation(df)

    elif mode == "llm":
        df_esc = llm.llm_escalation(df)
        eval.evaluate_escalation(df_esc)

    else:
        event_logger.error(f"Unknown mode: {mode}")
        return 

    if logging:
        approach = config["tags"]["approach"]
        run_name = f"{approach}_{session.now}"
        exp_logger.flush(run_name)
        session.save_snapshot()
        exp_logger.local_backup() 

    else:
        # log_path = Path(os.getenv("PATH_LOGDICT"))
        session.save_snapshot()
        exp_logger.local_backup()   # log_path


if __name__ == "__main__":
    escalation_check()


# -----------------------
# 
# ---------------------

# def mlflow_logging(logger, data_dict, save_json=None):

#     tags_dict = data_dict["tag"]
#     params_dict = data_dict["parameter"]
#     artifacts_set = data_dict["artifact"]

#     # set tags
#     for key, value in tags_dict.items():
#         logger.set_tag(key, value) 


    # logging information on dataset
    # exp_logger.log_artifact(log_dict["file_path_data"])
    # exp_logger.log_artifact(
    #         log_dict["file_name_data"]
    #         )
    # exp_logger.log_param()
    # exp_logger.log_param()
    # exp_logger.log_param("size_dataset", 
    #                  log_dict["size_dataset"])
    # exp_logger.log_param("version_dataset", 
    #                  log_dict["version_dataset"])
    
    

    # exp_logger.set_tag("approach", log_dict["approach"])
    # exp_logger.set_tag("version_approach", 
    #                log_dict["version_approach"])
    # exp_logger.log_param("git commit", 
    #                  log_dict["git_commit"])
    
    

    # log_dict["version_logic"] = config["vers_logic"]

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
        
            
   
