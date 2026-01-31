## 
# imports

import os
from pathlib import Path
import json
import click

import utils.general_helper as gh
import utils.escalation_helper as esc
# import utils.evaluation_helper as eval
import utils.path_helper as ph
import utils.file_helper as fh
import utils.decision_helper as dh

from B1_rule_escalation import escalate_by_rule
from B2_llm_escalation import escalate_by_llm
from C1_llm_postprocess import postprocess_escalation
from D1_evaluation import evaluate_escalation
# import utils.mlflow_helper as mh 


# from utils.mlflow_helper import mlflow_logging
from configuration.red_flags import RED_FLAGS

from core.session import session
from core.mlflow_logger import get_experiment_logger
from configuration.B4_llm_post import config

# --------------
# MAIN FUNCTION
# --------------

@click.command()
@click.option(
    "--logging/--no-logging",
    default=False,
    help="Enable MLflow logging"
)
def escalate_reports(logging=False):
    # load env variable and configurations
    gh.load_env_vars()

    folder = os.getenv("PATH_MODEL_PARAMETER")

    # setup session 
    session.load_config(config)
    session.save_session()
    # session.save_snapshot()

    # setup logger
    exp_name = session.experiment_name
    arti_loc = session.artifacts_location
    exp_logger = get_experiment_logger(
                experiment_name=exp_name,
                artifact_location=arti_loc
                            )
    exp_logger.setup_experiment()
    event_logger = exp_logger.logger

    session.backup_dir = exp_logger.backup_dir

    mode = session.mode
    if mode == "rule":
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
            event_logger.error("File '%s' already exists. Hence, no overwrite", path_flag)

    if mode == "llm":
        exp_logger.log_text(
                "system_prompt.txt", 
                session.prompt
                )

        exp_logger.log_text(
                "allowed_values.txt", 
                session.allowed_values
                )

        exp_logger.log_text(
                "json_scheme.json", 
                json.dumps(session.json_scheme, 
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
                            [session.prompt, session.allowed_values, session.json_scheme]):
            if not path.exists():
                if path != path_json:
                    ph.ensure_dir(path)
                    fh.save_text(path, file)
                    
                else:
                    fh.save_dict(path, file)
            else:
                event_logger.error("File '%s' already exists. Hence, no overwrite", path)
    
    # load data
    data = session.tags.get("vers_data")
    subapproach = session.tags.get("subapproach", "na")

    df = esc.get_data_df(data)
        
    if mode == "rule":
        df_esc = escalate_by_rule(df) # baseline_escalation(df)
        df_dict = dh.evaluate_rule_fn_fp(df_esc)
        evaluate_escalation(df_dict["all"], pred_column="expected_action_rule")

        esc.save_escalation_df(df_esc)

    elif mode == "llm":
        # if subapproach == "postprocessing":
        df_esc = escalate_by_llm(df)
        df_post = postprocess_escalation(df_esc)
        df_dict = dh.evaluate_llm_fn_fp(df_post)

        pred_column = ("expected_action_final" 
                       if subapproach != "baseline" 
                       else "expected_action_llm")
        
        # encode ?
        y_true, y_pred = eval.encode_labels(df_dict["all"], pred_column)

        evaluate_escalation(y_true, y_pred)

        esc.save_escalation_df(df_post)

    else:
        event_logger.error("Unknown mode: %s", mode)
        return 

    if logging:
        approach = session.tags.get("approach")
        vers_approach = session.tags.get("vers_approach")

        run_name = f"{approach}_{vers_approach} ({subapproach} - {session.now})"
        exp_logger.flush(run_name)
        session.save_snapshot()
        exp_logger.local_backup() 

    else:
        session.save_snapshot()
        exp_logger.local_backup()   # log_path

if __name__ == "__main__":
    escalate_reports()
