# imports
import json
import os
from pathlib import Path 

from core.mlflow_logger import get_experiment_logger
# from utils.escalation_baseline import baseline_escalation
from core.session import session
import utils.escalation_helper as esc
import utils.general_helper as gh
import utils.path_helper as ph
import utils.file_helper as fh


def escalate_by_llm(df_input):
    # load logger
    logger = get_experiment_logger()
    
    df = df_input.copy()    
    # if mode == "llm":
    prompt = session.prompt
    scheme = session.json_scheme
    allowed_values = session.allowed_values
    namespace = session.namespace
    batch_mode = session.tags.get("as batch", True)

    fct_escalate = esc.use_escalation_cache(prompt=prompt,
                                        scheme=scheme, 
                                        allowed_values=allowed_values,
                                        namespace=namespace,
                                        batch_mode=batch_mode)

    # mode = session.mode
    # version_run = session.tags.get("vers_approach", "tba") # config["vers_run"]
    result_df = esc.df_iteration(df, fct_escalate)
    result_df_re = result_df.rename(columns={"expected_action": "expected_action_llm"}).copy()
    result_df_re["expected_action"] = df["expected_action"]
    result_df_re["clarity"] = df["clarity"]
    result_df_re["domain"] = df["domain"]

    # df[f"risk_factors ({mode}_{version_run})"] = result_df["risk_factors"]
    # df[f"missing_information ({mode}_{version_run})"] = result_df["missing_information"]
    # df[f"severity ({mode}_{version_run})"] = result_df["severity"]
    # result_df[f"expected_action_llm ({mode}_{version_run})"] = result_df["expected_action"]
    # df[f"confidence ({mode}_{version_run})"] = result_df["confidence"]
    # df[f"uncertainty ({mode}_{version_run})"] = result_df["uncertainty_level"] # or result_df["uncertainty"]
    # df[f"confidence_derived ({mode}_{version_run})"] = result_df["confidence_derived"]
                            
    func_name = session.dep_function_name
    func = session.dep_function

    snapshots = gh.snapshot_dependent_functions(fct_escalate,
                                                dependencies=[func])

    # log_text
    logger.log_text("logic_complete.json",
                json.dumps(snapshots,
                        indent=2, ensure_ascii=False))

    logic_root = snapshots["root"]["source"]
    logger.log_text("logic_root.py",
                    logic_root) 

    logic_escalation = snapshots["dependencies"][f"{func_name}"]["source"]
    logger.log_text("logic_escalation.py",
                    logic_escalation) 
    

    # save texts local
    version = session.tags.get("vers_logic", "tba")
    folder = os.getenv("PATH_MODEL_PARAMETER")
    path_comp = Path(f"{folder}/logic_llm/{version}/logic_complete.json") 
    path_root = Path(f"{folder}/logic_llm/{version}/logic_root.py") 
    path_esc = Path(f"{folder}/logic_llm/{version}/logic_escalation.py") 

    for path, file in zip([path_comp, path_root, path_esc],
                    [snapshots, logic_root, logic_escalation]):
        if not path.exists():
            ph.ensure_dir(path)
            if path != path_comp:
                fh.save_text(path, file)
            else:
                fh.save_dict(path, file)
        else:
            print(f"File '{path}' already exists. Hence, no overwrite")

    # set_tag
    logger.set_tag("source", f"{func_name} ({version})")
        
    dep_func_hashed = snapshots["dependencies"][f"{func_name}"]["sha256"]
    logger.set_tag("source_hashed", dep_func_hashed)
   
    return result_df_re

if __name__ == "__main()__":
    escalate_by_llm()


