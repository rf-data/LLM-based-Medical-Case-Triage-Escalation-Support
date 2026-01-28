# imports 
from pathlib import Path
import os

from core.session import session
from core.mlflow_logger import get_experiment_logger
import utils.decision_helper as dh
import utils.path_helper as ph
import utils.file_helper as fh
import utils.general_helper as gh


def postprocess_escalation(df_input):
    # start logger
    exp_logger = get_experiment_logger()
    event_logger = exp_logger.logger

    version = session.tags.get("vers_logic", "tba")
    fn_path = f"logic_llm/{version}"
    folder = os.getenv("PATH_MODEL_PARAMETER")

    col_to_rename = {}
    for col in df_input.columns:
        parts = col.split(" ")
        if len(parts) >= 2:
            col_to_rename[col] = parts[0]

    df = df_input.rename(columns=col_to_rename)

    subapproach = session.tags.get("subapproach", None)
    if  subapproach == "post_processing":
        # create new features to ease decision wrt escalation
        df["n_risk_factors"] = df["risk_factors"].apply(len).astype(int)  
        df["n_missing_information"] = df["missing_information"].apply(len).astype(int)

        df_esc = dh.need_for_escalation(df)
        snapshot_dict = gh.snapshot_single_function(dh.need_for_escalation)

        # logging + saving files 
        exp_logger.set_tag("post_process_rule", snapshot_dict["name"])
        exp_logger.set_tag("post_process_rule_hashed", snapshot_dict["sha256"])
        exp_logger.log_text("logic_rules.py",
                        snapshot_dict["source"])

        logic_path = Path(f"{folder}/{fn_path}/logic_rules.py")
        if not logic_path.exists():
            ph.ensure_dir(logic_path)
            fh.save_text(logic_path, 
                         snapshot_dict["source"])
        else: 
            event_logger.error("File '%s' already exists. Hence, no overwrite", logic_path)
    else:
        df_esc = df.copy()

    return df_esc
    
if __name__ == "__main__":
    postprocess_escalation(df_input)