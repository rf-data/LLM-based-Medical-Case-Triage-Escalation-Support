import re
import json
import os
from pathlib import Path 

from configuration.red_flags import RED_FLAGS
from core.session import session
from core.mlflow_logger import get_experiment_logger
import utils.escalation_helper as esc
import utils.general_helper as gh


def escalate_by_rule(df_input):
    # load logger
    logger = get_experiment_logger()
    
    df = df_input.copy()

    def ruled_escalation(report: str) -> bool:
        text = report.lower()
        hits = []

        for category, keywords in RED_FLAGS.items():
            for kw in keywords:
                if re.search(rf"\b{kw}\b", text):
                    hits.append((category, kw))

        severe_markers = {"hypoton", "somnolent", "tachykard"}
        severe_hit = any(kw in text for kw in severe_markers)

        if len(hits) >= 1:
            return True
        if len(hits) >= 1 and severe_hit:
            return True

        return False
    
    fct_escalate = ruled_escalation

    # version_run = session.tags.get("vers_approach", "tba") # config["vers_run"]
    result_df = esc.df_iteration(df, fct_escalate)

    version = session.tags.get("vers_logic", "tba")
    # folder = os.getenv("PATH_MODEL_PARAMETER")
    fn_path = Path("logic_rule/{version}")

    df["expected_action_rule"] = result_df

    snapshot_dict = gh.snapshot_single_function(fct_escalate)

    # log_text
    logger.log_text(f"{fn_path}/logic_rule_complete.json",
                json.dumps(snapshot_dict,
                    indent=2, ensure_ascii=False))
        
    code = snapshot_dict["source"]
    logger.log_text(f"{fn_path}/logic_escalation_rule.py",
                    code) 

    func_name = snapshot_dict["name"]
    logger.set_tag("source", f"{func_name} ({version})")
        
    code_hashed = snapshot_dict["sha256"]
    logger.set_tag("source_hashed", code_hashed)

    return df




