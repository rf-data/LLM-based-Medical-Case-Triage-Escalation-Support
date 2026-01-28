import os
import utils.general_helper as gh

config = {
    "tags": {
        "approach": "rule",
        "subapproach": "baseline",
        "as batch": False,
        "vers_approach": "v1",
        "vers_data": "v2",
        "vers_logic": "v1",
        "vers_flags": "v1",
        "vers_prompt": "v1",
        "vers_values": "v1",
        "vers_json": "v1",
    },
    "parameter": {
        "git_commit": gh.get_git_commit(),
        "source_dataset": "synthetic",
    },
    "mode": "rule",
    "llm_model": None,
    "experiment_name": "escalation_clinical_reports",
    "artifacts_location": os.getenv("MLFLOW_ARTIFACTS"),
    "namespace": "escalation_check",
    "prompt": None,
    "allowed_values": None,
    "json_scheme": None,
    "dep_function": None,
    "dep_function_name": None,
}