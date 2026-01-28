from utils.model_helper import batch_escalation_by_llm
from configuration.prompt import prompt_v1, allowed_values_v1
from configuration.json_scheme import scheme_v1
import os
import utils.general_helper as gh

config = {
    "tags": {
        "approach": "llm",
        "subapproach": "baseline",
        "as batch": True,
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
    "mode": "llm",
    "llm_model": "openai",
    "experiment_name": "escalation_clinical_reports",
    "artifacts_location": os.getenv("MLFLOW_ARTIFACTS"),
    "namespace": "escalation_check",
    "prompt": prompt_v1.strip(),
    "allowed_values": allowed_values_v1.strip(),
    "json_scheme": scheme_v1,
    "dep_function": batch_escalation_by_llm,
    "dep_function_name": "batch_escalation_by_llm",
}