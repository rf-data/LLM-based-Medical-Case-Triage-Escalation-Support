##
# imports
from mlflow.tracking import MlflowClient

# import os
# import mlflow
import src.utils.general_helper as gh
import src.utils.mlflow_helper as mh
import src.utils.logger as log 


escalate_base = ""
escalate_base_hashed = ""
escalate_llm = ""
escalate_llm_hashed = ""


fct_base = baseline_escalation
# fct_llm_seq = eh.make_cached_escalation(prompt=prompt,
#                                                  scheme=scheme, 
#                                                  allowed_values=allowed_values,
#                                                  namespace=namespace) 

fct_llm_batch = eh.make_cached_escalation(prompt=prompt,
                                        scheme=scheme, 
                                        allowed_values=allowed_values,
                                        namespace=namespace)

func_name, dep_func = config.get("dependent_function", (None, None))

snapshot_base_dict = gh.snapshot_single_function(fct_base) 

snapshot_llm_seq = gh.snapshot_dependent_functions(fct_llm_seq,
                                                dependencies=[dep_func])

snapshot_llm_batch = gh.snapshot_dependent_functions(fct_llm_batch,
                                                dependencies=[dep_func])

update_dict = {
            "5efb973e89ca46a397c18a266c062781": 
                {"source": escalate_base,
                 "source_hashed": escalate_base_hashed},
            "afef7889edbf4227a329d3b451b91d79": 
                {"source": escalate_base,
                 "source_hashed": escalate_base_hashed},
            "6bf3a05ca3ed4612a4807dd2f953246e":
                {"source": escalate_llm,
                 "source_hashed": escalate_llm_hashed},
            "600d619c821b4c708c256d71d7972b7d":
                {"source": escalate_llm,
                 "source_hashed": escalate_llm_hashed}
                }

def update_artifacts(client, artifact_dict: dict, logger):
    for run_id, tags in artifact_dict.items():
        for k, v in tags.items():
            client.set_tag(run_id, k, v)
            logger.info(f"[TAG] successfully add artifact '{k}' for run '{run_id}'")


def update_tags(client, tag_dict: dict, logger):
    for run_id, tags in tag_dict.items():
        for k, v in tags.items():
            client.set_tag(run_id, k, v)
            logger.info(f"[TAG] successfully set tag '{k}:{v}' for run '{run_id}'")


def update_runs(client, update_dict: dict, logger):
    tags = ["source", "source_hashed", "version_approach", "approach", "visibility"]
    artifacts = ["json_scheme", 
                 "logic_complete", "logic_escalation", "logic_root", 
                 "allowed_values", "system_prompt",
                 "ClassReport", "Dataset_Prediction"]

    tag_dict = {}
    artifact_dict = {}
    for _, tags in update_dict.items():
        for k, _ in tags.items():
            if k in tags:
                tag_dict[k] = tags[k]
            
            if k in artifacts: 
                artifact_dict[k] = tags[k]

    if len(tag_dict) > 0:
        update_tags(client, tag_dict, logger)

    if len(artifact_dict) > 0:
        update_artifacts(client, artifact_dict, logger)

if __name__ == "__main__":
    client, mlflow_log = mh.create_mlflow_client()
    update_runs(client, update_dict, mlflow_log)
    # (client, tag_dict, mlflow_log)

# hide_runs(client, run_ids, mode="clear")