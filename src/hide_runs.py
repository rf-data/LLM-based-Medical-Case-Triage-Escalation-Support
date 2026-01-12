##
# imports
# from mlflow.tracking import MlflowClient
# import os
# import mlflow
# import src.utils.general_helper as gh
import src.utils.mlflow_helper as mh

run_ids = ["600d619c821b4c708c256d71d7972b7d",
           "6bf3a05ca3ed4612a4807dd2f953246e",
           "afef7889edbf4227a329d3b451b91d79",
           "5efb973e89ca46a397c18a266c062781"
           ]


def hide_runs(client, run_ids, mode: str="hide"):    
    # 
    value = "visible" if mode == "clear" else "hidden"

    for run_id in run_ids:
        client.set_tag(run_id, "visibility", value)
        print(f"[SUCCESS] set visibility to '{value}' for run '{run_id}'")
            
            # log_text(code, f"{fn_path}/logic_snapshot.py")
    #     

client = mh.create_mlflow_client()

hide_runs(client, run_ids, mode="clear")