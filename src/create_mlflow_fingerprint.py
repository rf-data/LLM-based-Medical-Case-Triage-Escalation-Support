
import os
# from pathlib import Path 
import src.utils.mlflow_helper as mh
import src.utils.general_helper as gh
# import src.utils.logger as log 

def create_mlflow_fingerprint(client, 
                              logger,
                              exp_name=None):
    
    if not exp_name:
        gh.load_env_vars()
        exp_name = os.getenv("FINGERPRINT_EXP")

    # 1. Experiment holen oder anlegen
    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        exp_id = client.create_experiment(exp_name)
    else:
        exp_id = experiment.experiment_id

    # 2. Expliziten Run erzeugen
    run = client.create_run(exp_id)

    # 3. Minimaler Fingerprint
    client.log_param(run.info.run_id, "fingerprint", "alive")
    client.set_tag(run.info.run_id, "purpose", "infrastructure_check")

    logger.info(f"Created fingerprint experiment ('{exp_name}', run id '{run.info.run_id}').")

    print(run.info.run_id)
    return run.info.run_id


if __name__ == "__main__":
    client, mlflow_log = mh.create_mlflow_client()
    create_mlflow_fingerprint(client, logger=mlflow_log)