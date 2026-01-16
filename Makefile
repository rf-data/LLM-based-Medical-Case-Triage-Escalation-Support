ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT := llm_escalation
# ENV ?= core 
# N_NEIGHBORS ?= 5
# MSG ?= auto
# MODE == 
# NUM ==

.PHONY: mlflow # etl setup_env repo_push    # etl create_embeds

# setup_env:
# 	bash "${ROOT}/scripts/0_init_setup.sh"

# repo_push:
# 	bash "${ROOT}/scripts/0_repo_push.sh" 

req_files:
	uv pip compile pyproject.toml -o requirements.txt
	uv pip compile pyproject.toml --group mlops -o requirements-mlops.txt
	uv pip compile pyproject.toml --group heavy -o requirements-heavy.txt
	uv pip compile pyproject.toml --group mlops --group heavy --group dev  -o requirements-dev.txt

mlflow_local:
	${ROOT}/scripts/0_setup_mlflow.sh

clear_cache:
	python ${ROOT}/src/clear_llm_cache.py
	
all_stop:
	@docker ps -aq | xargs -r docker stop

all_remove:
	@docker ps -aq | xargs -r docker rm -v

ml_docker:
	docker compose -p $(PROJECT) -f $(ROOT)/docker-compose.ml.yaml up --build -d

ml_stop:
	docker compose -p $(PROJECT)  -f $(ROOT)/docker-compose.api.yaml down
	
monitoring_docker:
	docker compose -p $(PROJECT) -f $(ROOT)/docker-compose.monitoring.yaml up --build -d

monitoring_stop:
	docker compose -p $(PROJECT) -f $(ROOT)/docker-compose.monitoring.yaml down

