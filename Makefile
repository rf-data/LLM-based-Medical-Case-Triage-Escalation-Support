ROOT := $(CURDIR)
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

mlflow:
	${ROOT}/scripts/0_setup_mlflow.sh

# etl:
# 	bash "${ROOT}/scripts/1_ETL.sh"

# all: 
# 	docker-compose up --build -d

# stop: 
# 	docker-compose down

# drift_reports:
# 	python3 src/??

# evaluation:
# 	docker-compose up -d --build evaluation

# fire-alert:
	
# create_embeds:
# 	bash ${ROOT}/scripts/2_create_embeds.sh 

# data_split:
# 	bash ${ROOT}/scripts/0_data_split.sh ${MODE}=split ${NUM}

# sample_data:
# 	bash ${ROOT}/scripts/0_sample_data.sh ${MODE}=split ${NUM}

# create_sim_mat:
# 	bash ${ROOT}/src/3_create_SimMat.sh ${ENV} ${N_NEIGHBORS}

# recommend:
# 	bash ${ROOT}/src/4_recommend.sh ${ENV} ${N_NEIGHBORS}

