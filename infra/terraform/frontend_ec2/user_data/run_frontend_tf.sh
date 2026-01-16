#!/usr/bin/env bash
set -euo pipefail

source .env.frontend

mkdir -p "${TF_LOG_DIR}"

echo "==== [TIME??] ==== Running Frontend-EC2 Terraform ===="
exec > >(tee -a "${TF_LOG_DIR}/${TF_STDOUT_LOG}") \
     2> >(tee -a "${TF_LOG_DIR}/${TF_STDERR_LOG}" >&2)

terraform init
terraform plan
terraform apply
