#!/usr/bin/env bash

# exit on error
set -e

mlflow experiments create \
  --experiment-name escalation_clinical_reports \
  --artifact-location file:///home/robfra/0_Portfolio_Projekte/LLM/mlflow/artifacts
