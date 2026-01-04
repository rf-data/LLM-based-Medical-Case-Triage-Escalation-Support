#!/usr/bin/env bash

# exit on error
set -e

# echo "DEBUG: SCRIPT REACHED"

# define paths
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

mkdir -p "$PROJECT_ROOT/logs"
LOGFILE="$PROJECT_ROOT/logs/0_mlflow.log"

# import env variables from .env
{
  if [ -f "$PROJECT_ROOT/.env" ]; then
      echo "Loading environment variables from ../.env"
      set -o allexport
      source "$PROJECT_ROOT/.env"
      set +o allexport
  fi
} >> "$LOGFILE" 2>&1

# Fallback falls kein .env vorhanden
MLFLOW_DB=${MLFLOW_DB:-"sqlite:///mlflow/mlflow.db"} 
ARTIFACT_DIR=${MLFLOW_ARTIFACTS:-"./mlflow/artifacts"}

# Kürzen auf die letzten 2–3 Teile des Pfads
short_db=$(echo "$MLFLOW_DB" | awk -F'/' '{print $(NF-2)"/"$(NF-1)"/"$NF}')
short_artifacts=$(echo "$ARTIFACT_DIR" | awk -F'/' '{print $(NF-1)"/"$NF}')

# run script
{
  echo ""
  echo "===== START MLFLOW_SERVER_SETUP [$(date '+%Y-%m-%d %H:%M:%S')] ===="
  echo "DB: .../$short_db"
  echo "Artifacts: .../$short_artifacts"

  # SQLite --> only local + one user access possible
  exec mlflow server \
    --backend-store-uri $MLFLOW_DB \
    --default-artifact-root $ARTIFACT_DIR \
    --host 127.0.0.1 \
    --port 5000 \
     >> "$LOGFILE" 2>&1 &
  
  echo "MLflow server started with PID $!"
  echo ""
  echo "===== END MLFLOW_SERVER_SETUP [$(date '+%Y-%m-%d %H:%M:%S')] ===="
}
{
  echo ""
  echo "===== START MLFLOW_SERVER_CHECK [$(date '+%Y-%m-%d %H:%M:%S')] ===="
  echo "WAIT....."
  # 1. Server läuft?
  until curl -sf http://127.0.0.1:5000/api/2.0/mlflow/experiments/list > /dev/null; do
    sleep 1
  done
  echo "MLflow server is ready"
  
  echo "Check #1: Python test script:"
  python3 -m src.mlflow_setup_test
  echo ""
  
  
  # curl http://127.0.0.1:5000/api/2.0/mlflow/experiments/list
  echo ""
  echo "Check #2: Retrieve URI by shell command"
  python - <<'EOF' 
  import mlflow
  print('URI =', mlflow.get_tracking_uri())
  EOF  
  echo ""
  echo "===== END MLFLOW_SERVER_CHECK [$(date '+%Y-%m-%d %H:%M:%S')] ===="
}  >> "$LOGFILE" 2>&1
