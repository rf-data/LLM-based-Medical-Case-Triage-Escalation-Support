#!/usr/bin/env bash

# exit on error
set -e

# ----------------------------
# Paths & logging
# ----------------------------
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

mkdir -p "$PROJECT_ROOT/logs"
LOGFILE="$PROJECT_ROOT/logs/0_mlflow_setup.log"

# ----------------------------
# Load environment variables
# ----------------------------
{
  echo ""
  echo "===== START MLFLOW_SERVER_SETUP [$(date '+%Y-%m-%d %H:%M:%S')] ===="
  if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment variables from ../.env"
    set -o allexport
    source "$PROJECT_ROOT/.env"
    set +o allexport
  else
    echo ""
    echo "No .env file found - relying on defaults"  
  fi
} >> "$LOGFILE" 2>&1

# ----------------------------
# Defaults (only for SERVER)
# ----------------------------
MLFLOW_DB=${MLFLOW_DB:-"sqlite:///mlflow/mlflow.db"} 
ARTIFACT_DIR=${MLFLOW_ARTIFACTS:-"./mlflow/artifacts"}

# Kürzen auf die letzten 2–3 Teile des Pfads
short_db=$(echo "$MLFLOW_DB" | awk -F'/' '{print $(NF-2)"/"$(NF-1)"/"$NF}')
short_artifacts=$(echo "$ARTIFACT_DIR" | awk -F'/' '{print $(NF-1)"/"$NF}')

# ----------------------------
# Start MLflow server
# ----------------------------
{
  echo "DB: .../$short_db"
  echo "Artifacts: .../$short_artifacts"

  # SQLite --> only local + one user access possible
  mlflow server \
    --backend-store-uri $MLFLOW_DB \
    --default-artifact-root $ARTIFACT_DIR \
    --host 127.0.0.1 \
    --port 5000 \
     >> "$LOGFILE" 2>&1 &
  
  echo "MLflow server started with PID $!"
  echo "===== END MLFLOW_SERVER_SETUP [$(date '+%Y-%m-%d %H:%M:%S')] ===="
} >> "$LOGFILE" 2>&1

# ----------------------------
# Wait until server is ready
# ----------------------------
{
  echo ""
  echo "===== START MLFLOW_SERVER_CHECK [$(date '+%Y-%m-%d %H:%M:%S')] ===="
  echo "Waiting for MLflow server to become available..."
  
  until ss -ltn | grep -q ':5000'; do sleep 1; done
  echo ""
  echo "Port is open."
  echo ""
  until curl -sf http://127.0.0.1:5000/ > /dev/null; do sleep 1; done
  echo "HTTP/API has been reached."
  # until curl -sf http://127.0.0.1:5000/api/2.0/mlflow/experiments/list > /dev/null; do
  #   sleep 1
  # done
  
  echo "MLflow server is ready"
} >> "$LOGFILE" 2>&1

# ----------------------------
# Check 1: Python test module
# ----------------------------
{
  echo ""
  echo "Check #1: Python test module (src.mlflow_setup_test)"
  python3 -m src.mlflow_setup_test
} >> "$LOGFILE" 2>&1

# ----------------------------
# Check 2: Raw Python URI check
# ----------------------------
{
  echo ""
  echo "Check #2: Raw Python tracking URI"
} >> "$LOGFILE" 2>&1

python - <<'EOF' >> "$LOGFILE" 2>&1
import mlflow
print("Tracking URI =", mlflow.get_tracking_uri())
EOF

{
  echo ""
  echo "===== END MLFLOW_SERVER_CHECK [$(date '+%Y-%m-%d %H:%M:%S')] ====="
} >> "$LOGFILE" 2>&1
