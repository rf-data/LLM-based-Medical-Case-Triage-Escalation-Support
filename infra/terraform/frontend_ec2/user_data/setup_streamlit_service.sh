#!/usr/bin/env bash
set -euo pipefail

source .env.frontend

# 
# cd "$PROJECT_PATH"
# source .venv/bin

sudo tee "/etc/systemd/system/streamlit.service" > /dev/null <<EOF
[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/$PROJECT_NAME
ExecStart=/home/ubuntu/$PROJECT_NAME/.venv/bin/streamlit run streamlit/main.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true

Restart=always 
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=30

EOF

# 
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl start streamlit

# check for streamlit status
echo "Status -- Streamlit"
systemctl status streamlit --no-pager

echo
echo "Check if streamlit port is reachable"
curl -I https://127.0.0.1:8501
