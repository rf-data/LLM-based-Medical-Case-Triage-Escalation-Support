#!/usr/bin/env bash
set -e

# set -euo pipefail

# load env variable
source .env.frontend

# --- Base Setup ---
echo "=== Frontend EC2 Base Setup ==="

# --- System update ---
sudo apt update && sudo apt upgrade -y

# --- Core packages ---
sudo apt install -y \
  ca-certificates \
  curl \
  gnupg \
  git \
  tmux \
  awscli \
  nginx \
  ufw \
  fail2ban \
  # unattended-upgrades \
  certbot \
  python3-certbot-nginx 

# --- firewall --- 
sudo ufw allow OpenSSH
sudo ufw allow 80
sudo ufw allow 443
sudo ufw --force enable

# --- Swap ---
SWAPFILE="/swapfile"
SWAPSIZE="4G" # oder "8G" je nach Bedarf

echo "[INFO] Creating swapfile (${SWAPSIZE}) at ${SWAPFILE}"
if ! swapon --show | grep -q "${SWAPFILE}"; then
  # create file
  sudo fallocate -l ${SWAPSIZE} ${SWAPFILE}
  # secure permissions
  sudo chmod 600 ${SWAPFILE}
  # format as swap
  sudo mkswap ${SWAPFILE}
  # enable swap
  sudo swapon ${SWAPFILE}
  echo "${SWAPFILE} none swap sw 0 0" | sudo tee -a /etc/fstab
fi
echo "[INFO] Swap enabled:"

# --- Locale / Time ---
sudo timedatectl set-timezone Europe/Berlin
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8


# nginx log directory (prepare only)
sudo mkdir -p /var/log/nginx/custom
sudo chown -R www-data:adm /var/log/nginx/custom

# --- Project workspace ---
mkdir -p ~/project/{monitoring,data,logs}

echo "=== Base setup finished ==="

