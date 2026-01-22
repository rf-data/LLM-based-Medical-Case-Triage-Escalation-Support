#!/usr/bin/env bash
set -e

# set -euo pipefail

# load env variable
# source .env.frontend

# --- Base Setup ---
echo "=== Frontend EC2 Base Setup ==="

# --- System update ---
if grep -qi ubuntu /etc/os-release; then
  sudo apt update && sudo apt upgrade -y

  # --- Core packages ---
  sudo apt install -y \
    ca-certificates \
    curl \
    gnupg \
    unzip \
    nginx \
    certbot \
    python3-certbot-nginx 
    # git \
    # tmux \
    # ufw \
    # fail2ban \
    # unattended-upgrades \
    # awscli \
else
  echo "Unsupported OS for certbot"
  exit 1
fi

# --- AWS CLI ---
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# --- firewall --- 
# sudo ufw allow OpenSSH
# sudo ufw allow 80
# sudo ufw allow 443
# sudo ufw --force enable

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

sudo mkdir -p /var/log/nginx
sudo touch /var/log/nginx/access.log /var/log/nginx/error.log
sudo chown -R www-data:adm /var/log/nginx
sudo chmod -R 750 /var/log/nginx

# --- Project workspace ---
# mkdir -p ~/project/{monitoring,data,logs}

echo "=== Base setup finished ==="

