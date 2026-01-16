#!/usr/bin/env bash
set -e

echo "=== Rakuten MLOps Base Setup ==="

# --- System update ---
sudo apt update && sudo apt upgrade -y

# --- Core packages ---
sudo apt install -y \
  ca-certificates \
  curl \
  gnupg \
  git \
  tmux \
  awscli

# --- Docker installation ---
if ! command -v docker &> /dev/null; then
  echo "Installing Docker..."
  sudo mkdir -p /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

  echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

  sudo apt update
  sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# --- Docker permissions ---
sudo usermod -aG docker $USER || true

# --- Project workspace ---
# mkdir -p ~/rakuten-mlops/{monitoring,data,logs}

echo "Setup finished. Please re-login to apply Docker group changes."
