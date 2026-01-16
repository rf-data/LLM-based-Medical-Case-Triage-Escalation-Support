#!/usr/bin/env bash
set -euo pipefail

SWAPFILE="/swapfile"
SWAPSIZE="4G" # oder "8G" je nach Bedarf

echo "[INFO] Adapting swapfile (${SWAPSIZE}) at ${SWAPFILE}"

# create file
fallocate -l ${SWAPSIZE} ${SWAPFILE}

# secure permissions
chmod 600 ${SWAPFILE}

# format as swap
mkswap ${SWAPFILE}

# enable swap
swapon ${SWAPFILE}

# persist across reboot
if ! grep -q "${SWAPFILE}" /etc/fstab; then
  echo "${SWAPFILE} none swap sw 0 0" >> /etc/fstab
fi

echo "[INFO] Swap enabled:"
swapon --show

# then 
# sysctl vm.swappiness=10
# echo "vm.swappiness=10" >> /etc/sysctl.conf