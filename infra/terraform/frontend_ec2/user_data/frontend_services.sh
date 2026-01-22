#!/usr/bin/env bash
set -euo pipefail

# load env variables
source .env.frontend

# : "${CODE_PORT:?CODE_PORT must be set}"
# : "${DOMAIN_CODE:?DOMAIN_CODE must be set}"
: "${NGINX_CONF_DIR:?NGINX_CONF_DIR must be set}"
: "${DOMAIN_NAME:?DOMAIN_NAME must be set}"
: "${DOMAIN_MAIN:?DOMAIN_MAIN must be set}"
: "${WEB_ROOT:?WEB_ROOT must be set}"

# --- Frontend Services Setup ---
echo "=== Frontend Services Setup ==="

### -------------------------------
### nginx vorbereiten
### -------------------------------
echo "[1/7] Preparing nginx base config"

sudo systemctl enable nginx
sudo systemctl start nginx

# Sauberer Default-vHost (kein Proxy!)
sudo rm -f /etc/nginx/sites-enabled/default

# sudo nginx -t
# sudo systemctl reload nginx

# sudo tee "$NGINX_CONF_DIR/00-default.conf" > /dev/null <<EOF
# server {
#     listen 80 default_server;
#     server_name _;

#     root $WEB_ROOT;
#     index index.html;

#     location / {
#         try_files \$uri \$uri/ =404;
#     }
# }
# EOF

sudo mkdir -p "$WEB_ROOT"
echo "nginx is alive (HTTP OK)" | sudo tee "$WEB_ROOT/index.html" > /dev/null

sudo logrotate -d /etc/logrotate.d/nginx

### -------------------------------
### Homepage vHost
### -------------------------------
echo "[2/7] Configuring homepage vHost"

echo "[x] Disabling nginx default site"
rm -f /etc/nginx/sites-enabled/default

sudo tee "$NGINX_CONF_DIR/$DOMAIN_NAME.conf" > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN_MAIN;

    root $WEB_ROOT;
    index index.html;

    location / {
        try_files \$uri \$uri/ =404;
    }
}
EOF

# ln 
### -------------------------------
### code-server vHost (Reverse Proxy)
### -------------------------------
echo "[3/7] Configuring code-server proxy --> SKIPPING"

# sudo tee "$NGINX_CONF_DIR/code-server.conf" > /dev/null <<EOF
# server {
#     listen 80;
#     server_name $DOMAIN_CODE;

#     location / {
#         proxy_pass http://127.0.0.1:$CODE_PORT;
#         proxy_http_version 1.1;

#         proxy_set_header Upgrade \$http_upgrade;
#         proxy_set_header Connection upgrade;

#         proxy_set_header Host \$host;
#         proxy_set_header X-Real-IP \$remote_addr;
#         proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
#         proxy_set_header X-Forwarded-Proto \$scheme;
#     }
# }
# EOF

sudo nginx -t
sudo systemctl reload nginx

### -------------------------------
### code-server via tmux sicherstellen
### -------------------------------
echo "[4/7] Ensuring code-server is running in tmux --> SKIPPING"

# if ! tmux has-session -t code 2>/dev/null; then
#   tmux new -d -s code \
#     "code-server --bind-addr 127.0.0.1:$CODE_PORT --auth password"
# fi

### -------------------------------
### HTTPS (OPTIONAL, bewusst spät)
### -------------------------------
echo "[5/7] HTTPS setup (certbot)"
echo ">>> Skipped by default."
echo ">>> Run manually when DNS is correct:"
echo "sudo certbot --nginx -d $DOMAIN_MAIN" 
# -d $DOMAIN_CODE

### -------------------------------
### Autostart code-server (@reboot)
### -------------------------------
echo "[6/7] Registering code-server autostart (@reboot) --> SKIPPING"

# (crontab -l 2>/dev/null | grep -v "code-server") | crontab -
# (crontab -l 2>/dev/null; echo "@reboot tmux new -d -s code 'code-server --bind-addr 127.0.0.1:$CODE_PORT --auth password'") | crontab -

### -------------------------------
### Final check
### -------------------------------
echo "[7/7] Final checks"
echo "Homepage:    http://$DOMAIN_MAIN"
# echo "Code-Server: http://$DOMAIN_CODE"
echo "HTTPS:       enable manually via certbot"

### -------------------------------
### potential additions
### -------------------------------
# sudo systemctl enable fail2ban
# sudo systemctl start fail2ban


echo "=== Frontend services setup finished ==="
