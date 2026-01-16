#!/usr/bin/env bash
set -e

source .env.frontend

echo "=== Frontend Finalization (HTTPS + nginx cleanup) ==="

echo "[1/6] Preconditions check"

command -v certbot >/dev/null || {
  echo "ERROR: certbot not installed"
  exit 1
}

sudo ss -lntp | grep -q ":80" || {
  echo "ERROR: nginx not listening on port 80"
  exit 1
}

sudo ss -lntp | grep -q ":443" || {
  echo "ERROR: nginx not listening on port 443 (run certbot first)"
  exit 1
}

echo "[2/6] Disable legacy / conflicting nginx configs"

sudo mkdir -p "$DISABLED_DIR"

for f in "$NGINX_CONF_DIR"/*.conf; do
  case "$f" in
    *frontend.conf) ;;   # keep canonical file
    *)
      echo "  disabling $(basename "$f")"
      sudo mv "$f" "$DISABLED_DIR"/
      ;;
  esac
done

echo "[3/6] Write canonical frontend nginx config"

sudo tee "$NGINX_CONF_DIR/frontend.conf" > /dev/null <<EOF
############################
# HTTP → HTTPS Redirects
############################

server {
    listen 80;
    server_name $DOMAIN_MAIN;
    return 301 https://$DOMAIN_MAIN\$request_uri;
}

server {
    listen 80;
    server_name $DOMAIN_CODE;
    return 301 https://$DOMAIN_CODE\$request_uri;
}

############################
# HTTPS – Homepage
############################

server {
    listen 443 ssl;
    server_name $DOMAIN_MAIN;

    ssl_certificate /etc/letsencrypt/live/$DOMAIN_MAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN_MAIN/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;

    root /usr/share/nginx/html;
    index index.html index.htm;

    location / {
        try_files \$uri \$uri/ =404;
    }
}

############################
# HTTPS – code-server
############################

server {
    listen 443 ssl;
    server_name $DOMAIN_CODE;

    ssl_certificate /etc/letsencrypt/live/$DOMAIN_MAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN_MAIN/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;

    location / {
        proxy_pass http://127.0.0.1:8443;
        proxy_http_version 1.1;

        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection upgrade;
        proxy_set_header Accept-Encoding "";

        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Proxy to Streamlit Portfolio App
    location /portfolio/ {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;

        # Path rewrite so Streamlit sees the correct path
        rewrite ^/portfolio/(.*)$ /$1 break;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Websockets
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

    }
}
EOF

echo "[4/6] nginx syntax check"
sudo nginx -t

echo "[5/6] Reload nginx"
sudo systemctl reload nginx

echo "[6/6] Final verification"

curl -s -I "https://$DOMAIN_MAIN" | head -n 1
curl -s -I "https://$DOMAIN_CODE" | head -n 1

echo "=== Frontend finalized successfully ==="
