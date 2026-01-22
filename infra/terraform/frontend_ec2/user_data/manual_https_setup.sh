# !/bin/bash

# install certbot and obtain SSL certificate for franke-data.dev
sudo certbot --nginx -d franke-data.dev

# check
sudo nginx -t || {
  echo "ERROR: nginx config test failed after certbot"
  exit 1
}

# 
sudo systemctl reload nginx

# 
curl -I https://franke-data.dev

# renew certificate
sudo certbot renew --dry-run
