# Frontend EC2 – Runbook

## Overview

This document describes the setup, recovery, and validation steps for the
Frontend EC2 instance serving:

- https://franke-data.dev (homepage)
- https://code.franke-data.dev (VS Code / code-server)

The stack consists of:
- Ubuntu EC2
- nginx (reverse proxy + TLS)
- Let’s Encrypt (certbot)
- code-server (via tmux)
- Optional Streamlit app at `/portfolio`

---

## Architecture

Internet
|
| :80 → redirect
| :443 → TLS
v
nginx (EC2)
├── static HTML (/)
├── code-server → 127.0.0.1:8443
└── streamlit → 127.0.0.1:8501


---

## Preconditions

- Elastic IP attached to the EC2
- DNS A-records:
  - franke-data.dev
  - code.franke-data.dev
- Ports open in Security Group:
  - 80 / 443 (public)
- OS-level firewall allows 80 / 443

---

## Setup Phases

### Phase 1 – Base OS Setup
Script: `base_setup.sh`

Includes:
- system update
- core packages
- nginx
- certbot
- firewall (ufw)
- swap
- locale & timezone

---

### Phase 2 – Frontend Services
Script: `frontend_services.sh`

Includes:
- nginx HTTP vHosts
- reverse proxy for code-server
- tmux-based code-server startup
- HTTP-only routing

---

### Phase 3 – Finalization (TLS & Cleanup)
Script: `frontend_finalize.sh`

Includes:
- canonical nginx config
- HTTPS-only routing
- removal of conflicting configs
- reload & verification

---

## Validation Checklist

**Run on the EC2:**
```bash
nginx -t
systemctl status nginx
ss -lntp | grep -E ":(80|443|8443)"
```

**From anywhere:**
```bash
curl -I https://franke-data.dev
curl -I https://code.franke-data.dev
```

*Expected output:*   
- Homepage: 200 OK   
- Code-server: 302 → /login


## Common Failure Modes
### Redirect Loop

**Cause:**   
- HTTPS server block contains redirect   
- Fix --> Redirects only on port 80

### 502 Bad Gateway

**Cause:**   
- code-server not running   
- Fix --> `tmux attach -t code` or   
```tmux new -d -s code "code-server --bind-addr 127.0.0.1:8443 --auth password"```

### TLS Errors

**Cause:**   
- certbot incomplete or DNS mismatch   
- Fix --> 
```
certbot certificates \
certbot --nginx -d franke-data.dev -d code.franke-data.dev
```

## Recovery (New EC2)
(1) Attach Elastic IP
(2) Verify DNS
(3) Run:
- `base_setup.sh`
- `frontend_services.sh`
(4) Run `certbot`
(5) Run `frontend_finalize.sh`
(6) Validate endpoints

## Notes
- No PHP / FPM is used   
- nginx configs are canonicalized in /etc/nginx/conf.d/frontend.conf   
- code-server runs via tmux (@reboot)