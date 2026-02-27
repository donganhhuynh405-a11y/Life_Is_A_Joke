# Security Best Practices

This document describes security measures for deploying and operating the trading bot safely.

---

## API Key Management

### Never hardcode credentials

All secrets must be supplied via environment variables or a secrets manager — never committed to source control.

```bash
# Correct: environment variables
export BINANCE_API_KEY="abc..."
export BINANCE_SECRET="xyz..."

# Correct: .env file (never commit this file)
echo ".env" >> .gitignore
```

### Principle of Least Privilege

Configure exchange API keys with only the permissions the bot needs:

| Permission | Required | Notes |
|------------|----------|-------|
| Read (market data) | ✅ Yes | Always required |
| Spot trading | ✅ Yes | If trading spot |
| Futures trading | Only if needed | Enable only for futures strategies |
| Withdrawals | ❌ No | **Never enable** |
| Transfer | ❌ No | **Never enable** |

### IP Whitelisting

Restrict API keys to your server's static IP address in the exchange dashboard. This limits damage if a key is leaked.

---

## Secret Storage

### Local Development

Use a `.env` file (excluded from git):

```bash
# .env  — never commit this
BINANCE_API_KEY=abc123
BINANCE_SECRET=xyz789
TELEGRAM_BOT_TOKEN=...
API_TOKEN=<random-32-char-string>
```

Generate a strong API token:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Production

Use a dedicated secrets manager:

- **Docker Swarm**: Docker Secrets
- **Kubernetes**: Kubernetes Secrets (prefer sealed-secrets or Vault)
- **Cloud**: AWS Secrets Manager / GCP Secret Manager / Azure Key Vault

Example Kubernetes secret:
```bash
kubectl create secret generic trading-bot-secrets \
  --from-env-file=.env \
  --namespace=trading
```

---

## Network Security

### TLS / HTTPS

Always run the API behind a TLS-terminating reverse proxy (Nginx/Caddy) in production. Never expose the bot's HTTP port directly to the internet.

```nginx
# deployment/nginx/trading-bot.conf
server {
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    location /api/ {
        proxy_pass http://127.0.0.1:8080;
    }
}
```

### Firewall Rules

```bash
# Allow only HTTPS and SSH
ufw default deny incoming
ufw allow 22/tcp   # SSH
ufw allow 443/tcp  # HTTPS (Nginx)
ufw enable
```

### VPN

For additional security, restrict API access to a VPN network only (WireGuard or OpenVPN).

---

## Docker Security

```yaml
# docker-compose.yml hardening
services:
  bot:
    user: "1000:1000"           # Non-root user
    read_only: true             # Read-only filesystem
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    tmpfs:
      - /tmp                    # Writable tmp in memory
```

---

## Data Security

### Encrypt Data at Rest

- Database files and backups should be stored on encrypted volumes.
- S3 buckets (if used for model storage) must have SSE enabled.

### Sensitive Logs

Ensure API keys and order details are redacted in logs:

```python
# src/utils.py
import re

def sanitize_log(message: str) -> str:
    return re.sub(r'(api[_-]?key|secret|token)["\s:=]+\S+', r'\1=***', message, flags=re.IGNORECASE)
```

### Log Rotation

```bash
# deployment/logrotate/tradingbot
/var/log/tradingbot/*.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
}
```

---

## Dependency Security

Keep dependencies updated and scan regularly:

```bash
# Check for known vulnerabilities
pip install pip-audit
pip-audit -r requirements.txt

# Keep dependencies updated
pip list --outdated
```

Pin exact versions in `requirements.txt` and update deliberately (not automatically in production).

---

## Authentication & Authorization

### Bot API

- All API endpoints require Bearer token authentication.
- Use a strong random token (32+ bytes).
- Rotate the token periodically or after any suspected compromise.

### Rate Limiting

The bot API enforces rate limits by default (100 req/min per IP). Configure in `config.yaml`:

```yaml
api:
  rate_limit_per_minute: 100
  rate_limit_burst: 20
```

---

## Incident Response

If you suspect a compromise:

1. **Immediately revoke** exchange API keys from the exchange dashboard.
2. **Stop the bot**: `docker-compose stop` or `systemctl stop tradingbot`.
3. **Rotate all secrets**: Generate new API keys, tokens, and passwords.
4. **Audit trade history** on the exchange for unauthorized activity.
5. **Review logs** for suspicious activity.
6. **Report** to the exchange's security team if unauthorized trades occurred.

---

## Reporting Vulnerabilities

See [SECURITY.md](../SECURITY.md) at the repository root for our security disclosure policy. Do **not** open public GitHub issues for security vulnerabilities.
