# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.3.x   | ✅ Yes |
| 1.2.x   | ✅ Yes (security fixes only) |
| < 1.2   | ❌ No |

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security issues privately via one of these channels:

1. **GitHub Private Vulnerability Reporting** — use the "Report a vulnerability" button on the [Security tab](../../security/advisories/new) of this repository.
2. **Email** — send details to `security@your-org.example.com` (replace with your actual address).

Include in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact (e.g., unauthorized trades, credential exposure)
- Any suggested mitigations

We will acknowledge receipt within **48 hours** and aim to provide an initial assessment within **7 days**.

---

## Disclosure Policy

- We follow [responsible disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure).
- Once a fix is available, we will coordinate with the reporter on public disclosure timing.
- Reporters who follow responsible disclosure will be credited in the release notes (unless they prefer anonymity).

---

## Security Considerations for Users

### API Keys

- **Never enable withdrawal permissions** on exchange API keys used by this bot.
- Enable **IP whitelisting** on all exchange API keys.
- Use **read-only** keys for monitoring-only instances.
- Rotate API keys **every 90 days** or immediately after any suspected compromise.

### Deployment

- Run the bot as a **non-root user**.
- Place the API endpoint behind a **TLS-terminating reverse proxy** (Nginx/Caddy).
- Restrict network access using a **firewall** — expose only ports 22 (SSH) and 443 (HTTPS).
- Store secrets in a **secrets manager** (Vault, AWS Secrets Manager, Kubernetes Secrets), not in `.env` files on production systems.

### Dependencies

- Pin dependency versions in `requirements.txt`.
- Run `pip-audit -r requirements.txt` regularly to check for known CVEs.
- Update dependencies on a scheduled basis; review changelogs before updating.

### Monitoring

- Enable Telegram alerts for all order events and critical errors.
- Monitor for unexpected API key usage on the exchange dashboard.
- Set up alerting for abnormal trading volume or unusual P&L swings.

For detailed security guidance see [docs/SECURITY.md](docs/SECURITY.md).
