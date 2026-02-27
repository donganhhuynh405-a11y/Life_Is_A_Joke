# Detailed Troubleshooting Guide

This guide supplements the basic [TROUBLESHOOTING.md](TROUBLESHOOTING.md) with in-depth diagnostics.

---

## Table of Contents

- [Exchange Connectivity](#exchange-connectivity)
- [ML / AI Issues](#ml--ai-issues)
- [Database Problems](#database-problems)
- [Order Execution Failures](#order-execution-failures)
- [Risk Manager Blocking Trades](#risk-manager-blocking-trades)
- [Telegram Notifications](#telegram-notifications)
- [Docker / Deployment](#docker--deployment)
- [Performance Issues](#performance-issues)
- [Log Analysis](#log-analysis)

---

## Exchange Connectivity

### `AuthenticationError` on startup

**Symptom:** `ccxt.base.errors.AuthenticationError: binance Invalid API-key`

**Causes & Fixes:**

1. **Wrong key** — double-check the key in your environment variables:
   ```bash
   echo $BINANCE_API_KEY | wc -c   # Should be ~65 chars
   ```

2. **IP whitelist** — Binance API keys can be restricted to specific IPs. Check the Binance dashboard.

3. **Clock skew** — HMAC signatures are time-sensitive:
   ```bash
   timedatectl status
   sudo ntpdate -u pool.ntp.org
   ```

4. **Testnet vs mainnet** — ensure `testnet: true/false` in `config.yaml` matches the key type.

---

### `NetworkError` / Timeout

**Symptom:** `ccxt.base.errors.NetworkError: binance GET ... timed out`

1. Check DNS resolution:
   ```bash
   nslookup api.binance.com
   curl -I https://api.binance.com/api/v3/ping
   ```

2. Increase timeout in config:
   ```yaml
   exchanges:
     binance:
       options:
         timeout: 30000  # ms
   ```

3. Check rate limiting — the bot may be hitting exchange limits. Reduce `poll_interval_seconds`.

---

### `ExchangeNotAvailable` (503)

Exchange is under maintenance. The bot will auto-retry with exponential backoff (up to 10 minutes). Monitor `logs/bot.log` for reconnection attempts.

---

## ML / AI Issues

### BERT / Sentiment model fails to load

**Symptom:** `OSError: Can't load tokenizer for 'ProsusAI/finbert'`

1. First-run downloads the model (~500 MB). Ensure internet access and disk space:
   ```bash
   df -h ~/.cache/huggingface
   ```

2. Pre-download manually:
   ```bash
   python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='ProsusAI/finbert')"
   ```

3. In air-gapped environments, copy the cached model:
   ```bash
   cp -r ~/.cache/huggingface /app/.cache/
   export TRANSFORMERS_CACHE=/app/.cache/huggingface
   ```

---

### ML predictions all return `HOLD`

**Causes:**

1. **Insufficient training data** — models need at least 500 candles to train. Check:
   ```bash
   python -c "from src.data_fetcher import DataFetcher; ..."
   ```

2. **Feature scaling mismatch** — retrain models:
   ```bash
   python train_ppo_agent.py --retrain
   ```

3. **Model file missing** — check `models/` directory:
   ```bash
   ls -lh models/
   ```

---

### PPO training crashes with `NaN loss`

1. Reduce learning rate: `--learning-rate 0.0001`
2. Clip gradient norms: `--max-grad-norm 0.5`
3. Check for NaN in input data:
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_parquet('data/BTC_USDT_1h.parquet')
   print(df.isnull().sum())
   "
   ```

---

## Database Problems

### `OperationalError: database is locked` (SQLite)

SQLite only supports one writer at a time. Solutions:

1. **Short-term**: Restart the bot; the lock will clear.
2. **Long-term**: Migrate to PostgreSQL:
   ```yaml
   database:
     type: postgresql
     host: localhost
     port: 5432
     name: tradingbot
   ```

### PostgreSQL connection refused

```bash
# Check PostgreSQL is running
systemctl status postgresql

# Test connection
psql -h localhost -U tradingbot -d tradingbot -c "\l"

# Check firewall
sudo ufw status
```

---

## Order Execution Failures

### `InsufficientFunds`

**Symptom:** `ccxt.base.errors.InsufficientFunds: binance Account has insufficient balance`

1. Check actual balance vs. configured `max_position_pct`.
2. In paper trading mode this should never occur — verify `environment: "paper"` is set.
3. Account for exchange fees in position sizing (`include_fees: true` in risk config).

### Orders placed but not filled

1. Check order book — market may have moved away from limit price.
2. Switch to market orders for better fill rate:
   ```yaml
   execution:
     order_type: "market"   # instead of "limit"
   ```
3. Review open orders on exchange dashboard; cancel stale orders.

---

## Risk Manager Blocking Trades

### "Daily loss limit reached" — bot not trading

The bot has hit `daily_loss_limit_pct`. This is by design. Options:

- Wait for the daily reset (midnight UTC).
- Adjust the limit in config (use caution):
  ```yaml
  risk:
    daily_loss_limit_pct: 5.0
  ```

### "Max consecutive losses" triggered

```yaml
risk:
  max_consecutive_losses: 5  # increase carefully
```

After three consecutive losses, the bot enters a cool-down. Review recent signals in logs before resuming.

---

## Telegram Notifications

### No messages received

1. Verify token and chat ID:
   ```bash
   curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe"
   curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage?chat_id=${TELEGRAM_CHAT_ID}&text=test"
   ```

2. Ensure the bot has been started (`/start` command in the chat).

3. Group chats require the bot to be an admin if the chat has privacy mode enabled.

---

## Docker / Deployment

### Container exits immediately

```bash
docker-compose logs bot
```

Common causes:
- Missing environment variables (check `.env` file)
- Config file not mounted: verify `docker-compose.yml` volume paths
- Port already in use: `lsof -i :8080`

### High memory usage

BERT models consume ~1.5 GB RAM. Ensure the container has sufficient memory:

```yaml
# docker-compose.yml
services:
  bot:
    mem_limit: 3g
```

Or disable BERT and use lightweight sentiment:

```yaml
sentiment:
  engine: "basic"  # instead of "bert"
```

---

## Performance Issues

### Bot is slow to generate signals

1. Profile with:
   ```bash
   python -m cProfile -o profile.out src/main.py
   python -m pstats profile.out
   ```

2. Enable async data fetching:
   ```yaml
   data_fetcher:
     async_enabled: true
     cache_ttl_seconds: 60
   ```

3. Run ML inference via Celery:
   ```bash
   celery -A src.celery_app worker --concurrency=4
   ```

---

## Log Analysis

### Log locations

| Environment | Path |
|-------------|------|
| Local | `logs/bot.log` |
| Docker | `docker-compose logs bot` |
| Systemd | `journalctl -u tradingbot -f` |

### Useful log grep patterns

```bash
# All errors
grep -E "ERROR|CRITICAL" logs/bot.log | tail -50

# Order events
grep "order\|position\|fill" logs/bot.log | tail -100

# ML signal history
grep "signal\|confidence\|action" logs/bot.log | tail -50

# Exchange errors
grep "ccxt\|ExchangeError\|NetworkError" logs/bot.log | tail -20
```

### Enabling debug logging

```yaml
log_level: "DEBUG"
```

> **Warning:** DEBUG level is very verbose — rotate logs frequently.

---

## Getting Help

1. Check this guide and [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
2. Search existing [GitHub Issues](https://github.com/your-org/Life_Is_A_Joke/issues).
3. Open a new issue with: Python version, OS, full error traceback, and sanitized config.
