# Usage Examples & Tutorials

## Quick Start

### 1. Paper Trading (Recommended First Step)

```bash
# Copy and configure
cp config.yaml config.local.yaml
# Edit config.local.yaml: set environment: "paper"

# Set required secrets
export BINANCE_API_KEY="your_key"
export BINANCE_SECRET="your_secret"
export TELEGRAM_BOT_TOKEN="your_token"  # optional
export TELEGRAM_CHAT_ID="your_chat_id"  # optional

# Run
python src/main.py --config config.local.yaml
```

### 2. Docker Compose (Recommended for Production)

```bash
cp .env.example .env
# Fill in your API keys in .env

docker-compose up -d
docker-compose logs -f bot
```

---

## Strategy Examples

### Classic MACD + RSI Strategy

```yaml
# config.yaml
trading:
  strategy: "classic_macd_rsi"
  symbols:
    - "BTC/USDT"
    - "ETH/USDT"
  timeframes:
    primary: "1h"

strategies:
  classic_macd_rsi:
    rsi_period: 14
    rsi_overbought: 70
    rsi_oversold: 30
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
```

### Enhanced Multi-Indicator Strategy

```yaml
trading:
  strategy: "enhanced_multi_indicator"

strategies:
  enhanced_multi_indicator:
    indicators:
      - macd
      - rsi
      - bollinger_bands
      - atr
    consensus_threshold: 0.6   # 60% of indicators must agree
    confidence_min: 0.65
```

### ML-Augmented Trading

```yaml
trading:
  strategy: "enhanced_multi_indicator"
  ml_enabled: true

ml:
  models:
    - gradient_boosting
    - random_forest
    - lstm
  ensemble_method: "weighted_vote"
  confidence_weight: 0.4    # ML signal weight in final decision
  retrain_interval_hours: 24
```

---

## Backtesting Examples

### Basic Backtest

```bash
python -m backtester.cli \
  --symbol BTC/USDT \
  --timeframe 1h \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --strategy classic_macd_rsi \
  --initial-capital 10000
```

**Sample output:**
```
=== Backtest Results ===
Period:          2023-01-01 → 2023-12-31
Total Trades:    147
Win Rate:        61.2%
Total Return:    +34.7%
Max Drawdown:    -8.3%
Sharpe Ratio:    1.87
Profit Factor:   1.63
```

### Multi-Symbol Backtest

```bash
python -m backtester.cli \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --timeframe 4h \
  --start 2023-06-01 \
  --end 2023-12-31 \
  --strategy enhanced_multi_indicator \
  --initial-capital 30000 \
  --output results/multi_symbol_backtest.json
```

### Comparing Strategies

```bash
for strategy in classic_macd_rsi enhanced_multi_indicator simple_trend; do
  python -m backtester.cli \
    --symbol BTC/USDT \
    --timeframe 1h \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --strategy $strategy \
    --output results/${strategy}.json
done
python scripts/compare_backtests.py results/
```

---

## ML Model Training

### Train PPO Reinforcement Learning Agent

```bash
# Train with default settings
python train_ppo_agent.py

# Train with custom parameters
python train_ppo_agent.py \
  --symbol BTC/USDT \
  --timeframe 1h \
  --episodes 1000 \
  --learning-rate 0.0003 \
  --output models/ppo_btc_1h.pkl
```

### Verify ML Models in Production

```bash
python verify_ml_production.py
python verify_real_bert.py
```

---

## API Usage Examples

### Python Client

```python
import requests

BASE_URL = "http://localhost:8080/api/v1"
HEADERS = {"Authorization": "Bearer your_api_token"}

# Get bot status
resp = requests.get(f"{BASE_URL}/status", headers=HEADERS)
print(resp.json())

# Get open positions
resp = requests.get(f"{BASE_URL}/positions", headers=HEADERS)
for pos in resp.json()["positions"]:
    print(f"{pos['symbol']}: {pos['unrealized_pnl_pct']:.2f}%")

# Get recent signals
resp = requests.get(
    f"{BASE_URL}/signals",
    params={"symbol": "BTC/USDT", "limit": 5},
    headers=HEADERS
)
for sig in resp.json()["signals"]:
    print(f"{sig['action']} @ confidence={sig['confidence']:.2f}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8080/api/v1/health

# Get status (authenticated)
curl -H "Authorization: Bearer $API_TOKEN" \
     http://localhost:8080/api/v1/status

# Pause bot
curl -X POST \
     -H "Authorization: Bearer $API_TOKEN" \
     http://localhost:8080/api/v1/bot/pause

# Close a position
curl -X DELETE \
     -H "Authorization: Bearer $API_TOKEN" \
     http://localhost:8080/api/v1/positions/pos_001
```

---

## Risk Management Configuration

### Conservative Profile

```yaml
risk:
  max_position_pct: 2.0       # Max 2% of portfolio per trade
  max_total_exposure_pct: 10.0 # Max 10% in open positions
  stop_loss_pct: 1.5
  take_profit_pct: 3.0
  daily_loss_limit_pct: 3.0
  max_consecutive_losses: 3
```

### Aggressive Profile

```yaml
risk:
  max_position_pct: 10.0
  max_total_exposure_pct: 50.0
  stop_loss_pct: 3.0
  take_profit_pct: 9.0
  daily_loss_limit_pct: 10.0
  max_consecutive_losses: 7
```

---

## Monitoring & Alerts

### Telegram Notifications Setup

```bash
# 1. Create a bot via @BotFather on Telegram
# 2. Get your chat ID: curl https://api.telegram.org/bot<TOKEN>/getUpdates

export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
export TELEGRAM_CHAT_ID="987654321"
```

### Prometheus + Grafana

```bash
# Start full monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana at http://localhost:3000
# Default credentials: admin / admin
```

---

## Custom Strategy Development

```python
# src/strategies/my_strategy.py
from src.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ema_period = config.get("ema_period", 20)

    async def generate_signal(self, symbol: str, ohlcv: list) -> dict:
        closes = [c[4] for c in ohlcv]
        ema = self._ema(closes, self.ema_period)

        if closes[-1] > ema[-1] * 1.002:
            action, confidence = "BUY", 0.75
        elif closes[-1] < ema[-1] * 0.998:
            action, confidence = "SELL", 0.75
        else:
            action, confidence = "HOLD", 0.50

        return {
            "action": action,
            "confidence": confidence,
            "reason": f"Price vs EMA{self.ema_period}",
            "price": closes[-1],
        }
```

Register in `config.yaml`:
```yaml
trading:
  strategy: "my_strategy"
```
