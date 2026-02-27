# Architecture Documentation

## Overview

Life Is A Joke is a modular, production-grade crypto trading bot with machine learning capabilities. It supports multiple exchanges, algorithmic strategies, and real-time sentiment analysis.

```
┌──────────────────────────────────────────────────────────┐
│                        Entry Points                       │
│              src/main.py  |  CLI  |  Dashboard            │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                      Core Bot Engine                      │
│                    src/core/bot.py                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ ExchangeMgr  │  │StrategyMgr   │  │  RiskManager   │  │
│  └──────────────┘  └──────────────┘  └────────────────┘  │
└──────┬─────────────────────┬──────────────────┬──────────┘
       │                     │                  │
┌──────▼──────┐  ┌───────────▼────────┐  ┌─────▼──────────┐
│  Exchange   │  │   ML / Prediction  │  │  Data / State  │
│  Adapters   │  │   Layer            │  │  Management    │
│  (CCXT)     │  │                    │  │                │
│  - Binance  │  │  - ml_models.py    │  │  - database.py │
│  - Bybit    │  │  - predictor.py    │  │  - reporter.py │
└─────────────┘  │  - sentiment*.py   │  └────────────────┘
                 │  - PPO Agent       │
                 └────────────────────┘
```

## Component Breakdown

### 1. Core (`src/core/`)

| File | Responsibility |
|------|---------------|
| `bot.py` | Main event loop, orchestrates all components |
| `exchange_manager.py` | Multi-exchange abstraction, order routing |
| `exchange_adapter.py` | Per-exchange normalization (CCXT wrapper) |
| `risk_manager.py` | Position sizing, drawdown limits, stop-loss |
| `confidence_position_sizer.py` | ML-confidence-weighted lot sizing |
| `database.py` | SQLite/PostgreSQL persistence layer |
| `telegram_notifier.py` | Alert delivery via Telegram Bot API |
| `elite_bot_integrator.py` | Premium signal aggregator integration |

### 2. Strategies (`src/strategies/`)

| File | Description |
|------|-------------|
| `base_strategy.py` | Abstract base class; defines signal interface |
| `strategy_manager.py` | Hot-loads and switches between strategies |
| `enhanced_multi_indicator.py` | MACD + RSI + Bollinger Bands ensemble |
| `simple_trend.py` | EMA crossover momentum strategy |

Strategy signals return a dict:
```python
{
    "action": "BUY" | "SELL" | "HOLD",
    "confidence": float,   # 0.0–1.0
    "reason": str,
    "price": float,
    "timestamp": str
}
```

### 3. ML Layer (`src/`)

| File | Description |
|------|-------------|
| `ml_models.py` | Gradient Boosting, Random Forest, LSTM wrappers |
| `predictor.py` | Aggregates ML signals; consensus voting |
| `sentiment.py` | Basic news/social sentiment (FinBERT-lite) |
| `sentiment_advanced.py` | Full BERT-based NLP sentiment pipeline |
| `train_ppo_agent.py` | Proximal Policy Optimization RL training loop |

### 4. Data Pipeline (`src/`)

| File | Description |
|------|-------------|
| `data_fetcher.py` | OHLCV + order-book fetcher with caching |
| `trend_analyzer.py` | Technical indicator computation (TA-Lib/pandas-ta) |
| `optimizer.py` | Hyperparameter search (Optuna) |

### 5. Backtester (`backtester/`)

| File | Description |
|------|-------------|
| `engine.py` | Event-driven backtesting engine |
| `cli.py` | Command-line interface for running backtests |

### 6. Infrastructure

| Path | Description |
|------|-------------|
| `deployment/` | Docker Compose, systemd units, Nginx, monitoring |
| `k8s/` | Kubernetes manifests |
| `metrics.py` | Prometheus metrics exporter |

## Data Flow

1. **Market Data Ingestion** – `data_fetcher.py` polls exchanges via CCXT at configurable intervals.
2. **Feature Engineering** – `trend_analyzer.py` computes indicators (RSI, MACD, BB, ATR, etc.).
3. **Signal Generation** – Strategies + ML ensemble produce directional signals with confidence scores.
4. **Risk Gating** – `risk_manager.py` validates position size against portfolio limits.
5. **Order Execution** – `executor.py` submits orders; `exchange_adapter.py` normalizes responses.
6. **Reporting** – Trade outcomes written to DB; notifications sent via Telegram.

## Configuration

All runtime configuration lives in `config.yaml`. Secrets (API keys, tokens) are injected via environment variables and never committed to source control. See `src/config.py` for the full schema.

## Scalability

- **Horizontal**: Multiple bot instances can run different symbol sets; shared PostgreSQL coordinates state.
- **Async**: Core event loop is `asyncio`-based; exchange I/O is non-blocking.
- **Task Queue**: Celery (`src/celery_app.py`) offloads heavy ML inference and backtests.
