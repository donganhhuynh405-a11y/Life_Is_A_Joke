# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- OpenAPI specification (`docs/api/openapi.yaml`)
- Mermaid architecture and data-flow diagrams (`docs/diagrams/`)
- Comprehensive documentation suite (`docs/ARCHITECTURE.md`, `docs/API.md`, etc.)

---

## [1.3.0] - 2024-01-15

### Added
- **PPO Reinforcement Learning Agent** — proximal policy optimization training loop (`train_ppo_agent.py`)
- **Advanced BERT Sentiment** — full FinBERT pipeline for news sentiment scoring (`src/sentiment_advanced.py`)
- **Confidence-based Position Sizing** — ML confidence scores now directly influence lot size (`src/core/confidence_position_sizer.py`)
- **Elite Bot Integrator** — premium signal aggregator integration (`src/core/elite_bot_integrator.py`)
- Kubernetes manifests (`k8s/`)
- Prometheus metrics endpoint (`metrics.py`)

### Changed
- Strategy manager now supports hot-reloading without restart
- Exchange adapter refactored to support Bybit futures alongside Binance spot
- Database layer supports both SQLite (dev) and PostgreSQL (production)

### Fixed
- Race condition in async order placement under high signal frequency
- MACD signal line calculation for edge candle counts
- Telegram notification backoff when rate-limited by Telegram API

---

## [1.2.0] - 2023-11-01

### Added
- **Enhanced Multi-Indicator Strategy** combining MACD, RSI, and Bollinger Bands
- **Backtesting CLI** (`backtester/cli.py`) with JSON output support
- **Celery task queue** for async ML inference (`src/celery_app.py`)
- **Health monitor** for automatic restart on critical errors (`src/health_monitor.py`)
- Nginx reverse proxy configuration (`deployment/nginx/`)
- Systemd service unit (`deployment/systemd/`)

### Changed
- ML ensemble now uses weighted voting (configurable weights per model)
- Risk manager daily loss limit resets at midnight UTC
- Log format standardized to structured JSON in production

### Fixed
- Memory leak in LSTM model when processing long OHLCV sequences
- Incorrect P&L calculation for fractional BTC positions
- Dashboard crash when no positions are open

### Security
- API token is now required on all endpoints (previously optional in development mode)
- Exchange API keys redacted from all log output

---

## [1.1.0] - 2023-08-15

### Added
- **ML Models** — Gradient Boosting and Random Forest price direction predictors (`src/ml_models.py`)
- **News Analysis** pipeline for social sentiment (`src/news/`)
- **Dashboard** web interface for real-time monitoring (`src/dashboard.py`)
- **Optimizer** for strategy hyperparameter search via Optuna (`src/optimizer.py`)
- Docker Compose setup (`docker-compose.yml`, `Dockerfile`)
- Multi-exchange support via CCXT abstraction

### Changed
- Configuration moved to `config.yaml` (was previously hardcoded)
- Strategy interface standardized — all strategies must return canonical signal dict
- Risk manager now supports `max_consecutive_losses` circuit breaker

### Fixed
- Bybit helper rate-limit handling
- Reporter summary calculation for mixed long/short portfolios

---

## [1.0.0] - 2023-05-01

### Added
- Initial release
- Classic MACD + RSI trading strategy
- Binance spot trading via CCXT
- SQLite trade journal
- Telegram trade notifications
- Simple trend (EMA crossover) strategy
- Paper trading mode
- Basic backtesting engine
- Risk manager with stop-loss and take-profit
- Configuration via `config.yaml`

---

[Unreleased]: https://github.com/your-org/Life_Is_A_Joke/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/your-org/Life_Is_A_Joke/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/your-org/Life_Is_A_Joke/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/your-org/Life_Is_A_Joke/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/your-org/Life_Is_A_Joke/releases/tag/v1.0.0
