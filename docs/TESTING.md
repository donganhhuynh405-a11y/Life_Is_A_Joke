# Testing Guide

## Overview

The test suite covers unit tests, integration tests, ML model validation, and backtesting verification. Tests are run with `pytest`.

---

## Test Structure

```
tests/
├── unit/
│   ├── test_risk_manager.py
│   ├── test_strategies.py
│   ├── test_data_fetcher.py
│   └── test_config.py
├── integration/
│   ├── test_exchange_adapter.py
│   └── test_database.py
├── conftest.py
└── fixtures/
    └── sample_ohlcv.json

test_ml_models.py          # ML model smoke tests (root)
test_ppo_implementation.py # PPO RL agent tests (root)
test_sentiment_bert.py     # BERT sentiment tests (root)
```

---

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### Specific Test Files

```bash
# Unit tests only
pytest tests/unit/ -v

# ML tests
pytest test_ml_models.py test_ppo_implementation.py test_sentiment_bert.py -v

# Single test
pytest tests/unit/test_risk_manager.py::TestRiskManager::test_position_size -v
```

### With Coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
open htmlcov/index.html
```

### Parallel Execution

```bash
pip install pytest-xdist
pytest tests/ -n auto
```

---

## Test Configuration

`pytest.ini` at the project root controls test discovery and markers:

```ini
[pytest]
testpaths = tests test_ml_models.py test_ppo_implementation.py test_sentiment_bert.py
markers =
    slow: marks tests as slow (ML training, backtests)
    integration: marks tests requiring live exchange connectivity
    unit: fast unit tests
```

Skip slow tests during development:

```bash
pytest tests/ -m "not slow" -v
```

Skip integration tests (no API keys needed):

```bash
pytest tests/ -m "not integration" -v
```

---

## Environment for Tests

Most tests use mocked exchange responses and do not require real API keys. Set a minimal environment:

```bash
export BINANCE_API_KEY="test_key"
export BINANCE_SECRET="test_secret"
export ENVIRONMENT="paper"
```

Integration tests require a `.env.test` file:

```bash
cp .env.example .env.test
# Fill in testnet credentials
pytest tests/integration/ --env-file .env.test
```

---

## Writing Tests

### Unit Test Example

```python
# tests/unit/test_risk_manager.py
import pytest
from src.core.risk_manager import RiskManager

@pytest.fixture
def risk_config():
    return {
        "max_position_pct": 5.0,
        "stop_loss_pct": 2.0,
        "daily_loss_limit_pct": 3.0,
        "max_consecutive_losses": 3,
    }

class TestRiskManager:
    def test_position_size_within_limit(self, risk_config):
        rm = RiskManager(risk_config)
        size = rm.calculate_position_size(
            portfolio_value=10000,
            price=50000,
            confidence=0.8,
        )
        assert size <= 10000 * 0.05 / 50000  # max 5% of portfolio

    def test_daily_loss_limit_blocks_trade(self, risk_config):
        rm = RiskManager(risk_config)
        rm.record_loss(pct=3.5)   # exceeds 3.0% limit
        assert rm.is_trading_allowed() is False
```

### Strategy Test Example

```python
# tests/unit/test_strategies.py
import pytest
from unittest.mock import AsyncMock
from src.strategies.simple_trend import SimpleTrendStrategy

@pytest.fixture
def sample_ohlcv():
    # 50 candles of synthetic data
    return [[i * 3600, 100 + i, 105 + i, 95 + i, 102 + i, 1000] for i in range(50)]

@pytest.mark.asyncio
async def test_simple_trend_returns_valid_signal(sample_ohlcv):
    strategy = SimpleTrendStrategy({"ema_period": 20})
    signal = await strategy.generate_signal("BTC/USDT", sample_ohlcv)

    assert signal["action"] in ("BUY", "SELL", "HOLD")
    assert 0.0 <= signal["confidence"] <= 1.0
    assert "reason" in signal
    assert "price" in signal
```

### Mock Exchange Example

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_order_placed_on_buy_signal():
    mock_exchange = AsyncMock()
    mock_exchange.create_order.return_value = {"id": "ord_123", "status": "filled"}

    with patch("src.core.exchange_manager.ExchangeManager.get_exchange",
               return_value=mock_exchange):
        # ... run bot logic ...
        mock_exchange.create_order.assert_called_once()
```

---

## ML Test Guidelines

ML tests validate model contracts, not prediction accuracy:

```python
# test_ml_models.py
def test_ml_model_output_shape():
    model = GradientBoostingPredictor()
    predictions = model.predict(sample_features)
    assert predictions.shape == (len(sample_features),)
    assert all(p in ("UP", "DOWN", "NEUTRAL") for p in predictions)

def test_model_confidence_range():
    model = EnsemblePredictor()
    _, confidence = model.predict_with_confidence(sample_features[0])
    assert 0.0 <= confidence <= 1.0
```

---

## Backtesting Tests

```bash
# Run a quick sanity backtest (uses fixture data, not live data)
python -m backtester.cli \
  --symbol BTC/USDT \
  --timeframe 1h \
  --start 2023-01-01 \
  --end 2023-03-01 \
  --strategy classic_macd_rsi \
  --dry-run
```

---

## Continuous Integration

Tests run automatically on every PR via GitHub Actions (`.github/workflows/`). PRs cannot be merged unless all tests pass.

The CI pipeline:
1. Installs dependencies from `requirements.txt` and `requirements-test.txt`
2. Runs `pytest tests/ -m "not integration and not slow"`
3. Reports coverage to the PR

---

## Troubleshooting Tests

### `ImportError` on test collection

```bash
# Ensure you're in the project root and venv is active
source .venv/bin/activate
python -m pytest tests/ -v
```

### Async test not running

Install `pytest-asyncio`:
```bash
pip install pytest-asyncio
```

And add to `conftest.py`:
```python
import pytest

@pytest.fixture(scope="session")
def event_loop():
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```
