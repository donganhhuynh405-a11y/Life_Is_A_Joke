"""
Pytest configuration and shared fixtures for the trading bot test suite.
"""
import sys
import os
import types
import pytest
from unittest.mock import MagicMock, AsyncMock

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Stub 'utils' module so src modules that do `from utils import WALLogger`
# don't fail when the src/utils package (which lacks WALLogger) shadows utils.py
if 'utils' not in sys.modules:
    _utils_stub = types.ModuleType('utils')
    _utils_stub.WALLogger = MagicMock
    _utils_stub.retry_async = lambda **kw: (lambda f: f)
    _utils_stub.retry_sync = lambda **kw: (lambda f: f)
    sys.modules['utils'] = _utils_stub

# Stub 'torch' so advanced_risk.py can be imported without PyTorch installed
if 'torch' not in sys.modules:
    _torch_stub = types.ModuleType('torch')
    _torch_stub.nn = MagicMock()
    _torch_stub.nn.Module = object
    _torch_stub.nn.Linear = MagicMock()
    _torch_stub.nn.Tanh = MagicMock()
    _torch_stub.nn.ReLU = MagicMock()
    _torch_stub.nn.Sequential = MagicMock()
    _torch_stub.nn.functional = MagicMock()
    _torch_stub.optim = MagicMock()
    _torch_stub.cuda = MagicMock()
    _torch_stub.cuda.is_available = MagicMock(return_value=False)
    _torch_stub.Tensor = MagicMock
    _torch_stub.FloatTensor = MagicMock(return_value=MagicMock())
    _torch_stub.tensor = MagicMock(return_value=MagicMock())
    _torch_stub.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))
    _torch_stub.zeros = MagicMock(return_value=MagicMock())
    _torch_stub.ones = MagicMock(return_value=MagicMock())
    _torch_stub.save = MagicMock()
    _torch_stub.load = MagicMock(return_value={})
    _torch_stub.clamp = MagicMock(side_effect=lambda x, *a, **kw: x)
    _torch_stub.cat = MagicMock()
    _torch_stub.distributions = MagicMock()
    _torch_stub.distributions.Normal = MagicMock()
    sys.modules['torch'] = _torch_stub
    sys.modules['torch.nn'] = _torch_stub.nn
    sys.modules['torch.nn.functional'] = _torch_stub.nn.functional
    sys.modules['torch.optim'] = _torch_stub.optim
    sys.modules['torch.distributions'] = _torch_stub.distributions


@pytest.fixture
def mock_config():
    """Minimal config object used across tests."""
    cfg = MagicMock()
    cfg.environment = 'test'
    cfg.secrets = {'binance_api_key': 'test_key', 'binance_api_secret': 'test_secret'}
    cfg.trading.symbols = ['BTC/USDT', 'ETH/USDT']
    cfg.trading.timeframes.primary = '1h'
    cfg.trading.filters.min_24h_volume = 1_000_000
    cfg.exchanges.binance.type = 'spot'
    cfg.exchanges.get.return_value = {'enabled': False}
    return cfg


@pytest.fixture
def mock_exchange():
    """Mock CCXT exchange instance."""
    exchange = MagicMock()
    exchange.fetch_balance = AsyncMock(return_value={
        'total': {'BTC': 0.5, 'USDT': 1000},
        'free': {'BTC': 0.5, 'USDT': 1000},
    })
    exchange.fetch_ticker = AsyncMock(return_value={
        'bid': 49900.0,
        'ask': 50100.0,
        'last': 50000.0,
        'baseVolume': 5_000_000,
    })
    exchange.markets = {
        'BTC/USDT': {
            'precision': {'amount': 6, 'price': 2},
            'limits': {'amount': {'min': 0.000001}},
        }
    }
    exchange.create_order = AsyncMock(return_value={'id': 'order123', 'price': 50000.0, 'amount': 0.001, 'side': 'buy'})
    exchange.close = AsyncMock()
    return exchange
