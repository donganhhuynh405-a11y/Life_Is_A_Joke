"""
Tests for Executor.
"""
import sys
import os
import types
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Insert src first so executor is found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Stub 'utils' so executor's `from utils import retry_async` succeeds
_utils_stub = types.ModuleType('utils')
_utils_stub.retry_async = lambda **kw: (lambda f: f)
_utils_stub.WALLogger = MagicMock
sys.modules.setdefault('utils', _utils_stub)


@pytest.fixture(autouse=True)
def patch_retry():
    """Strip retry decorator so tests don't loop."""
    with patch('executor.retry_async', lambda **kw: lambda f: f):
        yield


@pytest.fixture
def cfg():
    cfg = MagicMock()
    cfg.environment = 'test'
    cfg.secrets = {'binance_api_key': '', 'binance_api_secret': ''}
    return cfg


@pytest.fixture
def mock_exchange():
    ex = MagicMock()
    ex.fetch_balance = AsyncMock(return_value={
        'total': {'BTC': 0.0, 'USDT': 1000.0},
        'free':  {'BTC': 0.0, 'USDT': 1000.0},
    })
    ex.fetch_ticker = AsyncMock(return_value={
        'bid': 49900.0, 'ask': 50100.0, 'last': 50000.0,
    })
    ex.markets = {
        'BTC/USDT': {
            'precision': {'amount': 6, 'price': 2},
            'limits': {'amount': {'min': 0.000001}},
        }
    }
    ex.create_order = AsyncMock(return_value={
        'id': 'ord1', 'price': 50100.0, 'amount': 0.001, 'side': 'buy',
    })
    ex.close = AsyncMock()
    ex.load_markets = AsyncMock()
    ex.set_sandbox_mode = MagicMock()
    return ex


@pytest.fixture
def executor(cfg, mock_exchange):
    with patch('executor.ccxt') as mock_ccxt:
        mock_ccxt.binance = MagicMock(return_value=mock_exchange)
        from executor import Executor
        ex = Executor(cfg)
        ex.exchange = mock_exchange
        return ex


class TestExecutorInit:
    def test_running_is_false(self, executor):
        assert executor.running is False

    def test_open_positions_empty(self, executor):
        assert executor.open_positions == {}


class TestExecutorHasOpenPosition:
    @pytest.mark.asyncio
    async def test_no_position_when_zero_balance(self, executor, mock_exchange):
        result = await executor.has_open_position('BTC/USDT')
        assert result is False

    @pytest.mark.asyncio
    async def test_position_found_in_memory(self, executor):
        executor.open_positions['BTC/USDT'] = {'amount': 0.5}
        result = await executor.has_open_position('BTC/USDT')
        assert result is True

    @pytest.mark.asyncio
    async def test_position_detected_from_balance(self, executor, mock_exchange):
        mock_exchange.fetch_balance = AsyncMock(return_value={
            'total': {'BTC': 0.5, 'USDT': 1000.0},
        })
        result = await executor.has_open_position('BTC/USDT')
        assert result is True


class TestAdjustAmountToExchangeRules:
    @pytest.mark.asyncio
    async def test_rounds_to_precision(self, executor):
        adjusted = await executor.adjust_amount_to_exchange_rules('BTC/USDT', 0.0012345678)
        assert adjusted == round(0.0012345678, 6)

    @pytest.mark.asyncio
    async def test_returns_min_when_below_minimum(self, executor, mock_exchange):
        mock_exchange.markets['BTC/USDT']['limits']['amount']['min'] = 0.001
        adjusted = await executor.adjust_amount_to_exchange_rules('BTC/USDT', 0.0000001)
        assert adjusted == 0.001

    @pytest.mark.asyncio
    async def test_handles_unknown_symbol(self, executor):
        adjusted = await executor.adjust_amount_to_exchange_rules('XYZ/USDT', 1.23456789)
        assert adjusted == round(1.23456789, 8)

    @pytest.mark.asyncio
    async def test_no_exchange_returns_raw_amount(self, executor):
        executor.exchange = None
        adjusted = await executor.adjust_amount_to_exchange_rules('BTC/USDT', 0.5)
        assert adjusted == round(0.5, 8)


class TestExecutorStop:
    @pytest.mark.asyncio
    async def test_stop_closes_exchange(self, executor, mock_exchange):
        executor.running = True
        await executor.stop()
        mock_exchange.close.assert_called_once()
        assert executor.running is False

    @pytest.mark.asyncio
    async def test_shutdown_is_alias_for_stop(self, executor, mock_exchange):
        await executor.shutdown()
        mock_exchange.close.assert_called_once()


class TestPlaceOrder:
    @pytest.mark.asyncio
    async def test_place_order_calls_create_order(self, executor, mock_exchange):
        sync_exchange = MagicMock()
        sync_exchange.create_order = MagicMock(return_value={'id': 'test_order'})

        result = await executor.place_order(sync_exchange, 'BTC/USDT', 'buy', 0.001)
        sync_exchange.create_order.assert_called_once()
        assert result == {'id': 'test_order'}
