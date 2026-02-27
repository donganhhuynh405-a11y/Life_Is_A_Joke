"""
Tests for DataFetcher.
"""
import sys
import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


_BASE_TIMESTAMP_MS = 1_700_000_000_000  # milliseconds since epoch (~Nov 2023)


def _raw_ohlcv(n=10):
    """Return n fake OHLCV rows as CCXT would."""
    return [
        [_BASE_TIMESTAMP_MS + i * 3_600_000, 100 + i, 101 + i, 99 + i, 100.5 + i, 1000 + i]
        for i in range(n)
    ]


@pytest.fixture
def cfg():
    cfg = MagicMock()
    cfg.secrets = {'binance_api_key': '', 'binance_api_secret': ''}
    cfg.exchanges.binance.type = 'spot'
    cfg.exchanges.get.return_value = {'enabled': False}
    cfg.trading.symbols = ['BTC/USDT']
    cfg.trading.timeframes.primary = '1h'
    cfg.trading.filters.min_24h_volume = 100_000
    cfg.environment = 'test'
    return cfg


@pytest.fixture
def fetcher(cfg):
    with patch('data_fetcher.ccxt'):
        from data_fetcher import DataFetcher
        df = DataFetcher(cfg)
        return df


class TestDataFetcherInit:
    def test_not_initialized_on_creation(self, fetcher):
        assert fetcher.initialized is False

    def test_cache_is_empty(self, fetcher):
        assert len(fetcher.cache) == 0


class TestDataFetcherFetchOhlcv:
    @pytest.mark.asyncio
    async def test_returns_dataframe_on_success(self, fetcher):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=_raw_ohlcv(10))
        fetcher.exchanges = {'binance': mock_exchange}

        result = await fetcher.fetch_ohlcv('BTC/USDT', '1h', 10)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert 'close' in result.columns

    @pytest.mark.asyncio
    async def test_returns_none_when_no_exchanges(self, fetcher):
        fetcher.exchanges = {}
        result = await fetcher.fetch_ohlcv('BTC/USDT')
        assert result is None

    @pytest.mark.asyncio
    async def test_caches_result(self, fetcher):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=_raw_ohlcv(5))
        fetcher.exchanges = {'binance': mock_exchange}

        await fetcher.fetch_ohlcv('BTC/USDT', '1h', 5)
        await fetcher.fetch_ohlcv('BTC/USDT', '1h', 5)

        # Second call should use cache; exchange called only once
        assert mock_exchange.fetch_ohlcv.call_count == 1

    @pytest.mark.asyncio
    async def test_returns_none_on_exchange_error(self, fetcher):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv = AsyncMock(side_effect=Exception('network error'))
        fetcher.exchanges = {'binance': mock_exchange}

        result = await fetcher.fetch_ohlcv('ETH/USDT')
        assert result is None


class TestDataFetcherFetchTicker:
    @pytest.mark.asyncio
    async def test_returns_ticker_dict(self, fetcher):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker = AsyncMock(return_value={
            'bid': 49900.0, 'ask': 50100.0, 'last': 50000.0, 'baseVolume': 5_000_000,
        })
        fetcher.exchanges = {'binance': mock_exchange}

        result = await fetcher.fetch_ticker('BTC/USDT')
        assert result is not None
        assert result['bid'] == 49900.0
        assert result['ask'] == 50100.0

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self, fetcher):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker = AsyncMock(side_effect=Exception('timeout'))
        fetcher.exchanges = {'binance': mock_exchange}

        result = await fetcher.fetch_ticker('BTC/USDT')
        assert result is None


class TestDataFetcherFetchOrderBook:
    @pytest.mark.asyncio
    async def test_returns_order_book(self, fetcher):
        mock_exchange = MagicMock()
        mock_exchange.fetch_order_book = AsyncMock(return_value={
            'bids': [[49900, 1.5], [49800, 2.0]],
            'asks': [[50100, 1.0], [50200, 3.0]],
        })
        fetcher.exchanges = {'binance': mock_exchange}

        result = await fetcher.fetch_order_book('BTC/USDT', limit=2)
        assert result is not None
        assert 'bids' in result
        assert 'asks' in result


class TestDataFetcherShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_clears_exchanges(self, fetcher):
        mock_exchange = MagicMock()
        mock_exchange.close = AsyncMock()
        fetcher.exchanges = {'binance': mock_exchange}
        fetcher.initialized = True

        await fetcher.shutdown()

        mock_exchange.close.assert_called_once()
        assert fetcher.exchanges == {}
        assert fetcher.initialized is False
