"""
Integration tests for the trading bot components working together.
"""
import sys
import os
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def cfg():
    cfg = MagicMock()
    cfg.environment = 'test'
    cfg.secrets = {}
    return cfg


class TestRiskManagerIntegration:
    """Verify RiskManager position sizing integrates correctly with trade recording."""

    def test_size_then_record(self, cfg):
        with patch('risk_manager.WALLogger') as MockWAL:
            MockWAL.return_value.write = MagicMock()
            from risk_manager import RiskManager
            rm = RiskManager(cfg)
            size = rm.compute_position_size(10000, edge=0.1, winrate=0.6)
            trade = {'size': size, 'symbol': 'BTC/USDT'}
            rm.record_trade(trade)
            rm.wal.write.assert_called_once()
            assert size > 0


class TestSentimentIntegration:
    """Verify SentimentAnalyzer produces consistent output for known inputs."""

    def test_bullish_text_gives_high_score(self, cfg):
        from sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer(cfg)
        result = sa.analyze_texts(['bitcoin moon pump rally bullish buy hodl'])
        assert result['score'] > 0.5
        assert result['sentiment'] == 'bullish'

    def test_bearish_text_gives_low_score(self, cfg):
        from sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer(cfg)
        result = sa.analyze_texts(['crash dump bear sell rekt fud fear liquidation'])
        assert result['score'] < 0.5
        assert result['sentiment'] == 'bearish'

    def test_empty_text_gives_neutral(self, cfg):
        from sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer(cfg)
        result = sa.analyze_single_text('')
        assert result['score'] == 0.5
        assert result['sentiment'] == 'neutral'


class TestStrategyRiskIntegration:
    """Verify strategy signals flow correctly through risk sizing."""

    def test_signal_produces_positive_size(self, cfg):
        with patch('risk_manager.WALLogger'):
            from risk_manager import RiskManager
            rm = RiskManager(cfg)

        signal = {'confidence': 0.8, 'direction': 'buy', 'symbol': 'BTC/USDT'}
        account_balance = 5000
        edge = signal['confidence'] - 0.5  # crude edge estimate
        size = rm.compute_position_size(account_balance, edge=edge, winrate=0.6)
        assert size >= 0

    def test_low_confidence_signal_produces_zero_or_small_size(self, cfg):
        with patch('risk_manager.WALLogger'):
            from risk_manager import RiskManager
            rm = RiskManager(cfg)

        # confidence=0.3 → edge=−0.2 → kelly≤0
        size = rm.compute_position_size(10000, edge=-0.2, winrate=0.3)
        assert size == 0.0


@pytest.mark.asyncio
class TestExecutorIntegration:
    """Lightweight integration test for Executor with mocked exchange."""

    async def test_has_no_position_when_balance_zero(self, cfg):
        with patch('executor.ccxt') as mock_ccxt, \
             patch('executor.retry_async', lambda **kw: lambda f: f):
            mock_exchange_instance = MagicMock()
            mock_exchange_instance.fetch_balance = AsyncMock(return_value={
                'total': {'BTC': 0, 'USDT': 1000},
                'free': {'BTC': 0, 'USDT': 1000},
            })
            mock_exchange_instance.load_markets = AsyncMock()
            mock_exchange_instance.set_sandbox_mode = MagicMock()
            mock_exchange_cls = MagicMock(return_value=mock_exchange_instance)
            mock_ccxt.binance = mock_exchange_cls

            from executor import Executor
            ex = Executor(cfg)
            ex.exchange = mock_exchange_instance

            result = await ex.has_open_position('BTC/USDT')
            assert result is False
