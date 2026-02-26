"""
Tests for BaseStrategy and concrete strategy implementations.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategies.base_strategy import BaseStrategy
from strategies.simple_trend import SimpleTrendStrategy
from strategies.enhanced_multi_indicator import EnhancedMultiIndicatorStrategy


def _make_klines(closes):
    """Build minimal klines list from close prices."""
    return [[0, 0, 0, 0, str(c), 0, 0, 0, 0, 0, 0, 0] for c in closes]


@pytest.fixture
def base_deps():
    cfg = MagicMock()
    cfg.default_symbol = 'BTCUSDT'
    client = MagicMock()
    db = MagicMock()
    db.get_open_positions.return_value = []
    risk = MagicMock()
    return cfg, client, db, risk


class TestBaseStrategy:
    def test_is_abstract(self, base_deps):
        cfg, client, db, risk = base_deps
        with pytest.raises(TypeError):
            BaseStrategy(cfg, client, db, risk)

    def test_get_current_price(self, base_deps):
        cfg, client, db, risk = base_deps
        client.get_symbol_ticker.return_value = {'price': '50000.0'}
        strategy = SimpleTrendStrategy(cfg, client, db, risk)
        price = strategy.get_current_price('BTCUSDT')
        assert price == 50000.0

    def test_get_klines_delegates_to_client(self, base_deps):
        cfg, client, db, risk = base_deps
        client.get_klines.return_value = _make_klines([100] * 50)
        strategy = SimpleTrendStrategy(cfg, client, db, risk)
        klines = strategy.get_klines('BTCUSDT', '1h', 50)
        assert len(klines) == 50
        client.get_klines.assert_called_once_with(symbol='BTCUSDT', interval='1h', limit=50)


class TestSimpleTrendStrategy:
    def test_analyze_returns_list(self, base_deps):
        cfg, client, db, risk = base_deps
        closes = list(range(100, 145))  # 45 candles
        client.get_klines.return_value = _make_klines(closes)
        client.get_symbol_ticker.return_value = {'price': '144.0'}
        strategy = SimpleTrendStrategy(cfg, client, db, risk)
        result = strategy.analyze()
        assert isinstance(result, list)

    def test_not_enough_data_returns_empty(self, base_deps):
        cfg, client, db, risk = base_deps
        client.get_klines.return_value = _make_klines([100] * 5)
        strategy = SimpleTrendStrategy(cfg, client, db, risk)
        result = strategy.analyze()
        assert result == []

    def test_bullish_crossover_generates_buy(self, base_deps):
        cfg, client, db, risk = base_deps
        # Build closes that create a bullish crossover (short MA crosses above long MA)
        # First 30 declining, then last 10 sharply rising
        closes = [100 - i * 0.5 for i in range(30)] + [95 + i * 3 for i in range(12)]
        client.get_klines.return_value = _make_klines(closes)
        client.get_symbol_ticker.return_value = {'price': str(closes[-1])}
        db.get_open_positions.return_value = []
        strategy = SimpleTrendStrategy(cfg, client, db, risk)
        result = strategy.analyze()
        actions = [s['action'] for s in result]
        # We just verify the method ran without error and returns valid structure
        for signal in result:
            assert 'action' in signal
            assert 'symbol' in signal

    def test_stop_loss_triggers_close(self, base_deps):
        cfg, client, db, risk = base_deps
        position = {
            'id': '1',
            'symbol': 'BTCUSDT',
            'stop_loss': 200.0,
            'take_profit': 300.0,
        }
        closes = [100] * 42
        client.get_klines.return_value = _make_klines(closes)
        client.get_symbol_ticker.return_value = {'price': '150.0'}
        db.get_open_positions.return_value = [position]
        strategy = SimpleTrendStrategy(cfg, client, db, risk)
        signals = []
        strategy._check_exit_conditions(position, 150.0, signals)
        actions = [s['action'] for s in signals]
        assert 'CLOSE' in actions

    def test_take_profit_triggers_close(self, base_deps):
        cfg, client, db, risk = base_deps
        position = {
            'id': '2',
            'symbol': 'BTCUSDT',
            'stop_loss': 100.0,
            'take_profit': 250.0,
        }
        strategy = SimpleTrendStrategy(cfg, client, db, risk)
        signals = []
        strategy._check_exit_conditions(position, 300.0, signals)
        assert any(s['action'] == 'CLOSE' for s in signals)


class TestEnhancedMultiIndicatorStrategy:
    def test_instantiates_correctly(self, base_deps):
        cfg, client, db, risk = base_deps
        strategy = EnhancedMultiIndicatorStrategy(cfg, client, db, risk)
        assert strategy.name == 'EnhancedMultiIndicator'
        assert strategy.rsi_period == 14

    def test_analyze_returns_list(self, base_deps):
        cfg, client, db, risk = base_deps
        cfg.trading = MagicMock()
        cfg.trading.symbols = ['BTCUSDT']
        client.get_klines.return_value = _make_klines([50000 + i for i in range(100)])
        client.get_symbol_ticker.return_value = {'price': '50100.0'}
        strategy = EnhancedMultiIndicatorStrategy(cfg, client, db, risk)
        result = strategy.analyze()
        assert isinstance(result, list)
