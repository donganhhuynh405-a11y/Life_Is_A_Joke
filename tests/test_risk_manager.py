"""
Tests for RiskManager.
"""
from risk_manager import RiskManager
import sys
import os
import types
import pytest
from unittest.mock import MagicMock, patch

# Insert src first so risk_manager is found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Stub the 'utils' module so WALLogger import in risk_manager succeeds
# without the real utils package (which lives in src/utils/__init__.py and
# doesn't export WALLogger).
_utils_stub = types.ModuleType('utils')
_utils_stub.WALLogger = MagicMock
sys.modules.setdefault('utils', _utils_stub)


@pytest.fixture(autouse=True)
def mock_wal(monkeypatch):
    """Patch WALLogger so no filesystem writes occur during tests."""
    with patch('risk_manager.WALLogger') as MockWAL:
        MockWAL.return_value.write = MagicMock()
        yield MockWAL


@pytest.fixture
def cfg():
    return MagicMock()


@pytest.fixture
def rm(cfg):
    return RiskManager(cfg)


class TestRiskManager:
    def test_instantiation(self, rm):
        assert rm is not None
        assert rm.running is False

    @pytest.mark.asyncio
    async def test_start_sets_running(self, rm):
        await rm.start()
        assert rm.running is True

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, rm):
        await rm.start()
        await rm.stop()
        assert rm.running is False

    # --- compute_position_size tests ---

    def test_position_size_positive_edge(self, rm):
        size = rm.compute_position_size(10000, edge=0.1, winrate=0.6)
        assert size > 0

    def test_position_size_capped_at_20_percent(self, rm):
        size = rm.compute_position_size(10000, edge=1.0, winrate=0.99)
        assert size <= 10000 * 0.2

    def test_position_size_zero_when_negative_edge(self, rm):
        # winrate=0.3 => kelly fraction = (0.3 - 0.7) = -0.4 => clamped to 0
        size = rm.compute_position_size(10000, edge=0.0, winrate=0.3)
        assert size == 0.0

    def test_position_size_default_winrate(self, rm):
        # winrate=0.5 => kelly fraction = (0.5 - 0.5)/1 = 0 => size == 0
        size = rm.compute_position_size(10000, edge=0.0)
        assert size == 0.0

    def test_position_size_with_zero_balance(self, rm):
        size = rm.compute_position_size(0, edge=0.5, winrate=0.7)
        assert size == 0.0

    def test_position_size_scales_with_balance(self, rm):
        s1 = rm.compute_position_size(1000, edge=0.1, winrate=0.6)
        s2 = rm.compute_position_size(2000, edge=0.1, winrate=0.6)
        assert s2 == pytest.approx(s1 * 2)

    # --- record_trade tests ---

    def test_record_trade_calls_wal(self, rm):
        trade = {'symbol': 'BTC/USDT', 'side': 'buy', 'amount': 0.01}
        rm.record_trade(trade)
        rm.wal.write.assert_called_once()

    def test_record_trade_writes_string(self, rm):
        trade = {'symbol': 'ETH/USDT'}
        rm.record_trade(trade)
        args = rm.wal.write.call_args[0]
        assert isinstance(args[0], str)
