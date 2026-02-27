"""
Tests for advanced_risk.py components.
PyTorch is mocked in conftest.py so these tests run without GPU/heavy dependencies.
"""
import sys
import os
import pytest
import numpy as np
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestTradingEnvironment:
    @pytest.fixture
    def env(self):
        from advanced_risk import TradingEnvironment
        return TradingEnvironment(asset_count=4, initial_balance=10000)

    def test_reset_returns_state_array(self, env):
        state = env.reset()
        assert isinstance(state, np.ndarray)

    def test_initial_balance_set(self, env):
        assert env.initial_balance == 10000
        assert env.balance == 10000

    def test_portfolio_zeros_on_reset(self, env):
        env.reset()
        assert np.all(env.portfolio == 0)

    def test_get_portfolio_value_equals_balance_when_no_assets(self, env):
        env.reset()
        val = env._get_portfolio_value()
        assert val == pytest.approx(10000.0)

    def test_step_returns_tuple(self, env):
        env.reset()
        action = np.zeros(env.asset_count)
        result = env.step(action)
        # step returns (state, reward, done, info) - 4 values
        assert len(result) >= 3

    def test_step_increments_step_count(self, env):
        env.reset()
        env.step(np.zeros(env.asset_count))
        assert env.step_count == 1

    def test_action_to_weights_sums_to_one(self, env):
        env.reset()
        action = np.array([1.0, 2.0, 0.5, 1.5])
        weights = env._action_to_weights(action)
        assert weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_done_when_balance_halved(self, env):
        env.reset()
        env.balance = env.initial_balance * 0.4  # below 50%
        env.portfolio = np.zeros(env.asset_count)
        result = env.step(np.zeros(env.asset_count))
        done = result[2]
        assert done is True


class TestRLPortfolioManager:
    @pytest.fixture
    def manager(self, tmp_path):
        with patch('advanced_risk.PPOAgent'), patch('advanced_risk.ActorCriticNetwork'):
            from advanced_risk import RLPortfolioManager
            mgr = RLPortfolioManager(asset_count=4, model_path=str(tmp_path / 'model.pt'))
            return mgr

    def test_compute_leverage_within_range(self, manager):
        lev = manager.compute_leverage(volatility=0.05)
        assert manager.leverage_range[0] <= lev <= manager.leverage_range[1]

    def test_compute_leverage_high_vol_gives_low_leverage(self, manager):
        lev_high = manager.compute_leverage(volatility=0.5)
        lev_low = manager.compute_leverage(volatility=0.01)
        assert lev_high <= lev_low

    def test_apply_kelly_criterion_basic(self, manager):
        fraction = manager.apply_kelly_criterion(win_rate=0.6, avg_win=0.1, avg_loss=0.05)
        assert 0.0 <= fraction <= 1.0

    def test_apply_kelly_criterion_capped(self, manager):
        fraction = manager.apply_kelly_criterion(win_rate=0.99, avg_win=10.0, avg_loss=0.01)
        assert fraction <= 1.0


class TestPositionSizer:
    @pytest.fixture
    def sizer(self):
        from advanced_risk import PositionSizer
        return PositionSizer()

    def test_calculate_size_returns_positive(self, sizer):
        size = sizer.calculate_size(
            account_balance=10000,
            atr=500,
            entry_price=50000,
            stop_loss_points=0.03,
        )
        assert size > 0

    def test_calculate_size_zero_stop_loss_gives_small_default(self, sizer):
        size = sizer.calculate_size(
            account_balance=10000,
            atr=500,
            entry_price=50000,
            stop_loss_points=0,
        )
        # When stop_loss_points == 0, implementation returns 0.01 * balance
        assert size == pytest.approx(100.0)
