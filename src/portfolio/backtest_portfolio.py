"""Portfolio backtesting engine.

Simulates a multi-asset portfolio over historical data, computing
standard performance metrics.  Uses only ``numpy`` and the standard library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for portfolio backtest results."""

    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    portfolio_values: np.ndarray
    weights_history: np.ndarray
    rebalance_dates: List[int] = field(default_factory=list)
    turnover: float = 0.0
    transaction_costs: float = 0.0


class PortfolioBacktester:
    """Backtest a portfolio allocation strategy over historical prices.

    The backtester steps through the price history bar by bar,
    optionally rebalancing on a fixed frequency and recording all
    portfolio metrics.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        rebalance_frequency: int = 20,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> None:
        """Initialise the backtester.

        Args:
            initial_capital: Starting portfolio value in base currency.
            rebalance_frequency: Number of bars between rebalances.
            transaction_cost: Fractional cost applied to traded value.
            risk_free_rate: Annualised risk-free rate for Sharpe/Sortino.
            periods_per_year: Trading periods per year (252 for daily).
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_drawdown(self, portfolio_values: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute maximum drawdown and the drawdown series.

        Args:
            portfolio_values: Equity curve array.

        Returns:
            ``(max_drawdown, drawdown_series)`` where max_drawdown is a
            negative fraction.
        """
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / np.where(peak > 0, peak, 1)
        return float(drawdown.min()), drawdown

    def _annualised_return(self, total_return: float, n_periods: int) -> float:
        years = n_periods / self.periods_per_year
        return float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

    def _annualised_vol(self, returns: np.ndarray) -> float:
        return float(returns.std() * np.sqrt(self.periods_per_year))

    def _sharpe(self, ann_ret: float, ann_vol: float) -> float:
        return (ann_ret - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    def _sortino(self, returns: np.ndarray, ann_ret: float) -> float:
        downside = returns[returns < 0]
        down_vol = float(downside.std() * np.sqrt(self.periods_per_year)
                         ) if len(downside) > 0 else 0.0
        return (ann_ret - self.risk_free_rate) / down_vol if down_vol > 0 else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prices: np.ndarray,
        weight_fn: Callable[[np.ndarray, int], np.ndarray],
        initial_weights: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """Execute the backtest.

        Args:
            prices: Price matrix of shape ``(n_bars, n_assets)``.  Each
                row is one time period.
            weight_fn: Callable ``(prices_up_to_t, bar_index) → weights``
                called at each rebalance event to produce new target weights.
            initial_weights: Starting weights; equal-weight if *None*.

        Returns:
            :class:`BacktestResult` with all performance metrics.
        """
        n_bars, n_assets = prices.shape
        if initial_weights is None:
            initial_weights = np.ones(n_assets) / n_assets

        # Holdings in units; buy at the first close
        holdings = (self.initial_capital * initial_weights) / prices[0]
        portfolio_values = np.zeros(n_bars)
        weights_history = np.zeros((n_bars, n_assets))
        rebalance_dates: List[int] = []
        total_cost = 0.0
        total_turnover = 0.0

        for t in range(n_bars):
            current_prices = prices[t]
            port_val = (holdings * current_prices).sum()
            portfolio_values[t] = port_val
            cur_weights = (holdings * current_prices) / \
                port_val if port_val > 0 else np.zeros(n_assets)
            weights_history[t] = cur_weights

            # Rebalance
            if t > 0 and t % self.rebalance_frequency == 0:
                new_weights = weight_fn(prices[: t + 1], t)
                new_weights = np.clip(new_weights, 0, 1)
                new_weights /= new_weights.sum() if new_weights.sum() > 0 else 1.0

                delta_w = np.abs(new_weights - cur_weights)
                turnover = delta_w.sum() / 2
                total_turnover += turnover
                cost = turnover * port_val * self.transaction_cost
                total_cost += cost
                port_val_after_cost = port_val - cost

                holdings = (port_val_after_cost * new_weights) / \
                    np.where(current_prices > 0, current_prices, 1)
                rebalance_dates.append(t)

        # Performance metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        bar_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        ann_ret = self._annualised_return(total_return, n_bars)
        ann_vol = self._annualised_vol(bar_returns)
        sharpe = self._sharpe(ann_ret, ann_vol)
        max_dd, _ = self._compute_drawdown(portfolio_values)
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
        sortino = self._sortino(bar_returns, ann_ret)

        logger.info(
            "Backtest complete: total_return=%.2f%% | sharpe=%.2f | max_dd=%.2f%%",
            total_return * 100,
            sharpe,
            max_dd * 100,
        )

        return BacktestResult(
            total_return=total_return,
            annualised_return=ann_ret,
            annualised_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            portfolio_values=portfolio_values,
            weights_history=weights_history,
            rebalance_dates=rebalance_dates,
            turnover=total_turnover,
            transaction_costs=total_cost,
        )

    def benchmark_comparison(
        self,
        portfolio_values: np.ndarray,
        benchmark_prices: np.ndarray,
    ) -> Dict[str, float]:
        """Compare portfolio against a buy-and-hold benchmark.

        Args:
            portfolio_values: Equity curve from :meth:`run`.
            benchmark_prices: Single-asset benchmark price series
                ``(n_bars,)``.

        Returns:
            Dict with ``"alpha"``, ``"beta"``, and ``"information_ratio"``.
        """
        port_ret = np.diff(portfolio_values) / portfolio_values[:-1]
        bench_ret = np.diff(benchmark_prices) / benchmark_prices[:-1]
        n = min(len(port_ret), len(bench_ret))
        port_ret, bench_ret = port_ret[:n], bench_ret[:n]

        if bench_ret.std() > 0:
            beta = float(np.cov(port_ret, bench_ret)[0, 1] / bench_ret.var())
        else:
            beta = 0.0
        alpha = float(port_ret.mean() - beta * bench_ret.mean()) * self.periods_per_year

        active_ret = port_ret - bench_ret
        info_ratio = float(active_ret.mean() / active_ret.std()) * \
            np.sqrt(self.periods_per_year) if active_ret.std() > 0 else 0.0

        return {"alpha": alpha, "beta": beta, "information_ratio": info_ratio}
