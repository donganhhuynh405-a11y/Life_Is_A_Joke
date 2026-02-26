"""Automatic portfolio rebalancing strategies.

Uses only ``numpy`` for numerical computations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PortfolioRebalancer:
    """Handles threshold-based and calendar-based portfolio rebalancing.

    Given current holdings and target weights, this class computes the
    required trades to bring the portfolio back to its target allocation,
    optionally taking transaction costs into account.
    """

    def __init__(
        self,
        target_weights: np.ndarray,
        asset_names: Optional[List[str]] = None,
        rebalance_threshold: float = 0.05,
        transaction_cost: float = 0.001,
        min_trade_size: float = 0.0,
    ) -> None:
        """Initialise the rebalancer.

        Args:
            target_weights: Target allocation weights (must sum to 1).
            asset_names: Optional human-readable asset labels.
            rebalance_threshold: Drift (in absolute weight units) that
                triggers a rebalance for *threshold* mode.
            transaction_cost: Fractional transaction cost applied to
                each trade value when computing net cost.
            min_trade_size: Minimum absolute trade value; smaller trades
                are skipped to avoid excessive churn.
        """
        if not np.isclose(target_weights.sum(), 1.0):
            raise ValueError(f"target_weights must sum to 1.0; got {target_weights.sum():.4f}")

        self.target_weights = target_weights
        self.n_assets = len(target_weights)
        self.asset_names = asset_names or [f"asset_{i}" for i in range(self.n_assets)]
        self.rebalance_threshold = rebalance_threshold
        self.transaction_cost = transaction_cost
        self.min_trade_size = min_trade_size

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def current_weights(self, holdings: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Compute the current portfolio weight vector from holdings.

        Args:
            holdings: Number of units held per asset.
            prices: Current price per asset.

        Returns:
            Weight vector of shape ``(n_assets,)``.
        """
        values = holdings * prices
        total = values.sum()
        if total <= 0:
            return np.zeros(self.n_assets)
        return values / total

    def drift(self, current: np.ndarray) -> np.ndarray:
        """Compute the signed drift from target weights.

        Args:
            current: Current weight vector.

        Returns:
            Drift array (current − target).
        """
        return current - self.target_weights

    def needs_rebalance(self, current: np.ndarray) -> bool:
        """Decide whether rebalancing is warranted.

        Args:
            current: Current weight vector.

        Returns:
            *True* if any asset drifts beyond *rebalance_threshold*.
        """
        return bool(np.any(np.abs(self.drift(current)) > self.rebalance_threshold))

    def compute_trades(
        self,
        current_weights: np.ndarray,
        portfolio_value: float,
        prices: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute the required trades to reach target weights.

        Args:
            current_weights: Current weight vector.
            portfolio_value: Total portfolio value in base currency.
            prices: Current asset prices.

        Returns:
            A tuple ``(trade_units, total_cost)`` where *trade_units* is
            positive for buys and negative for sells, and *total_cost* is
            the estimated transaction cost.
        """
        target_values = self.target_weights * portfolio_value
        current_values = current_weights * portfolio_value
        delta_values = target_values - current_values

        # Filter out trades below min_trade_size
        delta_values = np.where(np.abs(delta_values) >= self.min_trade_size, delta_values, 0.0)

        trade_units = delta_values / np.where(prices > 0, prices, np.inf)
        total_cost = np.abs(delta_values).sum() * self.transaction_cost

        for name, units, val in zip(self.asset_names, trade_units, delta_values):
            if val != 0:
                action = "BUY" if val > 0 else "SELL"
                logger.debug("%s %s %.4f units ($%.2f)", action, name, abs(units), abs(val))

        return trade_units, total_cost

    def rebalance(
        self,
        holdings: np.ndarray,
        prices: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool]:
        """Perform a full rebalance check and compute required trades.

        Args:
            holdings: Current unit holdings per asset.
            prices: Current prices per asset.

        Returns:
            ``(trade_units, cost, did_rebalance)`` where *did_rebalance*
            is *False* when drift is below the threshold.
        """
        cur = self.current_weights(holdings, prices)
        if not self.needs_rebalance(cur):
            logger.info("Portfolio within threshold; no rebalance required.")
            return np.zeros(self.n_assets), 0.0, False

        portfolio_value = (holdings * prices).sum()
        trade_units, cost = self.compute_trades(cur, portfolio_value, prices)
        logger.info("Rebalancing portfolio. Estimated cost: $%.4f", cost)
        return trade_units, cost, True

    # ------------------------------------------------------------------
    # Calendar-based rebalancing
    # ------------------------------------------------------------------

    def calendar_rebalance(
        self,
        holdings: np.ndarray,
        prices: np.ndarray,
        force: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Rebalance unconditionally (calendar-triggered).

        Args:
            holdings: Current unit holdings.
            prices: Current prices.
            force: When *True* ignore the drift threshold.

        Returns:
            ``(trade_units, estimated_cost)``.
        """
        cur = self.current_weights(holdings, prices)
        if not force and not self.needs_rebalance(cur):
            return np.zeros(self.n_assets), 0.0
        portfolio_value = (holdings * prices).sum()
        return self.compute_trades(cur, portfolio_value, prices)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def allocation_report(self, holdings: np.ndarray, prices: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Produce a human-readable allocation report.

        Args:
            holdings: Current unit holdings.
            prices: Current prices.

        Returns:
            Nested dict ``{asset: {current_weight, target_weight, drift}}``.
        """
        cur = self.current_weights(holdings, prices)
        drifts = self.drift(cur)
        return {
            name: {
                "current_weight": float(cw),
                "target_weight": float(tw),
                "drift": float(d),
            }
            for name, cw, tw, d in zip(self.asset_names, cur, self.target_weights, drifts)
        }
