"""Modern Portfolio Theory and Markowitz mean-variance optimisation.

Uses only ``numpy`` and ``scipy``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Mean-variance portfolio optimiser (Markowitz framework).

    Supports maximum Sharpe ratio, minimum volatility, and
    target-return efficient-frontier objectives.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        allow_short: bool = False,
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Initialise the optimiser.

        Args:
            risk_free_rate: Annualised risk-free rate used in Sharpe calculation.
            allow_short: When *True* relax the lower bound to -1.
            weight_bounds: ``(min_weight, max_weight)`` per asset.
        """
        self.risk_free_rate = risk_free_rate
        self.allow_short = allow_short
        lb = -1.0 if allow_short else weight_bounds[0]
        ub = weight_bounds[1]
        self.weight_bounds = (lb, ub)
        self._mean_returns: Optional[np.ndarray] = None
        self._cov_matrix: Optional[np.ndarray] = None
        self._n_assets: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> "PortfolioOptimizer":
        """Compute expected returns and the covariance matrix.

        Args:
            returns: Asset return matrix of shape ``(n_periods, n_assets)``.

        Returns:
            ``self``.
        """
        self._mean_returns = returns.mean(axis=0)
        self._cov_matrix = np.cov(returns, rowvar=False)
        self._n_assets = returns.shape[1]
        logger.info("PortfolioOptimizer fitted on %d assets, %d periods.", self._n_assets, returns.shape[0])
        return self

    # ------------------------------------------------------------------
    # Portfolio statistics
    # ------------------------------------------------------------------

    def portfolio_return(self, weights: np.ndarray) -> float:
        """Compute portfolio expected return.

        Args:
            weights: Asset weight vector.
        """
        return float(np.dot(weights, self._mean_returns))

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Compute portfolio standard deviation.

        Args:
            weights: Asset weight vector.
        """
        return float(np.sqrt(weights @ self._cov_matrix @ weights))

    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """Compute the Sharpe ratio.

        Args:
            weights: Asset weight vector.
        """
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol if vol > 0 else 0.0

    # ------------------------------------------------------------------
    # Optimisation objectives
    # ------------------------------------------------------------------

    def _constraints(self) -> List[Dict]:
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def _bounds(self) -> List[Tuple[float, float]]:
        return [self.weight_bounds] * self._n_assets

    def _initial_weights(self) -> np.ndarray:
        return np.ones(self._n_assets) / self._n_assets

    def maximize_sharpe(self) -> np.ndarray:
        """Find the maximum Sharpe ratio portfolio.

        Returns:
            Optimal weight vector.
        """
        self._check_fitted()
        result = minimize(
            fun=lambda w: -self.sharpe_ratio(w),
            x0=self._initial_weights(),
            method="SLSQP",
            bounds=self._bounds(),
            constraints=self._constraints(),
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if not result.success:
            logger.warning("maximize_sharpe did not converge: %s", result.message)
        return result.x

    def minimize_volatility(self) -> np.ndarray:
        """Find the global minimum-variance portfolio.

        Returns:
            Optimal weight vector.
        """
        self._check_fitted()
        result = minimize(
            fun=lambda w: self.portfolio_volatility(w),
            x0=self._initial_weights(),
            method="SLSQP",
            bounds=self._bounds(),
            constraints=self._constraints(),
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if not result.success:
            logger.warning("minimize_volatility did not converge: %s", result.message)
        return result.x

    def efficient_return(self, target_return: float) -> np.ndarray:
        """Find the minimum-variance portfolio for a given target return.

        Args:
            target_return: Desired annualised portfolio return.

        Returns:
            Optimal weight vector.
        """
        self._check_fitted()
        constraints = self._constraints() + [
            {"type": "eq", "fun": lambda w: self.portfolio_return(w) - target_return}
        ]
        result = minimize(
            fun=lambda w: self.portfolio_volatility(w),
            x0=self._initial_weights(),
            method="SLSQP",
            bounds=self._bounds(),
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if not result.success:
            logger.warning("efficient_return did not converge: %s", result.message)
        return result.x

    def efficient_frontier(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate efficient frontier portfolios.

        Args:
            n_points: Number of frontier points to compute.

        Returns:
            Three arrays ``(returns, volatilities, sharpe_ratios)`` each
            of length *n_points*.
        """
        self._check_fitted()
        min_ret = float(np.min(self._mean_returns))
        max_ret = float(np.max(self._mean_returns))
        target_returns = np.linspace(min_ret, max_ret, n_points)

        rets, vols, sharpes = [], [], []
        for tr in target_returns:
            try:
                w = self.efficient_return(tr)
                rets.append(self.portfolio_return(w))
                vols.append(self.portfolio_volatility(w))
                sharpes.append(self.sharpe_ratio(w))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Frontier point failed: %s", exc)

        return np.array(rets), np.array(vols), np.array(sharpes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._mean_returns is None:
            raise RuntimeError("Optimiser not fitted. Call fit() first.")

    def get_portfolio_stats(self, weights: np.ndarray) -> Dict[str, float]:
        """Return a summary dict of portfolio statistics.

        Args:
            weights: Asset weight vector.
        """
        return {
            "expected_return": self.portfolio_return(weights),
            "volatility": self.portfolio_volatility(weights),
            "sharpe_ratio": self.sharpe_ratio(weights),
        }
