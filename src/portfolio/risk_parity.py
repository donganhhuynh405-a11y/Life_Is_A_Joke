"""Risk parity portfolio allocation.

Each asset (or asset cluster) contributes equally to total portfolio risk.
Uses only ``numpy`` and ``scipy``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class RiskParityOptimizer:
    """Equal Risk Contribution (ERC) portfolio optimizer.

    The target is to find weights **w** such that every asset's marginal
    risk contribution (MRC) is equal:

        MRC_i = w_i · (Σw)_i / √(wᵀΣw)  ∀ i

    Reference:
        Maillard, S., Roncalli, T. & Teiletche, J. (2010).
        *The Properties of Equally Weighted Risk Contribution Portfolios*.
        Journal of Portfolio Management, 36(4), 60-70.
    """

    def __init__(
        self,
        asset_names: Optional[List[str]] = None,
        risk_budget: Optional[np.ndarray] = None,
        max_iter: int = 2000,
        tol: float = 1e-10,
    ) -> None:
        """Initialise the risk parity optimizer.

        Args:
            asset_names: Optional asset labels.
            risk_budget: Target risk-contribution fractions ``(n_assets,)``.
                Must sum to 1.  Defaults to equal risk budget.
            max_iter: Maximum SLSQP iterations.
            tol: Convergence tolerance.
        """
        self.asset_names = asset_names
        self.risk_budget = risk_budget
        self.max_iter = max_iter
        self.tol = tol
        self._cov: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _portfolio_vol(self, w: np.ndarray) -> float:
        return float(np.sqrt(w @ self._cov @ w))

    def _risk_contributions(self, w: np.ndarray) -> np.ndarray:
        """Compute each asset's fractional risk contribution.

        Args:
            w: Weight vector.

        Returns:
            Fractional contribution vector summing to 1.
        """
        sigma = self._portfolio_vol(w)
        if sigma == 0:
            return np.zeros_like(w)
        mrc = (self._cov @ w) * w / sigma
        return mrc / mrc.sum()

    def _erc_objective(self, w: np.ndarray, budget: np.ndarray) -> float:
        """Sum of squared deviations from the target risk budget.

        Args:
            w: Current weight vector.
            budget: Target fractional risk budget.
        """
        rc = self._risk_contributions(w)
        return float(np.sum((rc - budget) ** 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, cov_matrix: np.ndarray) -> "RiskParityOptimizer":
        """Fit the optimizer by storing the covariance matrix.

        Args:
            cov_matrix: Asset return covariance matrix ``(n_assets, n_assets)``.

        Returns:
            ``self``.
        """
        self._cov = cov_matrix
        n = cov_matrix.shape[0]
        if self.asset_names is None:
            self.asset_names = [f"asset_{i}" for i in range(n)]
        if self.risk_budget is None:
            self.risk_budget = np.ones(n) / n
        elif not np.isclose(self.risk_budget.sum(), 1.0):
            raise ValueError("risk_budget must sum to 1.0")
        logger.info("RiskParityOptimizer fitted on %d assets.", n)
        return self

    def optimize(self) -> np.ndarray:
        """Solve for the risk parity weights.

        Returns:
            Optimal weight vector ``(n_assets,)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._cov is None:
            raise RuntimeError("Optimiser not fitted. Call fit() first.")
        n = self._cov.shape[0]
        x0 = np.ones(n) / n

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(1e-6, 1.0)] * n  # Long-only; avoid exact zero for stability

        result = minimize(
            fun=self._erc_objective,
            x0=x0,
            args=(self.risk_budget,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )
        if not result.success:
            logger.warning("Risk parity optimisation did not converge: %s", result.message)

        self._weights = result.x
        return self._weights

    def risk_contribution_report(self) -> Dict[str, Dict[str, float]]:
        """Return a per-asset risk contribution report.

        Returns:
            Nested dict ``{asset: {weight, risk_contribution, target_budget}}``.

        Raises:
            RuntimeError: If :meth:`optimize` has not been called.
        """
        if self._weights is None:
            raise RuntimeError("Call optimize() first.")
        rc = self._risk_contributions(self._weights)
        return {
            name: {
                "weight": float(w),
                "risk_contribution": float(r),
                "target_budget": float(b),
            }
            for name, w, r, b in zip(self.asset_names, self._weights, rc, self.risk_budget)
        }

    def portfolio_volatility(self) -> float:
        """Return the optimised portfolio volatility.

        Raises:
            RuntimeError: If :meth:`optimize` has not been called.
        """
        if self._weights is None:
            raise RuntimeError("Call optimize() first.")
        return self._portfolio_vol(self._weights)


class HierarchicalRiskParity:
    """Hierarchical Risk Parity (HRP) using scipy hierarchical clustering.

    HRP does not require matrix inversion and is therefore more robust
    when the covariance matrix is near-singular.

    Reference:
        López de Prado, M. (2016). *Building Diversified Portfolios that
        Outperform Out-of-Sample*. Journal of Portfolio Management, 42(4).
    """

    def __init__(self, linkage_method: str = "single") -> None:
        """Initialise HRP.

        Args:
            linkage_method: Scipy linkage method used for clustering.
        """
        self.linkage_method = linkage_method
        self._weights: Optional[np.ndarray] = None
        self._asset_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Internal helpers (recursive bisection)
    # ------------------------------------------------------------------

    def _get_cluster_var(self, cov: np.ndarray, cluster_items: List[int]) -> float:
        cov_slice = cov[np.ix_(cluster_items, cluster_items)]
        w = self._get_ivp(cov_slice)
        return float(w @ cov_slice @ w)

    @staticmethod
    def _get_ivp(cov: np.ndarray) -> np.ndarray:
        """Inverse-variance portfolio weights for a given covariance block."""
        ivp = 1.0 / np.diag(cov)
        return ivp / ivp.sum()

    def _recursive_bisection(self, cov: np.ndarray, sorted_items: List[int]) -> np.ndarray:
        """Recursively allocate weight between two halves of *sorted_items*."""
        w = np.ones(len(sorted_items))
        cluster_items = [list(range(len(sorted_items)))]

        while cluster_items:
            cluster_items = [
                i[j:k]
                for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(cluster_items), 2):
                if i + 1 >= len(cluster_items):
                    break
                left = cluster_items[i]
                right = cluster_items[i + 1]
                left_var = self._get_cluster_var(cov, [sorted_items[x] for x in left])
                right_var = self._get_cluster_var(cov, [sorted_items[x] for x in right])
                alloc_left = 1 - left_var / (left_var + right_var)
                w[left] *= alloc_left
                w[right] *= 1 - alloc_left

        return w

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_optimize(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Compute HRP weights from a return matrix.

        Args:
            returns: Asset return matrix ``(n_periods, n_assets)``.
            asset_names: Optional asset labels.

        Returns:
            HRP weight vector ``(n_assets,)``.
        """
        from scipy.cluster.hierarchy import linkage, leaves_list  # noqa: PLC0415

        n = returns.shape[1]
        self._asset_names = asset_names or [f"asset_{i}" for i in range(n)]
        corr = np.corrcoef(returns, rowvar=False)
        cov = np.cov(returns, rowvar=False)

        # Correlation-based distance
        dist = np.sqrt((1 - corr) / 2)
        np.fill_diagonal(dist, 0.0)
        condensed = dist[np.triu_indices_from(dist, k=1)]
        Z = linkage(condensed, method=self.linkage_method)
        sorted_items = list(leaves_list(Z))

        w_sorted = self._recursive_bisection(cov, sorted_items)
        # Map back to original asset order
        self._weights = np.empty(n)
        for rank, original_idx in enumerate(sorted_items):
            self._weights[original_idx] = w_sorted[rank]

        logger.info("HRP weights computed for %d assets.", n)
        return self._weights

    def weights(self) -> np.ndarray:
        """Return the computed HRP weights.

        Raises:
            RuntimeError: If :meth:`fit_optimize` has not been called.
        """
        if self._weights is None:
            raise RuntimeError("Call fit_optimize() first.")
        return self._weights
