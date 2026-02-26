"""Black-Litterman portfolio optimisation model.

Combines market-implied equilibrium returns (reverse-optimised from the
market cap weights) with investor views to produce a posterior return
distribution used for mean-variance optimisation.

Uses only ``numpy`` and ``scipy``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class BlackLitterman:
    """Black-Litterman model for incorporating investor views into MPT.

    Reference:
        Black, F. & Litterman, R. (1992). Global portfolio optimisation.
        *Financial Analysts Journal*, 48(5), 28-43.
    """

    def __init__(
        self,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        asset_names: Optional[List[str]] = None,
    ) -> None:
        """Initialise the Black-Litterman model.

        Args:
            cov_matrix: Covariance matrix of asset returns
                ``(n_assets, n_assets)``.
            market_weights: Market-capitalisation weights ``(n_assets,)``.
            risk_aversion: Risk-aversion coefficient (δ) for the
                reverse-optimisation step.
            tau: Uncertainty scaling factor applied to the prior
                covariance (typically 0.01–0.10).
            asset_names: Optional labels for each asset.
        """
        n = cov_matrix.shape[0]
        if cov_matrix.shape != (n, n):
            raise ValueError("cov_matrix must be square.")
        if market_weights.shape[0] != n:
            raise ValueError("market_weights length must equal cov_matrix size.")

        self.cov_matrix = cov_matrix
        self.market_weights = market_weights / market_weights.sum()
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.asset_names = asset_names or [f"asset_{i}" for i in range(n)]
        self.n_assets = n

        # Derived
        self._pi: np.ndarray = self._implied_equilibrium_returns()
        self._posterior_returns: Optional[np.ndarray] = None
        self._posterior_cov: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Reverse-optimisation
    # ------------------------------------------------------------------

    def _implied_equilibrium_returns(self) -> np.ndarray:
        """Compute market-implied equilibrium excess returns (Π).

        Π = δ · Σ · w_mkt

        Returns:
            Equilibrium return vector ``(n_assets,)``.
        """
        pi = self.risk_aversion * self.cov_matrix @ self.market_weights
        logger.debug("Implied equilibrium returns: %s", pi)
        return pi

    @property
    def equilibrium_returns(self) -> np.ndarray:
        """Market-implied equilibrium excess return vector Π."""
        return self._pi

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def add_views(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: Optional[np.ndarray] = None,
        confidence: float = 0.5,
    ) -> None:
        """Incorporate investor views and compute posterior estimates.

        Args:
            P: Pick matrix of shape ``(n_views, n_assets)`` – each row
                defines one relative or absolute view.
            Q: View return vector of shape ``(n_views,)`` – the expected
                excess return for each view.
            Omega: View uncertainty covariance ``(n_views, n_views)``.
                When *None* it is set to
                ``diag(τ · P · Σ · Pᵀ) / confidence``.
            confidence: Scalar in ``(0, 1]`` used to scale *Omega* when
                *Omega* is *None*.  Higher values → more confident views.

        Raises:
            ValueError: On shape mismatches.
        """
        n_views = P.shape[0]
        if P.shape[1] != self.n_assets:
            raise ValueError(f"P must have {self.n_assets} columns; got {P.shape[1]}.")
        if Q.shape[0] != n_views:
            raise ValueError(f"Q must have {n_views} elements; got {Q.shape[0]}.")

        if Omega is None:
            diag_vals = self.tau * np.diag(P @ self.cov_matrix @ P.T)
            Omega = np.diag(diag_vals / confidence)

        prior_cov = self.tau * self.cov_matrix  # τΣ
        M = np.linalg.inv(np.linalg.inv(prior_cov) + P.T @ np.linalg.inv(Omega) @ P)
        self._posterior_returns = M @ (np.linalg.inv(prior_cov) @ self._pi + P.T @ np.linalg.inv(Omega) @ Q)
        self._posterior_cov = M + self.cov_matrix  # add back sampling uncertainty

        logger.info("Black-Litterman posterior computed with %d views.", n_views)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimal_weights(
        self,
        risk_free_rate: float = 0.0,
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> np.ndarray:
        """Compute maximum Sharpe ratio weights using the posterior estimates.

        Args:
            risk_free_rate: Annualised risk-free rate.
            weight_bounds: ``(min, max)`` weight bounds per asset.

        Returns:
            Optimal weight vector ``(n_assets,)``.

        Raises:
            RuntimeError: If :meth:`add_views` has not been called.
        """
        if self._posterior_returns is None:
            raise RuntimeError("No views added. Call add_views() first.")

        mu = self._posterior_returns
        sigma = self._posterior_cov

        def neg_sharpe(w: np.ndarray) -> float:
            ret = float(w @ mu)
            vol = float(np.sqrt(w @ sigma @ w))
            return -(ret - risk_free_rate) / vol if vol > 0 else 0.0

        x0 = np.ones(self.n_assets) / self.n_assets
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [weight_bounds] * self.n_assets

        result = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if not result.success:
            logger.warning("BL optimisation did not fully converge: %s", result.message)
        return result.x

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def posterior_summary(self) -> Dict[str, np.ndarray]:
        """Return posterior return and covariance estimates.

        Returns:
            Dict with keys ``"returns"`` and ``"covariance"``.

        Raises:
            RuntimeError: If :meth:`add_views` has not been called.
        """
        if self._posterior_returns is None:
            raise RuntimeError("No views added. Call add_views() first.")
        return {
            "returns": self._posterior_returns.copy(),
            "covariance": self._posterior_cov.copy(),
        }

    def view_diagnostics(
        self,
        P: np.ndarray,
        Q: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute diagnostics comparing equilibrium vs view-implied returns.

        Args:
            P: Pick matrix (same as passed to :meth:`add_views`).
            Q: View return vector.

        Returns:
            Dict with keys ``"equilibrium"``, ``"views"``,
            ``"posterior"``.
        """
        if self._posterior_returns is None:
            raise RuntimeError("No views added. Call add_views() first.")
        return {
            "equilibrium": self._pi,
            "views": P.T @ Q,
            "posterior": self._posterior_returns,
        }
