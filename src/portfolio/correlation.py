"""Asset correlation analysis for portfolio construction.

Uses only ``numpy`` and ``scipy`` for all computations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Compute, visualise, and cluster asset return correlations.

    Supports Pearson, Spearman, and rolling correlations as well as
    hierarchical clustering for portfolio diversification analysis.
    """

    def __init__(
        self,
        asset_names: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> None:
        """Initialise the analyser.

        Args:
            asset_names: Optional list of asset names for labelling.
            method: Default correlation method – ``"pearson"`` or
                ``"spearman"``.
        """
        self.asset_names = asset_names
        self.method = method
        self._corr_matrix: Optional[np.ndarray] = None
        self._returns: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Correlation computation
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> "CorrelationAnalyzer":
        """Compute the correlation matrix from a returns array.

        Args:
            returns: Return matrix of shape ``(n_periods, n_assets)``.

        Returns:
            ``self``.
        """
        self._returns = returns
        n = returns.shape[1]
        if self.asset_names is None:
            self.asset_names = [f"asset_{i}" for i in range(n)]

        if self.method == "pearson":
            self._corr_matrix = np.corrcoef(returns, rowvar=False)
        elif self.method == "spearman":
            corr, _ = stats.spearmanr(returns)
            self._corr_matrix = np.atleast_2d(corr)
        else:
            raise ValueError(f"Unknown method: {self.method!r}. Choose 'pearson' or 'spearman'.")

        logger.info(
            "CorrelationAnalyzer fitted on %d assets, %d periods (%s).",
            n,
            returns.shape[0],
            self.method,
        )
        return self

    def correlation_matrix(self) -> np.ndarray:
        """Return the fitted correlation matrix.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._corr_matrix is None:
            raise RuntimeError("Analyser not fitted. Call fit() first.")
        return self._corr_matrix

    def rolling_correlation(
        self,
        asset_a: int,
        asset_b: int,
        window: int = 60,
    ) -> np.ndarray:
        """Compute rolling pairwise correlation between two assets.

        Args:
            asset_a: Column index of the first asset.
            asset_b: Column index of the second asset.
            window: Rolling window size (in periods).

        Returns:
            Array of rolling correlation values (NaN for initial periods).
        """
        if self._returns is None:
            raise RuntimeError("Analyser not fitted. Call fit() first.")
        a = self._returns[:, asset_a]
        b = self._returns[:, asset_b]
        n = len(a)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            slice_a = a[i - window + 1 : i + 1]
            slice_b = b[i - window + 1 : i + 1]
            if np.std(slice_a) > 0 and np.std(slice_b) > 0:
                result[i] = np.corrcoef(slice_a, slice_b)[0, 1]
        return result

    # ------------------------------------------------------------------
    # Derived statistics
    # ------------------------------------------------------------------

    def average_correlation(self) -> float:
        """Return the mean off-diagonal correlation.

        Returns:
            Scalar average correlation coefficient.
        """
        corr = self.correlation_matrix()
        n = corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return float(corr[mask].mean())

    def most_correlated_pairs(self, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """Return the most positively correlated asset pairs.

        Args:
            top_n: Number of pairs to return.

        Returns:
            List of ``(asset_a, asset_b, correlation)`` tuples.
        """
        corr = self.correlation_matrix()
        n = corr.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((self.asset_names[i], self.asset_names[j], corr[i, j]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_n]

    def least_correlated_pairs(self, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """Return the least correlated (most diversifying) asset pairs.

        Args:
            top_n: Number of pairs to return.

        Returns:
            List of ``(asset_a, asset_b, correlation)`` tuples.
        """
        corr = self.correlation_matrix()
        n = corr.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((self.asset_names[i], self.asset_names[j], corr[i, j]))
        pairs.sort(key=lambda x: x[2])
        return pairs[:top_n]

    # ------------------------------------------------------------------
    # Hierarchical clustering
    # ------------------------------------------------------------------

    def cluster_assets(
        self,
        n_clusters: int = 3,
        linkage_method: str = "ward",
    ) -> Dict[str, int]:
        """Cluster assets hierarchically based on correlation distance.

        Args:
            n_clusters: Number of clusters to form.
            linkage_method: Scipy linkage method (``"ward"``, ``"average"``, etc.).

        Returns:
            Dictionary mapping asset name to cluster label.
        """
        corr = self.correlation_matrix()
        distance = np.sqrt((1 - corr) / 2)
        # Make symmetric and zero diagonal to satisfy pdist-like requirements
        np.fill_diagonal(distance, 0.0)

        condensed = distance[np.triu_indices_from(distance, k=1)]
        Z = linkage(condensed, method=linkage_method)
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        return {name: int(label) for name, label in zip(self.asset_names, labels)}

    def diversification_ratio(self, weights: np.ndarray, volatilities: np.ndarray) -> float:
        """Compute the diversification ratio of a portfolio.

        The diversification ratio is the weighted average asset volatility
        divided by the portfolio volatility.

        Args:
            weights: Portfolio weight vector.
            volatilities: Per-asset standard deviation vector.

        Returns:
            Diversification ratio (≥ 1; higher is more diversified).
        """
        corr = self.correlation_matrix()
        cov = np.diag(volatilities) @ corr @ np.diag(volatilities)
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        weighted_avg_vol = np.dot(weights, volatilities)
        return float(weighted_avg_vol / portfolio_vol) if portfolio_vol > 0 else 1.0
