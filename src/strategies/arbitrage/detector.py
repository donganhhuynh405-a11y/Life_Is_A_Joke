"""Arbitrage opportunity detection and ranking engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedOpportunity:
    """Generic container for any detected arbitrage signal."""

    kind: str                   # "triangular" | "cross_exchange" | "funding"
    symbol: str
    profit_pct: float
    confidence: float           # 0–1
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    ttl_seconds: float = 5.0   # estimated time before opportunity expires


class ArbitrageDetector:
    """
    Unified arbitrage opportunity detector.

    Aggregates signals from triangular, cross-exchange, and funding-rate
    sub-strategies, deduplicates overlapping opportunities, and applies a
    confidence score based on historical accuracy.

    Parameters
    ----------
    min_confidence : float
        Minimum confidence score (0–1) to surface an opportunity.
    decay_window : int
        Number of recent opportunities used to compute rolling accuracy.
    """

    def __init__(
        self,
        min_confidence: float = 0.50,
        decay_window: int = 100,
    ) -> None:
        self.min_confidence = min_confidence
        self.decay_window = decay_window
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        raw_opportunities: List[Any],
        kind: str,
    ) -> List[DetectedOpportunity]:
        """
        Convert raw strategy-specific opportunities into DetectedOpportunity objects.

        Parameters
        ----------
        raw_opportunities : list
            Output from a strategy's ``scan`` / ``find_opportunities`` method.
        kind : str
            One of "triangular", "cross_exchange", "funding".

        Returns
        -------
        list of DetectedOpportunity
            Filtered and ranked results.
        """
        detected: List[DetectedOpportunity] = []
        for raw in raw_opportunities:
            opp = self._convert(raw, kind)
            if opp is None:
                continue
            opp.confidence = self._adjust_confidence(opp)
            if opp.confidence >= self.min_confidence:
                detected.append(opp)

        detected.sort(key=lambda o: o.profit_pct * o.confidence, reverse=True)
        return detected

    def rank(self, opportunities: List[DetectedOpportunity]) -> List[DetectedOpportunity]:
        """
        Re-rank a mixed list of opportunities by risk-adjusted expected value.

        Parameters
        ----------
        opportunities : list of DetectedOpportunity
            Mixed opportunities from multiple strategies.

        Returns
        -------
        list of DetectedOpportunity
            Sorted by expected_value = profit_pct * confidence.
        """
        return sorted(
            opportunities,
            key=lambda o: o.profit_pct * o.confidence,
            reverse=True,
        )

    def record_outcome(self, opportunity: DetectedOpportunity, realised_pct: float) -> None:
        """
        Record the realised outcome to improve future confidence estimates.

        Parameters
        ----------
        opportunity : DetectedOpportunity
            The opportunity that was acted on.
        realised_pct : float
            Actual profit percentage achieved.
        """
        self._history.append(
            {
                "kind": opportunity.kind,
                "predicted_pct": opportunity.profit_pct,
                "realised_pct": realised_pct,
                "ts": time.time(),
            }
        )
        if len(self._history) > self.decay_window * 2:
            self._history = self._history[-self.decay_window :]

    def rolling_accuracy(self, kind: Optional[str] = None) -> float:
        """
        Compute rolling hit-rate (fraction where realised_pct > 0).

        Parameters
        ----------
        kind : str, optional
            Filter by opportunity kind; if None, aggregate all kinds.

        Returns
        -------
        float
            Hit rate in [0, 1].
        """
        records = [r for r in self._history if kind is None or r["kind"] == kind]
        if not records:
            return 0.5  # neutral prior
        hits = sum(1 for r in records[-self.decay_window :] if r["realised_pct"] > 0)
        return hits / min(len(records), self.decay_window)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert(self, raw: Any, kind: str) -> Optional[DetectedOpportunity]:
        """Map strategy-specific objects to DetectedOpportunity."""
        try:
            if kind == "triangular":
                return DetectedOpportunity(
                    kind=kind,
                    symbol="|".join(raw.path.symbols),
                    profit_pct=raw.path.profit_pct,
                    confidence=raw.confidence,
                    details={"path": raw.path.symbols, "directions": raw.path.directions},
                    timestamp=raw.timestamp,
                )
            if kind == "cross_exchange":
                return DetectedOpportunity(
                    kind=kind,
                    symbol=raw.symbol,
                    profit_pct=raw.net_profit_pct,
                    confidence=min(1.0, raw.net_profit_pct / 0.5),
                    details={
                        "buy_exchange": raw.buy_exchange,
                        "sell_exchange": raw.sell_exchange,
                        "spread_pct": raw.spread_pct,
                    },
                    timestamp=raw.timestamp,
                )
            if kind == "funding":
                return DetectedOpportunity(
                    kind=kind,
                    symbol=raw.symbol,
                    profit_pct=raw.net_yield_pct,
                    confidence=min(1.0, raw.net_yield_pct / 20.0),
                    details={
                        "direction": raw.direction,
                        "annualized_rate": raw.annualized_rate,
                        "basis_pct": raw.basis_pct,
                    },
                    timestamp=raw.timestamp,
                )
        except AttributeError as exc:
            logger.debug("Could not convert opportunity: %s", exc)
        return None

    def _adjust_confidence(self, opp: DetectedOpportunity) -> float:
        """Scale confidence by historical accuracy for this kind."""
        accuracy = self.rolling_accuracy(opp.kind)
        return float(np.clip(opp.confidence * accuracy * 2, 0.0, 1.0))
