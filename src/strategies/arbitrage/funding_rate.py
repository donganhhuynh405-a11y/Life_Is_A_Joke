"""Funding rate arbitrage strategy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundingOpportunity:
    """Represents a funding-rate arbitrage setup."""

    symbol: str
    exchange: str
    funding_rate: float         # current 8-hour funding rate (can be negative)
    annualized_rate: float      # funding_rate * 3 * 365 * 100 (%)
    spot_price: float
    perp_price: float
    basis_pct: float            # (perp - spot) / spot * 100
    net_yield_pct: float        # estimated annual net yield after fees
    direction: str              # "long_spot_short_perp" or "short_spot_long_perp"
    timestamp: float = 0.0


class FundingRateArbitrage:
    """
    Captures funding rate payments by holding opposite positions in spot
    and perpetual futures markets.

    When the funding rate is positive, perp longs pay shorts; the strategy
    goes long spot and short perp to collect funding.  When negative, the
    reverse applies.

    Parameters
    ----------
    min_annual_yield_pct : float
        Minimum annualised net yield (%) to consider an opportunity.
    fee_rate : float
        Average trading fee rate per side.
    funding_intervals_per_day : int
        Number of funding settlements per day (typically 3 for 8-h funding).
    max_basis_pct : float
        Maximum tolerated basis percentage before rejecting the opportunity.
    """

    def __init__(
        self,
        min_annual_yield_pct: float = 10.0,
        fee_rate: float = 0.001,
        funding_intervals_per_day: int = 3,
        max_basis_pct: float = 1.0,
    ) -> None:
        self.min_annual_yield_pct = min_annual_yield_pct
        self.fee_rate = fee_rate
        self.funding_intervals_per_day = funding_intervals_per_day
        self.max_basis_pct = max_basis_pct
        self._intervals_per_year = funding_intervals_per_day * 365

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        funding_data: Dict[str, Dict],
        spot_prices: Dict[str, float],
    ) -> List[FundingOpportunity]:
        """
        Scan funding rates for arbitrage opportunities.

        Parameters
        ----------
        funding_data : dict
            Mapping of symbol → {"exchange": str, "funding_rate": float,
            "perp_price": float, "open_interest": float}.
        spot_prices : dict
            Mapping of symbol → spot price.

        Returns
        -------
        list of FundingOpportunity
            Sorted by net_yield_pct descending.
        """
        opportunities: List[FundingOpportunity] = []
        for symbol, data in funding_data.items():
            spot = spot_prices.get(symbol)
            if spot is None or spot <= 0:
                continue
            opp = self._evaluate(symbol, data, spot)
            if opp is not None:
                opportunities.append(opp)
        opportunities.sort(key=lambda o: o.net_yield_pct, reverse=True)
        return opportunities

    def estimate_holding_period_return(
        self, opportunity: FundingOpportunity, days: int
    ) -> float:
        """
        Estimate total return for holding the position for *days* days.

        Parameters
        ----------
        opportunity : FundingOpportunity
            The opportunity to evaluate.
        days : int
            Holding period in days.

        Returns
        -------
        float
            Estimated percentage return over the holding period.
        """
        intervals = days * self.funding_intervals_per_day
        gross = opportunity.funding_rate * intervals * 100
        entry_cost = self.fee_rate * 2 * 100  # open both legs
        exit_cost = self.fee_rate * 2 * 100   # close both legs
        return gross - entry_cost - exit_cost

    def predict_funding_direction(
        self, historical_rates: List[float], window: int = 8
    ) -> float:
        """
        Simple momentum estimate of future funding rate.

        Parameters
        ----------
        historical_rates : list of float
            Past funding rates in chronological order.
        window : int
            Look-back window for trend calculation.

        Returns
        -------
        float
            Predicted next funding rate.
        """
        if len(historical_rates) < 2:
            return historical_rates[-1] if historical_rates else 0.0
        arr = np.array(historical_rates[-window:])
        weights = np.arange(1, len(arr) + 1, dtype=float)
        return float(np.average(arr, weights=weights))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self, symbol: str, data: Dict, spot: float
    ) -> Optional[FundingOpportunity]:
        funding_rate = float(data.get("funding_rate", 0))
        perp_price = float(data.get("perp_price", spot))
        exchange = str(data.get("exchange", "unknown"))

        basis_pct = (perp_price - spot) / spot * 100
        if abs(basis_pct) > self.max_basis_pct:
            return None

        annualized = funding_rate * self._intervals_per_year * 100
        # Subtract round-trip fee cost annualised once (entry + exit amortised)
        entry_exit_annual = self.fee_rate * 4 * 100  # conservative
        net_yield = abs(annualized) - entry_exit_annual

        if net_yield < self.min_annual_yield_pct:
            return None

        direction = (
            "long_spot_short_perp" if funding_rate > 0 else "short_spot_long_perp"
        )
        return FundingOpportunity(
            symbol=symbol,
            exchange=exchange,
            funding_rate=funding_rate,
            annualized_rate=annualized,
            spot_price=spot,
            perp_price=perp_price,
            basis_pct=basis_pct,
            net_yield_pct=net_yield,
            direction=direction,
            timestamp=time.time(),
        )
