"""Cross-exchange arbitrage strategy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class CrossExchangeOpportunity:
    """Describes a price discrepancy across two exchanges."""

    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct: float           # (sell - buy) / buy * 100
    net_profit_pct: float       # after fees and estimated transfer costs
    max_size: float             # limited by order-book depth
    timestamp: float = 0.0


class CrossExchangeArbitrage:
    """
    Identifies and sizes cross-exchange arbitrage opportunities.

    The strategy compares the best ask on one exchange with the best bid
    on another for the same symbol and flags cases where the spread
    exceeds total round-trip fees.

    Parameters
    ----------
    fee_rates : dict
        Per-exchange taker fee rate, e.g. {"binance": 0.001, "kraken": 0.002}.
    transfer_cost_pct : float
        Estimated transfer / withdrawal cost as a percentage of trade value.
    min_net_profit_pct : float
        Minimum net profit percentage after all costs.
    depth_levels : int
        Number of order-book levels to consider for size estimation.
    """

    def __init__(
        self,
        fee_rates: Optional[Dict[str, float]] = None,
        transfer_cost_pct: float = 0.05,
        min_net_profit_pct: float = 0.10,
        depth_levels: int = 5,
    ) -> None:
        self.fee_rates = fee_rates or {}
        self.transfer_cost_pct = transfer_cost_pct
        self.min_net_profit_pct = min_net_profit_pct
        self.depth_levels = depth_levels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        ticker_data: Dict[str, Dict[str, Dict]],
    ) -> List[CrossExchangeOpportunity]:
        """
        Scan ticker data across exchanges for arbitrage opportunities.

        Parameters
        ----------
        ticker_data : dict
            Nested mapping: exchange → symbol → {"ask": float, "bid": float,
            "ask_size": float, "bid_size": float}.

        Returns
        -------
        list of CrossExchangeOpportunity
            Sorted by net_profit_pct descending.
        """
        opportunities: List[CrossExchangeOpportunity] = []
        exchanges = list(ticker_data.keys())

        for i, ex_buy in enumerate(exchanges):
            for ex_sell in exchanges[i + 1:]:
                for symbol in ticker_data[ex_buy]:
                    if symbol not in ticker_data[ex_sell]:
                        continue
                    opp = self._check_pair(
                        symbol,
                        ex_buy,
                        ex_sell,
                        ticker_data[ex_buy][symbol],
                        ticker_data[ex_sell][symbol],
                    )
                    if opp:
                        opportunities.append(opp)
                    # Check the reverse direction
                    opp_rev = self._check_pair(
                        symbol,
                        ex_sell,
                        ex_buy,
                        ticker_data[ex_sell][symbol],
                        ticker_data[ex_buy][symbol],
                    )
                    if opp_rev:
                        opportunities.append(opp_rev)

        opportunities.sort(key=lambda o: o.net_profit_pct, reverse=True)
        return opportunities

    def estimate_execution_time(self, opportunity: CrossExchangeOpportunity) -> float:
        """
        Rough estimate of execution time in seconds for the given opportunity.

        Returns a conservative estimate based on typical API latencies.
        """
        # Two API calls in parallel + network overhead
        return 2.5

    def calculate_position_size(
        self,
        opportunity: CrossExchangeOpportunity,
        capital: float,
        max_pct: float = 0.20,
    ) -> float:
        """
        Calculate safe position size capped by available depth and capital limit.

        Parameters
        ----------
        opportunity : CrossExchangeOpportunity
            The detected opportunity.
        capital : float
            Available capital in quote currency.
        max_pct : float
            Maximum fraction of capital to risk on one trade.

        Returns
        -------
        float
            Position size in base currency.
        """
        capital_limit = capital * max_pct
        depth_limit = opportunity.max_size * opportunity.buy_price
        usable_capital = min(capital_limit, depth_limit)
        return usable_capital / opportunity.buy_price if opportunity.buy_price > 0 else 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fee_for(self, exchange: str) -> float:
        return self.fee_rates.get(exchange, 0.001)

    def _check_pair(
        self,
        symbol: str,
        ex_buy: str,
        ex_sell: str,
        buy_tick: Dict,
        sell_tick: Dict,
    ) -> Optional[CrossExchangeOpportunity]:
        ask = float(buy_tick.get("ask", 0))
        bid = float(sell_tick.get("bid", 0))
        if ask <= 0 or bid <= 0 or bid <= ask:
            return None

        spread_pct = (bid - ask) / ask * 100
        total_fee_pct = (self._fee_for(ex_buy) + self._fee_for(ex_sell)) * 100
        net_profit_pct = spread_pct - total_fee_pct - self.transfer_cost_pct

        if net_profit_pct < self.min_net_profit_pct:
            return None

        ask_size = float(buy_tick.get("ask_size", 0))
        bid_size = float(sell_tick.get("bid_size", 0))
        max_size = min(ask_size, bid_size)

        return CrossExchangeOpportunity(
            symbol=symbol,
            buy_exchange=ex_buy,
            sell_exchange=ex_sell,
            buy_price=ask,
            sell_price=bid,
            spread_pct=spread_pct,
            net_profit_pct=net_profit_pct,
            max_size=max_size,
            timestamp=time.time(),
        )
