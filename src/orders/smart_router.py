"""Smart order routing across multiple venues."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Venue:
    """Represents a trading venue with its fee structure and order book."""

    name: str
    fee_rate: float                         # taker fee rate
    asks: List[List[float]] = field(default_factory=list)   # [[price, qty], ...]
    bids: List[List[float]] = field(default_factory=list)
    latency_ms: float = 0.0                 # estimated order placement latency

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else float("inf")

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0


@dataclass
class RoutedOrder:
    """A child order routed to a specific venue."""

    venue: str
    side: str
    quantity: float
    price: Optional[float]      # None → market
    expected_cost: float        # inclusive of fees


class SmartOrderRouter:
    """
    Routes a parent order across multiple venues to minimise total cost.

    The router sweeps the consolidated order book (all venues combined),
    greedily filling at the best available prices while accounting for
    per-venue fee rates.

    Parameters
    ----------
    venues : list of Venue
        Available venues with current order books loaded.
    max_venues : int
        Maximum number of venues to split an order across.
    prioritise_latency : bool
        If True, break price ties by choosing the lowest-latency venue.
    """

    def __init__(
        self,
        venues: Optional[List[Venue]] = None,
        max_venues: int = 3,
        prioritise_latency: bool = False,
    ) -> None:
        self.venues: List[Venue] = venues or []
        self.max_venues = max_venues
        self.prioritise_latency = prioritise_latency

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        side: str,
        quantity: float,
        symbol: str = "",
    ) -> List[RoutedOrder]:
        """
        Compute optimal routing for a parent order.

        Parameters
        ----------
        side : str
            "buy" or "sell".
        quantity : float
            Total quantity to fill.
        symbol : str
            Symbol name (informational).

        Returns
        -------
        list of RoutedOrder
            Child orders, sorted by venue name.
        """
        consolidated = self._build_consolidated_book(side)
        routed: List[RoutedOrder] = []
        venue_quantities: Dict[str, float] = {}

        remaining = quantity
        for price, qty, venue_name, fee_rate in consolidated:
            if remaining <= 0:
                break
            if len(venue_quantities) >= self.max_venues and venue_name not in venue_quantities:
                continue
            take = min(qty, remaining)
            cost = take * price * (1 + fee_rate)
            venue_quantities[venue_name] = venue_quantities.get(venue_name, 0) + take
            remaining -= take

        # Collapse per-venue quantities into child orders
        for v_name, qty in venue_quantities.items():
            venue = next((v for v in self.venues if v.name == v_name), None)
            if venue is None:
                continue
            book_side = venue.asks if side == "buy" else venue.bids
            avg_price = self._vwap_fill(book_side, qty)
            routed.append(
                RoutedOrder(
                    venue=v_name,
                    side=side,
                    quantity=qty,
                    price=avg_price,
                    expected_cost=qty * avg_price * (1 + venue.fee_rate),
                )
            )

        routed.sort(key=lambda o: o.venue)
        logger.debug("Routed %s %s across %d venue(s)", side, symbol, len(routed))
        return routed

    def estimated_cost(self, routed_orders: List[RoutedOrder]) -> float:
        """Total expected cost (including fees) for a set of routed orders."""
        return sum(o.expected_cost for o in routed_orders)

    def add_venue(self, venue: Venue) -> None:
        """Register a new venue."""
        self.venues.append(venue)

    def remove_venue(self, name: str) -> None:
        """Deregister a venue by name."""
        self.venues = [v for v in self.venues if v.name != name]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_consolidated_book(
        self, side: str
    ) -> List[Tuple[float, float, str, float]]:
        """Return a merged, sorted list of (price, qty, venue, fee) tuples."""
        levels: List[Tuple[float, float, str, float]] = []
        for venue in self.venues:
            book_side = venue.asks if side == "buy" else venue.bids
            for price, qty in book_side:
                adj_price = price * (1 + venue.fee_rate)
                levels.append((adj_price, qty, venue.name, venue.fee_rate))
        # Sort asks ascending, bids descending
        levels.sort(key=lambda x: x[0], reverse=(side == "sell"))
        return levels

    @staticmethod
    def _vwap_fill(book_side: List[List[float]], qty: float) -> float:
        """Volume-weighted average price for filling *qty* from a book side."""
        filled, cost = 0.0, 0.0
        for price, size in book_side:
            take = min(size, qty - filled)
            cost += take * price
            filled += take
            if filled >= qty:
                break
        return cost / filled if filled else 0.0
