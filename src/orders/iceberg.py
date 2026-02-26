"""Iceberg order implementation."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IcebergSlice:
    """A single visible slice of an iceberg order."""

    slice_id: int
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    status: str = "pending"     # pending | submitted | filled | cancelled
    filled_qty: float = 0.0
    filled_price: float = 0.0


@dataclass
class IcebergOrder:
    """
    Iceberg (reserve) order that exposes only a small visible quantity.

    The order splits a large total quantity into *peak_size* slices and
    submits the next slice only after the current one is filled.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g. "BTC/USDT").
    side : str
        "buy" or "sell".
    total_quantity : float
        Full order size.
    peak_size : float
        Visible quantity per slice.
    price : float, optional
        Limit price; if None, slices are submitted as market orders.
    randomize_peak : bool
        Slightly randomise each slice size (±10 %) to reduce detectability.
    """

    symbol: str
    side: str
    total_quantity: float
    peak_size: float
    price: Optional[float] = None
    randomize_peak: bool = True

    slices: List[IcebergSlice] = field(default_factory=list, init=False)
    _remaining: float = field(init=False)
    _slice_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._remaining = self.total_quantity
        self._build_slices()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_slices(self) -> None:
        """Pre-compute all slices."""
        try:
            import random
            remaining = self.total_quantity
            while remaining > 0:
                base = min(self.peak_size, remaining)
                if self.randomize_peak and remaining > base:
                    jitter = random.uniform(0.90, 1.10)
                    size = min(base * jitter, remaining)
                else:
                    size = base
                self.slices.append(
                    IcebergSlice(
                        slice_id=self._slice_counter,
                        symbol=self.symbol,
                        side=self.side,
                        quantity=round(size, 8),
                        price=self.price,
                    )
                )
                self._slice_counter += 1
                remaining -= size
        except Exception as exc:  # noqa: BLE001
            logger.error("Error building iceberg slices: %s", exc)

    def next_slice(self) -> Optional[IcebergSlice]:
        """Return the next pending slice, or None if all slices are submitted."""
        for sl in self.slices:
            if sl.status == "pending":
                return sl
        return None

    def mark_filled(self, slice_id: int, filled_qty: float, filled_price: float) -> None:
        """
        Mark a slice as filled and update remaining quantity.

        Parameters
        ----------
        slice_id : int
            The slice that was filled.
        filled_qty : float
            Quantity actually filled.
        filled_price : float
            Average fill price.
        """
        for sl in self.slices:
            if sl.slice_id == slice_id:
                sl.filled_qty = filled_qty
                sl.filled_price = filled_price
                sl.status = "filled"
                break

    @property
    def total_filled(self) -> float:
        """Total quantity filled across all slices."""
        return sum(sl.filled_qty for sl in self.slices)

    @property
    def is_complete(self) -> bool:
        """True when all slices have been filled."""
        return all(sl.status == "filled" for sl in self.slices)

    @property
    def average_fill_price(self) -> float:
        """Volume-weighted average fill price across completed slices."""
        filled_slices = [s for s in self.slices if s.status == "filled"]
        if not filled_slices:
            return 0.0
        total_value = sum(s.filled_qty * s.filled_price for s in filled_slices)
        total_qty = sum(s.filled_qty for s in filled_slices)
        return total_value / total_qty if total_qty else 0.0
