"""Time-Weighted Average Price (TWAP) execution algorithm."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TWAPSlice:
    """A single scheduled TWAP slice."""

    index: int
    scheduled_time: float       # Unix timestamp when slice should be submitted
    quantity: float
    status: str = "pending"     # pending | submitted | filled | skipped
    filled_qty: float = 0.0
    filled_price: float = 0.0
    submitted_time: float = 0.0


class TWAPExecutor:
    """
    Executes a large order by splitting it into equal time-based slices.

    The executor schedules *num_slices* child orders evenly over
    *duration_seconds* and optionally adds small random jitter to each
    interval to reduce market impact.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    side : str
        "buy" or "sell".
    total_quantity : float
        Full order quantity.
    duration_seconds : float
        Total time window over which to execute.
    num_slices : int
        Number of equal time slices.
    add_jitter : bool
        If True, randomises each slice time by ±10 % of interval.
    price_limit : float, optional
        Maximum (buy) or minimum (sell) acceptable price; slices outside
        this limit are skipped.
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        duration_seconds: float = 3600.0,
        num_slices: int = 12,
        add_jitter: bool = True,
        price_limit: Optional[float] = None,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.duration_seconds = duration_seconds
        self.num_slices = num_slices
        self.add_jitter = add_jitter
        self.price_limit = price_limit
        self.slices: List[TWAPSlice] = []
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, start_time: Optional[float] = None) -> None:
        """
        Initialise execution schedule.

        Parameters
        ----------
        start_time : float, optional
            Unix timestamp to begin execution; defaults to now.
        """
        self._start_time = start_time or time.time()
        interval = self.duration_seconds / self.num_slices
        base_qty = self.total_quantity / self.num_slices

        rng = np.random.default_rng()
        for i in range(self.num_slices):
            jitter = rng.uniform(-0.1, 0.1) * interval if self.add_jitter else 0.0
            scheduled = self._start_time + i * interval + jitter
            self.slices.append(
                TWAPSlice(
                    index=i,
                    scheduled_time=scheduled,
                    quantity=round(base_qty, 8),
                )
            )
        logger.info(
            "TWAP scheduled: %d slices over %.0fs for %s %s",
            self.num_slices,
            self.duration_seconds,
            self.side,
            self.symbol,
        )

    def due_slices(self, now: Optional[float] = None) -> List[TWAPSlice]:
        """
        Return slices that are due for submission.

        Parameters
        ----------
        now : float, optional
            Current Unix timestamp; defaults to ``time.time()``.

        Returns
        -------
        list of TWAPSlice
            Slices whose ``scheduled_time`` has passed and are still pending.
        """
        now = now or time.time()
        return [s for s in self.slices if s.status == "pending" and s.scheduled_time <= now]

    def mark_filled(self, index: int, filled_qty: float, filled_price: float) -> None:
        """Record fill for slice *index*."""
        for sl in self.slices:
            if sl.index == index:
                sl.filled_qty = filled_qty
                sl.filled_price = filled_price
                sl.status = "filled"
                break

    def skip_slice(self, index: int, reason: str = "") -> None:
        """Skip a slice (e.g. due to price limit breach)."""
        for sl in self.slices:
            if sl.index == index:
                sl.status = "skipped"
                logger.debug("TWAP slice %d skipped: %s", index, reason)
                break

    @property
    def progress_pct(self) -> float:
        """Percentage of total quantity filled."""
        filled = sum(s.filled_qty for s in self.slices if s.status == "filled")
        return filled / self.total_quantity * 100 if self.total_quantity else 0.0

    @property
    def vwap(self) -> float:
        """Volume-weighted average fill price so far."""
        filled = [s for s in self.slices if s.status == "filled"]
        if not filled:
            return 0.0
        total_value = sum(s.filled_qty * s.filled_price for s in filled)
        total_qty = sum(s.filled_qty for s in filled)
        return total_value / total_qty if total_qty else 0.0

    @property
    def is_complete(self) -> bool:
        """True when all slices are no longer pending."""
        return all(s.status != "pending" for s in self.slices)
