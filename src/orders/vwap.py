"""Volume-Weighted Average Price (VWAP) execution algorithm."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfile:
    """Historical intraday volume distribution."""

    buckets: np.ndarray         # relative volume weights per time bucket
    bucket_duration_seconds: float  # e.g. 300 for 5-minute buckets

    @classmethod
    def uniform(cls, n_buckets: int = 48, bucket_seconds: float = 1800.0) -> "VolumeProfile":
        """Create a flat (uniform) volume profile as a baseline."""
        return cls(
            buckets=np.ones(n_buckets) / n_buckets,
            bucket_duration_seconds=bucket_seconds,
        )

    @classmethod
    def from_historical(
        cls, volumes: np.ndarray, bucket_seconds: float = 1800.0
    ) -> "VolumeProfile":
        """
        Build a profile from historical per-bucket volume observations.

        Parameters
        ----------
        volumes : np.ndarray
            1-D array of historical volumes per bucket (raw counts or sums).
        bucket_seconds : float
            Duration of each bucket in seconds.
        """
        total = volumes.sum()
        weights = volumes / total if total > 0 else np.ones_like(volumes) / len(volumes)
        return cls(buckets=weights, bucket_duration_seconds=bucket_seconds)


@dataclass
class VWAPSlice:
    """Execution slice targeting a specific volume bucket."""

    bucket_index: int
    target_qty: float
    status: str = "pending"     # pending | submitted | filled | skipped
    filled_qty: float = 0.0
    filled_price: float = 0.0


class VWAPExecutor:
    """
    Executes a large order in proportion to historical intraday volume patterns.

    Instead of splitting time equally (TWAP), VWAP allocates more quantity to
    time buckets that historically see higher trading volume, minimising market
    impact and tracking error versus the market VWAP.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    side : str
        "buy" or "sell".
    total_quantity : float
        Full order quantity.
    volume_profile : VolumeProfile
        Intraday volume distribution to target.
    start_bucket : int
        Index of the first bucket to participate in.
    end_bucket : int, optional
        Last bucket index (inclusive); defaults to last bucket in profile.
    participation_rate : float
        Target fraction of market volume per bucket (0–1). Slices are
        scaled so the total equals total_quantity.
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        volume_profile: Optional[VolumeProfile] = None,
        start_bucket: int = 0,
        end_bucket: Optional[int] = None,
        participation_rate: float = 0.10,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.total_quantity = total_quantity
        self.volume_profile = volume_profile or VolumeProfile.uniform()
        self.start_bucket = start_bucket
        self.end_bucket = end_bucket if end_bucket is not None else len(
            self.volume_profile.buckets) - 1
        self.participation_rate = participation_rate
        self.slices: List[VWAPSlice] = []
        self._build_slices()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_slices(self) -> None:
        """Allocate quantities across active buckets."""
        profile = self.volume_profile.buckets
        active = profile[self.start_bucket: self.end_bucket + 1]
        total_weight = active.sum()
        if total_weight <= 0:
            active = np.ones(len(active)) / len(active)
            total_weight = 1.0

        for i, weight in enumerate(active):
            bucket_idx = self.start_bucket + i
            qty = round(self.total_quantity * weight / total_weight, 8)
            self.slices.append(VWAPSlice(bucket_index=bucket_idx, target_qty=qty))

    def slice_for_bucket(self, bucket_index: int) -> Optional[VWAPSlice]:
        """Return the pending slice for a given bucket, or None."""
        for sl in self.slices:
            if sl.bucket_index == bucket_index and sl.status == "pending":
                return sl
        return None

    def mark_filled(self, bucket_index: int, filled_qty: float, filled_price: float) -> None:
        """Record a fill for the slice at *bucket_index*."""
        for sl in self.slices:
            if sl.bucket_index == bucket_index:
                sl.filled_qty = filled_qty
                sl.filled_price = filled_price
                sl.status = "filled"
                break

    def adjust_remaining(self, bucket_index: int) -> float:
        """
        Re-distribute unfilled quantity from missed/skipped buckets into
        remaining pending slices.

        Parameters
        ----------
        bucket_index : int
            Current bucket (buckets before this are considered past).

        Returns
        -------
        float
            Quantity redistributed.
        """
        missed = sum(
            sl.target_qty - sl.filled_qty
            for sl in self.slices
            if sl.bucket_index < bucket_index and sl.status != "filled"
        )
        pending = [sl for sl in self.slices if sl.bucket_index >=
                   bucket_index and sl.status == "pending"]
        if pending and missed > 0:
            extra_per_slice = missed / len(pending)
            for sl in pending:
                sl.target_qty += extra_per_slice
        return missed

    @property
    def vwap(self) -> float:
        """Volume-weighted average fill price across completed slices."""
        filled = [s for s in self.slices if s.status == "filled" and s.filled_qty > 0]
        if not filled:
            return 0.0
        total_value = sum(s.filled_qty * s.filled_price for s in filled)
        total_qty = sum(s.filled_qty for s in filled)
        return total_value / total_qty if total_qty else 0.0

    @property
    def tracking_error(self, market_vwap: float = 0.0) -> float:
        """Absolute tracking error versus market VWAP."""
        return abs(self.vwap - market_vwap) if market_vwap else 0.0

    @property
    def is_complete(self) -> bool:
        """True when no pending slices remain."""
        return all(s.status != "pending" for s in self.slices)
