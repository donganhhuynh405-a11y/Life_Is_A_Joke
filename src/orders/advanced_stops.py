"""Advanced stop-loss order types."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trailing Stop
# ---------------------------------------------------------------------------

@dataclass
class TrailingStopOrder:
    """
    Trailing stop that moves with price but never regresses.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    side : str
        "long" (stop triggers on downside) or "short" (stop triggers on upside).
    trail_pct : float
        Trailing distance as a percentage of the reference price.
    initial_price : float
        Starting reference price.
    """

    symbol: str
    side: str
    trail_pct: float
    initial_price: float

    _reference_price: float = field(init=False)
    _stop_price: float = field(init=False)
    triggered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._reference_price = self.initial_price
        self._stop_price = self._compute_stop(self.initial_price)

    def _compute_stop(self, ref: float) -> float:
        multiplier = 1 - self.trail_pct / 100 if self.side == "long" else 1 + self.trail_pct / 100
        return ref * multiplier

    def update(self, current_price: float) -> bool:
        """
        Feed the latest price and check whether the stop has triggered.

        Parameters
        ----------
        current_price : float
            Latest market price.

        Returns
        -------
        bool
            True if the stop has just been triggered.
        """
        if self.triggered:
            return True

        if self.side == "long" and current_price > self._reference_price:
            self._reference_price = current_price
            self._stop_price = self._compute_stop(current_price)
        elif self.side == "short" and current_price < self._reference_price:
            self._reference_price = current_price
            self._stop_price = self._compute_stop(current_price)

        if self.side == "long" and current_price <= self._stop_price:
            self.triggered = True
            logger.info("Trailing stop triggered for %s at %.4f", self.symbol, current_price)
        elif self.side == "short" and current_price >= self._stop_price:
            self.triggered = True
            logger.info("Trailing stop triggered for %s at %.4f", self.symbol, current_price)

        return self.triggered

    @property
    def stop_price(self) -> float:
        """Current stop level."""
        return self._stop_price


# ---------------------------------------------------------------------------
# Bracket Order
# ---------------------------------------------------------------------------

@dataclass
class BracketOrder:
    """
    Bracket order combining a target (take-profit) and a stop-loss.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    side : str
        "long" or "short".
    entry_price : float
        Entry fill price.
    take_profit_pct : float
        Target profit as a percentage of entry price.
    stop_loss_pct : float
        Stop distance as a percentage of entry price.
    """

    symbol: str
    side: str
    entry_price: float
    take_profit_pct: float
    stop_loss_pct: float

    take_profit_price: float = field(init=False)
    stop_loss_price: float = field(init=False)
    status: str = field(default="open", init=False)  # open | tp_hit | sl_hit

    def __post_init__(self) -> None:
        if self.side == "long":
            self.take_profit_price = self.entry_price * (1 + self.take_profit_pct / 100)
            self.stop_loss_price = self.entry_price * (1 - self.stop_loss_pct / 100)
        else:
            self.take_profit_price = self.entry_price * (1 - self.take_profit_pct / 100)
            self.stop_loss_price = self.entry_price * (1 + self.stop_loss_pct / 100)

    def update(self, current_price: float) -> str:
        """
        Update the bracket with the latest price.

        Returns
        -------
        str
            Current status: "open", "tp_hit", or "sl_hit".
        """
        if self.status != "open":
            return self.status
        if self.side == "long":
            if current_price >= self.take_profit_price:
                self.status = "tp_hit"
            elif current_price <= self.stop_loss_price:
                self.status = "sl_hit"
        else:
            if current_price <= self.take_profit_price:
                self.status = "tp_hit"
            elif current_price >= self.stop_loss_price:
                self.status = "sl_hit"
        return self.status

    @property
    def risk_reward_ratio(self) -> float:
        """Risk-to-reward ratio (TP distance / SL distance)."""
        sl_dist = abs(self.entry_price - self.stop_loss_price)
        tp_dist = abs(self.take_profit_price - self.entry_price)
        return tp_dist / sl_dist if sl_dist else 0.0


# ---------------------------------------------------------------------------
# Time Stop
# ---------------------------------------------------------------------------

@dataclass
class TimeStopOrder:
    """
    Exits a position if it has not reached the target within a time limit.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    max_duration_seconds : float
        Maximum holding period before forced exit.
    """

    symbol: str
    max_duration_seconds: float

    _opened_at: float = field(init=False)
    triggered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._opened_at = time.time()

    def update(self, now: Optional[float] = None) -> bool:
        """
        Check if the time limit has elapsed.

        Returns
        -------
        bool
            True if the time stop has triggered.
        """
        if self.triggered:
            return True
        elapsed = (now or time.time()) - self._opened_at
        if elapsed >= self.max_duration_seconds:
            self.triggered = True
            logger.info("Time stop triggered for %s after %.1fs", self.symbol, elapsed)
        return self.triggered

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._opened_at


# ---------------------------------------------------------------------------
# Volatility Stop (ATR-based)
# ---------------------------------------------------------------------------

class VolatilityStop:
    """
    ATR-based stop that widens during high-volatility regimes.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    side : str
        "long" or "short".
    atr_multiplier : float
        Number of ATR units for the stop distance.
    atr_period : int
        Look-back period for ATR calculation.
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        atr_multiplier: float = 2.0,
        atr_period: int = 14,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period
        self._prices: List[float] = []
        self._highs: List[float] = []
        self._lows: List[float] = []
        self.stop_price: float = 0.0
        self.triggered: bool = False

    def update(self, high: float, low: float, close: float) -> bool:
        """
        Feed a new OHLC bar and update the volatility stop.

        Parameters
        ----------
        high, low, close : float
            Bar high, low, and close prices.

        Returns
        -------
        bool
            True if the stop has triggered.
        """
        self._highs.append(high)
        self._lows.append(low)
        self._prices.append(close)

        if len(self._prices) < 2:
            return False

        atr = self._compute_atr()
        stop_dist = atr * self.atr_multiplier

        if self.side == "long":
            new_stop = close - stop_dist
            self.stop_price = max(self.stop_price, new_stop)
            if close <= self.stop_price:
                self.triggered = True
        else:
            new_stop = close + stop_dist
            self.stop_price = min(self.stop_price, new_stop) if self.stop_price else new_stop
            if close >= self.stop_price:
                self.triggered = True

        return self.triggered

    def _compute_atr(self) -> float:
        """Compute the Average True Range over the look-back period."""
        highs = np.array(self._highs[-self.atr_period :])
        lows = np.array(self._lows[-self.atr_period :])
        closes = np.array(self._prices[-self.atr_period :])
        prev_closes = np.array(self._prices[-(self.atr_period + 1) : -1])

        if len(prev_closes) == 0:
            return highs[-1] - lows[-1]

        n = min(len(highs), len(prev_closes))
        tr = np.maximum(
            highs[-n:] - lows[-n:],
            np.maximum(
                np.abs(highs[-n:] - prev_closes[-n:]),
                np.abs(lows[-n:] - prev_closes[-n:]),
            ),
        )
        return float(tr.mean())
