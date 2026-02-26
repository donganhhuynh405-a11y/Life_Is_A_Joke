"""Triangular arbitrage strategy using pure Python and NumPy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrianglePath:
    """Represents a triangular arbitrage path through three currency pairs."""

    symbols: List[str]          # e.g. ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
    directions: List[str]       # "buy" or "sell" for each leg
    implied_rate: float = 0.0   # product of conversion rates
    profit_pct: float = 0.0     # estimated profit percentage after fees


@dataclass
class ArbitrageResult:
    """Result of a triangular arbitrage scan."""

    path: TrianglePath
    entry_capital: float
    estimated_profit: float
    confidence: float           # 0–1 score based on liquidity/spread
    timestamp: float = 0.0


class TriangularArbitrage:
    """
    Detects and sizes triangular arbitrage opportunities within a single exchange.

    A triangular path converts base_currency → A → B → base_currency and
    profits when the round-trip rate exceeds 1 + total_fee_factor.

    Parameters
    ----------
    fee_rate : float
        Maker/taker fee per leg (e.g. 0.001 for 0.1 %).
    min_profit_pct : float
        Minimum net profit percentage to consider an opportunity valid.
    max_slippage_pct : float
        Maximum allowed slippage per leg before discarding the path.
    """

    def __init__(
        self,
        fee_rate: float = 0.001,
        min_profit_pct: float = 0.05,
        max_slippage_pct: float = 0.05,
    ) -> None:
        self.fee_rate = fee_rate
        self.min_profit_pct = min_profit_pct
        self.max_slippage_pct = max_slippage_pct
        # fee_multiplier across all three legs
        self._fee_factor = (1 - fee_rate) ** 3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_opportunities(
        self,
        order_books: Dict[str, Dict],
        base_currency: str = "USDT",
    ) -> List[ArbitrageResult]:
        """
        Scan order books for profitable triangular paths.

        Parameters
        ----------
        order_books : dict
            Mapping of symbol → {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}.
        base_currency : str
            The currency used to start and end each triangle.

        Returns
        -------
        list of ArbitrageResult
            Sorted by estimated_profit descending.
        """
        paths = self._enumerate_paths(list(order_books.keys()), base_currency)
        results: List[ArbitrageResult] = []

        for path in paths:
            result = self._evaluate_path(path, order_books, capital=1.0)
            if result is not None:
                results.append(result)

        results.sort(key=lambda r: r.estimated_profit, reverse=True)
        return results

    def calculate_optimal_size(
        self,
        result: ArbitrageResult,
        available_capital: float,
        max_position_pct: float = 0.10,
    ) -> float:
        """
        Calculate optimal trade size using a simple Kelly-fraction heuristic.

        Parameters
        ----------
        result : ArbitrageResult
            Opportunity to size.
        available_capital : float
            Total capital available.
        max_position_pct : float
            Hard cap as fraction of available capital.

        Returns
        -------
        float
            Recommended capital to deploy.
        """
        kelly = result.confidence * (result.path.profit_pct / 100)
        fraction = min(kelly, max_position_pct)
        return available_capital * fraction

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enumerate_paths(
        self, symbols: List[str], base: str
    ) -> List[TrianglePath]:
        """Build all valid triangular paths starting and ending in *base*."""
        paths: List[TrianglePath] = []
        pairs = self._parse_pairs(symbols)

        for sym_a, (base_a, quote_a) in pairs.items():
            if base_a != base and quote_a != base:
                continue
            # First leg: buy or sell to leave *base*
            if quote_a == base:
                mid_currency = base_a
                dir_a = "buy"
            else:
                mid_currency = quote_a
                dir_a = "sell"

            for sym_b, (base_b, quote_b) in pairs.items():
                if sym_b == sym_a:
                    continue
                if base_b == mid_currency:
                    next_currency = quote_b
                    dir_b = "sell"
                elif quote_b == mid_currency:
                    next_currency = base_b
                    dir_b = "buy"
                else:
                    continue

                # Third leg back to *base*
                for sym_c, (base_c, quote_c) in pairs.items():
                    if sym_c in (sym_a, sym_b):
                        continue
                    if base_c == next_currency and quote_c == base:
                        dir_c = "sell"
                    elif quote_c == next_currency and base_c == base:
                        dir_c = "buy"
                    else:
                        continue
                    paths.append(
                        TrianglePath(
                            symbols=[sym_a, sym_b, sym_c],
                            directions=[dir_a, dir_b, dir_c],
                        )
                    )
        return paths

    def _parse_pairs(self, symbols: List[str]) -> Dict[str, Tuple[str, str]]:
        """Parse symbol strings into (base, quote) tuples."""
        result: Dict[str, Tuple[str, str]] = {}
        for sym in symbols:
            if "/" in sym:
                base, quote = sym.split("/", 1)
                result[sym] = (base, quote)
        return result

    def _best_price(
        self, book_side: List[List[float]], direction: str, qty: float = 1.0
    ) -> Optional[float]:
        """Return volume-weighted average fill price from an order book side."""
        if not book_side:
            return None
        levels = np.array(book_side, dtype=float)
        prices, sizes = levels[:, 0], levels[:, 1]

        filled, cost = 0.0, 0.0
        for price, size in zip(prices, sizes):
            take = min(size, qty - filled)
            cost += take * price
            filled += take
            if filled >= qty:
                break
        if filled == 0:
            return None
        return cost / filled

    def _evaluate_path(
        self,
        path: TrianglePath,
        order_books: Dict[str, Dict],
        capital: float,
    ) -> Optional[ArbitrageResult]:
        """Evaluate a single triangular path and return a result if profitable."""
        import time

        rate = capital
        confidences: List[float] = []

        for sym, direction in zip(path.symbols, path.directions):
            if sym not in order_books:
                return None
            book = order_books[sym]
            side = book.get("asks", []) if direction == "buy" else book.get("bids", [])
            price = self._best_price(side, direction, qty=rate)
            if price is None or price <= 0:
                return None

            # Apply spread confidence
            best_ask = book["asks"][0][0] if book.get("asks") else price
            best_bid = book["bids"][0][0] if book.get("bids") else price
            spread_pct = (best_ask - best_bid) / best_bid * 100 if best_bid else 0
            if spread_pct > self.max_slippage_pct:
                return None
            confidences.append(max(0.0, 1 - spread_pct / self.max_slippage_pct))

            if direction == "buy":
                rate = rate / price * (1 - self.fee_rate)
            else:
                rate = rate * price * (1 - self.fee_rate)

        profit_pct = (rate - capital) / capital * 100
        if profit_pct < self.min_profit_pct:
            return None

        path.implied_rate = rate / capital
        path.profit_pct = profit_pct

        return ArbitrageResult(
            path=path,
            entry_capital=capital,
            estimated_profit=rate - capital,
            confidence=float(np.mean(confidences)),
            timestamp=time.time(),
        )
