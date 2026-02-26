"""Fast arbitrage execution engine."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionLeg:
    """Single order leg within an arbitrage execution."""

    symbol: str
    side: str           # "buy" or "sell"
    amount: float
    price: Optional[float] = None   # None → market order
    exchange: str = ""
    order_id: str = ""
    status: str = "pending"         # pending | filled | partial | failed
    filled_price: float = 0.0
    filled_qty: float = 0.0
    latency_ms: float = 0.0


@dataclass
class ExecutionResult:
    """Aggregated result of a multi-leg arbitrage execution."""

    legs: List[ExecutionLeg] = field(default_factory=list)
    success: bool = False
    realised_pnl: float = 0.0
    total_fees: float = 0.0
    total_latency_ms: float = 0.0
    error: str = ""


class ArbitrageExecutor:
    """
    Executes arbitrage legs as fast as possible using async order placement.

    Orders are submitted concurrently via an injected async order-placement
    callable.  A circuit breaker halts execution if too many legs fail in a
    rolling window.

    Parameters
    ----------
    order_fn : callable, optional
        Async function with signature
        ``async (exchange, symbol, side, amount, price) -> dict``.
        Must return a dict with keys ``order_id``, ``filled_price``,
        ``filled_qty``, ``fee``.
    max_retries : int
        Number of retries per leg on transient failures.
    circuit_breaker_threshold : int
        Consecutive failures before execution is halted.
    timeout_seconds : float
        Per-leg timeout.
    """

    def __init__(
        self,
        order_fn: Optional[Callable] = None,
        max_retries: int = 1,
        circuit_breaker_threshold: int = 3,
        timeout_seconds: float = 2.0,
    ) -> None:
        self.order_fn = order_fn or self._noop_order
        self.max_retries = max_retries
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.timeout_seconds = timeout_seconds
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, legs: List[ExecutionLeg]) -> ExecutionResult:
        """
        Execute all legs concurrently and collect results.

        Parameters
        ----------
        legs : list of ExecutionLeg
            Legs to execute (all submitted simultaneously).

        Returns
        -------
        ExecutionResult
            Aggregated outcome.
        """
        if self._consecutive_failures >= self.circuit_breaker_threshold:
            logger.warning("Circuit breaker open – execution halted.")
            return ExecutionResult(legs=legs, error="circuit_breaker_open")

        start = time.perf_counter()
        tasks = [self._execute_leg(leg) for leg in legs]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        result = ExecutionResult(legs=legs)
        total_fees = 0.0
        all_ok = True

        for leg, outcome in zip(legs, completed):
            if isinstance(outcome, Exception):
                leg.status = "failed"
                all_ok = False
                result.error = str(outcome)
                self._consecutive_failures += 1
            else:
                leg.status = outcome.get("status", "filled")
                leg.order_id = outcome.get("order_id", "")
                leg.filled_price = float(outcome.get("filled_price", 0))
                leg.filled_qty = float(outcome.get("filled_qty", 0))
                leg.latency_ms = float(outcome.get("latency_ms", 0))
                total_fees += float(outcome.get("fee", 0))
                if leg.status not in ("filled", "partial"):
                    all_ok = False
                    self._consecutive_failures += 1
                else:
                    self._consecutive_failures = 0

        result.success = all_ok
        result.total_fees = total_fees
        result.total_latency_ms = (time.perf_counter() - start) * 1000
        result.realised_pnl = self._calculate_pnl(legs) - total_fees
        return result

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker after investigating failures."""
        self._consecutive_failures = 0
        logger.info("Circuit breaker reset.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _execute_leg(self, leg: ExecutionLeg) -> Dict[str, Any]:
        """Submit a single leg with retry logic."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                result = await asyncio.wait_for(
                    self.order_fn(
                        leg.exchange, leg.symbol, leg.side, leg.amount, leg.price
                    ),
                    timeout=self.timeout_seconds,
                )
                result["latency_ms"] = (time.perf_counter() - t0) * 1000
                return result
            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.warning("Leg %s timed out (attempt %d)", leg.symbol, attempt + 1)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("Leg %s error: %s (attempt %d)", leg.symbol, exc, attempt + 1)
        raise last_exc or RuntimeError("Unknown execution error")

    @staticmethod
    async def _noop_order(
        exchange: str,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float],
    ) -> Dict[str, Any]:
        """Stub order function – replace with real exchange adapter."""
        await asyncio.sleep(0)
        return {
            "order_id": f"stub_{symbol}_{side}",
            "status": "filled",
            "filled_price": price or 0.0,
            "filled_qty": amount,
            "fee": amount * (price or 1.0) * 0.001,
        }

    @staticmethod
    def _calculate_pnl(legs: List[ExecutionLeg]) -> float:
        """Approximate PnL from filled legs (positive = profit in quote currency)."""
        pnl = 0.0
        for leg in legs:
            value = leg.filled_qty * leg.filled_price
            pnl += value if leg.side == "sell" else -value
        return pnl
