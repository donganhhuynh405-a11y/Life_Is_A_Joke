"""API rate limiting using token bucket and sliding window algorithms.

Pure stdlib implementation using ``collections.deque`` – no external
dependencies required.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple


class TokenBucketRateLimiter:
    """Token-bucket rate limiter.

    Tokens are replenished continuously at *rate* tokens per second up to a
    maximum of *capacity* tokens.  Each call to :meth:`acquire` consumes one
    token.  If no token is available the call blocks (if ``block=True``) or
    returns ``False`` immediately.

    Args:
        capacity: Maximum number of tokens in the bucket.
        rate: Token replenishment rate (tokens per second).
    """

    def __init__(self, capacity: float, rate: float) -> None:
        if capacity <= 0 or rate <= 0:
            raise ValueError("capacity and rate must be positive")
        self.capacity = float(capacity)
        self.rate = float(rate)
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Replenish tokens based on elapsed time (call with lock held)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * self.rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_refill = now

    def acquire(self, tokens: float = 1.0, block: bool = True, timeout: float = -1.0) -> bool:
        """Attempt to acquire *tokens* tokens.

        Args:
            tokens: Number of tokens to consume (default 1).
            block: When ``True``, wait until tokens are available or *timeout*
                   expires.
            timeout: Maximum seconds to block (``-1`` means wait forever).

        Returns:
            ``True`` if tokens were acquired; ``False`` otherwise.
        """
        deadline = time.monotonic() + timeout if (block and timeout >= 0) else None

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

            if not block:
                return False

            if deadline is not None and time.monotonic() >= deadline:
                return False

            # Sleep for ~half the time needed to accumulate the missing tokens
            with self._lock:
                missing = tokens - self._tokens
            sleep_time = (missing / self.rate) * 0.5
            time.sleep(max(0.001, sleep_time))

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens (approximate)."""
        with self._lock:
            self._refill()
            return self._tokens


class SlidingWindowRateLimiter:
    """Sliding-window rate limiter backed by a ``deque``.

    Maintains a timestamped record of recent requests within a rolling time
    window.  A request is allowed only when the count of requests within the
    last *window_seconds* is below *max_requests*.

    Args:
        max_requests: Maximum number of allowed requests per window.
        window_seconds: Duration of the sliding window in seconds.
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        if max_requests <= 0 or window_seconds <= 0:
            raise ValueError("max_requests and window_seconds must be positive")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def _evict_old(self, now: float) -> None:
        """Remove timestamps older than the current window (call with lock held)."""
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def is_allowed(self) -> bool:
        """Return ``True`` and record the request if within rate limits.

        Returns:
            ``True`` when the request is permitted; ``False`` when rate-limited.
        """
        now = time.monotonic()
        with self._lock:
            self._evict_old(now)
            if len(self._timestamps) < self.max_requests:
                self._timestamps.append(now)
                return True
            return False

    def remaining_requests(self) -> int:
        """Return the number of requests still permitted in the current window."""
        now = time.monotonic()
        with self._lock:
            self._evict_old(now)
            return max(0, self.max_requests - len(self._timestamps))

    def retry_after(self) -> float:
        """Return the seconds until at least one slot becomes available."""
        now = time.monotonic()
        with self._lock:
            self._evict_old(now)
            if len(self._timestamps) < self.max_requests:
                return 0.0
            oldest = self._timestamps[0]
        return max(0.0, (oldest + self.window_seconds) - now)


class RateLimiter:
    """Composite rate limiter that manages per-key token-bucket limiters.

    Useful for enforcing per-exchange or per-endpoint API rate limits.

    Args:
        default_capacity: Default token bucket capacity for new keys.
        default_rate: Default replenishment rate (tokens/s) for new keys.
    """

    def __init__(self, default_capacity: float = 10.0, default_rate: float = 1.0) -> None:
        self.default_capacity = default_capacity
        self.default_rate = default_rate
        self._limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, key: str) -> TokenBucketRateLimiter:
        with self._lock:
            if key not in self._limiters:
                self._limiters[key] = TokenBucketRateLimiter(
                    capacity=self.default_capacity,
                    rate=self.default_rate,
                )
            return self._limiters[key]

    def register(self, key: str, capacity: float, rate: float) -> None:
        """Register a custom token-bucket limiter for *key*.

        Args:
            key: Identifier (e.g. ``"bybit_order"``).
            capacity: Bucket capacity.
            rate: Replenishment rate (tokens/s).
        """
        with self._lock:
            self._limiters[key] = TokenBucketRateLimiter(capacity=capacity, rate=rate)

    def acquire(self, key: str, block: bool = False, timeout: float = 1.0) -> bool:
        """Acquire one token for *key*.

        Args:
            key: Rate-limit key.
            block: Whether to block until a token is available.
            timeout: Maximum wait time when blocking.

        Returns:
            ``True`` if a token was acquired; ``False`` otherwise.
        """
        limiter = self._get_or_create(key)
        return limiter.acquire(block=block, timeout=timeout)

    def available(self, key: str) -> float:
        """Return available tokens for *key* (approximate)."""
        return self._get_or_create(key).available_tokens

    def status(self) -> Dict[str, float]:
        """Return a snapshot of available tokens for all registered keys."""
        with self._lock:
            keys = list(self._limiters.keys())
        return {k: self._limiters[k].available_tokens for k in keys}
