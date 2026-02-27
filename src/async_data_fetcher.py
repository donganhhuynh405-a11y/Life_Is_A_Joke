"""Fully async data fetching wrapper with caching and retry logic."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Container for a single fetch result."""

    key: str
    data: Any
    source: str             # e.g. "cache" or "remote"
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    error: str = ""


class AsyncDataFetcher:
    """
    Fully async data fetching wrapper with in-memory caching and retry.

    Provides a unified interface for fetching data from multiple async
    sources concurrently, with configurable TTL-based caching, exponential
    back-off retries, and request rate limiting.

    Parameters
    ----------
    ttl_seconds : float
        Cache time-to-live.  Data older than this is re-fetched.
    max_retries : int
        Maximum retry attempts on transient failures.
    base_backoff : float
        Initial back-off delay in seconds (doubles on each retry).
    rate_limit_rps : float
        Maximum requests per second across all fetch calls.
    timeout_seconds : float
        Per-request timeout.
    """

    def __init__(
        self,
        ttl_seconds: float = 60.0,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        rate_limit_rps: float = 10.0,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.timeout_seconds = timeout_seconds
        self._cache: Dict[str, Dict] = {}
        self._semaphore = asyncio.Semaphore(int(rate_limit_rps))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        force_refresh: bool = False,
    ) -> FetchResult:
        """
        Fetch a single item, using the cache when available.

        Parameters
        ----------
        key : str
            Cache key identifying the data item.
        fetch_fn : callable
            Async (or sync) callable that fetches the data.
        force_refresh : bool
            If True, bypass the cache.

        Returns
        -------
        FetchResult
        """
        if not force_refresh:
            cached = self._get_cached(key)
            if cached is not None:
                return FetchResult(key=key, data=cached, source="cache", latency_ms=0.0)

        return await self._fetch_with_retry(key, fetch_fn)

    async def fetch_many(
        self,
        requests: Dict[str, Callable[[], Any]],
        force_refresh: bool = False,
    ) -> Dict[str, FetchResult]:
        """
        Fetch multiple items concurrently.

        Parameters
        ----------
        requests : dict
            Mapping of cache_key → fetch_callable.
        force_refresh : bool
            Whether to bypass cache for all items.

        Returns
        -------
        dict mapping key → FetchResult
        """
        tasks = {
            key: self.fetch(key, fn, force_refresh)
            for key, fn in requests.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        output: Dict[str, FetchResult] = {}
        for key, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                output[key] = FetchResult(key=key, data=None, source="error",
                                          latency_ms=0.0, error=str(result))
            else:
                output[key] = result
        return output

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        self._cache.pop(key, None)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    def cache_stats(self) -> Dict[str, int]:
        """Return cache size and number of expired entries."""
        now = time.time()
        total = len(self._cache)
        expired = sum(1 for v in self._cache.values() if now - v["ts"] > self.ttl_seconds)
        return {"total": total, "expired": expired, "valid": total - expired}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_cached(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry["ts"]) < self.ttl_seconds:
            return entry["data"]
        return None

    async def _fetch_with_retry(
        self, key: str, fetch_fn: Callable[[], Any]
    ) -> FetchResult:
        backoff = self.base_backoff
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            async with self._semaphore:
                t0 = time.perf_counter()
                try:
                    if asyncio.iscoroutinefunction(fetch_fn):
                        data = await asyncio.wait_for(fetch_fn(), timeout=self.timeout_seconds)
                    else:
                        data = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(None, fetch_fn),
                            timeout=self.timeout_seconds,
                        )
                    latency_ms = (time.perf_counter() - t0) * 1000
                    self._cache[key] = {"data": data, "ts": time.time()}
                    return FetchResult(key=key, data=data, source="remote", latency_ms=latency_ms)
                except asyncio.TimeoutError as exc:
                    last_exc = exc
                    logger.warning("Fetch timeout for key '%s' (attempt %d)", key, attempt + 1)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "Fetch error for key '%s': %s (attempt %d)",
                        key,
                        exc,
                        attempt + 1)

            if attempt < self.max_retries:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

        return FetchResult(
            key=key, data=None, source="error", latency_ms=0.0,
            error=str(last_exc or "Unknown error"),
        )
