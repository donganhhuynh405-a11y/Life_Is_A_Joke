"""Caching strategies: LRU eviction and TTL expiry.

Pure stdlib implementation using ``collections.OrderedDict`` and ``deque``.
These strategies can be used independently of any cache backend.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class CacheStrategy(ABC, Generic[K, V]):
    """Abstract base class for cache eviction/expiry strategies."""

    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """Return the value for *key*, or ``None`` when absent/expired."""

    @abstractmethod
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Store *value* under *key*, optionally with a TTL in seconds."""

    @abstractmethod
    def delete(self, key: K) -> bool:
        """Remove *key*; return ``True`` when it existed."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of entries currently in the cache."""


class LRUStrategy(CacheStrategy[K, V]):
    """Least-Recently-Used eviction strategy.

    Evicts the least-recently accessed entry when the cache reaches
    *max_size*.  Supports per-entry TTL.

    Args:
        max_size: Maximum number of entries before eviction occurs.
    """

    def __init__(self, max_size: int = 1024) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        # OrderedDict: key → (value, expiry_timestamp_or_None)
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def _is_expired(self, expiry: Optional[float]) -> bool:
        return expiry is not None and time.monotonic() > expiry

    def get(self, key: K) -> Optional[V]:
        """Return the value for *key*, moving it to the end (most-recently used).

        Returns ``None`` when the key is absent or has expired.
        """
        with self._lock:
            if key not in self._store:
                return None
            value, expiry = self._store[key]
            if self._is_expired(expiry):
                del self._store[key]
                return None
            # Move to end to mark as recently used
            self._store.move_to_end(key)
            return value

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Store *value* under *key*.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Time-to-live in seconds; ``None`` means no expiry.
        """
        expiry = time.monotonic() + ttl if ttl is not None else None
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expiry)
            # Evict oldest entry when over capacity
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def delete(self, key: K) -> bool:
        """Remove *key*; return ``True`` when it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    def evict_expired(self) -> int:
        """Remove all expired entries; return the count removed."""
        now = time.monotonic()
        with self._lock:
            expired_keys = [
                k for k, (_, exp) in self._store.items()
                if exp is not None and now > exp
            ]
            for k in expired_keys:
                del self._store[k]
        return len(expired_keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def info(self) -> dict:
        """Return a summary of cache statistics."""
        with self._lock:
            now = time.monotonic()
            total = len(self._store)
            expired = sum(
                1 for _, exp in self._store.values()
                if exp is not None and now > exp
            )
        return {
            "strategy": "LRU",
            "max_size": self.max_size,
            "total_entries": total,
            "expired_entries": expired,
            "live_entries": total - expired,
        }


class TTLStrategy(CacheStrategy[K, V]):
    """Pure TTL-based strategy: entries expire after a fixed duration.

    Unlike :class:`LRUStrategy`, no eviction based on access order is
    performed.  Expired entries are lazily removed on access.

    Args:
        default_ttl: Default TTL in seconds (default ``60``).
        max_size: Maximum number of live entries; oldest inserted entry is
                  evicted when the limit is reached (FIFO).
    """

    def __init__(self, default_ttl: float = 60.0, max_size: int = 10_000) -> None:
        if default_ttl <= 0 or max_size <= 0:
            raise ValueError("default_ttl and max_size must be positive")
        self.default_ttl = default_ttl
        self.max_size = max_size
        # Insertion-ordered store: key → (value, expiry)
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> Optional[V]:
        """Return the value for *key*, or ``None`` when absent or expired."""
        with self._lock:
            if key not in self._store:
                return None
            value, expiry = self._store[key]
            if time.monotonic() > expiry:
                del self._store[key]
                return None
            return value

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Store *value* under *key* with optional TTL override."""
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        with self._lock:
            self._store[key] = (value, expiry)
            # FIFO eviction when over capacity
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def delete(self, key: K) -> bool:
        """Remove *key*; return ``True`` when it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    def evict_expired(self) -> int:
        """Proactively remove all expired entries; return the count removed."""
        now = time.monotonic()
        with self._lock:
            expired_keys = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired_keys:
                del self._store[k]
        return len(expired_keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def info(self) -> dict:
        """Return a summary of cache state."""
        with self._lock:
            now = time.monotonic()
            total = len(self._store)
            expired = sum(1 for _, exp in self._store.values() if now > exp)
        return {
            "strategy": "TTL",
            "default_ttl": self.default_ttl,
            "max_size": self.max_size,
            "total_entries": total,
            "expired_entries": expired,
            "live_entries": total - expired,
        }


class MultiLevelStrategy(CacheStrategy[K, V]):
    """Two-level cache: fast L1 (LRU) backed by a larger L2 (TTL).

    A cache miss on L1 promotes the entry from L2 to L1.

    Args:
        l1_size: Maximum entries in the L1 (LRU) cache.
        l2_size: Maximum entries in the L2 (TTL) cache.
        l1_ttl: TTL for L1 entries in seconds.
        l2_ttl: TTL for L2 entries in seconds.
    """

    def __init__(
        self,
        l1_size: int = 256,
        l2_size: int = 4096,
        l1_ttl: float = 30.0,
        l2_ttl: float = 300.0,
    ) -> None:
        self._l1: LRUStrategy = LRUStrategy(max_size=l1_size)
        self._l2: TTLStrategy = TTLStrategy(default_ttl=l2_ttl, max_size=l2_size)
        self._l1_ttl = l1_ttl

    def get(self, key: K) -> Optional[V]:
        """Return value from L1 (fast) or L2 (slower), promoting to L1 on miss."""
        value = self._l1.get(key)
        if value is not None:
            return value
        value = self._l2.get(key)
        if value is not None:
            # Promote to L1
            self._l1.set(key, value, ttl=self._l1_ttl)
        return value

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Store *value* in both L1 and L2."""
        self._l1.set(key, value, ttl=self._l1_ttl)
        self._l2.set(key, value, ttl=ttl)

    def delete(self, key: K) -> bool:
        """Remove *key* from both levels."""
        r1 = self._l1.delete(key)
        r2 = self._l2.delete(key)
        return r1 or r2

    def clear(self) -> None:
        """Clear both cache levels."""
        self._l1.clear()
        self._l2.clear()

    def __len__(self) -> int:
        return len(self._l2)

    def info(self) -> dict:
        """Return combined statistics for both cache levels."""
        return {
            "strategy": "MultiLevel",
            "l1": self._l1.info(),
            "l2": self._l2.info(),
        }
