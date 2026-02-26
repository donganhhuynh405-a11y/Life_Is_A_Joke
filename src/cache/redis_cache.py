"""Redis caching layer with lazy redis import.

Provides a simple key-value cache backed by Redis with TTL support,
serialisation via JSON, and a transparent fallback to an in-memory dict
when Redis is unavailable.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_redis_available = False


def _get_redis():  # type: ignore[return]
    """Lazily import redis-py."""
    global _redis_available
    try:
        import redis  # type: ignore[import]

        _redis_available = True
        return redis
    except ImportError as exc:
        raise ImportError(
            "redis package is required. Install with: pip install redis"
        ) from exc


class _InMemoryFallback:
    """Simple TTL-aware in-memory store used when Redis is unavailable."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}

    def get(self, key: str) -> Optional[str]:
        exp = self._expiry.get(key)
        if exp is not None and time.monotonic() > exp:
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return None
        return self._store.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self._store[key] = value
        if ex is not None:
            self._expiry[key] = time.monotonic() + ex

    def delete(self, key: str) -> int:
        existed = key in self._store
        self._store.pop(key, None)
        self._expiry.pop(key, None)
        return int(existed)

    def exists(self, key: str) -> int:
        return int(self.get(key) is not None)

    def flushdb(self) -> None:
        self._store.clear()
        self._expiry.clear()

    def ping(self) -> bool:
        return True


class RedisCache:
    """Redis-backed cache with JSON serialisation and in-memory fallback.

    Args:
        host: Redis host (default ``localhost``).
        port: Redis port (default ``6379``).
        db: Redis database index (default ``0``).
        password: Optional Redis password.
        default_ttl: Default key TTL in seconds (default ``300``).
        key_prefix: Prefix prepended to all cache keys.
        use_fallback: When ``True`` (default), fall back to in-memory storage
                      if Redis is unreachable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 300,
        key_prefix: str = "trading_bot:",
        use_fallback: bool = True,
    ) -> None:
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._client: Any = None
        self._using_fallback = False

        try:
            redis = _get_redis()
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
            logger.info("Connected to Redis at %s:%d db=%d", host, port, db)
        except Exception as exc:
            if use_fallback:
                logger.warning(
                    "Redis unavailable (%s); using in-memory fallback cache", exc
                )
                self._client = _InMemoryFallback()
                self._using_fallback = True
            else:
                raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _full_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    @staticmethod
    def _serialise(value: Any) -> str:
        return json.dumps(value, default=str)

    @staticmethod
    def _deserialise(raw: str) -> Any:
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Retrieve the value for *key*, or ``None`` when absent.

        Args:
            key: Cache key (prefix is applied automatically).

        Returns:
            Deserialised value, or ``None``.
        """
        try:
            raw = self._client.get(self._full_key(key))
            if raw is None:
                return None
            return self._deserialise(raw)
        except Exception as exc:
            logger.warning("Cache GET error for key '%s': %s", key, exc)
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store *value* under *key* with an optional TTL.

        Args:
            key: Cache key.
            value: JSON-serialisable value to store.
            ttl: Time-to-live in seconds (uses :attr:`default_ttl` when omitted).

        Returns:
            ``True`` on success; ``False`` on error.
        """
        try:
            self._client.set(
                self._full_key(key),
                self._serialise(value),
                ex=ttl if ttl is not None else self.default_ttl,
            )
            return True
        except Exception as exc:
            logger.warning("Cache SET error for key '%s': %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        """Delete *key* from the cache.

        Args:
            key: Cache key to remove.

        Returns:
            ``True`` when the key existed; ``False`` otherwise.
        """
        try:
            return bool(self._client.delete(self._full_key(key)))
        except Exception as exc:
            logger.warning("Cache DELETE error for key '%s': %s", key, exc)
            return False

    def exists(self, key: str) -> bool:
        """Return ``True`` when *key* exists in the cache."""
        try:
            return bool(self._client.exists(self._full_key(key)))
        except Exception:
            return False

    def get_or_set(
        self,
        key: str,
        default_factory: Any,
        ttl: Optional[int] = None,
    ) -> Any:
        """Return the cached value for *key*, computing and caching it if absent.

        Args:
            key: Cache key.
            default_factory: Zero-argument callable that produces the value.
            ttl: TTL for the computed value.

        Returns:
            Cached or freshly computed value.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        value = default_factory()
        self.set(key, value, ttl=ttl)
        return value

    def clear(self) -> None:
        """Flush all keys with the configured prefix from the cache."""
        try:
            if self._using_fallback:
                self._client.flushdb()
            else:
                # Scan-based prefix deletion to avoid FLUSHDB on shared Redis
                cursor: int = 0
                pattern = f"{self.key_prefix}*"
                while True:
                    cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                    if keys:
                        self._client.delete(*keys)
                    if cursor == 0:
                        break
            logger.info("Cache cleared (prefix='%s')", self.key_prefix)
        except Exception as exc:
            logger.warning("Cache CLEAR error: %s", exc)

    @property
    def is_connected(self) -> bool:
        """Return ``True`` when the cache backend is reachable."""
        try:
            return bool(self._client.ping())
        except Exception:
            return False
