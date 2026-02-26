"""Advanced metrics collection using prometheus_client with lazy imports.

Falls back to no-op counters/gauges/histograms when prometheus_client is
not installed so the rest of the application continues to function.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

_prom_available = False
_registry: Any = None


def _init_prometheus() -> bool:
    """Attempt to import prometheus_client; returns True on success."""
    global _prom_available, _registry
    if _prom_available:
        return True
    try:
        import prometheus_client as prom  # noqa: F401

        _registry = prom.REGISTRY
        _prom_available = True
        logger.info("prometheus_client available; metrics enabled")
        return True
    except ImportError:
        logger.debug("prometheus_client not installed; metrics disabled")
        return False


# ---------------------------------------------------------------------------
# No-op metric stubs
# ---------------------------------------------------------------------------


class _NoOpMetric:
    """Stub used when prometheus_client is unavailable."""

    def labels(self, **_: Any) -> "_NoOpMetric":
        return self

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, value: float) -> None:
        pass

    @contextmanager
    def time(self) -> Generator[None, None, None]:
        yield


# ---------------------------------------------------------------------------
# Metric factory
# ---------------------------------------------------------------------------


def _make_counter(
    name: str,
    documentation: str,
    labelnames: Optional[List[str]] = None,
) -> Any:
    if not _init_prometheus():
        return _NoOpMetric()
    import prometheus_client as prom

    try:
        return prom.Counter(name, documentation, labelnames or [])
    except ValueError:
        # Metric already registered – retrieve it from the registry
        return prom.REGISTRY._names_to_collectors.get(name, _NoOpMetric())  # type: ignore[attr-defined]


def _make_gauge(
    name: str,
    documentation: str,
    labelnames: Optional[List[str]] = None,
) -> Any:
    if not _init_prometheus():
        return _NoOpMetric()
    import prometheus_client as prom

    try:
        return prom.Gauge(name, documentation, labelnames or [])
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name, _NoOpMetric())  # type: ignore[attr-defined]


def _make_histogram(
    name: str,
    documentation: str,
    labelnames: Optional[List[str]] = None,
    buckets: Optional[List[float]] = None,
) -> Any:
    if not _init_prometheus():
        return _NoOpMetric()
    import prometheus_client as prom

    kwargs: Dict[str, Any] = {}
    if buckets:
        kwargs["buckets"] = buckets
    try:
        return prom.Histogram(name, documentation, labelnames or [], **kwargs)
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name, _NoOpMetric())  # type: ignore[attr-defined]


def _make_summary(
    name: str,
    documentation: str,
    labelnames: Optional[List[str]] = None,
) -> Any:
    if not _init_prometheus():
        return _NoOpMetric()
    import prometheus_client as prom

    try:
        return prom.Summary(name, documentation, labelnames or [])
    except ValueError:
        return prom.REGISTRY._names_to_collectors.get(name, _NoOpMetric())  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Main collector class
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Centralised metrics registry for the trading bot.

    All metrics are lazily created on first access so that the class can be
    instantiated even when prometheus_client is not installed.
    """

    def __init__(self, namespace: str = "trading_bot") -> None:
        self.namespace = namespace
        self._metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Trading metrics
    # ------------------------------------------------------------------

    @property
    def orders_total(self) -> Any:
        """Counter: total number of orders placed."""
        if "orders_total" not in self._metrics:
            self._metrics["orders_total"] = _make_counter(
                f"{self.namespace}_orders_total",
                "Total number of orders placed",
                ["symbol", "side", "status"],
            )
        return self._metrics["orders_total"]

    @property
    def pnl_gauge(self) -> Any:
        """Gauge: current unrealised PnL in USD."""
        if "pnl_gauge" not in self._metrics:
            self._metrics["pnl_gauge"] = _make_gauge(
                f"{self.namespace}_pnl_usd",
                "Current unrealised PnL in USD",
                ["symbol"],
            )
        return self._metrics["pnl_gauge"]

    @property
    def portfolio_value(self) -> Any:
        """Gauge: total portfolio value in USD."""
        if "portfolio_value" not in self._metrics:
            self._metrics["portfolio_value"] = _make_gauge(
                f"{self.namespace}_portfolio_value_usd",
                "Total portfolio value in USD",
            )
        return self._metrics["portfolio_value"]

    @property
    def trade_latency(self) -> Any:
        """Histogram: order execution latency in seconds."""
        if "trade_latency" not in self._metrics:
            self._metrics["trade_latency"] = _make_histogram(
                f"{self.namespace}_trade_latency_seconds",
                "Order execution latency in seconds",
                ["exchange"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            )
        return self._metrics["trade_latency"]

    # ------------------------------------------------------------------
    # ML model metrics
    # ------------------------------------------------------------------

    @property
    def model_predictions_total(self) -> Any:
        """Counter: total ML model predictions."""
        if "model_predictions_total" not in self._metrics:
            self._metrics["model_predictions_total"] = _make_counter(
                f"{self.namespace}_model_predictions_total",
                "Total number of ML model predictions",
                ["model_name", "symbol"],
            )
        return self._metrics["model_predictions_total"]

    @property
    def model_inference_latency(self) -> Any:
        """Histogram: ML model inference latency in seconds."""
        if "model_inference_latency" not in self._metrics:
            self._metrics["model_inference_latency"] = _make_histogram(
                f"{self.namespace}_model_inference_latency_seconds",
                "ML model inference latency in seconds",
                ["model_name"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            )
        return self._metrics["model_inference_latency"]

    @property
    def model_accuracy(self) -> Any:
        """Gauge: rolling model prediction accuracy."""
        if "model_accuracy" not in self._metrics:
            self._metrics["model_accuracy"] = _make_gauge(
                f"{self.namespace}_model_accuracy",
                "Rolling model prediction accuracy (0-1)",
                ["model_name", "symbol"],
            )
        return self._metrics["model_accuracy"]

    # ------------------------------------------------------------------
    # System metrics
    # ------------------------------------------------------------------

    @property
    def api_requests_total(self) -> Any:
        """Counter: total API requests made."""
        if "api_requests_total" not in self._metrics:
            self._metrics["api_requests_total"] = _make_counter(
                f"{self.namespace}_api_requests_total",
                "Total number of API requests",
                ["exchange", "endpoint", "status"],
            )
        return self._metrics["api_requests_total"]

    @property
    def api_request_latency(self) -> Any:
        """Histogram: external API request latency in seconds."""
        if "api_request_latency" not in self._metrics:
            self._metrics["api_request_latency"] = _make_histogram(
                f"{self.namespace}_api_request_latency_seconds",
                "External API request latency in seconds",
                ["exchange", "endpoint"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            )
        return self._metrics["api_request_latency"]

    @property
    def cache_hits_total(self) -> Any:
        """Counter: cache hit/miss totals."""
        if "cache_hits_total" not in self._metrics:
            self._metrics["cache_hits_total"] = _make_counter(
                f"{self.namespace}_cache_hits_total",
                "Total cache hits and misses",
                ["cache_name", "result"],
            )
        return self._metrics["cache_hits_total"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @contextmanager
    def measure_latency(self, metric: Any, **label_kwargs: Any) -> Generator[None, None, None]:
        """Context manager that records execution time to a histogram metric.

        Args:
            metric: A histogram metric (real or no-op).
            **label_kwargs: Labels to apply before observing.

        Yields:
            Nothing – just times the block.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            try:
                if label_kwargs:
                    metric.labels(**label_kwargs).observe(elapsed)
                else:
                    metric.observe(elapsed)
            except Exception:  # pragma: no cover
                pass

    def start_http_server(self, port: int = 8000) -> None:
        """Start the Prometheus HTTP metrics exposition server on *port*.

        Does nothing when prometheus_client is not available.

        Args:
            port: TCP port to listen on.
        """
        if not _init_prometheus():
            logger.warning("prometheus_client unavailable; metrics server not started")
            return
        try:
            import prometheus_client as prom

            prom.start_http_server(port)
            logger.info("Prometheus metrics server started on port %d", port)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to start metrics server: %s", exc)
