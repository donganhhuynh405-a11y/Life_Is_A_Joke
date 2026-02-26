"""OpenTelemetry distributed tracing with lazy imports.

Provides span creation, context propagation, and trace export without
requiring opentelemetry packages to be installed.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded OpenTelemetry components
_tracer_provider = None
_tracer = None
_otel_available = False


def _init_otel() -> bool:
    """Attempt to initialise OpenTelemetry; returns True on success."""
    global _tracer_provider, _tracer, _otel_available
    if _otel_available:
        return True
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({"service.name": "trading-bot"})
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
        trace.set_tracer_provider(_tracer_provider)
        _tracer = trace.get_tracer("trading-bot")
        _otel_available = True
        logger.info("OpenTelemetry tracing initialised")
        return True
    except ImportError:
        logger.debug("opentelemetry-sdk not installed; tracing disabled")
        return False
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to initialise OpenTelemetry: %s", exc)
        return False


class _NoOpSpan:
    """Fallback span used when OpenTelemetry is not available."""

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        pass

    def record_exception(self, exc: Exception) -> None:  # noqa: D401
        pass

    def set_status(self, status: Any) -> None:  # noqa: D401
        pass


class TracingManager:
    """Manages distributed tracing for the trading bot."""

    def __init__(self, service_name: str = "trading-bot") -> None:
        self.service_name = service_name
        self._available = _init_otel()

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """Context manager that creates a tracing span.

        Falls back to a no-op span when OpenTelemetry is unavailable.

        Args:
            name: Span name.
            attributes: Optional key/value attributes to attach.

        Yields:
            An OpenTelemetry Span or a :class:`_NoOpSpan`.
        """
        if not self._available or _tracer is None:
            yield _NoOpSpan()
            return

        try:
            from opentelemetry import trace

            with _tracer.start_as_current_span(name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                try:
                    yield span
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, str(exc))
                    )
                    raise
        except ImportError:
            yield _NoOpSpan()

    def trace(
        self,
        span_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Decorator that wraps a function in a tracing span.

        Args:
            span_name: Override span name (defaults to function name).
            attributes: Static attributes to attach to the span.

        Returns:
            Decorated function.
        """

        def decorator(func: Callable) -> Callable:
            name = span_name or func.__qualname__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.start_span(name, attributes):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def configure_otlp_exporter(self, endpoint: str) -> None:
        """Configure an OTLP gRPC exporter at *endpoint*.

        This replaces the default console exporter.  Requires
        ``opentelemetry-exporter-otlp-proto-grpc`` to be installed.

        Args:
            endpoint: OTLP receiver endpoint, e.g. ``http://localhost:4317``.
        """
        if not self._available or _tracer_provider is None:
            logger.warning("OTel not available; cannot configure OTLP exporter")
            return
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = OTLPSpanExporter(endpoint=endpoint)
            _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("OTLP exporter configured at %s", endpoint)
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp-proto-grpc not installed"
            )


# Module-level singleton
_default_manager: Optional[TracingManager] = None


def get_tracing_manager(service_name: str = "trading-bot") -> TracingManager:
    """Return (or create) the module-level :class:`TracingManager` singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = TracingManager(service_name=service_name)
    return _default_manager
