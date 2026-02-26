"""Structured logging using Python's built-in logging module.

Provides JSON-formatted, context-aware loggers with no external dependencies.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    _RESERVED = frozenset(
        {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.threadName,
            "process": record.process,
        }

        # Include any extra fields attached to the record
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)  # serialisability check
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str)


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that injects a persistent context dict into every record."""

    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(logger, extra=context or {})
        self._context: Dict[str, Any] = dict(context or {})
        self._lock = threading.Lock()

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple:
        extra = dict(self._context)
        extra.update(kwargs.get("extra", {}))
        kwargs["extra"] = extra
        return msg, kwargs

    def bind(self, **fields: Any) -> "ContextLogger":
        """Return a *new* :class:`ContextLogger` with additional context fields."""
        new_context = dict(self._context)
        new_context.update(fields)
        return ContextLogger(self.logger, new_context)

    def unbind(self, *keys: str) -> "ContextLogger":
        """Return a new logger with the specified context keys removed."""
        new_context = {k: v for k, v in self._context.items() if k not in keys}
        return ContextLogger(self.logger, new_context)


def _build_handler(
    handler_type: str,
    log_file: Optional[str] = None,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Handler:
    """Build and return a log handler of the requested *handler_type*."""
    if handler_type == "file" and log_file:
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(_JsonFormatter())
    return handler


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_output: bool = True,
) -> None:
    """Configure the root logger.

    Args:
        level: Logging level string (``DEBUG``, ``INFO``, etc.).
        log_file: Optional path to a rotating log file.
        json_output: When *True* (default), emit JSON-formatted records.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove pre-existing handlers to avoid duplicate output
    for existing in list(root.handlers):
        root.removeHandler(existing)

    stream_handler = _build_handler("stream")
    if not json_output:
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    root.addHandler(stream_handler)

    if log_file:
        root.addHandler(_build_handler("file", log_file=log_file))


def get_logger(
    name: str,
    context: Optional[Dict[str, Any]] = None,
) -> ContextLogger:
    """Return a :class:`ContextLogger` for *name* with optional context.

    Args:
        name: Logger name (usually ``__name__``).
        context: Optional dictionary of persistent context fields.

    Returns:
        A :class:`ContextLogger` instance.

    Example::

        logger = get_logger(__name__, {"service": "trading-bot"})
        logger.info("Order placed", extra={"symbol": "BTCUSDT"})
    """
    base = logging.getLogger(name)
    return ContextLogger(base, context)
