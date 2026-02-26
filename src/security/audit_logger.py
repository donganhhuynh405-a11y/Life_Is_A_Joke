"""Comprehensive audit logging using Python's stdlib only.

Emits structured, tamper-evident audit records for security-sensitive events
such as authentication attempts, order placement, and configuration changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class AuditEventType(str, Enum):
    """Enumeration of recognised audit event types."""

    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_FAILURE = "auth.failure"
    ORDER_PLACED = "order.placed"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_FILLED = "order.filled"
    CONFIG_CHANGED = "config.changed"
    SECRET_ACCESSED = "secret.accessed"
    RISK_BREACH = "risk.breach"
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    RATE_LIMITED = "security.rate_limited"
    PERMISSION_DENIED = "security.permission_denied"


class AuditLogger:
    """Thread-safe audit logger that writes JSON records with hash chaining.

    Each record includes a ``chain_hash`` that is the SHA-256 of the previous
    record's hash concatenated with the current record's content.  This makes
    it computationally hard to tamper with historical records without
    detection.

    Args:
        log_file: Path to the audit log file.  Defaults to
                  ``AUDIT_LOG_FILE`` env var or ``./audit.log``.
        app_name: Application identifier embedded in every record.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        app_name: str = "trading-bot",
    ) -> None:
        self.app_name = app_name
        _path = log_file or os.environ.get("AUDIT_LOG_FILE", "audit.log")
        self._log_path = Path(_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prev_hash = "0" * 64  # genesis hash
        self._logger = logging.getLogger(f"audit.{app_name}")

        # Also emit to a dedicated rotating file handler
        self._file_handler = logging.FileHandler(str(self._log_path), encoding="utf-8")
        self._file_handler.setFormatter(logging.Formatter("%(message)s"))
        audit_log = logging.getLogger(f"audit_file.{app_name}")
        audit_log.setLevel(logging.INFO)
        audit_log.propagate = False
        audit_log.addHandler(self._file_handler)
        self._audit_file_logger = audit_log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_record(
        self,
        event_type: AuditEventType,
        actor: str,
        details: Dict[str, Any],
        outcome: str = "success",
        severity: str = "INFO",
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "app": self.app_name,
            "event_type": event_type.value,
            "actor": actor,
            "outcome": outcome,
            "severity": severity,
            "details": details,
        }
        content = json.dumps(record, sort_keys=True, default=str)
        record["chain_hash"] = hashlib.sha256(
            (self._prev_hash + content).encode()
        ).hexdigest()
        return record

    def _emit(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        self._prev_hash = record["chain_hash"]
        self._audit_file_logger.info(line)
        # Mirror to application logger at the appropriate level
        level = logging.getLevelName(record.get("severity", "INFO"))
        self._logger.log(level if isinstance(level, int) else logging.INFO, line)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: AuditEventType,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        severity: str = "INFO",
    ) -> str:
        """Emit an audit record.

        Args:
            event_type: The type of event being recorded.
            actor: Identity of the actor (user ID, service name, IP, etc.).
            details: Optional dict of event-specific details.
            outcome: ``"success"`` or ``"failure"``.
            severity: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, or ``"CRITICAL"``.

        Returns:
            The unique record ID (UUID4 string).
        """
        with self._lock:
            record = self._build_record(
                event_type=event_type,
                actor=actor,
                details=details or {},
                outcome=outcome,
                severity=severity,
            )
            self._emit(record)
            return record["id"]

    # Convenience methods -------------------------------------------------

    def auth_login(self, actor: str, ip_address: str, success: bool = True) -> str:
        """Record an authentication event."""
        return self.log(
            AuditEventType.AUTH_LOGIN if success else AuditEventType.AUTH_FAILURE,
            actor=actor,
            details={"ip_address": ip_address},
            outcome="success" if success else "failure",
            severity="INFO" if success else "WARNING",
        )

    def order_placed(self, actor: str, order: Dict[str, Any]) -> str:
        """Record an order placement event."""
        return self.log(
            AuditEventType.ORDER_PLACED,
            actor=actor,
            details=order,
            severity="INFO",
        )

    def config_changed(self, actor: str, key: str, old_value: Any, new_value: Any) -> str:
        """Record a configuration change."""
        return self.log(
            AuditEventType.CONFIG_CHANGED,
            actor=actor,
            details={"key": key, "old_value": str(old_value), "new_value": str(new_value)},
            severity="WARNING",
        )

    def risk_breach(self, actor: str, details: Dict[str, Any]) -> str:
        """Record a risk limit breach."""
        return self.log(
            AuditEventType.RISK_BREACH,
            actor=actor,
            details=details,
            outcome="failure",
            severity="CRITICAL",
        )

    def permission_denied(self, actor: str, resource: str, action: str) -> str:
        """Record a permission-denied event."""
        return self.log(
            AuditEventType.PERMISSION_DENIED,
            actor=actor,
            details={"resource": resource, "action": action},
            outcome="failure",
            severity="WARNING",
        )

    def verify_chain(self) -> bool:
        """Verify the integrity of the on-disk audit log chain.

        Reads every record from the log file and recomputes each ``chain_hash``
        to detect tampering.

        Returns:
            ``True`` when all hashes match; ``False`` otherwise.
        """
        prev_hash = "0" * 64
        try:
            with open(self._log_path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    stored_hash = record.pop("chain_hash", None)
                    content = json.dumps(record, sort_keys=True, default=str)
                    expected = hashlib.sha256(
                        (prev_hash + content).encode()
                    ).hexdigest()
                    if stored_hash != expected:
                        self._logger.error(
                            "Chain hash mismatch at line %d (record id=%s)",
                            lineno,
                            record.get("id"),
                        )
                        return False
                    prev_hash = stored_hash
        except FileNotFoundError:
            return True  # Empty log is considered valid
        except Exception as exc:
            self._logger.error("Error verifying audit chain: %s", exc)
            return False

        return True
