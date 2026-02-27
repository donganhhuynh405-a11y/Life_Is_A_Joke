"""Authentication and authorization using JWT (PyJWT lazy import).

Provides token issuance, validation, role-based access control, and API key
management.  No credentials are hard-coded; secrets must be supplied via
environment variables or a secrets manager.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """Built-in roles for the trading bot."""

    ADMIN = "admin"
    TRADER = "trader"
    READ_ONLY = "read_only"
    SYSTEM = "system"


# Map roles → allowed permissions
ROLE_PERMISSIONS: Dict[Role, Set[str]] = {
    Role.ADMIN: {
        "orders:read", "orders:write", "orders:cancel",
        "config:read", "config:write",
        "users:read", "users:write",
        "metrics:read",
    },
    Role.TRADER: {
        "orders:read", "orders:write", "orders:cancel",
        "config:read",
        "metrics:read",
    },
    Role.READ_ONLY: {
        "orders:read",
        "config:read",
        "metrics:read",
    },
    Role.SYSTEM: {
        "orders:read", "orders:write", "orders:cancel",
        "config:read", "config:write",
        "metrics:read",
    },
}


@dataclass
class TokenClaims:
    """Decoded JWT claims."""

    subject: str
    roles: List[str]
    issued_at: float
    expires_at: float
    jti: str  # JWT ID for revocation
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Return ``True`` when the token has passed its expiry time."""
        return time.time() > self.expires_at

    def has_role(self, role: Role) -> bool:
        """Return ``True`` when *role* is in the token's role list."""
        return role.value in self.roles

    def has_permission(self, permission: str) -> bool:
        """Return ``True`` when any of the token's roles grants *permission*."""
        for role_str in self.roles:
            try:
                role = Role(role_str)
                if permission in ROLE_PERMISSIONS.get(role, set()):
                    return True
            except ValueError:
                continue
        return False


def _get_jwt():  # type: ignore[return]
    """Lazily import PyJWT."""
    try:
        import jwt  # type: ignore[import]

        return jwt
    except ImportError as exc:
        raise ImportError(
            "PyJWT is required for JWT auth. Install with: pip install PyJWT"
        ) from exc


class AuthManager:
    """Issues and validates JWT tokens; manages API keys.

    Args:
        secret_key: HMAC secret used to sign tokens.  Defaults to the
                    ``JWT_SECRET_KEY`` environment variable.  A random key
                    is generated (and logged as a warning) when neither
                    is provided.
        algorithm: JWT signing algorithm (default ``HS256``).
        token_ttl: Token time-to-live in seconds (default 3600 = 1 hour).
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        token_ttl: int = 3600,
    ) -> None:
        self._secret = secret_key or os.environ.get("JWT_SECRET_KEY", "")
        if not self._secret:
            self._secret = secrets.token_urlsafe(32)
            logger.warning(
                "JWT_SECRET_KEY not set; using an ephemeral random key. "
                "Tokens will be invalidated on restart."
            )
        self.algorithm = algorithm
        self.token_ttl = token_ttl
        # Simple in-memory revocation set (use Redis in production)
        self._revoked_jtis: Set[str] = set()
        # API key store: {key_hash: {"subject": ..., "roles": [...]}}
        self._api_keys: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # JWT operations
    # ------------------------------------------------------------------

    def issue_token(
        self,
        subject: str,
        roles: Optional[List[str]] = None,
        extra_claims: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> str:
        """Issue a signed JWT for *subject*.

        Args:
            subject: Identity the token represents (user ID, service name).
            roles: List of role strings (defaults to ``["read_only"]``).
            extra_claims: Optional additional payload claims.
            ttl: Override token TTL in seconds.

        Returns:
            Encoded JWT string.
        """
        jwt = _get_jwt()
        now = time.time()
        jti = secrets.token_urlsafe(16)
        payload: Dict[str, Any] = {
            "sub": subject,
            "iat": now,
            "exp": now + (ttl or self.token_ttl),
            "jti": jti,
            "roles": roles or [Role.READ_ONLY.value],
        }
        if extra_claims:
            payload.update(extra_claims)
        token: str = jwt.encode(payload, self._secret, algorithm=self.algorithm)
        logger.debug("Issued token for subject '%s' (jti=%s)", subject, jti)
        return token

    def validate_token(self, token: str) -> TokenClaims:
        """Validate a JWT and return its decoded :class:`TokenClaims`.

        Args:
            token: Encoded JWT string.

        Returns:
            Decoded :class:`TokenClaims`.

        Raises:
            ValueError: On invalid, expired, or revoked tokens.
        """
        jwt = _get_jwt()
        try:
            payload: Dict[str, Any] = jwt.decode(
                token,
                self._secret,
                algorithms=[self.algorithm],
            )
        except Exception as exc:
            raise ValueError(f"Invalid token: {exc}") from exc

        jti: str = payload.get("jti", "")
        if jti in self._revoked_jtis:
            raise ValueError("Token has been revoked")

        return TokenClaims(
            subject=payload["sub"],
            roles=payload.get("roles", []),
            issued_at=payload["iat"],
            expires_at=payload["exp"],
            jti=jti,
            extra={
                k: v
                for k, v in payload.items()
                if k not in {"sub", "iat", "exp", "jti", "roles"}
            },
        )

    def revoke_token(self, jti: str) -> None:
        """Add *jti* to the in-memory revocation list.

        Args:
            jti: The JWT ID claim from the token to revoke.
        """
        self._revoked_jtis.add(jti)
        logger.info("Revoked token jti=%s", jti)

    def require_permission(self, token: str, permission: str) -> TokenClaims:
        """Validate *token* and assert it grants *permission*.

        Args:
            token: Encoded JWT string.
            permission: Required permission string (e.g. ``"orders:write"``).

        Returns:
            Decoded :class:`TokenClaims` when authorised.

        Raises:
            PermissionError: When the token lacks the required permission.
            ValueError: On invalid or expired tokens.
        """
        claims = self.validate_token(token)
        if not claims.has_permission(permission):
            raise PermissionError(
                f"Subject '{claims.subject}' lacks permission '{permission}'"
            )
        return claims

    # ------------------------------------------------------------------
    # API key management
    # ------------------------------------------------------------------

    def generate_api_key(
        self,
        subject: str,
        roles: Optional[List[str]] = None,
    ) -> str:
        """Generate a random API key and store its hash.

        Args:
            subject: Owner identity.
            roles: Roles to associate with this key.

        Returns:
            The plaintext API key (store securely; not recoverable later).
        """
        raw_key = secrets.token_urlsafe(32)
        key_hash = self._hash_api_key(raw_key)
        self._api_keys[key_hash] = {
            "subject": subject,
            "roles": roles or [Role.READ_ONLY.value],
            "created_at": time.time(),
        }
        logger.info("Generated API key for subject '%s'", subject)
        return raw_key

    def validate_api_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return its metadata.

        Args:
            raw_key: Plaintext API key supplied by the caller.

        Returns:
            Key metadata dict, or ``None`` when the key is invalid.
        """
        key_hash = self._hash_api_key(raw_key)
        # Constant-time lookup to prevent timing attacks
        result = None
        for stored_hash, metadata in self._api_keys.items():
            if hmac.compare_digest(stored_hash, key_hash):
                result = metadata
                break
        return result

    @staticmethod
    def _hash_api_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()
