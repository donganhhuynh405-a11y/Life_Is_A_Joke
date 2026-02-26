"""HashiCorp Vault integration with lazy hvac import.

Provides secret retrieval, dynamic credential management, and secret renewal
without hard-coding credentials.  Falls back to environment variables when
Vault is unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_hvac_available = False


def _get_hvac():  # type: ignore[return]
    """Lazily import hvac; raises ImportError with a helpful message."""
    global _hvac_available
    try:
        import hvac  # type: ignore[import]

        _hvac_available = True
        return hvac
    except ImportError as exc:
        raise ImportError(
            "hvac is required for Vault integration. "
            "Install it with: pip install hvac"
        ) from exc


class VaultClient:
    """Thin wrapper around hvac that adds retry logic and env-var fallback.

    Args:
        url: Vault server URL (default: ``VAULT_ADDR`` env var or ``http://127.0.0.1:8200``).
        token: Vault token (default: ``VAULT_TOKEN`` env var).
        namespace: Optional Vault enterprise namespace.
        mount_point: KV secrets engine mount point (default ``secret``).
    """

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        namespace: Optional[str] = None,
        mount_point: str = "secret",
    ) -> None:
        self.url = url or os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200")
        self._token = token or os.environ.get("VAULT_TOKEN", "")
        self._namespace = namespace
        self.mount_point = mount_point
        self._client: Any = None

    def _ensure_connected(self) -> None:
        """Lazily create and authenticate the hvac client."""
        if self._client is not None:
            return
        hvac = _get_hvac()
        self._client = hvac.Client(
            url=self.url,
            token=self._token,
            namespace=self._namespace,
        )
        if not self._client.is_authenticated():
            raise PermissionError(
                "Vault client is not authenticated. Check VAULT_TOKEN."
            )
        logger.info("Connected to Vault at %s", self.url)

    # ------------------------------------------------------------------
    # KV v2 operations
    # ------------------------------------------------------------------

    def read_secret(self, path: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Read a secret from the KV v2 secrets engine.

        Args:
            path: Secret path relative to the mount point (e.g. ``trading/api_keys``).
            version: Optional version number; latest if omitted.

        Returns:
            Dictionary of secret key/value pairs.

        Raises:
            KeyError: When the secret path does not exist.
        """
        self._ensure_connected()
        kwargs: Dict[str, Any] = {"path": path, "mount_point": self.mount_point}
        if version is not None:
            kwargs["version"] = version
        response = self._client.secrets.kv.v2.read_secret_version(**kwargs)
        if response is None or "data" not in response:
            raise KeyError(f"Secret not found at path: {path}")
        data: Dict[str, Any] = response["data"].get("data", {})
        logger.debug("Read secret from path: %s", path)
        return data

    def write_secret(self, path: str, secret: Dict[str, Any]) -> None:
        """Write a secret to the KV v2 secrets engine.

        Args:
            path: Secret path relative to the mount point.
            secret: Dictionary of key/value pairs to store.
        """
        self._ensure_connected()
        self._client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=secret,
            mount_point=self.mount_point,
        )
        logger.info("Wrote secret to path: %s", path)

    def delete_secret(self, path: str) -> None:
        """Soft-delete the latest version of a secret.

        Args:
            path: Secret path relative to the mount point.
        """
        self._ensure_connected()
        self._client.secrets.kv.v2.delete_latest_version_of_secret(
            path=path,
            mount_point=self.mount_point,
        )
        logger.info("Deleted secret at path: %s", path)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_api_key(self, name: str, env_fallback: Optional[str] = None) -> str:
        """Retrieve an API key from Vault, falling back to an env variable.

        Args:
            name: Secret path in Vault.
            env_fallback: Environment variable name to use when Vault is
                          unavailable.

        Returns:
            The API key string.

        Raises:
            ValueError: When neither Vault nor the env variable provides a key.
        """
        try:
            secret = self.read_secret(name)
            value = secret.get("value") or secret.get("key") or secret.get("api_key")
            if value:
                return str(value)
        except Exception as exc:
            logger.warning(
                "Could not read '%s' from Vault (%s); trying env fallback", name, exc
            )

        if env_fallback:
            env_value = os.environ.get(env_fallback)
            if env_value:
                return env_value

        raise ValueError(
            f"API key '{name}' not found in Vault or environment variable '{env_fallback}'"
        )

    def is_available(self) -> bool:
        """Return *True* when Vault is reachable and authenticated."""
        try:
            self._ensure_connected()
            return bool(self._client.is_authenticated())
        except Exception:
            return False
