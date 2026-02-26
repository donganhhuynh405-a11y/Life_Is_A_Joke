"""Data encryption utilities using AES-256 via the cryptography library.

All imports from ``cryptography`` are lazy so the module can be imported
even when the library is not installed.  Key material is never hard-coded;
it must be provided by the caller or loaded from environment variables /
a secrets manager.
"""

from __future__ import annotations

import base64
import logging
import os
import secrets
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _get_cryptography():  # type: ignore[return]
    """Lazily import cryptography primitives."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.backends import default_backend

        return AESGCM, hashes, crypto_hmac, PBKDF2HMAC, default_backend
    except ImportError as exc:
        raise ImportError(
            "cryptography package is required. "
            "Install it with: pip install cryptography"
        ) from exc


class EncryptionManager:
    """AES-256-GCM authenticated encryption manager.

    AES-GCM provides both confidentiality and integrity; no separate HMAC
    is needed.  Each call to :meth:`encrypt` generates a fresh 96-bit
    random nonce so that the same plaintext never produces identical
    ciphertext.

    Args:
        key: 32-byte (256-bit) encryption key.  When omitted, the key is
             read from the ``ENCRYPTION_KEY_B64`` environment variable
             (base64-encoded).  A random key is generated when neither
             source provides one (useful for development; not suitable for
             production).
    """

    KEY_LENGTH = 32  # 256 bits
    NONCE_LENGTH = 12  # 96 bits (GCM standard)

    def __init__(self, key: Optional[bytes] = None) -> None:
        if key is not None:
            if len(key) != self.KEY_LENGTH:
                raise ValueError(
                    f"Key must be exactly {self.KEY_LENGTH} bytes; "
                    f"got {len(key)}."
                )
            self._key = key
        else:
            env_key_b64 = os.environ.get("ENCRYPTION_KEY_B64", "")
            if env_key_b64:
                try:
                    decoded = base64.b64decode(env_key_b64)
                    if len(decoded) != self.KEY_LENGTH:
                        raise ValueError("ENCRYPTION_KEY_B64 must decode to 32 bytes")
                    self._key = decoded
                    logger.debug("Encryption key loaded from ENCRYPTION_KEY_B64")
                except Exception as exc:
                    raise ValueError(
                        f"Failed to decode ENCRYPTION_KEY_B64: {exc}"
                    ) from exc
            else:
                self._key = secrets.token_bytes(self.KEY_LENGTH)
                logger.warning(
                    "No encryption key provided; using an ephemeral random key. "
                    "Data encrypted now CANNOT be decrypted after restart."
                )

    # ------------------------------------------------------------------
    # Core encrypt / decrypt
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Encrypt *plaintext* using AES-256-GCM.

        The returned bytes are ``nonce || ciphertext+tag`` where the 12-byte
        nonce is prepended to the GCM output (ciphertext + 16-byte auth tag).

        Args:
            plaintext: Data to encrypt.
            associated_data: Optional authenticated-but-not-encrypted data
                             (e.g. metadata that must not be tampered with).

        Returns:
            ``nonce + ciphertext + tag`` as raw bytes.
        """
        AESGCM, *_ = _get_cryptography()
        nonce = secrets.token_bytes(self.NONCE_LENGTH)
        aesgcm = AESGCM(self._key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        return nonce + ciphertext

    def decrypt(self, token: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt a token produced by :meth:`encrypt`.

        Args:
            token: ``nonce + ciphertext + tag`` bytes returned by :meth:`encrypt`.
            associated_data: Must match the value supplied during encryption.

        Returns:
            Original plaintext bytes.

        Raises:
            ValueError: On authentication failure or malformed input.
        """
        if len(token) < self.NONCE_LENGTH + 16:  # 16 = GCM tag size
            raise ValueError("Token is too short to be valid")
        AESGCM, *_ = _get_cryptography()
        nonce = token[: self.NONCE_LENGTH]
        ciphertext = token[self.NONCE_LENGTH :]
        aesgcm = AESGCM(self._key)
        try:
            return aesgcm.decrypt(nonce, ciphertext, associated_data)
        except Exception as exc:
            raise ValueError("Decryption failed: authentication tag mismatch") from exc

    # ------------------------------------------------------------------
    # String helpers
    # ------------------------------------------------------------------

    def encrypt_str(self, plaintext: str, associated_data: Optional[str] = None) -> str:
        """Encrypt a string and return a URL-safe base64 token.

        Args:
            plaintext: UTF-8 string to encrypt.
            associated_data: Optional UTF-8 string of authenticated data.

        Returns:
            URL-safe base64-encoded ciphertext token.
        """
        ad_bytes = associated_data.encode() if associated_data else None
        raw = self.encrypt(plaintext.encode(), ad_bytes)
        return base64.urlsafe_b64encode(raw).decode()

    def decrypt_str(self, token: str, associated_data: Optional[str] = None) -> str:
        """Decrypt a base64 token produced by :meth:`encrypt_str`.

        Args:
            token: URL-safe base64-encoded ciphertext token.
            associated_data: Must match the value supplied during encryption.

        Returns:
            Decrypted UTF-8 string.
        """
        ad_bytes = associated_data.encode() if associated_data else None
        raw = base64.urlsafe_b64decode(token.encode())
        return self.decrypt(raw, ad_bytes).decode()

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    @staticmethod
    def derive_key(
        password: str,
        salt: Optional[bytes] = None,
        iterations: int = 260_000,
    ) -> Tuple[bytes, bytes]:
        """Derive a 256-bit key from *password* using PBKDF2-HMAC-SHA256.

        Args:
            password: User-supplied passphrase.
            salt: 16-byte random salt.  A new random salt is generated when
                  omitted – callers must persist this salt for later decryption.
            iterations: PBKDF2 iteration count (NIST recommends ≥ 210,000
                        for SHA-256 as of 2023).

        Returns:
            Tuple of ``(derived_key_bytes, salt_bytes)``.
        """
        _, hashes, _, PBKDF2HMAC, default_backend = _get_cryptography()
        if salt is None:
            salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=EncryptionManager.KEY_LENGTH,
            salt=salt,
            iterations=iterations,
            backend=default_backend(),
        )
        key = kdf.derive(password.encode())
        return key, salt

    @staticmethod
    def generate_key() -> Tuple[bytes, str]:
        """Generate a fresh random 256-bit key.

        Returns:
            Tuple of ``(raw_key_bytes, base64_encoded_key_string)``.
        """
        key = secrets.token_bytes(EncryptionManager.KEY_LENGTH)
        return key, base64.urlsafe_b64encode(key).decode()
