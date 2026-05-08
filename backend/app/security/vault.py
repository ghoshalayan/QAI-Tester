"""Phase 3 — credential vault: encryption-at-rest + TOTP code generation.

Three concerns this module owns:

1. **Key resolution.** Where does the Fernet master key live?
    Order:
      a) ``QAI_VAULT_KEY`` env var (production deploy override)
      b) ``data/.vault_key`` file with 0600 perms
      c) Generate a fresh key on first vault read, write to file
    Env var wins — friendly default for local dev, swappable for
    production.

2. **Encryption / decryption.** Fernet (AES-128-CBC + HMAC-SHA256
   from the ``cryptography`` package). Symmetric, authenticated,
   small ciphertext overhead (~100 bytes), URL-safe base64. Same
   shape as Django's signed cookies.

3. **TOTP code generation.** Wraps ``pyotp`` so callers can ask
   "give me the current code for this credential" without dragging
   pyotp into qa_agent. Empty / invalid secret yields ``None`` — the
   auth flow falls back to HITL prompt for OTP.

Read path (vault.read_credential):
- Look up the row.
- If row.encrypted=False: pass through plaintext (legacy MVP rows).
- Else: decrypt username / password / totp_secret with the resolved
  Fernet key. Decryption failure raises ``VaultError`` — caller
  should NOT silently fall back to raw bytes (that'd be a key/data
  drift).

Write path (vault.write_credential):
- Always encrypt new writes with the resolved Fernet key.
- Set row.encrypted=True.
"""

from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.test_plan import TestPlanCredential

logger = logging.getLogger(__name__)


_VAULT_KEY_ENV = "QAI_VAULT_KEY"
_VAULT_KEY_FILENAME = ".vault_key"


class VaultError(RuntimeError):
    """Raised when vault encrypt / decrypt fails — usually a key/data drift.

    Callers should NOT silently fall through; surface it as a 500 to
    the user with a clear "the vault key is wrong or the data is
    corrupt" message so they can rotate / re-enter credentials.
    """


@dataclass(frozen=True)
class CredentialPlaintext:
    """Decrypted credential payload returned by ``read_credential``."""

    username: str
    password: str
    totp_secret: str | None


# Module-level cache so we don't re-resolve the key on every vault
# call. Invalidated on process exit; tests use ``_reset_key_cache``.
_key_cache: bytes | None = None


def _reset_key_cache() -> None:
    """Test helper — wipes the cached key so the next call re-resolves."""
    global _key_cache
    _key_cache = None


def _data_dir() -> Path:
    """Resolve the v2 backend's ``data/`` directory.

    Lazy import of settings to avoid a cycle (settings → vault on
    early-stage credential operations during app boot).
    """
    from app.config import settings  # noqa: PLC0415

    raw = getattr(settings, "data_dir", None) or "data"
    p = Path(raw)
    if not p.is_absolute():
        # Relative paths anchor at the backend root. ``app/security/
        # vault.py`` → parents[2] is the backend root.
        p = Path(__file__).resolve().parents[2] / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_key() -> bytes:
    """Return the Fernet key, generating + persisting one on first use.

    Order: env var → file → generate. The generated key is written
    with 0600 perms on POSIX so other local users can't read it;
    Windows ACLs don't have a clean equivalent so we accept the
    default user-owned permissions.
    """
    global _key_cache
    if _key_cache is not None:
        return _key_cache

    env_key = os.environ.get(_VAULT_KEY_ENV)
    if env_key:
        _key_cache = env_key.encode("utf-8") if isinstance(env_key, str) else env_key
        return _key_cache

    key_path = _data_dir() / _VAULT_KEY_FILENAME
    if key_path.exists():
        try:
            data = key_path.read_bytes().strip()
            if data:
                _key_cache = data
                return _key_cache
        except OSError as e:
            raise VaultError(
                f"Could not read vault key at {key_path}: {e}",
            ) from e

    # First-run: generate + persist.
    try:
        from cryptography.fernet import Fernet  # noqa: PLC0415
    except ImportError as e:
        raise VaultError(
            "cryptography package not installed — run `uv sync` "
            "to install Fernet support.",
        ) from e

    new_key = Fernet.generate_key()
    try:
        key_path.write_bytes(new_key)
        # Tighten perms on POSIX. Best-effort; ignored on Windows.
        try:
            os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    except OSError as e:
        raise VaultError(
            f"Could not write vault key to {key_path}: {e}",
        ) from e
    logger.info(
        "Generated fresh vault key at %s (%d bytes)",
        key_path, len(new_key),
    )
    _key_cache = new_key
    return _key_cache


def _fernet():
    """Build a Fernet instance from the resolved key."""
    try:
        from cryptography.fernet import Fernet  # noqa: PLC0415
    except ImportError as e:
        raise VaultError(
            "cryptography package not installed — run `uv sync`.",
        ) from e
    return Fernet(_resolve_key())


def encrypt_str(plaintext: str) -> str:
    """Encrypt a string. Returns URL-safe base64 ciphertext."""
    f = _fernet()
    try:
        return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")
    except Exception as e:
        raise VaultError(f"vault encrypt failed: {e}") from e


def decrypt_str(ciphertext: str) -> str:
    """Decrypt a Fernet ciphertext. Raises ``VaultError`` on bad data /
    bad key."""
    f = _fernet()
    try:
        return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except Exception as e:
        raise VaultError(
            f"vault decrypt failed (key drift or corrupt data?): {e}",
        ) from e


def read_credential(row: "TestPlanCredential") -> CredentialPlaintext:
    """Return decrypted plaintext for a credential row.

    Branches on ``row.encrypted``:
    - False (legacy plaintext) → pass through as-is.
    - True → Fernet-decrypt each field. ``totp_secret`` is None when
      the column was empty / unset.
    """
    encrypted = bool(getattr(row, "encrypted", False))
    if not encrypted:
        return CredentialPlaintext(
            username=row.username or "",
            password=row.password or "",
            totp_secret=(row.totp_secret or None),
        )
    return CredentialPlaintext(
        username=decrypt_str(row.username) if row.username else "",
        password=decrypt_str(row.password) if row.password else "",
        totp_secret=(
            decrypt_str(row.totp_secret) if row.totp_secret else None
        ),
    )


def encrypt_for_write(
    username: str,
    password: str,
    totp_secret: str | None,
) -> tuple[str, str, str | None]:
    """Encrypt fields for a fresh write. Returns the trio ready to set
    on a credential row, plus the caller should set ``encrypted=True``.

    TOTP-secret normalization: strip whitespace and a leading
    ``otpauth://`` prefix if the user pasted a full URI; pyotp's
    ``parse_uri`` handles those, but for the raw secret column we
    want just the base32 seed.
    """
    enc_user = encrypt_str(username) if username else ""
    enc_pass = encrypt_str(password) if password else ""
    enc_totp: str | None = None
    if totp_secret:
        seed = _normalize_totp_seed(totp_secret)
        if seed:
            enc_totp = encrypt_str(seed)
    return enc_user, enc_pass, enc_totp


def _normalize_totp_seed(raw: str) -> str:
    """Accept a base32 seed OR an ``otpauth://...`` URI; return the
    base32 seed (uppercase, no whitespace). Empty when the input
    can't be parsed."""
    s = (raw or "").strip()
    if not s:
        return ""
    if s.lower().startswith("otpauth://"):
        try:
            import pyotp  # noqa: PLC0415

            otp = pyotp.parse_uri(s)
            seed = getattr(otp, "secret", "") or ""
            return seed.replace(" ", "").upper()
        except Exception as e:
            logger.warning(
                "TOTP URI parse failed (%s); treating as raw seed",
                e,
            )
            return s.replace(" ", "").upper()
    return s.replace(" ", "").upper()


def generate_totp_code(seed: str) -> str | None:
    """Return the current 6-digit TOTP code for ``seed``, or ``None``
    when the seed is empty / pyotp isn't installed / generation
    fails. Auth flow falls back to HITL prompt on ``None``.
    """
    if not seed:
        return None
    try:
        import pyotp  # noqa: PLC0415
    except ImportError:
        logger.info(
            "pyotp not installed — cannot generate TOTP; HITL fallback",
        )
        return None
    try:
        return pyotp.TOTP(seed).now()
    except Exception as e:
        logger.warning("TOTP generation failed: %s", e)
        return None
