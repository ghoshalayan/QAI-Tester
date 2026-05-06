"""Provider factory + cached singleton.

The cache is keyed on the full ``(provider, model, api_key, base_url)`` tuple so
any settings change automatically invalidates it.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from app.llm.base import LLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.openai_provider import OpenAIProvider

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

_lock = threading.Lock()
_cache: tuple[tuple, LLMProvider] | None = None  # (cache_key, instance)


def build_provider(
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
) -> LLMProvider:
    """Construct a fresh provider instance. No caching — use ``get_provider`` for that."""
    if provider == "gemini":
        return GeminiProvider(api_key=api_key, model=model)
    if provider == "openai":
        return OpenAIProvider(api_key=api_key, model=model, is_compat=False)
    if provider == "openai_compat":
        if not base_url:
            raise ValueError("base_url is required when provider is 'openai_compat'")
        return OpenAIProvider(
            api_key=api_key, model=model, base_url=base_url, is_compat=True,
        )
    raise ValueError(f"Unknown provider: {provider!r}")


def get_provider(
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
) -> LLMProvider:
    """Return the cached provider, rebuilding if any field changed."""
    global _cache
    key = (provider, model, api_key, base_url)
    with _lock:
        if _cache is not None and _cache[0] == key:
            return _cache[1]
        instance = build_provider(provider, model, api_key, base_url)
        _cache = (key, instance)
        return instance


def invalidate_cache() -> None:
    global _cache
    with _lock:
        _cache = None


def get_provider_from_db(db: "Session") -> LLMProvider:
    """Build (or fetch from cache) the provider configured in ``app_settings``.

    Raises ``RuntimeError`` if the user hasn't configured an LLM yet —
    callers (router endpoints, agent services) should surface this as a 400.
    """
    # Local import to avoid a cycle with app.models
    from app.models.app_settings import AppSettings

    row = db.query(AppSettings).filter(AppSettings.id == 1).first()
    if not row or not row.api_key:
        raise RuntimeError(
            "LLM not configured. Configure it in Settings before running agents.",
        )
    return get_provider(
        provider=row.provider,
        model=row.model,
        api_key=row.api_key,
        base_url=row.base_url,
    )
