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
# Migration 0025 — per-tier providers can use different (provider,
# model, key) combos so the single-entry cache from v1 isn't enough.
# Switch to a dict keyed on the cache_key tuple. Eviction is implicit
# via :func:`invalidate_cache` (called on settings save).
_cache: dict[tuple, LLMProvider] = {}


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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
    if provider == "openrouter":
        # OpenRouter is OpenAI-API-compatible with a fixed endpoint.
        # We surface it as its own provider value so the UI can show
        # OpenRouter-specific model suggestions + skip the base_url
        # field. The actual transport is the OpenAI SDK with
        # base_url=https://openrouter.ai/api/v1.
        effective_base_url = base_url or OPENROUTER_BASE_URL
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=effective_base_url,
            is_compat=True,
        )
    raise ValueError(f"Unknown provider: {provider!r}")


def get_provider(
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
) -> LLMProvider:
    """Return the cached provider, rebuilding if any field changed."""
    key = (provider, model, api_key, base_url)
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        instance = build_provider(provider, model, api_key, base_url)
        _cache[key] = instance
        return instance


def invalidate_cache() -> None:
    with _lock:
        _cache.clear()


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
