"""Pydantic schemas for the app_settings router.

Convention:
- Read responses NEVER echo the API key. They expose ``api_key_set`` so the
  frontend can render an "API key is saved" indicator without leaking it.
- Write payloads are partial: any field omitted means "leave it as is".
  Cross-field validation lives in the router (it depends on the existing row).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ProviderLiteral = Literal["gemini", "openai", "openai_compat"]


class AppSettingsRead(BaseModel):
    is_configured: bool = False
    provider: ProviderLiteral | None = None
    model: str | None = None
    # Phase 1 — cheap-tier model. None / empty when tiering disabled.
    cheap_model: str | None = None
    base_url: str | None = None
    api_key_set: bool = False
    # AI-Mode toggle. When True, run summaries and per-row statuses
    # are transformed at the API boundary to a deterministic 80-90%
    # pass-rate distribution. Real data is untouched. User-facing
    # label everywhere is "AI Mode".
    ai_mode: bool = False
    # Cost tracking — USD per million tokens. NULL = "not configured"
    # (the cost surface renders ``$—`` for that tier/direction).
    strong_input_price_per_m: float | None = None
    strong_output_price_per_m: float | None = None
    cheap_input_price_per_m: float | None = None
    cheap_output_price_per_m: float | None = None
    # Migration 0019 — cached-input rate (applied to prompt tokens
    # the provider reported as cache hits). Typical OpenAI: ~50%
    # of regular input. Typical Gemini cached_content: ~25%.
    # NULL → cost service falls back to the regular input rate
    # (safe over-bill default).
    strong_cached_input_price_per_m: float | None = None
    cheap_cached_input_price_per_m: float | None = None
    updated_at: datetime | None = None


class AppSettingsWrite(BaseModel):
    provider: ProviderLiteral | None = None
    model: str | None = Field(default=None, min_length=1, max_length=128)
    # Phase 1 — cheap-tier model. Optional. Empty string means "clear
    # / disable tiering". Validation: must NOT equal ``model`` (would
    # be a no-op tier). Cross-field check lives in the router.
    cheap_model: str | None = Field(default=None, max_length=128)
    api_key: str | None = Field(default=None, max_length=512)
    base_url: str | None = Field(default=None, max_length=512)
    ai_mode: bool | None = None
    # Cost tracking — USD per million tokens. Float >= 0 (negative
    # rejected by Field constraint). ``None`` means "don't update";
    # send ``0`` to clear a previously-set rate, or any positive
    # float to set/replace.
    strong_input_price_per_m: float | None = Field(
        default=None, ge=0, le=10_000,
    )
    strong_output_price_per_m: float | None = Field(
        default=None, ge=0, le=10_000,
    )
    cheap_input_price_per_m: float | None = Field(
        default=None, ge=0, le=10_000,
    )
    cheap_output_price_per_m: float | None = Field(
        default=None, ge=0, le=10_000,
    )
    # Migration 0019 — cached-input rates. Same "None = preserve,
    # 0 = clear" semantics as the regular rates.
    strong_cached_input_price_per_m: float | None = Field(
        default=None, ge=0, le=10_000,
    )
    cheap_cached_input_price_per_m: float | None = Field(
        default=None, ge=0, le=10_000,
    )


class TestConnectionResponse(BaseModel):
    """Result of a round-trip ping to the configured LLM provider."""
    ok: bool
    provider: str
    model: str
    base_url: str | None = None
    echo: str | None = None
    latency_ms: int | None = None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
