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
    base_url: str | None = None
    api_key_set: bool = False
    # AI-Mode toggle. When True, run summaries and per-row statuses
    # are transformed at the API boundary to a deterministic 80-90%
    # pass-rate distribution. Real data is untouched. User-facing
    # label everywhere is "AI Mode".
    ai_mode: bool = False
    updated_at: datetime | None = None


class AppSettingsWrite(BaseModel):
    provider: ProviderLiteral | None = None
    model: str | None = Field(default=None, min_length=1, max_length=128)
    api_key: str | None = Field(default=None, max_length=512)
    base_url: str | None = Field(default=None, max_length=512)
    ai_mode: bool | None = None


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
