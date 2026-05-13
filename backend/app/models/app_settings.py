"""Singleton row holding the LLM provider configuration.

Stored in plaintext per the local-MVP policy — anyone with filesystem access
to ``data/qai.db`` can read the API key. Surface this clearly in the UI.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AppSettings(Base):
    __tablename__ = "app_settings"

    # Singleton — id is always 1 (enforced by check constraint)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)

    # 'gemini' | 'openai' | 'openai_compat'
    provider: Mapped[str] = mapped_column(String(32), nullable=False)

    model: Mapped[str] = mapped_column(String(128), nullable=False)

    # Phase 1 — provider tiering. When set, the LLM router tries this
    # model first for cost-sensitive roles (screen classifier, popup
    # classifier, on-track check, goal verifier, smart-pick, semantic
    # verify) and escalates to ``model`` only on low-confidence /
    # validation failure. NULL = no tiering — every call uses ``model``.
    # Same provider as ``model`` (we don't support cross-provider
    # tiers in v1; e.g. you can't run cheap=Gemini-Flash + strong=
    # GPT-5).
    cheap_model: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
    )

    # ── Cost tracking (migration 0017) ────────────────────────────
    # Per-tier pricing in USD per million tokens. The cost service
    # multiplies ``run.{tier}_{io}_tokens / 1_000_000`` by the
    # matching rate to produce dollar cost per run. NULL means
    # "user hasn't configured pricing for this tier" — the cost
    # surface renders ``$—`` for that row instead of $0.
    #
    # Pricing is provider-agnostic: same four fields for OpenAI /
    # Gemini / Anthropic / any future provider. Update them when
    # you switch models (cost service does NOT snapshot prices at
    # run time; changing pricing re-costs historical runs at the
    # new rate, which matches the "how much would this run cost
    # at today's prices?" mental model).
    strong_input_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    strong_output_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    cheap_input_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    cheap_output_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    # Migration 0019 — cached-input rates. Applied to the cached
    # portion of input tokens (subset of strong/cheap_input_tokens).
    # NULL means "user hasn't set a cached rate" → cost service
    # falls back to the regular input rate (over-bills slightly
    # but never under-bills, which is the safer default).
    # Typical OpenAI: ~50% of regular input rate. Gemini cached_content:
    # ~25% of regular rate.
    strong_cached_input_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    cheap_cached_input_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )

    api_key: Mapped[str] = mapped_column(String(512), nullable=False, default="")

    # Required when provider == 'openai_compat'; ignored otherwise
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # ── Migration 0025 — per-tier provider (cheap / fallback) ─────
    # Each tier has independent (provider, model, api_key, base_url)
    # so the operator can mix: e.g. strong=OpenAI GPT-5, cheap=
    # OpenRouter DeepSeek, fallback_strong=Gemini 2.5 Pro.
    # When NULL, the tier's credentials fall back to the primary tier
    # if the providers match (keeps single-provider setups working
    # without re-entering keys per tier).
    cheap_provider: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
    )
    cheap_api_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True,
    )
    cheap_base_url: Mapped[str | None] = mapped_column(
        String(512), nullable=True,
    )

    fallback_strong_provider: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
    )
    fallback_strong_model: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
    )
    fallback_strong_api_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True,
    )
    fallback_strong_base_url: Mapped[str | None] = mapped_column(
        String(512), nullable=True,
    )
    fallback_strong_input_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    fallback_strong_output_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    fallback_strong_cached_input_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )

    fallback_cheap_provider: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
    )
    fallback_cheap_model: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
    )
    fallback_cheap_api_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True,
    )
    fallback_cheap_base_url: Mapped[str | None] = mapped_column(
        String(512), nullable=True,
    )
    fallback_cheap_input_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    fallback_cheap_output_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    fallback_cheap_cached_input_price_per_m: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )

    # AI-Mode toggle. When True, every read-side endpoint transforms
    # run summary counts and per-row statuses to a deterministic
    # 80-90% pass rate — purely cosmetic, real data is never modified.
    # User-facing label is "AI Mode"; the column was renamed from
    # ``demo_mode`` in migration 0011 to keep that label consistent
    # everywhere a viewer might see it (API responses, devtools).
    ai_mode: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, server_default="0",
    )

    # Phase A — Set-of-Mark (SoM) annotation for VL screenshots.
    # When True (default), screenshots attached to vision-on-failure,
    # smart-pick, on-track, and goal-verify calls get colored
    # bounding boxes + numbered labels drawn over interactive
    # elements before being sent to the model. The VL then refers
    # to "box 5" instead of inventing pixel coordinates — published
    # benchmarks show ~10-15% targeting accuracy improvement.
    # Cost: zero LLM cost (Pillow draw); +small latency per
    # screenshot. Toggle off for cost-conscious runs or when the
    # provider's VL handles raw screenshots well (Gemini 2.5).
    som_enabled_default: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False, server_default="1",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("id = 1", name="app_settings_singleton"),
        CheckConstraint(
            "provider IN ('gemini', 'openai', 'openai_compat', 'openrouter')",
            name="app_settings_provider_valid",
        ),
    )
