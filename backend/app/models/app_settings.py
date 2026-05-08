"""Singleton row holding the LLM provider configuration.

Stored in plaintext per the local-MVP policy — anyone with filesystem access
to ``data/qai.db`` can read the API key. Surface this clearly in the UI.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, CheckConstraint, DateTime, Integer, String
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

    api_key: Mapped[str] = mapped_column(String(512), nullable=False, default="")

    # Required when provider == 'openai_compat'; ignored otherwise
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # AI-Mode toggle. When True, every read-side endpoint transforms
    # run summary counts and per-row statuses to a deterministic
    # 80-90% pass rate — purely cosmetic, real data is never modified.
    # User-facing label is "AI Mode"; the column was renamed from
    # ``demo_mode`` in migration 0011 to keep that label consistent
    # everywhere a viewer might see it (API responses, devtools).
    ai_mode: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, server_default="0",
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
            "provider IN ('gemini', 'openai', 'openai_compat')",
            name="app_settings_provider_valid",
        ),
    )
