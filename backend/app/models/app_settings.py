"""Singleton row holding the LLM provider configuration.

Stored in plaintext per the local-MVP policy — anyone with filesystem access
to ``data/qai.db`` can read the API key. Surface this clearly in the UI.
"""

from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, Integer, String
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

    api_key: Mapped[str] = mapped_column(String(512), nullable=False, default="")

    # Required when provider == 'openai_compat'; ignored otherwise
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

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
