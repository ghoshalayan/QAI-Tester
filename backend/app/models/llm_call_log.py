"""Per-LLM-call telemetry row.

One row per LLM round-trip the agent makes during a run. Lets the
Cost Logs UI drill into "what calls happened, in what order, to
which model, with what tokens" so users can investigate why a run
cost what it did.

Lifecycle
---------
- Written by ``app.llm.cost_tracker.end_run`` (in a single batch
  flush so the call log doesn't add per-call DB chatter).
- Read by the Cost Logs drill-in view through
  ``GET /api/settings/cost/runs/{run_id}/calls``.
- Cascade-deleted with the parent run.

``model`` is snapshotted at call time — changing the model later
doesn't rewrite this column. Cost is NOT stored; the read endpoint
joins against current ``app_settings`` pricing so rate updates
re-cost history.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LlmCallLog(Base):
    __tablename__ = "llm_call_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    step_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("execution_steps.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    tier: Mapped[str] = mapped_column(String(16), nullable=False)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    # Migration 0019 — cached portion of input_tokens (subset, not
    # additive). Populated from the provider's response (OpenAI's
    # ``prompt_tokens_details.cached_tokens`` / Gemini's
    # ``cached_content_token_count``). 0 when the prompt didn't hit
    # cache OR the provider didn't report it.
    cached_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    escalated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0",
    )
    duration_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
