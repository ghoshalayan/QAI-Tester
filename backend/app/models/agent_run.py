"""AgentRun — tracks every BRD→FRD / FRD→TC / execute / reporter invocation.

Status transitions
------------------
    queued → running → completed
                     ↘ failed
                     ↘ cancelled    (user pressed Cancel; check_cancellation is
                                     polled between phases — week 3 cancels
                                     before persist, week 5 will cancel mid-run)
    running ↔ paused                (week 5+; not used yet)

``input_json`` and ``output_summary_json`` hold kind-specific payloads:

    brd_to_frd → input  {source_document_ids: [int], cap_chunks: int}
                 output {generated: int, approved: int, llm_input_tokens: int,
                         llm_output_tokens: int}
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("test_plans.id", ondelete="SET NULL"),
        nullable=True,
    )

    # 'brd_to_frd' | 'frd_to_tc' | 'execute' | 'reporter'
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    # 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'
    status: Mapped[str] = mapped_column(
        String(16), default="queued", nullable=False,
    )

    # Phase 6 — agent strategy. Only meaningful for ``kind='execute'``
    # runs in agentic mode. ``hybrid`` (default) = DOM-first ladder
    # with vision rescue. ``vision_only`` = pure VL+coords (slower /
    # costlier but works on apps DOM resolution can't reach: heavy
    # canvas, sealed shadow DOM, hostile sites).
    agent_strategy: Mapped[str] = mapped_column(
        String(16),
        default="hybrid",
        server_default="hybrid",
        nullable=False,
    )

    input_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    output_summary_json: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False,
    )
    # Production-α — plan-scoped WorldState. Carried across submodules
    # within one run so submodule N can assert preconditions set up
    # by submodule N-1 (cart_count, logged_in_as, current_url, etc.).
    # Updated by the agent via verify-success and explicit asserts.
    # NULL on legacy runs that predate this column; the runtime
    # treats NULL as an empty dict.
    world_state_json: Mapped[dict | None] = mapped_column(
        JSON, nullable=True,
    )

    # ── Cost tracking (migration 0017) ────────────────────────────
    # Per-tier token counters. The agent loop maintains four counters
    # (cheap_in, cheap_out, strong_in, strong_out) and writes them
    # here at run end. The aggregate ``output_summary_json`` still
    # carries the totals for back-compat.
    #
    # Pre-feature runs have all four = 0 (default). The cost helper
    # detects that case and treats the aggregate tokens (from
    # output_summary_json) as strong-tier — preserving a sensible
    # cost number on historical runs without a DB backfill.
    strong_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    strong_output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    cheap_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    cheap_output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    # Migration 0019 — cached portion of the per-tier input counters
    # above (SUBSET, not additive). Cost service derives
    # ``regular_input = input_tokens - cached_input_tokens`` and
    # applies the cached rate to the cached portion. Default 0
    # so historical runs read as "no cache hit" (cost unchanged).
    strong_cached_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    cheap_cached_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )
    # Model names captured AT RUN TIME so cost calc stays correct
    # after the user changes their LLM config. Snapshotted from
    # AppSettings.{model, cheap_model} when the run starts. NULL on
    # legacy / pre-feature runs.
    strong_model_snapshot: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
    )
    cheap_model_snapshot: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    __table_args__ = (
        CheckConstraint(
            "kind IN ('brd_to_frd', 'frd_to_tc', 'execute', "
            "'reporter', 'recon', 'record')",
            name="agent_run_kind_valid",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'paused', 'completed', "
            "'failed', 'cancelled')",
            name="agent_run_status_valid",
        ),
    )
