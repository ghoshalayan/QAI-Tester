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

    input_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    output_summary_json: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False,
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
            "kind IN ('brd_to_frd', 'frd_to_tc', 'execute', 'reporter')",
            name="agent_run_kind_valid",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'paused', 'completed', "
            "'failed', 'cancelled')",
            name="agent_run_status_valid",
        ),
    )
