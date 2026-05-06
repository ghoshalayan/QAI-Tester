"""ExecutionStep — per-step result row produced by the executor agent.

A row is created for every leaf ``step`` TcNode that's about to run (filtered
by ``selectable_default`` and the deselected-ancestor cut). Modules and
submodules are *not* persisted — they don't execute; their pass/fail
aggregate is derived from the rows under them at render time.

Lifecycle of a single row
-------------------------
    pending → running → passed
                     ↘ failed       (assertion or selector miss)
                     ↘ skipped      (an ancestor failed → cut the branch)
                     ↘ blocked      (needs HITL — credentials / OTP / confirm;
                                     wired in week 6)

Snapshot fields
---------------
``title_snapshot``, ``path_snapshot``, ``action_type_snapshot``,
``target_hint_snapshot``, ``expected_snapshot``, ``narrative_snapshot`` —
frozen at run-time. Editing or deleting the source ``TcNode`` later doesn't
mutate history. ``tc_node_id`` is SET NULL on tc_node delete so the row
survives.

Cascade
-------
- ``run_id``     CASCADE — a run delete (rare; usually retained) wipes its rows
- ``project_id`` CASCADE — wiping the project wipes everything
- ``plan_id``   SET NULL — plan could be deleted, but run history stays
- ``tc_node_id`` SET NULL — same reason
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExecutionStep(Base):
    __tablename__ = "execution_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
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
    tc_node_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("tc_nodes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── Snapshots (frozen at run-time, never mutated) ─────────────
    title_snapshot: Mapped[str] = mapped_column(String(512), nullable=False)
    path_snapshot: Mapped[str] = mapped_column(String(2048), nullable=False)
    action_type_snapshot: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
    )
    target_hint_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    narrative_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Run-time state ────────────────────────────────────────────
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="pending", nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Outputs ───────────────────────────────────────────────────
    screenshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    narration: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

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
        CheckConstraint(
            "status IN ('pending', 'running', 'passed', 'failed', "
            "'skipped', 'blocked', 'inconclusive')",
            name="execution_step_status_valid",
        ),
        # Ordered scan within a run for the live timeline
        Index("ix_execution_steps_run_ordinal", "run_id", "ordinal"),
    )
