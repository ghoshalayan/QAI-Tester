"""TcNode — N-level test-case tree, scoped per plan.

A TC tree lives under a single ``TestPlan``. Roots are ``kind='module'``
(``parent_id`` IS NULL); their children are typically ``submodule`` rows;
leaves are ``step`` rows that the executor walks during a run.

Schema supports arbitrary depth via ``parent_id`` self-FK; week 4 generates
3 levels (module → submodule → step). Deeper trees are explicitly allowed
by the data model and may be produced by future agent versions.

Field semantics
---------------
- ``ordinal``       : order among siblings sharing a ``parent_id``. Stable
                      across re-renders; insertion code maintains uniqueness
                      by always inserting with ``max(ordinal)+1``.
- ``depth``         : 0 = module, 1 = submodule, 2 = step (leaf). Cached at
                      insert; speeds tree queries.
- ``path_cached``   : denormalized full path string for citing nodes in
                      reports without joining ancestors.
- Step fields       : ``action_type``, ``target_hint``, ``narrative``,
                      ``expected``, ``data_needs_json``. NULL for non-leaf
                      nodes — week 5's executor only consumes leaves.
- ``selectable_default``: initial tick state for the run-selection UI. The
                      user adjusts before launching a run; per-run picks live
                      in a separate ``run_selections`` table (week 5).
- ``source_requirement_ids``: list of FRD ``Requirement.id`` values that
                      motivated this node (week-4 traceability).

Cascade
-------
- ``project_id``  CASCADE — wipe with project
- ``plan_id``     CASCADE — wipe with plan delete
- ``parent_id``   CASCADE — deleting a parent node removes the subtree
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TcNode(Base):
    __tablename__ = "tc_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("test_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("tc_nodes.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    path_cached: Mapped[str] = mapped_column(String(2048), nullable=False)

    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description_md: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Step-only fields ──────────────────────────────────────────
    action_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    target_hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_needs_json: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # ── Phase E: frozen replay path (submodule-level) ─────────────
    # When an agentic run passes vision verification on this
    # submodule, we serialize the agent's working tool sequence here
    # (with successful selectors after fuzzy / vision substitution).
    # Replay-mode runs walk this list deterministically — no LLM
    # calls. NULL for nodes that have never been successfully
    # agent-run; replay falls back to agentic for them.
    frozen_path: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # ── Production-α: goal preconditions / postconditions / signals ─
    # Richer goal shape so the agent can reason about WHEN it's done
    # rather than relying on narrative interpretation. All four are
    # ``list[str]`` of short conditions; NULL = legacy run that
    # predates this column (agent infers from goal description).
    #
    # preconditions       : state that must hold BEFORE the agent
    #                       acts. Asserted against WorldState at
    #                       submodule start; mismatch → auto-dispute
    #                       with kind=precondition_failed.
    # postconditions      : state that must hold AFTER for the goal
    #                       to count as passed. Used to update
    #                       WorldState on success.
    # evidence_signals    : observable signals that prove the post-
    #                       conditions met (e.g. "Subtotal text"
    #                       AND "cart icon badge ≥ 1"). The agent's
    #                       verify is N-of-M — claim done when ≥
    #                       threshold of signals match.
    # alternative_paths   : human-readable hints about other ways
    #                       to reach the postconditions when the
    #                       primary flow is blocked (e.g. "use the
    #                       cart icon in header instead of the
    #                       Cart link in footer").
    preconditions: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True,
    )
    postconditions: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True,
    )
    evidence_signals: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True,
    )
    alternative_paths: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True,
    )

    # ── Selection + status ────────────────────────────────────────
    selectable_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="draft",
    )

    # ── Traceability ──────────────────────────────────────────────
    source_requirement_ids: Mapped[list[int]] = mapped_column(
        JSON, nullable=False, default=list,
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
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Self-referential relationship for tree walk
    children: Mapped[list["TcNode"]] = relationship(
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="TcNode.ordinal",
        # Avoid eager auto-fetch; queries explicitly load via selectinload
        lazy="select",
    )

    __table_args__ = (
        CheckConstraint(
            "kind IN ('module', 'submodule', 'step')",
            name="tc_node_kind_valid",
        ),
        CheckConstraint(
            "status IN ('draft', 'approved', 'archived')",
            name="tc_node_status_valid",
        ),
        # Composite index for ordered tree walks within a plan
        Index(
            "ix_tc_nodes_plan_parent_ordinal",
            "plan_id",
            "parent_id",
            "ordinal",
        ),
    )
