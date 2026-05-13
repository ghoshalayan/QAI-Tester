"""Phase C.3 — Test-case versioning.

Per the user's spec, refinement runs produce **new versions** of the
test-case tree rather than mutating in-place. v1 is the BRD-derived
baseline; v2+ come from app-map refinement or manual edits.

Storage shape
-------------
- ``test_plan_tc_versions``: header rows. One per refinement run. The
  plan's ``current_tc_version_id`` points at the version that should
  be used at run-start unless the user picks a different one.
- ``tc_node_snapshots``: the frozen TcNode tree as it looked at
  version-creation time. Same fields as TcNode but with
  ``tc_version_id`` FK + ``original_tc_node_id`` traceability.

Why snapshots and not just diffs:
- Diffs would require reconstructing the tree at read time.
- A snapshot is a direct copy; reading version V means listing
  ``tc_node_snapshots`` where ``tc_version_id = V``.
- Refinement-time storage is cheap (~kB per submodule).

Run-time selection
------------------
``test_plans.current_tc_version_id`` defaults to NULL meaning "use
the live TcNode tree". When set, the agent run service materialises
the TcNode tree FROM the matching snapshot rows for the duration of
the run. The live TcNode tree never gets touched.

Audit trail
-----------
Each version records ``source`` (brd_initial / app_map_refined /
manual) plus ``created_at`` so the report can attribute outcomes to
the exact TC version used.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TcVersion(Base):
    __tablename__ = "test_plan_tc_versions"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )
    plan_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("test_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Monotonic per plan — 1 for the first version, 2 for the second,
    # etc. Reset isn't supported; new refinements always increment.
    version_number: Mapped[int] = mapped_column(
        Integer, nullable=False,
    )

    # 'brd_initial' | 'app_map_refined' | 'manual'
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="manual",
    )

    # User-facing label. Defaults to "v{N} ({source})" if not provided.
    label: Mapped[str] = mapped_column(
        String(120), nullable=False, default="",
    )

    # Free-form notes about WHY this refinement was created — the
    # source AppMap's pages_scouted, the run_id that triggered, the
    # diff summary, etc. Rendered in the version picker dialog.
    notes_json: Mapped[dict | None] = mapped_column(
        JSON, nullable=True,
    )

    # Snapshot of the BRD chunks used as input (chunk ids + content
    # hashes). Lets the refinement service detect "BRD has changed
    # since this version was created — refresh recommended".
    source_doc_snapshot: Mapped[dict | None] = mapped_column(
        JSON, nullable=True,
    )

    # Snapshot of the AppMap version used as input (when source =
    # app_map_refined). NULL for manual / brd_initial.
    source_app_map_run_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False,
    )

    snapshots: Mapped[list["TcNodeSnapshot"]] = relationship(
        back_populates="version",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="TcNodeSnapshot.ordinal",
    )


class TcNodeSnapshot(Base):
    """One TcNode as it looked at a given TcVersion's creation time.

    Mirrors the shape of TcNode (kind / title / description /
    action_type / target_hint / narrative / expected / data_needs /
    selectable_default) PLUS:

    - ``tc_version_id`` — owning version
    - ``original_tc_node_id`` — the live TcNode this row was snapshotted
      from, when one existed; NULL for "added" rows the refiner emitted
    - ``parent_snapshot_id`` — link to the parent snapshot to preserve
      tree shape inside this version (don't reuse live tc_nodes.parent_id;
      a future refinement might add/remove parents)
    - ``change_kind`` — telemetry for the report:
      ``kept | rewritten | added | flagged_missing``
    """

    __tablename__ = "tc_node_snapshots"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )
    tc_version_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "test_plan_tc_versions.id", ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # Optional pointer to the original live TcNode (when one existed).
    # NULL when this is a brand-new row emitted by the refiner.
    original_tc_node_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("tc_nodes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Parent snapshot in THIS version's tree. NULL for the root
    # module(s). Self-referential to keep the snapshot tree
    # self-contained (refinement may change parentage).
    parent_snapshot_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("tc_node_snapshots.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Same content fields as TcNode.
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    path_cached: Mapped[str] = mapped_column(String(2048), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description_md: Mapped[str | None] = mapped_column(Text, nullable=True)
    action_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    target_hint: Mapped[str | None] = mapped_column(Text, nullable=True)
    narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_needs_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    selectable_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )

    # 'kept' | 'rewritten' | 'added' | 'flagged_missing'
    change_kind: Mapped[str] = mapped_column(
        String(24), nullable=False, default="kept",
    )

    # Short LLM-emitted reason explaining the change — surfaced in
    # the diff dialog so the user understands WHY the refiner did
    # what it did.
    change_reason: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )

    # Phase D — live-UI validation. After refinement, an optional
    # dry-run probes each step against the running app (DOM
    # locate without dispatching) and records its result here.
    # Distinguishes "the refiner said this step exists" from "the
    # app actually has it RIGHT NOW".
    #
    # Status semantics:
    # - ``pending``    : not validated yet (default)
    # - ``confirmed``  : target_hint + expected text both resolved
    # - ``partial``    : one of two probes resolved (medium confidence)
    # - ``unresolved`` : target_hint didn't resolve (low confidence,
    #                    but the step might still work via fuzzy/vision
    #                    at runtime — warn, don't block)
    # - ``unreachable``: validator couldn't reach the predicted page
    #                    (nav broke, login wall, etc.) — block until
    #                    investigated
    # - ``skipped``    : step's change_kind was flagged_missing, so
    #                    the validator deliberately didn't probe
    validation_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending",
        server_default="pending",
    )
    # Confidence 0.0 - 1.0. Combines change_kind weight + validation
    # outcome + refiner self-confidence. The rollup view (dialog +
    # test-cases viewer) renders a per-step badge derived from this.
    validation_confidence: Mapped[float | None] = mapped_column(
        Float, nullable=True,
    )
    # Short human-readable explanation (rendered as a tooltip on
    # validation badges). E.g. "target_hint 'Display Name input'
    # didn't resolve on the page; nearest visible textbox is
    # 'Display name'".
    validation_reason: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    validation_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    version: Mapped["TcVersion"] = relationship(
        back_populates="snapshots",
    )
