"""Phase E — Reusable sub-flow modules.

A SubFlowModule is a named, reusable bundle of frozen test steps
extracted from a passed submodule. Other plans against the same app
(matched by ``target_url_pattern``) can IMPORT the module, getting
a pre-frozen submodule that replays deterministically on first run.

Why this exists
---------------
The "complex flow gets confused" problem on apps like Solar isn't
usually a vision / refinement issue — it's that every plan
re-discovers the same building blocks from scratch (login, create
role, assign permissions, etc.). Tosca handles this with a Test
Module Repository where stable building blocks are authored once
and composed.

We achieve the same outcome from a different angle: any submodule
that PASSES end-to-end has its v2 frozen segments serialized onto
the TcNode (Phase B). Phase E lets the user "save that as a
reusable module", giving it a name + description, after which any
other plan against the same target_url can import it as a fresh
submodule with the v2 segments pre-populated. The agent doesn't
re-discover anything — the next run replays.

Lifecycle
---------
- **Promote**: user clicks "Save as module" on a passed submodule
  in the test cases tab. We snapshot the TcNode's frozen_path +
  its title/description into a SubFlowModule row.
- **Import**: user picks a module from the library and imports it
  into a plan. We create a new TcNode submodule under a chosen
  parent module, copy the frozen segments into the new TcNode's
  ``frozen_path``, and seed its children (step TcNodes) from the
  segments' descriptions so the test-cases viewer still shows the
  steps. Status is ``approved`` since the source was a proven run.
- **Update**: when a future run re-passes the imported submodule
  with improvements (self-heal patches), the user can "republish"
  to overwrite the source module so all OTHER plans benefit.

Scope
-----
Modules are PROJECT-scoped (not global). A project's modules can
be imported across all plans within that project. Cross-project
sharing would require an extra "Share to library" surface; left
out for v1.

target_url_pattern semantics
----------------------------
Same semantics as :class:`AppKnowledge.target_url_pattern` — when
importing, the plan's ``target_url`` is matched against the
module's pattern (substring match by default). A module for
``solar.com`` can be imported into a plan for
``staging.solar.com`` if the user picks it explicitly; we don't
auto-filter by pattern.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
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


class SubFlowModule(Base):
    __tablename__ = "sub_flow_modules"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True,
    )
    project_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Display fields surfaced in the library + import dialog.
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Matching key for cross-plan import filtering. URL substring
    # (e.g. ``solar.com``, ``trycloudflare.com``). NULL = unrestricted.
    target_url_pattern: Mapped[str | None] = mapped_column(
        String(512), nullable=True, index=True,
    )

    # Free-text tags for filtering / search. Examples:
    # ``["auth", "admin", "create"]``. Empty list when not categorized.
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # The v2 frozen segments (same shape as TcNode.frozen_path with
    # version=2 + segments[]). This is what gets copied onto the
    # imported TcNode's frozen_path so replay can walk it.
    frozen_segments: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Step summaries (title + action_type + target_hint + narrative
    # + expected) extracted from the source submodule's children. The
    # import path uses these to create the new TcNode's step children
    # so the test-cases viewer renders the same tree shape as the
    # original. Distinct from frozen_segments which is replay-only.
    step_snapshots: Mapped[list] = mapped_column(JSON, nullable=False)

    # Provenance — where this module came from. Useful for the
    # library UI ("last refreshed from run #81 on 2026-05-12").
    source_plan_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    source_submodule_tc_node_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
    )
    source_run_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
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
