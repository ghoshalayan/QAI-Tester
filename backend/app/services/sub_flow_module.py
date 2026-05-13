"""Phase E — Sub-flow module service.

Three flows:

1. **Promote** — turn a passed submodule's frozen v2 segments into
   a named reusable module. Reads the source TcNode + its
   children, snapshots ``frozen_path`` and step summaries.
2. **Import** — apply a module to a destination plan. Creates a
   new submodule TcNode under the target module (or auto-finds /
   creates a default module bucket), copies the frozen segments
   onto the new submodule's ``frozen_path``, and creates step
   TcNodes mirroring the module's step snapshots.
3. **CRUD** — list / get / update metadata / delete.

What changes vs. plain TC import
--------------------------------
The imported TcNode has its ``frozen_path`` pre-populated with the
SOURCE run's proven steps. The first run against the imported
submodule REPLAYS deterministically (no LLM cost, no re-discovery)
through :mod:`app.agents.replay`. Per-sub-goal handoff to agentic
remains intact — if a segment fails, only that segment re-runs
under the agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────


@dataclass
class PromoteResult:
    module_id: int
    name: str
    segments: int
    steps: int


@dataclass
class ImportResult:
    new_submodule_id: int
    parent_module_id: int
    steps_created: int


# ── Promote ───────────────────────────────────────────────────────


def promote_submodule_to_module(
    db: "Session",
    *,
    project_id: int,
    submodule_tc_node_id: int,
    name: str,
    description: str = "",
    target_url_pattern: str | None = None,
    tags: list[str] | None = None,
    source_run_id: int | None = None,
) -> PromoteResult:
    """Snapshot a passed submodule into a reusable module row.

    Requires the source TcNode to already carry a v2 ``frozen_path``
    (otherwise there's nothing replay-worthy to capture). Raises
    ``ValueError`` if the submodule isn't frozen or doesn't belong
    to the project.

    ``target_url_pattern`` defaults to the source plan's target URL
    (substring-match-friendly). Override when you want the module
    to match a broader pattern (e.g. only the host).
    """
    from app.models.sub_flow_module import SubFlowModule  # noqa: PLC0415
    from app.models.tc_node import TcNode  # noqa: PLC0415
    from app.models.test_plan import TestPlan  # noqa: PLC0415
    from sqlalchemy import select  # noqa: PLC0415

    sm = db.get(TcNode, submodule_tc_node_id)
    if sm is None:
        raise ValueError(
            f"submodule {submodule_tc_node_id} not found",
        )
    if sm.project_id != project_id:
        raise ValueError(
            "submodule belongs to a different project",
        )
    if sm.kind != "submodule":
        raise ValueError(
            f"tc_node {submodule_tc_node_id} is a {sm.kind!r}, "
            "not a submodule — only submodules can be promoted",
        )

    frozen = sm.frozen_path
    if not isinstance(frozen, dict):
        raise ValueError(
            "submodule has no frozen_path — run agentic mode "
            "successfully against it first",
        )
    # We accept BOTH v1 (steps[]) and v2 (segments[]) here so legacy
    # frozen paths can still be promoted. The replay walker handles
    # both shapes already.
    has_segments = (
        frozen.get("version") == 2
        and isinstance(frozen.get("segments"), list)
        and frozen.get("segments")
    )
    has_steps = (
        isinstance(frozen.get("steps"), list)
        and frozen.get("steps")
    )
    if not has_segments and not has_steps:
        raise ValueError(
            "frozen_path is malformed (no segments / steps)",
        )

    # Default target_url_pattern from the source plan's target URL.
    if target_url_pattern is None and sm.plan_id is not None:
        plan = db.get(TestPlan, sm.plan_id)
        if plan is not None:
            target_url_pattern = (plan.target_url or "").strip() or None

    # Collect step snapshots — the child step TcNodes' content so
    # the import path can recreate the same tree shape downstream.
    child_steps = list(db.execute(
        select(TcNode)
        .where(
            TcNode.parent_id == sm.id,
            TcNode.kind == "step",
        )
        .order_by(TcNode.ordinal),
    ).scalars())
    step_snapshots: list[dict[str, Any]] = []
    for s in child_steps:
        step_snapshots.append({
            "ordinal": s.ordinal,
            "title": s.title,
            "description_md": s.description_md,
            "action_type": s.action_type,
            "target_hint": s.target_hint,
            "narrative": s.narrative,
            "expected": s.expected,
            "data_needs_json": s.data_needs_json,
        })

    seg_count = (
        len(frozen.get("segments", []))
        if has_segments
        else 0
    )
    step_count = (
        sum(len(seg.get("steps") or []) for seg in frozen.get("segments", []))
        if has_segments
        else len(frozen.get("steps", []))
    )

    module = SubFlowModule(
        project_id=project_id,
        name=name.strip()[:255],
        description=description.strip() or None,
        target_url_pattern=target_url_pattern,
        tags=list(tags or []) or None,
        frozen_segments=frozen,
        step_snapshots=step_snapshots,
        source_plan_id=sm.plan_id,
        source_submodule_tc_node_id=sm.id,
        source_run_id=source_run_id,
    )
    db.add(module)
    db.flush()
    db.commit()

    return PromoteResult(
        module_id=module.id,
        name=module.name,
        segments=seg_count,
        steps=step_count,
    )


# ── Import ────────────────────────────────────────────────────────


def import_module_into_plan(
    db: "Session",
    *,
    project_id: int,
    plan_id: int,
    module_id: int,
    parent_module_tc_node_id: int | None = None,
) -> ImportResult:
    """Apply a module to a plan: create a new submodule TcNode with
    the module's frozen segments + step children.

    ``parent_module_tc_node_id`` lets the user place the imported
    submodule under a specific top-level module in the plan's TC
    tree. When None, we use the first existing module — or create a
    "Imported modules" module if none exists.

    Returns the new submodule's id + step count.
    """
    from app.models.sub_flow_module import SubFlowModule  # noqa: PLC0415
    from app.models.tc_node import TcNode  # noqa: PLC0415
    from app.models.test_plan import TestPlan  # noqa: PLC0415
    from sqlalchemy import select, func  # noqa: PLC0415

    module = db.get(SubFlowModule, module_id)
    if module is None:
        raise ValueError(f"module {module_id} not found")
    if module.project_id != project_id:
        raise ValueError("module belongs to a different project")
    plan = db.get(TestPlan, plan_id)
    if plan is None:
        raise ValueError(f"plan {plan_id} not found")
    if plan.project_id != project_id:
        raise ValueError("plan belongs to a different project")

    # Resolve / create the parent module TcNode.
    parent: TcNode | None = None
    if parent_module_tc_node_id is not None:
        parent = db.get(TcNode, parent_module_tc_node_id)
        if (
            parent is None
            or parent.plan_id != plan_id
            or parent.kind != "module"
        ):
            raise ValueError(
                "parent_module_tc_node_id must point at a module "
                "in this plan",
            )
    if parent is None:
        # Find the first module in the plan.
        parent = db.execute(
            select(TcNode)
            .where(
                TcNode.plan_id == plan_id,
                TcNode.kind == "module",
            )
            .order_by(TcNode.ordinal),
        ).scalars().first()
    if parent is None:
        # Create an "Imported modules" bucket.
        parent = TcNode(
            project_id=project_id,
            plan_id=plan_id,
            parent_id=None,
            kind="module",
            ordinal=0,
            depth=0,
            path_cached="Imported modules",
            title="Imported modules",
            description_md=(
                "Auto-created container for modules imported from "
                "the project's sub-flow library."
            ),
            selectable_default=True,
            status="approved",
            source_requirement_ids=[],
        )
        db.add(parent)
        db.flush()

    # Next ordinal under the parent.
    next_sm_ordinal = (
        db.execute(
            select(func.max(TcNode.ordinal))
            .where(TcNode.parent_id == parent.id),
        ).scalar() or -1
    ) + 1

    sm_title = module.name
    sm_path = f"{parent.path_cached or parent.title} > {sm_title}"
    submodule = TcNode(
        project_id=project_id,
        plan_id=plan_id,
        parent_id=parent.id,
        kind="submodule",
        ordinal=next_sm_ordinal,
        depth=parent.depth + 1,
        path_cached=sm_path[:2048],
        title=sm_title[:512],
        description_md=(
            module.description
            or f"Imported from module #{module.id} "
            f"(originally from plan #{module.source_plan_id})"
        ),
        selectable_default=True,
        status="approved",
        source_requirement_ids=[],
        # The big one — the imported submodule starts with the
        # frozen path already wired. First run replays.
        frozen_path=module.frozen_segments,
    )
    db.add(submodule)
    db.flush()

    # Re-create child step TcNodes from the module's step snapshots
    # so the test-cases viewer renders the tree the same way.
    step_count = 0
    for step_idx, snap in enumerate(module.step_snapshots or []):
        if not isinstance(snap, dict):
            continue
        title = str(snap.get("title", "(untitled step)"))[:512]
        db.add(TcNode(
            project_id=project_id,
            plan_id=plan_id,
            parent_id=submodule.id,
            kind="step",
            ordinal=step_idx,
            depth=submodule.depth + 1,
            path_cached=f"{sm_path} > {title}"[:2048],
            title=title,
            description_md=snap.get("description_md"),
            action_type=(snap.get("action_type") or None),
            target_hint=(snap.get("target_hint") or None),
            narrative=(snap.get("narrative") or None),
            expected=(snap.get("expected") or None),
            data_needs_json=snap.get("data_needs_json"),
            selectable_default=True,
            status="approved",
            source_requirement_ids=[],
        ))
        step_count += 1

    db.commit()

    return ImportResult(
        new_submodule_id=submodule.id,
        parent_module_id=parent.id,
        steps_created=step_count,
    )


# ── Listing + metadata edit + delete ──────────────────────────────


def list_modules(
    db: "Session",
    *,
    project_id: int,
) -> list[dict[str, Any]]:
    from app.models.sub_flow_module import SubFlowModule  # noqa: PLC0415
    from sqlalchemy import select  # noqa: PLC0415

    rows = list(db.execute(
        select(SubFlowModule)
        .where(SubFlowModule.project_id == project_id)
        .order_by(SubFlowModule.updated_at.desc()),
    ).scalars())
    out: list[dict[str, Any]] = []
    for m in rows:
        seg_count = 0
        step_total = 0
        fs = m.frozen_segments or {}
        if isinstance(fs, dict):
            if fs.get("version") == 2:
                segments = fs.get("segments") or []
                seg_count = len(segments)
                step_total = sum(
                    len(s.get("steps") or [])
                    for s in segments
                    if isinstance(s, dict)
                )
            else:
                step_total = len(fs.get("steps") or [])
        out.append({
            "id": m.id,
            "project_id": m.project_id,
            "name": m.name,
            "description": m.description,
            "target_url_pattern": m.target_url_pattern,
            "tags": m.tags or [],
            "segments": seg_count,
            "steps": step_total,
            "step_snapshot_count": len(m.step_snapshots or []),
            "source_plan_id": m.source_plan_id,
            "source_submodule_tc_node_id": (
                m.source_submodule_tc_node_id
            ),
            "source_run_id": m.source_run_id,
            "frozen_path_version": (
                fs.get("version") if isinstance(fs, dict) else 1
            ),
            "created_at": (
                m.created_at.isoformat() if m.created_at else None
            ),
            "updated_at": (
                m.updated_at.isoformat() if m.updated_at else None
            ),
        })
    return out


def get_module(
    db: "Session",
    *,
    project_id: int,
    module_id: int,
) -> dict[str, Any] | None:
    from app.models.sub_flow_module import SubFlowModule  # noqa: PLC0415

    m = db.get(SubFlowModule, module_id)
    if m is None or m.project_id != project_id:
        return None
    return {
        "id": m.id,
        "project_id": m.project_id,
        "name": m.name,
        "description": m.description,
        "target_url_pattern": m.target_url_pattern,
        "tags": m.tags or [],
        "frozen_segments": m.frozen_segments,
        "step_snapshots": m.step_snapshots,
        "source_plan_id": m.source_plan_id,
        "source_submodule_tc_node_id": m.source_submodule_tc_node_id,
        "source_run_id": m.source_run_id,
        "created_at": (
            m.created_at.isoformat() if m.created_at else None
        ),
        "updated_at": (
            m.updated_at.isoformat() if m.updated_at else None
        ),
    }


def update_module_metadata(
    db: "Session",
    *,
    project_id: int,
    module_id: int,
    name: str | None = None,
    description: str | None = None,
    target_url_pattern: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    from app.models.sub_flow_module import SubFlowModule  # noqa: PLC0415

    m = db.get(SubFlowModule, module_id)
    if m is None or m.project_id != project_id:
        return False
    if name is not None:
        m.name = name.strip()[:255]
    if description is not None:
        m.description = description.strip() or None
    if target_url_pattern is not None:
        m.target_url_pattern = target_url_pattern.strip() or None
    if tags is not None:
        m.tags = list(tags) or None
    db.commit()
    return True


def delete_module(
    db: "Session",
    *,
    project_id: int,
    module_id: int,
) -> bool:
    from app.models.sub_flow_module import SubFlowModule  # noqa: PLC0415

    m = db.get(SubFlowModule, module_id)
    if m is None or m.project_id != project_id:
        return False
    db.delete(m)
    db.commit()
    return True
