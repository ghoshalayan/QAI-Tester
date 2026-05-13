"""TC nodes router — list (tree-shape) / get / PATCH / DELETE / bulk-update.

Mounted at ``/api/projects/{project_id}/plans/{plan_id}/tc-nodes``.

Tree shape
----------
``GET /tc-nodes`` returns ``list[TcNodeTreeRead]`` — one entry per root module,
with ``children`` populated recursively. The frontend renders this with a
single recursive component.

``GET /tc-nodes/{node_id}`` returns the subtree rooted at that node as a
single ``TcNodeTreeRead``.

``DELETE`` cascades — the FK self-FK ``parent_id ON DELETE CASCADE`` removes
the entire subtree atomically.

``PATCH`` with a title change recomputes ``path_cached`` for the node and
all its descendants so search/citation strings stay correct.

Route ordering note
-------------------
Literal ``/bulk-update`` is declared before parametric ``/{node_id}``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.project import Project
from app.models.tc_node import TcNode
from app.models.test_plan import TestPlan
from app.schemas.tc_node import (
    TcNodeBulkUpdateRequest,
    TcNodeBulkUpdateResponse,
    TcNodeRead,
    TcNodeTreeRead,
    TcNodeUpdate,
)
from app.services.tc_export_service import (
    _select_export_nodes,
    export_to_json,
    export_to_markdown,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/projects/{project_id}/plans/{plan_id}/tc-nodes",
    tags=["TC Nodes"],
)


# ── Helpers ───────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _require_plan(
    db: Session, project_id: int, plan_id: int,
) -> TestPlan:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    plan = db.get(TestPlan, plan_id)
    if not plan or plan.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Plan not found")
    return plan


def _require_node(
    db: Session, project_id: int, plan_id: int, node_id: int,
) -> TcNode:
    _require_plan(db, project_id, plan_id)
    node = db.get(TcNode, node_id)
    if (
        not node
        or node.plan_id != plan_id
        or node.project_id != project_id
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "TC node not found")
    return node


def _node_to_dict(node: TcNode) -> dict[str, Any]:
    """Flat dict matching ``TcNodeRead`` (no children)."""
    fp = node.frozen_path
    has_frozen = isinstance(fp, dict) and bool(
        fp.get("segments") or fp.get("steps"),
    )
    fp_version = (
        fp.get("version") if isinstance(fp, dict) and has_frozen
        else None
    )
    return {
        "id": node.id,
        "project_id": node.project_id,
        "plan_id": node.plan_id,
        "parent_id": node.parent_id,
        "kind": node.kind,
        "ordinal": node.ordinal,
        "depth": node.depth,
        "path_cached": node.path_cached,
        "title": node.title,
        "description_md": node.description_md,
        "action_type": node.action_type,
        "target_hint": node.target_hint,
        "narrative": node.narrative,
        "expected": node.expected,
        "data_needs_json": node.data_needs_json,
        "selectable_default": node.selectable_default,
        "status": node.status,
        "source_requirement_ids": list(node.source_requirement_ids or []),
        "has_frozen_path": has_frozen,
        "frozen_path_version": fp_version,
        "created_at": node.created_at,
        "updated_at": node.updated_at,
        "reviewed_at": node.reviewed_at,
    }


def _build_tree(
    nodes: list[TcNode], from_root_id: int | None = None,
) -> list[dict[str, Any]]:
    """Build tree-shape list from a flat ordered set of nodes.

    If ``from_root_id`` is given → return only that subtree (single-element list).
    Otherwise → return all roots (``parent_id IS NULL``).
    """
    children_by_parent: dict[int | None, list[TcNode]] = defaultdict(list)
    by_id: dict[int, TcNode] = {}
    for n in nodes:
        children_by_parent[n.parent_id].append(n)
        by_id[n.id] = n

    # Stable sibling order
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda n: n.ordinal)

    def to_dict_with_children(n: TcNode) -> dict[str, Any]:
        d = _node_to_dict(n)
        d["children"] = [
            to_dict_with_children(c) for c in children_by_parent.get(n.id, [])
        ]
        return d

    if from_root_id is not None:
        root = by_id.get(from_root_id)
        if not root:
            return []
        return [to_dict_with_children(root)]

    roots = children_by_parent.get(None, [])
    return [to_dict_with_children(r) for r in roots]


def _rebuild_path_cached_subtree(db: Session, root: TcNode) -> None:
    """After a title change on ``root``, refresh ``path_cached`` for the node
    and every descendant so future citations remain accurate."""
    # Compute root's path: ancestors' titles + root.title
    ancestor_titles: list[str] = []
    cursor: TcNode | None = root
    while cursor and cursor.parent_id is not None:
        parent = db.get(TcNode, cursor.parent_id)
        if not parent:
            break
        ancestor_titles.insert(0, parent.title)
        cursor = parent

    root_path = " > ".join([*ancestor_titles, root.title])[:2048]
    root.path_cached = root_path

    # DFS through descendants, updating each
    stack: list[tuple[int, str]] = [(root.id, root_path)]
    while stack:
        parent_id, parent_path = stack.pop()
        children = list(
            db.scalars(
                select(TcNode)
                .where(TcNode.parent_id == parent_id)
                .order_by(TcNode.ordinal),
            ),
        )
        for c in children:
            c.path_cached = f"{parent_path} > {c.title}"[:2048]
            stack.append((c.id, c.path_cached))

    db.flush()


# ── Bulk update (literal — before /{node_id}) ─────────────────────


@router.post("/bulk-update", response_model=TcNodeBulkUpdateResponse)
def bulk_update(
    project_id: int,
    plan_id: int,
    payload: TcNodeBulkUpdateRequest,
    db: Session = Depends(get_db),
):
    """Apply approve/archive/delete to a set of nodes selected by
    ``node_ids`` and/or ``filter_status`` and/or ``filter_kind``.

    Multiple filters are AND-combined. At least one filter is required.
    """
    _require_plan(db, project_id, plan_id)

    if (
        not payload.node_ids
        and payload.filter_status is None
        and payload.filter_kind is None
    ):
        raise HTTPException(
            400,
            "Provide at least one filter: node_ids, filter_status, or filter_kind",
        )

    stmt = select(TcNode).where(TcNode.plan_id == plan_id)
    if payload.node_ids:
        stmt = stmt.where(TcNode.id.in_(payload.node_ids))
    if payload.filter_status is not None:
        stmt = stmt.where(TcNode.status == payload.filter_status)
    if payload.filter_kind is not None:
        stmt = stmt.where(TcNode.kind == payload.filter_kind)

    rows = list(db.scalars(stmt))
    if not rows:
        return TcNodeBulkUpdateResponse(
            affected=0, affected_ids=[], action=payload.action,
        )

    affected_ids = [r.id for r in rows]
    now = _utcnow()

    if payload.action == "approve":
        for r in rows:
            r.status = "approved"
            r.reviewed_at = now
    elif payload.action == "archive":
        for r in rows:
            r.status = "archived"
            r.reviewed_at = now
    elif payload.action == "delete":
        # Order doesn't matter — CASCADE handles descendants of any deleted node
        for r in rows:
            db.delete(r)
    elif payload.action == "select":
        for r in rows:
            r.selectable_default = True
    elif payload.action == "deselect":
        for r in rows:
            r.selectable_default = False

    db.commit()

    logger.info(
        "Bulk %s applied to %d TC nodes in plan %s",
        payload.action,
        len(rows),
        plan_id,
    )
    return TcNodeBulkUpdateResponse(
        affected=len(rows),
        affected_ids=affected_ids if payload.action != "delete" else [],
        action=payload.action,
    )


# ── Export (literal — before /{node_id}) ──────────────────────────


@router.get("/export")
def export_nodes(
    project_id: int,
    plan_id: int,
    fmt: str = Query(default="json", alias="format", pattern="^(json|md)$"),
    node_ids: str | None = Query(
        default=None,
        description=(
            "Optional comma-separated list of node ids. When provided, only "
            "those nodes (plus their ancestor chain for context) are exported."
        ),
    ),
    selected_only: bool = Query(
        default=False,
        description=(
            "When true, export step rows with selectable_default=True plus "
            "their ancestors. Equivalent to 'what would run on the next "
            "execute'. Ignored when node_ids is set."
        ),
    ),
    db: Session = Depends(get_db),
):
    """Download the plan's TC tree as ``json`` (re-importable) or
    ``md`` (human-readable Markdown).

    Selection:
    - ``node_ids=1,2,3`` → those nodes + their ancestors
    - ``selected_only=true`` → all selectable_default steps + ancestors
    - neither → entire tree

    Returned with ``Content-Disposition: attachment`` so a frontend ``<a
    href download>`` triggers a download. Empty selections still return
    a valid empty document.
    """
    plan = _require_plan(db, project_id, plan_id)

    parsed_ids: list[int] | None = None
    if node_ids:
        try:
            parsed_ids = [
                int(x) for x in node_ids.split(",") if x.strip()
            ]
        except ValueError:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "node_ids must be a comma-separated list of integers",
            )

    nodes = _select_export_nodes(
        db,
        plan_id=plan_id,
        node_ids=parsed_ids,
        selected_only=selected_only,
    )

    safe_name = "".join(
        c if c.isalnum() or c in "-_." else "_" for c in (plan.name or "plan")
    )[:60] or "plan"

    if fmt == "md":
        body = export_to_markdown(db, plan, nodes=nodes)
        filename = f"{safe_name}-test-cases.md"
        media_type = "text/markdown; charset=utf-8"
    else:
        body = export_to_json(db, plan, nodes=nodes)
        filename = f"{safe_name}-test-cases.json"
        media_type = "application/json"

    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


# ── List (tree-shaped) ────────────────────────────────────────────


@router.get("", response_model=list[TcNodeTreeRead])
def list_tree(
    project_id: int,
    plan_id: int,
    db: Session = Depends(get_db),
):
    """Return the entire TC tree for the plan, rooted at modules."""
    _require_plan(db, project_id, plan_id)

    stmt = (
        select(TcNode)
        .where(TcNode.plan_id == plan_id)
        .order_by(TcNode.depth, TcNode.parent_id, TcNode.ordinal)
    )
    nodes = list(db.scalars(stmt))
    return _build_tree(nodes, from_root_id=None)


# ── Parametric routes ────────────────────────────────────────────


@router.get("/{node_id}", response_model=TcNodeTreeRead)
def get_node(
    project_id: int,
    plan_id: int,
    node_id: int,
    db: Session = Depends(get_db),
):
    """Return the subtree rooted at ``node_id``."""
    _require_node(db, project_id, plan_id, node_id)

    stmt = (
        select(TcNode)
        .where(TcNode.plan_id == plan_id)
        .order_by(TcNode.depth, TcNode.parent_id, TcNode.ordinal)
    )
    nodes = list(db.scalars(stmt))
    trees = _build_tree(nodes, from_root_id=node_id)
    if not trees:
        raise HTTPException(404, "Node not found")
    return trees[0]


@router.patch("/{node_id}", response_model=TcNodeRead)
def update_node(
    project_id: int,
    plan_id: int,
    node_id: int,
    payload: TcNodeUpdate,
    db: Session = Depends(get_db),
):
    """Partial edit. Title changes also recompute ``path_cached`` for the
    node and every descendant so citations stay accurate.

    Status changes stamp ``reviewed_at``. Body edits don't auto-demote
    (TC nodes have no ``edited`` status — week 3's pattern doesn't apply)."""
    node = _require_node(db, project_id, plan_id, node_id)

    title_changed = False
    if payload.title is not None:
        new_title = payload.title.strip()
        if new_title != node.title:
            node.title = new_title
            title_changed = True

    if payload.description_md is not None:
        node.description_md = payload.description_md or None

    # Step-only fields — schema allows setting on any kind; non-step rows
    # just won't have them rendered by the executor.
    if payload.action_type is not None:
        node.action_type = payload.action_type or None
    if payload.target_hint is not None:
        node.target_hint = payload.target_hint or None
    if payload.narrative is not None:
        node.narrative = payload.narrative or None
    if payload.expected is not None:
        node.expected = payload.expected or None
    if payload.data_needs_json is not None:
        node.data_needs_json = payload.data_needs_json

    if payload.selectable_default is not None:
        node.selectable_default = payload.selectable_default

    if payload.status is not None and payload.status != node.status:
        node.status = payload.status
        node.reviewed_at = _utcnow()

    if title_changed:
        _rebuild_path_cached_subtree(db, node)

    db.commit()
    db.refresh(node)
    return _node_to_dict(node)


@router.delete("/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_node(
    project_id: int,
    plan_id: int,
    node_id: int,
    db: Session = Depends(get_db),
):
    """Hard delete. CASCADE removes the entire subtree."""
    node = _require_node(db, project_id, plan_id, node_id)
    db.delete(node)
    db.commit()
