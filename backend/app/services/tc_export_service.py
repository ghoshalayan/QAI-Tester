"""Export selected TC nodes as JSON (re-importable) or Markdown (review).

Two consumers:
- ``GET /tc-nodes/export?format=json`` — full structured tree
- ``GET /tc-nodes/export?format=md``   — human-readable nested headers

Selection semantics
-------------------
Three modes, evaluated in this order:

1. ``node_ids`` provided  → export those exact nodes plus their ancestors
   (so the Markdown / JSON keeps module → submodule → step context).
2. ``selected_only=True`` → all step rows with ``selectable_default=True``
   plus their ancestor chain. Matches what would run on the next execute.
3. neither              → the entire plan tree.

The export is read-only and cheap; we don't bother caching.
"""

from __future__ import annotations

import json as _json
import logging
from collections import defaultdict
from io import StringIO
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.tc_node import TcNode
from app.models.test_plan import TestPlan

logger = logging.getLogger(__name__)


# ── Node selection ───────────────────────────────────────────────


def _select_export_nodes(
    db: Session,
    *,
    plan_id: int,
    node_ids: list[int] | None,
    selected_only: bool,
) -> list[TcNode]:
    """Apply the selection mode and return the nodes to export, including
    every ancestor of every requested leaf so the output preserves the
    Module → Submodule → Step structure.
    """
    stmt = (
        select(TcNode)
        .where(TcNode.plan_id == plan_id)
        .order_by(TcNode.depth, TcNode.parent_id, TcNode.ordinal)
    )
    all_nodes = list(db.scalars(stmt))
    by_id = {n.id: n for n in all_nodes}

    if node_ids:
        # Walk up to gather ancestors; preserves tree completeness.
        wanted: set[int] = set()
        for nid in node_ids:
            cur = by_id.get(nid)
            while cur is not None:
                if cur.id in wanted:
                    break
                wanted.add(cur.id)
                cur = by_id.get(cur.parent_id) if cur.parent_id else None
        return [n for n in all_nodes if n.id in wanted]

    if selected_only:
        # All steps the executor would run, plus ancestors for context.
        wanted = set()
        for n in all_nodes:
            if n.kind == "step" and n.selectable_default:
                cur = n
                while cur is not None:
                    if cur.id in wanted:
                        break
                    wanted.add(cur.id)
                    cur = (
                        by_id.get(cur.parent_id) if cur.parent_id else None
                    )
        return [n for n in all_nodes if n.id in wanted]

    # No filter — return everything for the plan
    return all_nodes


# ── JSON export ──────────────────────────────────────────────────


def export_to_json(
    db: Session,
    plan: TestPlan,
    *,
    nodes: list[TcNode],
) -> str:
    """Serialize the plan + filtered tree to a re-importable JSON string.

    Output shape::

        {
          "plan": {id, name, target_url, scope, description},
          "exported_at": "...",
          "node_count": N,
          "modules": [
            {id, title, ordinal, depth, kind, status, selectable_default,
             description_md, source_requirement_ids,
             children: [submodule, ...]}
          ]
        }

    Round-trippable as long as a future ``POST /tc-nodes/import`` reuses
    these fields. (Future scope.)
    """
    children_by_parent: dict[int | None, list[TcNode]] = defaultdict(list)
    for n in nodes:
        children_by_parent[n.parent_id].append(n)
    for sibs in children_by_parent.values():
        sibs.sort(key=lambda n: n.ordinal)

    def serialize(node: TcNode) -> dict[str, Any]:
        out = {
            "id": node.id,
            "kind": node.kind,
            "ordinal": node.ordinal,
            "depth": node.depth,
            "title": node.title,
            "description_md": node.description_md,
            "status": node.status,
            "selectable_default": node.selectable_default,
            "source_requirement_ids": list(
                node.source_requirement_ids or [],
            ),
        }
        if node.kind == "step":
            out["action_type"] = node.action_type
            out["target_hint"] = node.target_hint
            out["narrative"] = node.narrative
            out["expected"] = node.expected
            out["data_needs_json"] = list(node.data_needs_json or [])
        children = children_by_parent.get(node.id, [])
        if children:
            out["children"] = [serialize(c) for c in children]
        return out

    from datetime import datetime, timezone

    payload = {
        "plan": {
            "id": plan.id,
            "name": plan.name,
            "target_url": plan.target_url or "",
            "scope": list(plan.scope or []),
            "description": plan.description or "",
        },
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "node_count": len(nodes),
        "modules": [serialize(r) for r in children_by_parent.get(None, [])],
    }
    return _json.dumps(payload, indent=2, ensure_ascii=False)


# ── Markdown export ──────────────────────────────────────────────


def export_to_markdown(
    db: Session,
    plan: TestPlan,
    *,
    nodes: list[TcNode],
) -> str:
    """Hierarchical Markdown — easy to paste into docs or review.

    Structure::

        # Plan: <name>
        Target URL: <url>
        Scope: <s1, s2, ...>

        ## <Module title>
        ### <Submodule title>
        #### <Step ordinal>. <Step title>

        - Action: click
        - Target hint: ``button[data-testid='go']``
        - Narrative: Click the Go button
        - Expected: Submit succeeds
        - Data needs:
          - **credentials** — admin password
    """
    children_by_parent: dict[int | None, list[TcNode]] = defaultdict(list)
    for n in nodes:
        children_by_parent[n.parent_id].append(n)
    for sibs in children_by_parent.values():
        sibs.sort(key=lambda n: n.ordinal)

    out = StringIO()
    w = out.write

    w(f"# Plan: {plan.name}\n\n")
    if plan.target_url:
        w(f"**Target URL:** {plan.target_url}\n\n")
    if plan.scope:
        w(f"**Scope:** {', '.join(plan.scope)}\n\n")
    if plan.description:
        w(f"{plan.description}\n\n")
    w(f"_Exporting {len(nodes)} node(s)._\n\n---\n\n")

    def _esc(text: str | None) -> str:
        return (text or "").replace("\n", " ").strip()

    def render_step(node: TcNode) -> None:
        w(f"#### {node.ordinal + 1}. {node.title}\n\n")
        if node.action_type:
            w(f"- **Action:** `{node.action_type}`\n")
        if node.target_hint:
            w(f"- **Target hint:** `{_esc(node.target_hint)}`\n")
        if node.narrative:
            w(f"- **Narrative:** {_esc(node.narrative)}\n")
        if node.expected:
            w(f"- **Expected:** {_esc(node.expected)}\n")
        if node.data_needs_json:
            w("- **Data needs:**\n")
            for dn in node.data_needs_json:
                kind = str(dn.get("kind", "")).strip() or "data"
                notes = str(dn.get("notes", "")).strip()
                w(f"  - **{kind}** — {notes or '_(no notes)_'}\n")
        if node.description_md:
            w(f"\n{node.description_md.strip()}\n")
        sel = (
            "[x] selected" if node.selectable_default else "[ ] not selected"
        )
        w(f"\n_{sel} · status: `{node.status}`_\n\n")

    def render_submodule(node: TcNode) -> None:
        w(f"### {node.title}\n\n")
        if node.description_md:
            w(f"{node.description_md.strip()}\n\n")
        for child in children_by_parent.get(node.id, []):
            if child.kind == "step":
                render_step(child)
            else:
                render_submodule(child)  # tolerate deeper nesting

    def render_module(node: TcNode) -> None:
        w(f"## {node.title}\n\n")
        if node.description_md:
            w(f"{node.description_md.strip()}\n\n")
        for child in children_by_parent.get(node.id, []):
            if child.kind == "submodule":
                render_submodule(child)
            elif child.kind == "step":
                render_step(child)

    for root in children_by_parent.get(None, []):
        render_module(root)

    return out.getvalue()
