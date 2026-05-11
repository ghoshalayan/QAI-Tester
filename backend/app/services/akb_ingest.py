"""Production-α.3 — AKB ingestion hooks.

Bridges existing artifacts → ``app_knowledge`` table so the runtime
agent's RAG sees them. Two pipelines:

1. ``ingest_plan_documents_to_akb(db, plan)`` — on plan save, walk
   the plan's linked BRD/FRD documents and copy their chunks into
   AKB with ``kind="brd_chunk"``, scoped to ``plan.target_url``.
   Idempotent (deduplicated on (pattern, kind, content) by the
   AKB write_chunk helper).

2. ``ingest_frozen_path_summary(db, run_id, tc_node)`` — γ.2 hook
   called when a submodule's frozen path is first captured. Writes
   a one-paragraph human-readable summary so future runs that
   query AKB for "how do I do X on this app" get back the proven
   working flow.

Why eager (on-save) vs lazy (on-first-query):
- Eager: pays the embed cost ONCE when the user explicitly says
  "this BRD is for this app". Subsequent runs see knowledge
  immediately. Cost is bounded — embed is local CPU.
- Lazy would mean "first run on a new app pays the embed latency"
  which is exactly when the user is watching the live feed.

Re-ingestion behavior: when a plan's target_url changes, old
brd_chunk rows under the previous pattern stay (other plans on
that URL might still reference them). The new URL gets a fresh
ingest. Use the AKB browser's "clear app knowledge" button to
force-reset a target_url's chunks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.models.test_plan import TestPlan
    from app.models.tc_node import TcNode

logger = logging.getLogger(__name__)


def ingest_plan_documents_to_akb(
    db: "Session",
    plan: "TestPlan",
) -> int:
    """Walk the plan's linked BRD/FRD documents and ingest their
    chunks into AKB under ``plan.target_url``.

    Returns the number of chunks newly added (excludes deduplicated
    repeat saves). Skips silently when:
    - ``plan.target_url`` is empty (nothing to scope to)
    - The plan has no linked documents
    - A linked doc has no chunks (still parsing / failed to ingest)
    """
    target_url = (plan.target_url or "").strip()
    if not target_url:
        logger.debug(
            "AKB ingest skipped: plan %s has no target_url", plan.id,
        )
        return 0

    from app.models.document import Document, DocumentChunk  # noqa: PLC0415
    from app.services.akb import write_chunk  # noqa: PLC0415

    # Pull every chunk from every linked document.
    linked = list(getattr(plan, "linked_docs", []) or [])
    if not linked:
        return 0

    total_added = 0
    for link in linked:
        doc_id = getattr(link, "document_id", None)
        if doc_id is None:
            continue
        doc = db.get(Document, doc_id)
        if doc is None:
            continue
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc_id)
            .order_by(DocumentChunk.ordinal)
            .all()
        )
        if not chunks:
            continue
        for ch in chunks:
            text = (ch.text or "").strip()
            if not text:
                continue
            # Light annotation so the agent's prompt block knows
            # "this came from a BRD section called X" — better
            # signal than raw chunks.
            heading = (ch.heading_path or "").strip()
            content = (
                f"[Section: {heading}] {text}" if heading else text
            )
            tags = ["brd"]
            if doc.kind:
                tags.append(str(doc.kind).lower())
            row_id = write_chunk(
                db,
                target_url_pattern=target_url,
                kind="brd_chunk",
                content=content[:2000],  # bound chunk size
                tags=tags,
                source_doc_id=doc.id,
            )
            if row_id is not None:
                total_added += 1

    logger.info(
        "AKB ingest: plan %s -> %d chunk(s) under target %r",
        plan.id, total_added, target_url,
    )
    return total_added


def ingest_frozen_path_summary(
    db: "Session",
    *,
    target_url: str,
    submodule_title: str,
    frozen_path: dict,
    source_run_id: int | None = None,
) -> int | None:
    """γ.2 hook — write a frozen path's one-paragraph summary to AKB.

    Future runs querying "how do I do <similar task> on this app"
    get a proven working sequence as a hint. The summary stays
    text-only (no selectors); the actual replayable path lives on
    ``tc_nodes.frozen_path``.
    """
    from app.services.akb import write_chunk  # noqa: PLC0415

    steps = frozen_path.get("steps") if isinstance(frozen_path, dict) else None
    if not isinstance(steps, list) or not steps:
        return None
    parts = [
        f"On {target_url}, the working flow for "
        f"\"{submodule_title}\" is:",
    ]
    for i, step in enumerate(steps[:25], start=1):
        if not isinstance(step, dict):
            continue
        tool = step.get("tool", "?")
        args = step.get("args") or {}
        hint = (
            args.get("target_hint")
            or args.get("url")
            or args.get("value")
            or args.get("key")
            or ""
        )
        # Trim long values for prompt economy.
        if isinstance(hint, str) and len(hint) > 80:
            hint = hint[:77] + "..."
        parts.append(f"  {i}. {tool}({hint!r})" if hint else f"  {i}. {tool}")
    content = "\n".join(parts)
    return write_chunk(
        db,
        target_url_pattern=target_url,
        kind="frozen_path_summary",
        content=content,
        tags=["frozen", "flow"],
        source_run_id=source_run_id,
    )
