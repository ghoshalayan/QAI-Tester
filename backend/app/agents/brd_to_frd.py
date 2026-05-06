"""BRD → FRD synthesis orchestrator (pure function).

Given one or more BRD documents in a project, asks the configured LLM to
derive functional requirements with traceability back to source BRD chunks.
Persists each as a ``Requirement`` row with ``status='proposed'``.

Pipeline
--------
    load_chunks → build_prompt → chat_structured → assign_codes → persist

Cancellation
------------
A caller-supplied ``is_cancelled`` callback is polled at safe boundaries
(before LLM call; before persist). On cancel, raises :class:`AgentCancelled`
which the runtime catches and marks ``agent_run.status = 'cancelled'``.

Multi-BRD synthesis
-------------------
When ``source_document_ids`` lists multiple BRDs, chunks are interleaved by
``(document_id, ordinal)`` and the LLM sees them as one numbered list.
Each generated requirement's ``source_document_id`` is set to the BRD that
contributed the most chunks among those it cites (majority vote); ties go
to the lowest ``document_id``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.base import ChatMessage, LLMProvider
from app.models.document import Document, DocumentChunk
from app.models.requirement import Requirement

logger = logging.getLogger(__name__)


class AgentCancelled(Exception):
    """Raised when ``is_cancelled()`` returned True at a checkpoint."""


@dataclass
class SynthesisResult:
    generated_count: int
    requirement_ids: list[int]
    chunks_seen: int
    truncated: bool                   # True if cap_chunks limit was hit
    input_tokens: int | None
    output_tokens: int | None


# ── Prompt + schema ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior business analyst converting a BRD (business requirements document) into an FRD (functional requirements document).

For each functional requirement you derive, output:
- title          : 5-15 words, action-oriented (e.g. "Validate email format on signup form")
- body_md        : Markdown body — concise acceptance criteria a QA engineer can verify
- rationale      : 1-2 sentences: why this FRD follows from the BRD chunks you cite
- source_chunk_indices : array of integers — the [N] indices of BRD chunks that support this FRD
- confidence     : 0.0 (weak/inferred) to 1.0 (directly stated in the BRD)

Quality bar:
- Specific and testable. A QA engineer should be able to confirm pass/fail by inspection.
- Atomic — one requirement = one observable behavior. Split compound requirements.
- Restate functionally; don't echo BRD text verbatim.
- Don't invent requirements not implied by the BRD chunks.
- If a BRD chunk is ambiguous, lower the confidence rather than guess.
- Aim for thorough coverage — produce as many distinct, atomic FRDs as the BRD warrants."""

# JSON schema is OpenAI-strict-friendly (additionalProperties:false everywhere,
# every property in required) so it works on Gemini, OpenAI, and compat alike.
FRD_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body_md": {"type": "string"},
                    "rationale": {"type": "string"},
                    "source_chunk_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "confidence": {"type": "number"},
                },
                "required": [
                    "title",
                    "body_md",
                    "rationale",
                    "source_chunk_indices",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["requirements"],
    "additionalProperties": False,
}


# ── Helpers ───────────────────────────────────────────────────────


def _check_cancel(
    is_cancelled: Callable[[], bool] | None, where: str,
) -> None:
    if is_cancelled and is_cancelled():
        raise AgentCancelled(f"Cancelled at: {where}")


def _emit(
    emit_event: Callable[[str, dict], None] | None,
    event_type: str,
    data: dict,
) -> None:
    if emit_event:
        try:
            emit_event(event_type, data)
        except Exception as e:  # never let event-emit kill the agent
            logger.warning("emit_event raised, continuing: %s", e)


def _next_frd_code_number(db: Session, project_id: int) -> int:
    """Next FRD-N number for the project (1 if none exist)."""
    rows = list(
        db.scalars(
            select(Requirement.code).where(
                Requirement.project_id == project_id,
                Requirement.kind == "FRD",
            ),
        ),
    )
    max_num = 0
    for code in rows:
        m = re.match(r"^FRD-(\d+)$", code or "")
        if m:
            n = int(m.group(1))
            if n > max_num:
                max_num = n
    return max_num + 1


def _build_user_prompt(chunks: list[DocumentChunk]) -> str:
    lines = [
        "BRD CHUNKS — each prefixed with its index in [N] and its heading path:",
        "",
    ]
    for i, c in enumerate(chunks):
        heading = c.heading_path or "(no heading)"
        lines.append(f"[{i}] {heading}")
        lines.append(c.text.strip())
        lines.append("")
    lines.append(
        "Generate functional requirements as JSON. Cite the BRD chunk(s) "
        "that motivated each FRD via source_chunk_indices.",
    )
    return "\n".join(lines)


def _validate_source_docs(
    db: Session, project_id: int, doc_ids: list[int],
) -> list[Document]:
    if not doc_ids:
        raise ValueError("source_document_ids must not be empty")
    unique = list(dict.fromkeys(doc_ids))
    docs = list(
        db.scalars(select(Document).where(Document.id.in_(unique))),
    )
    by_id = {d.id: d for d in docs}
    missing = [i for i in unique if i not in by_id]
    if missing:
        raise ValueError(f"Documents not found: {missing}")

    bad_project = [d.id for d in docs if d.project_id != project_id]
    if bad_project:
        raise ValueError(
            f"Documents belong to a different project: {bad_project}",
        )

    bad_kind = [d.id for d in docs if d.kind != "BRD"]
    if bad_kind:
        raise ValueError(
            f"Only BRD documents can be source for BRD→FRD synthesis. "
            f"Wrong kind: {bad_kind}",
        )

    not_parsed = [d.id for d in docs if d.status != "parsed"]
    if not_parsed:
        raise ValueError(
            f"Documents must be parsed before synthesis: {not_parsed}",
        )
    return docs


def _load_chunks(
    db: Session, doc_ids: list[int], cap_chunks: int,
) -> tuple[list[DocumentChunk], bool]:
    """Return (chunks, truncated). Ordered by (document_id, ordinal) for determinism."""
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id.in_(doc_ids))
        .order_by(DocumentChunk.document_id, DocumentChunk.ordinal)
    )
    all_chunks = list(db.scalars(stmt))
    if len(all_chunks) > cap_chunks:
        return all_chunks[:cap_chunks], True
    return all_chunks, False


def _primary_doc_id(
    chunks: list[DocumentChunk], indices: list[int],
) -> int | None:
    """Pick the document_id that contributed the most cited chunks."""
    valid = [
        chunks[i].document_id
        for i in indices
        if isinstance(i, int) and 0 <= i < len(chunks)
    ]
    if not valid:
        return None
    counter = Counter(valid)
    most_common = counter.most_common()
    # Ties → smallest doc_id for determinism
    top_count = most_common[0][1]
    candidates = [d for d, c in most_common if c == top_count]
    return min(candidates)


def _persist_requirements(
    db: Session,
    project_id: int,
    chunks: list[DocumentChunk],
    parsed_items: list[dict[str, Any]],
    start_code_num: int,
) -> list[Requirement]:
    requirements: list[Requirement] = []

    for offset, item in enumerate(parsed_items):
        title = (item.get("title") or "").strip()
        body_md = (item.get("body_md") or "").strip()
        rationale = (item.get("rationale") or "").strip() or None
        if not title or not body_md:
            logger.warning(
                "Skipping requirement with empty title or body: %r", item,
            )
            continue

        raw_indices = item.get("source_chunk_indices") or []
        chunk_ids = [
            chunks[i].id
            for i in raw_indices
            if isinstance(i, int) and 0 <= i < len(chunks)
        ]
        primary_doc_id = _primary_doc_id(chunks, raw_indices)

        try:
            confidence = float(item.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = None

        req = Requirement(
            project_id=project_id,
            source_document_id=primary_doc_id,
            source_chunk_ids=chunk_ids,
            kind="FRD",
            code=f"FRD-{start_code_num + offset}",
            title=title[:512],
            body_md=body_md,
            rationale=rationale,
            confidence=confidence,
            status="proposed",
        )
        db.add(req)
        requirements.append(req)

    db.flush()  # populate ids without committing yet
    return requirements


# ── Orchestrator entry point ──────────────────────────────────────


def synthesize_frd(
    db: Session,
    provider: LLMProvider,
    project_id: int,
    source_document_ids: list[int],
    *,
    cap_chunks: int = 50,
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> SynthesisResult:
    """Synthesize FRD requirements from one or more BRD documents.

    Args:
        db: SQLAlchemy session. Caller commits/rolls back.
        provider: LLMProvider supporting ``chat_structured``.
        project_id: Owns the resulting requirements.
        source_document_ids: BRD docs to read from. Must belong to ``project_id``,
            ``kind == 'BRD'``, ``status == 'parsed'``.
        cap_chunks: Max chunks to send the LLM in one call. Default 50.
        emit_event: Optional callback ``(event_type, data) -> None`` for SSE.
        is_cancelled: Optional callback returning True to cancel.

    Returns:
        SynthesisResult with generated count, ids, token usage, and truncation flag.

    Raises:
        ValueError: invalid inputs (no docs, wrong project/kind/status, etc.)
        AgentCancelled: caller flagged cancellation between phases.
        RuntimeError: LLM returned invalid JSON despite the schema.
    """
    # ── Validate ────────────────────────────────────────────────
    _emit(emit_event, "phase", {"phase": "validating", "message": "Validating source documents"})
    docs = _validate_source_docs(db, project_id, source_document_ids)

    # ── Load chunks ─────────────────────────────────────────────
    _emit(
        emit_event,
        "phase",
        {"phase": "loading", "message": f"Loading chunks from {len(docs)} BRD(s)"},
    )
    _check_cancel(is_cancelled, "before loading chunks")

    chunks, truncated = _load_chunks(
        db, [d.id for d in docs], cap_chunks=cap_chunks,
    )
    if not chunks:
        raise ValueError(
            "No chunks found in the selected BRDs. Are they ingested?",
        )

    _emit(
        emit_event,
        "phase",
        {
            "phase": "loading",
            "message": (
                f"Loaded {len(chunks)} chunk(s)"
                + (f" (capped at {cap_chunks})" if truncated else "")
            ),
            "chunks_seen": len(chunks),
            "truncated": truncated,
        },
    )

    # ── LLM call ────────────────────────────────────────────────
    _check_cancel(is_cancelled, "before LLM call")
    _emit(
        emit_event,
        "phase",
        {
            "phase": "calling_llm",
            "message": (
                f"Calling {provider.provider_id} ({provider.model}) — "
                f"may take 10-60s depending on chunk count"
            ),
        },
    )

    user_prompt = _build_user_prompt(chunks)
    try:
        chat_result = provider.chat_structured(
            messages=[
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_prompt),
            ],
            schema=FRD_SYNTHESIS_SCHEMA,
            schema_name="frd_synthesis",
            temperature=0.2,  # low temperature for stable, reproducible output
            max_output_tokens=4096,
        )
    except Exception as e:
        # Surface as a regular error — runtime decides the agent_run status
        raise RuntimeError(
            f"LLM call failed: {type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = chat_result.parsed or {}
    items = parsed.get("requirements") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise RuntimeError(
            f"LLM returned an unexpected shape: top-level 'requirements' "
            f"missing or not a list. Got: {type(parsed).__name__}",
        )

    _emit(
        emit_event,
        "phase",
        {
            "phase": "calling_llm",
            "message": f"LLM produced {len(items)} candidate FRD(s)",
            "candidates": len(items),
            "input_tokens": chat_result.input_tokens,
            "output_tokens": chat_result.output_tokens,
        },
    )

    # ── Persist ─────────────────────────────────────────────────
    _check_cancel(is_cancelled, "before persist")
    _emit(
        emit_event,
        "phase",
        {"phase": "persisting", "message": f"Persisting {len(items)} FRD(s)"},
    )

    start_code_num = _next_frd_code_number(db, project_id)
    requirements = _persist_requirements(
        db, project_id, chunks, items, start_code_num,
    )

    db.commit()
    requirement_ids = [r.id for r in requirements]

    _emit(
        emit_event,
        "done",
        {
            "generated": len(requirements),
            "requirement_ids": requirement_ids,
            "input_tokens": chat_result.input_tokens,
            "output_tokens": chat_result.output_tokens,
        },
    )

    return SynthesisResult(
        generated_count=len(requirements),
        requirement_ids=requirement_ids,
        chunks_seen=len(chunks),
        truncated=truncated,
        input_tokens=chat_result.input_tokens,
        output_tokens=chat_result.output_tokens,
    )
