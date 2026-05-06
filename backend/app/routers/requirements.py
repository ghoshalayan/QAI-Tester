"""Requirements router — CRUD + bulk actions for FRD items.

Mounted at ``/api/projects/{project_id}/requirements``.

This step (week 3 step 6) keeps FAISS untouched; status changes only update
the DB. Step 7 wires up the approval→FAISS-embedding side-effects so that
approved requirements become searchable for the week-4 TC agent.

Route ordering note
-------------------
Literal ``/bulk-update`` and ``""`` come before parametric ``/{req_id}``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.document import Document, DocumentChunk
from app.models.project import Project
from app.models.requirement import Requirement
from app.schemas.requirement import (
    BulkUpdateRequest,
    BulkUpdateResponse,
    RequirementDetail,
    RequirementKind,
    RequirementRead,
    RequirementStatus,
    RequirementUpdate,
    SourceChunkRef,
)
from app.services.requirement_embed_service import (
    remove_before_delete,
    sync_after_bulk,
    sync_after_change,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/projects/{project_id}/requirements",
    tags=["Requirements"],
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _require_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


def _require_requirement(
    db: Session, project_id: int, req_id: int,
) -> Requirement:
    req = db.get(Requirement, req_id)
    if not req or req.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Requirement not found")
    return req


# ── Bulk update (literal — declared before /{req_id}) ─────────────


@router.post("/bulk-update", response_model=BulkUpdateResponse)
def bulk_update(
    project_id: int,
    payload: BulkUpdateRequest,
    db: Session = Depends(get_db),
):
    """Apply one of {approve, reject, delete} to a set of requirements.

    Specify either ``requirement_ids`` (explicit list), ``filter_status``
    (e.g. all proposed), or both (intersection). Returns ids that were
    affected (empty for ``delete`` since the rows are gone).
    """
    _require_project(db, project_id)

    if not payload.requirement_ids and not payload.filter_status:
        raise HTTPException(
            400,
            "Provide either requirement_ids or filter_status (or both)",
        )

    stmt = select(Requirement).where(Requirement.project_id == project_id)
    if payload.requirement_ids:
        stmt = stmt.where(Requirement.id.in_(payload.requirement_ids))
    if payload.filter_status:
        stmt = stmt.where(Requirement.status == payload.filter_status)

    rows = list(db.scalars(stmt))
    if not rows:
        return BulkUpdateResponse(
            affected=0, affected_ids=[], action=payload.action,
        )

    affected_ids = [r.id for r in rows]
    now = _utcnow()

    if payload.action == "delete":
        # Drop FAISS vectors first; then delete rows in same transaction
        for r in rows:
            remove_before_delete(r)
        for r in rows:
            db.delete(r)
        db.commit()
        return BulkUpdateResponse(
            affected=len(rows),
            affected_ids=[],
            action=payload.action,
        )

    # Capture prior statuses BEFORE we overwrite them — sync_after_bulk
    # uses these to figure out which way each row is moving (approve/un-approve).
    prior_pairs: list[tuple[Requirement, str | None]] = [
        (r, r.status) for r in rows
    ]

    if payload.action == "approve":
        for r in rows:
            r.status = "approved"
            r.reviewed_at = now
    elif payload.action == "reject":
        for r in rows:
            r.status = "rejected"
            r.reviewed_at = now

    # Reconcile FAISS in the same transaction. If the embedding model is not
    # loaded yet, this is the moment it loads (~5-10s on first call).
    sync_after_bulk(db, prior_pairs)

    db.commit()

    logger.info(
        "Bulk %s applied to %d requirements in project %s",
        payload.action, len(rows), project_id,
    )
    return BulkUpdateResponse(
        affected=len(rows),
        affected_ids=affected_ids if payload.action != "delete" else [],
        action=payload.action,
    )


# ── List ──────────────────────────────────────────────────────────


@router.get("", response_model=list[RequirementRead])
def list_requirements(
    project_id: int,
    status_filter: RequirementStatus | None = Query(default=None, alias="status"),
    kind: RequirementKind | None = None,
    source_document_id: int | None = None,
    db: Session = Depends(get_db),
):
    """List requirements in the project, newest first.

    Optional filters: ``status`` (proposed/edited/approved/rejected),
    ``kind`` (currently only FRD), ``source_document_id`` (the BRD that
    inspired them).
    """
    _require_project(db, project_id)
    stmt = select(Requirement).where(Requirement.project_id == project_id)
    if status_filter is not None:
        stmt = stmt.where(Requirement.status == status_filter)
    if kind is not None:
        stmt = stmt.where(Requirement.kind == kind)
    if source_document_id is not None:
        stmt = stmt.where(Requirement.source_document_id == source_document_id)
    stmt = stmt.order_by(Requirement.created_at.desc())
    return list(db.scalars(stmt))


# ── Parametric routes ────────────────────────────────────────────


@router.get("/{req_id}", response_model=RequirementDetail)
def get_requirement(
    project_id: int, req_id: int, db: Session = Depends(get_db),
):
    """Full requirement detail with source chunks resolved.

    Single request returns everything the review card needs — body, rationale,
    confidence, and the actual BRD chunks the agent cited.
    """
    req = _require_requirement(db, project_id, req_id)

    source_doc_filename: str | None = None
    if req.source_document_id is not None:
        d = db.get(Document, req.source_document_id)
        if d is not None:
            source_doc_filename = d.filename

    source_chunks: list[SourceChunkRef] = []
    if req.source_chunk_ids:
        rows = list(
            db.execute(
                select(DocumentChunk, Document)
                .join(Document, DocumentChunk.document_id == Document.id)
                .where(DocumentChunk.id.in_(req.source_chunk_ids)),
            ).all(),
        )
        # Preserve the order from source_chunk_ids
        by_id = {ch.id: (ch, d) for ch, d in rows}
        for cid in req.source_chunk_ids:
            entry = by_id.get(cid)
            if entry is None:
                continue  # chunk was deleted (BRD removed) — skip silently
            ch, d = entry
            source_chunks.append(
                SourceChunkRef(
                    chunk_id=ch.id,
                    document_id=d.id,
                    document_filename=d.filename,
                    heading_path=ch.heading_path,
                    anchor=ch.anchor,
                    text=ch.text,
                    char_count=ch.char_count,
                    ordinal=ch.ordinal,
                ),
            )

    return RequirementDetail(
        id=req.id,
        project_id=req.project_id,
        source_document_id=req.source_document_id,
        source_chunk_ids=list(req.source_chunk_ids or []),
        kind=req.kind,  # type: ignore[arg-type]
        code=req.code,
        title=req.title,
        body_md=req.body_md,
        status=req.status,  # type: ignore[arg-type]
        confidence=req.confidence,
        rationale=req.rationale,
        embedding_id=req.embedding_id,
        created_at=req.created_at,
        updated_at=req.updated_at,
        reviewed_at=req.reviewed_at,
        source_document_filename=source_doc_filename,
        source_chunks=source_chunks,
    )


@router.patch("/{req_id}", response_model=RequirementRead)
def update_requirement(
    project_id: int,
    req_id: int,
    payload: RequirementUpdate,
    db: Session = Depends(get_db),
):
    """Edit a requirement (partial). Status transitions:

    - Explicit ``status`` field: use it (and stamp ``reviewed_at``)
    - Otherwise, if ``title`` or ``body_md`` changed: move to ``edited``
      (so an approved item that gets edited returns to review)
    - ``rationale``-only edits don't change status
    """
    req = _require_requirement(db, project_id, req_id)
    prior_status = req.status  # snapshot for FAISS sync

    body_or_title_changed = False
    if payload.title is not None:
        new_title = payload.title.strip()
        if new_title != req.title:
            req.title = new_title
            body_or_title_changed = True
    if payload.body_md is not None:
        if payload.body_md != req.body_md:
            req.body_md = payload.body_md
            body_or_title_changed = True
    if payload.rationale is not None:
        req.rationale = payload.rationale or None

    if payload.status is not None:
        if payload.status != req.status:
            req.status = payload.status
            req.reviewed_at = _utcnow()
    elif body_or_title_changed:
        # Edits revert approved/rejected back to "edited" so the user
        # re-reviews before re-approving.
        req.status = "edited"
        req.reviewed_at = _utcnow()

    # Reconcile FAISS in the same transaction:
    # - newly approved → embed
    # - was approved, now anything else → drop vector
    # - approved AND title/body changed → re-embed (upsert)
    if req.status == "approved" and body_or_title_changed:
        # Force re-embed even if status didn't change (stayed "approved")
        sync_after_change(db, req, prior_status="proposed")  # any non-approved sentinel
    else:
        sync_after_change(db, req, prior_status=prior_status)

    db.commit()
    db.refresh(req)
    return req


@router.delete("/{req_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_requirement(
    project_id: int, req_id: int, db: Session = Depends(get_db),
):
    """Hard delete. Drops FAISS vector first if the row was approved."""
    req = _require_requirement(db, project_id, req_id)
    remove_before_delete(req)  # no-op if not embedded
    db.delete(req)
    db.commit()
