"""Documents router — upload / paste / list / detail / parsed / chunks / delete / search / events.

Mounted at ``/api/projects/{project_id}/documents``. Cleanup contract:

- ``DELETE /{doc_id}``   — wipes the doc's vectors from FAISS, its disk dir, and its
                           chunk rows (FK CASCADE).
- ``DELETE /api/projects/{id}`` (handled by projects router) — wipes the entire
  ``data/docs/<project_id>/`` tree and the project's FAISS dir; chunk rows
  cascade through ``documents → projects``.

Route ordering note
-------------------
Literal routes (``/upload``, ``/paste``, ``/search``, ``/events``) are
declared **before** the parametric ``/{doc_id}`` route so the router
matches them as literals, not as a doc id.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.embeddings.bge import get_embedder
from app.faiss_store.store import get_store
from app.ingest import parse_paste
from app.models.document import Document, DocumentChunk
from app.models.project import Project
from app.schemas.document import (
    ChunkRead,
    DocumentKind,
    DocumentParsed,
    DocumentRead,
    PasteRequest,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.services.ingest_service import (
    CHUNKS_NAMESPACE,
    ingest_document,
    topic_for_project,
)
from app.sse.response import sse_for_topic

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/projects/{project_id}/documents",
    tags=["Documents"],
)

# 50 MB upload cap — tunable; large enough for typical BRDs/FRDs
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

_EXT_TO_SOURCE: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "md",
    ".markdown": "md",
}


def _require_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


def _require_doc(db: Session, project_id: int, doc_id: int) -> Document:
    doc = db.get(Document, doc_id)
    if not doc or doc.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


def _safe_filename(raw: str | None) -> str:
    """Strip path separators and weird characters; keep extension intact."""
    if not raw:
        return "upload"
    cleaned = Path(raw).name  # strip directory components
    cleaned = re.sub(r"[\x00-\x1f]", "", cleaned)  # control chars
    return cleaned[:255] or "upload"


# ── Upload (multipart) ─────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    project_id: int,
    background_tasks: BackgroundTasks,
    kind: DocumentKind = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a PDF, DOCX, or Markdown file. Ingest runs in the background."""
    _require_project(db, project_id)

    filename = _safe_filename(file.filename)
    ext = Path(filename).suffix.lower()
    source_type = _EXT_TO_SOURCE.get(ext)
    if not source_type:
        raise HTTPException(
            400,
            f"Unsupported file extension: {ext or '(none)'}. "
            f"Allowed: {sorted(_EXT_TO_SOURCE.keys())}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large ({len(contents):,} bytes). "
            f"Max {MAX_UPLOAD_BYTES:,} bytes.",
        )

    # 1. Insert doc row first so we have an id for the disk path
    doc = Document(
        project_id=project_id,
        kind=kind,
        source_type=source_type,
        filename=filename,
        status="pending",
        sha256=hashlib.sha256(contents).hexdigest(),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # 2. Save the original bytes
    doc_dir = settings.docs_dir / str(project_id) / str(doc.id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    original_path = doc_dir / f"original{ext}"
    original_path.write_bytes(contents)

    doc.original_path = str(original_path)
    db.commit()
    db.refresh(doc)

    # 3. Schedule ingest pipeline
    background_tasks.add_task(ingest_document, doc.id)

    return doc


# ── Paste (JSON) ────────────────────────────────────────────────────


@router.post(
    "/paste",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
def paste_document(
    project_id: int,
    payload: PasteRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Submit pasted text as a document. Treated as canonical Markdown."""
    _require_project(db, project_id)

    parsed_md = parse_paste(payload.content, title=payload.title)

    title = (payload.title or "Pasted").strip() or "Pasted"
    safe_title = re.sub(r"[^\w\s.-]", "", title).strip()[:100] or "pasted"
    filename = f"{safe_title}.md"

    doc = Document(
        project_id=project_id,
        kind=payload.kind,
        source_type="paste",
        filename=filename,
        status="pending",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    doc_dir = settings.docs_dir / str(project_id) / str(doc.id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    original_path = doc_dir / "original.md"
    original_path.write_text(parsed_md, encoding="utf-8")

    doc.original_path = str(original_path)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(ingest_document, doc.id)
    return doc


# ── Search (literal route — must come before /{doc_id}) ────────────


@router.post("/search", response_model=SearchResponse)
def search_documents(
    project_id: int,
    payload: SearchRequest,
    db: Session = Depends(get_db),
):
    """Semantic search across all parsed chunks of this project's documents."""
    _require_project(db, project_id)

    embedder = get_embedder()
    query_vec = embedder.embed_query(payload.query)

    # Search wider when filtering by kind so the post-filter has headroom
    k_search = payload.k * 3 if payload.kind else payload.k
    hits_raw = get_store().search(
        project_id, CHUNKS_NAMESPACE, query_vec, k=k_search,
    )

    if not hits_raw:
        return SearchResponse(query=payload.query, k=payload.k, hits=[])

    chunk_ids = [h[0] for h in hits_raw]
    score_by_id = dict(hits_raw)

    stmt = (
        select(DocumentChunk, Document)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(DocumentChunk.id.in_(chunk_ids))
    )
    rows = list(db.execute(stmt).all())
    by_id: dict[int, tuple[DocumentChunk, Document]] = {
        ch.id: (ch, d) for ch, d in rows
    }

    hits: list[SearchHit] = []
    for chunk_id in chunk_ids:  # preserve FAISS ranking order
        if chunk_id not in by_id:
            continue  # row was deleted between FAISS search and DB query
        ch, d = by_id[chunk_id]
        if payload.kind and d.kind != payload.kind:
            continue
        hits.append(
            SearchHit(
                chunk_id=ch.id,
                document_id=d.id,
                document_kind=d.kind,  # type: ignore[arg-type]
                document_filename=d.filename,
                heading_path=ch.heading_path,
                anchor=ch.anchor,
                text=ch.text,
                score=score_by_id[chunk_id],
            ),
        )
        if len(hits) >= payload.k:
            break

    return SearchResponse(query=payload.query, k=payload.k, hits=hits)


# ── SSE events stream (literal route — before /{doc_id}) ──────────


@router.get("/events")
async def doc_events_stream(
    request: Request,
    project_id: int,
    since_seq: int = 0,
):
    """Subscribe to live ingest events for this project's documents.

    Emits:
    - ``doc_started``  / ``doc_progress`` / ``doc_completed`` / ``doc_failed``

    Reconnects honor ``Last-Event-ID``; missed events are replayed.
    """
    return sse_for_topic(
        request, topic_for_project(project_id), since_seq=since_seq,
    )


# ── List ────────────────────────────────────────────────────────────


@router.get("", response_model=list[DocumentRead])
def list_documents(project_id: int, db: Session = Depends(get_db)):
    _require_project(db, project_id)
    stmt = (
        select(Document)
        .where(Document.project_id == project_id)
        .order_by(Document.created_at.desc())
    )
    return list(db.scalars(stmt))


# ── Detail / parsed / chunks / delete (parametric — last) ─────────


@router.get("/{doc_id}", response_model=DocumentRead)
def get_document(
    project_id: int, doc_id: int, db: Session = Depends(get_db),
):
    return _require_doc(db, project_id, doc_id)


@router.get("/{doc_id}/parsed", response_model=DocumentParsed)
def get_parsed_md(
    project_id: int, doc_id: int, db: Session = Depends(get_db),
):
    doc = _require_doc(db, project_id, doc_id)
    if not doc.parsed_md_path:
        raise HTTPException(
            409,
            f"Document not parsed yet (status: {doc.status})",
        )
    path = Path(doc.parsed_md_path)
    if not path.exists():
        raise HTTPException(500, "Parsed markdown file missing on disk")

    md = path.read_text(encoding="utf-8")
    return DocumentParsed(document_id=doc.id, parsed_md=md, char_count=len(md))


@router.get("/{doc_id}/chunks", response_model=list[ChunkRead])
def list_chunks(
    project_id: int, doc_id: int, db: Session = Depends(get_db),
):
    _require_doc(db, project_id, doc_id)
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc_id)
        .order_by(DocumentChunk.ordinal)
    )
    return list(db.scalars(stmt))


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    project_id: int, doc_id: int, db: Session = Depends(get_db),
):
    """Delete a document. Removes vectors from FAISS + wipes the disk dir."""
    doc = _require_doc(db, project_id, doc_id)

    # Snapshot chunk ids BEFORE delete (FK CASCADE will remove rows)
    chunk_ids = list(
        db.scalars(
            select(DocumentChunk.id).where(
                DocumentChunk.document_id == doc_id,
            ),
        ),
    )

    # 1. Drop vectors from FAISS
    if chunk_ids:
        get_store().remove(project_id, CHUNKS_NAMESPACE, chunk_ids)

    # 2. Wipe disk dir
    doc_dir = settings.docs_dir / str(project_id) / str(doc_id)
    if doc_dir.exists():
        shutil.rmtree(doc_dir, ignore_errors=True)

    # 3. Delete row (cascades to chunks)
    db.delete(doc)
    db.commit()
