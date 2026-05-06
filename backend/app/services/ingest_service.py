"""Document ingest orchestrator.

Pipeline (run as a background task by the documents router):

    1. parsing   — load original_path, run the format-specific parser,
                   save canonical Markdown to data/docs/<pid>/<did>/parsed.md
    2. chunking  — heading-aware splitter
    3. embedding — for each batch:
                     a. INSERT chunk rows (flush to get ids)
                     b. embed batch via BGE
                     c. upsert vectors into FAISS namespace "chunks"
                        with chunk.id as the FAISS id
                     d. commit DB transaction (chunks become visible)
                     e. emit SSE progress

SSE events
----------
Topic: ``project:<project_id>:docs``

- ``doc_started``    {doc_id, filename, kind}
- ``doc_progress``   {doc_id, phase, message, current?, total?}
- ``doc_completed``  {doc_id, chunk_count, char_count}
- ``doc_failed``     {doc_id, error}

Concurrency
-----------
Runs synchronously inside FastAPI's ``BackgroundTasks`` threadpool. The SSE
bus's ``publish`` is thread-safe and bridges back to the FastAPI event loop
via ``loop.call_soon_threadsafe``. The FAISS store has per-(project, ns)
locks, so multiple ingests on the same project serialize on FAISS access
but parse/embed in parallel.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.embeddings.bge import get_embedder
from app.faiss_store.store import get_store
from app.ingest import (
    chunk_markdown,
    parse_docx,
    parse_markdown,
    parse_paste,
    parse_pdf,
)
from app.models.document import Document, DocumentChunk
from app.sse.bus import get_bus

logger = logging.getLogger(__name__)

CHUNK_TARGET_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 32
CHUNKS_NAMESPACE = "chunks"


def topic_for_project(project_id: int) -> str:
    return f"project:{project_id}:docs"


def _emit(project_id: int, event_type: str, **data) -> None:
    get_bus().publish(topic_for_project(project_id), event_type, data)


def _run_parser(doc: Document) -> str:
    """Pick + run the parser for ``doc.source_type``. Returns canonical Markdown."""
    if not doc.original_path:
        raise RuntimeError(f"document {doc.id} has no original_path")

    path = Path(doc.original_path)
    if not path.exists():
        raise RuntimeError(f"original file missing on disk: {path}")

    if doc.source_type == "pdf":
        return parse_pdf(path)
    if doc.source_type == "docx":
        return parse_docx(path)
    if doc.source_type == "md":
        return parse_markdown(path.read_bytes())
    if doc.source_type == "paste":
        # Paste files are written as canonical MD already; parse_paste here just
        # re-normalizes (idempotent) in case the file was edited externally.
        return parse_paste(path.read_text(encoding="utf-8"))
    raise RuntimeError(f"unknown source_type: {doc.source_type!r}")


def _mark_failed(
    db: Session, doc: Document, project_id: int, message: str,
) -> None:
    logger.warning("ingest failed for doc %s: %s", doc.id, message)
    doc.status = "failed"
    doc.error_message = message[:2000]
    db.commit()
    _emit(project_id, "doc_failed", doc_id=doc.id, error=message)


def ingest_document(doc_id: int) -> None:
    """Full ingest pipeline for one document. Safe to schedule via ``BackgroundTasks``."""
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            logger.warning("ingest_document: doc %s not found", doc_id)
            return

        project_id = doc.project_id
        _emit(
            project_id,
            "doc_started",
            doc_id=doc.id,
            filename=doc.filename,
            kind=doc.kind,
            source_type=doc.source_type,
        )

        # ── Phase 1: parse ──────────────────────────────────────
        try:
            doc.status = "parsing"
            db.commit()
            _emit(
                project_id,
                "doc_progress",
                doc_id=doc.id,
                phase="parsing",
                message=f"Parsing {doc.filename}",
            )

            parsed_md = _run_parser(doc)

            doc_dir = settings.docs_dir / str(project_id) / str(doc.id)
            doc_dir.mkdir(parents=True, exist_ok=True)
            md_path = doc_dir / "parsed.md"
            md_path.write_text(parsed_md, encoding="utf-8")

            doc.parsed_md_path = str(md_path)
            doc.char_count = len(parsed_md)
            db.commit()

            _emit(
                project_id,
                "doc_progress",
                doc_id=doc.id,
                phase="parsing",
                message=f"Parsed {len(parsed_md):,} chars",
            )
        except Exception as e:
            _mark_failed(db, doc, project_id, f"Parse failed: {type(e).__name__}: {e}")
            return

        # ── Phase 2: chunk ──────────────────────────────────────
        try:
            chunks = chunk_markdown(
                parsed_md,
                target_size=CHUNK_TARGET_SIZE,
                overlap=CHUNK_OVERLAP,
            )
            _emit(
                project_id,
                "doc_progress",
                doc_id=doc.id,
                phase="chunking",
                message=f"Produced {len(chunks)} chunk(s)",
            )
        except Exception as e:
            _mark_failed(db, doc, project_id, f"Chunking failed: {type(e).__name__}: {e}")
            return

        if not chunks:
            doc.status = "parsed"
            doc.chunk_count = 0
            db.commit()
            _emit(
                project_id,
                "doc_completed",
                doc_id=doc.id,
                chunk_count=0,
                char_count=doc.char_count,
            )
            return

        # ── Phase 3: embed + persist ─────────────────────────────
        try:
            doc.status = "embedding"
            db.commit()

            embedder = get_embedder()
            store = get_store()
            total = len(chunks)

            for i in range(0, total, EMBED_BATCH_SIZE):
                batch = chunks[i : i + EMBED_BATCH_SIZE]

                # 1. INSERT chunk rows; flush to populate auto-incremented ids
                rows = [
                    DocumentChunk(
                        document_id=doc.id,
                        ordinal=c.ordinal,
                        heading_path=c.heading_path,
                        anchor=c.anchor,
                        text=c.text,
                        char_count=c.char_count,
                    )
                    for c in batch
                ]
                db.add_all(rows)
                db.flush()
                ids = [r.id for r in rows]

                # 2. Embed the batch (CPU; ~50–150 ms per chunk on first run)
                vectors = embedder.embed_documents([c.text for c in batch])

                # 3. Upsert vectors into FAISS using chunk.id as the FAISS id
                store.upsert(project_id, CHUNKS_NAMESPACE, ids, vectors)

                # 4. Commit so chunks are visible to concurrent reads
                db.commit()

                done = min(i + EMBED_BATCH_SIZE, total)
                _emit(
                    project_id,
                    "doc_progress",
                    doc_id=doc.id,
                    phase="embedding",
                    message=f"Embedded {done}/{total} chunk(s)",
                    current=done,
                    total=total,
                )

            doc.status = "parsed"
            doc.chunk_count = total
            db.commit()
            _emit(
                project_id,
                "doc_completed",
                doc_id=doc.id,
                chunk_count=total,
                char_count=doc.char_count,
            )
        except Exception as e:
            _mark_failed(
                db, doc, project_id, f"Embedding failed: {type(e).__name__}: {e}",
            )
            return

    finally:
        db.close()
