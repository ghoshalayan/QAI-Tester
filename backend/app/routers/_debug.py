"""Debug-only endpoints for verifying internal services.

Mounted under ``/api/_debug`` so it's clear these aren't part of the public
API contract. Safe to leave on locally; remove or gate before any deployment.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.embeddings.bge import get_embedder
from app.faiss_store.store import get_store
from app.ingest import (
    chunk_markdown,
    parse_docx,
    parse_markdown,
    parse_paste,
    parse_pdf,
)
from app.sse.bus import get_bus
from app.sse.response import sse_for_topic

router = APIRouter(prefix="/api/_debug", tags=["Debug"])


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    is_query: bool = Field(
        default=False,
        description="If true, applies the BGE query prefix. Use for search queries, not for documents being indexed.",
    )


class EmbedResponse(BaseModel):
    model: str
    dim: int
    count: int
    is_query: bool
    sample: list[float] = Field(
        ...,
        description="First 8 dims of the first vector — sanity-check that it isn't all zeros.",
    )
    norms: list[float] = Field(
        ...,
        description="L2 norm of each output vector. Should be ~1.0 since the embedder normalizes.",
    )
    elapsed_ms: int


@router.post("/embed", response_model=EmbedResponse)
def debug_embed(payload: EmbedRequest):
    import numpy as np

    embedder = get_embedder()
    started = time.monotonic()

    if payload.is_query:
        vecs = np.stack([embedder.embed_query(t) for t in payload.texts])
    else:
        vecs = embedder.embed_documents(payload.texts)

    elapsed_ms = int((time.monotonic() - started) * 1000)

    return EmbedResponse(
        model=embedder.MODEL_NAME,
        dim=int(vecs.shape[1]),
        count=int(vecs.shape[0]),
        is_query=payload.is_query,
        sample=vecs[0][:8].tolist(),
        norms=[float(np.linalg.norm(v)) for v in vecs],
        elapsed_ms=elapsed_ms,
    )


@router.get("/embed/status")
def debug_embed_status():
    """Quick check whether the embedding model has been loaded."""
    embedder = get_embedder()
    return {
        "model": embedder.MODEL_NAME,
        "dim": embedder.DIM,
        "device": embedder.DEVICE,
        "loaded": embedder.is_loaded,
    }


# ── FAISS store ───────────────────────────────────────────────────


class FaissDoc(BaseModel):
    id: int
    text: str


class FaissAddRequest(BaseModel):
    project_id: int = Field(..., gt=0)
    namespace: str = Field(..., min_length=1, max_length=64)
    docs: list[FaissDoc] = Field(..., min_length=1)
    upsert: bool = Field(
        default=False,
        description="If true, removes any existing vectors with these ids before adding.",
    )


class FaissAddResponse(BaseModel):
    project_id: int
    namespace: str
    added: int
    total_in_index: int
    elapsed_ms: int


class FaissSearchRequest(BaseModel):
    project_id: int = Field(..., gt=0)
    namespace: str = Field(..., min_length=1, max_length=64)
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=100)


class FaissHit(BaseModel):
    id: int
    score: float


class FaissSearchResponse(BaseModel):
    project_id: int
    namespace: str
    query: str
    k: int
    hits: list[FaissHit]
    elapsed_ms: int


@router.post("/faiss/add", response_model=FaissAddResponse)
def debug_faiss_add(payload: FaissAddRequest):
    """Embed the given docs (no query prefix) and add them to the FAISS index."""
    embedder = get_embedder()
    store = get_store()
    started = time.monotonic()

    ids = [d.id for d in payload.docs]
    texts = [d.text for d in payload.docs]
    vectors = embedder.embed_documents(texts)

    if payload.upsert:
        store.upsert(payload.project_id, payload.namespace, ids, vectors)
    else:
        store.add(payload.project_id, payload.namespace, ids, vectors)

    return FaissAddResponse(
        project_id=payload.project_id,
        namespace=payload.namespace,
        added=len(ids),
        total_in_index=store.count(payload.project_id, payload.namespace),
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )


@router.post("/faiss/search", response_model=FaissSearchResponse)
def debug_faiss_search(payload: FaissSearchRequest):
    """Embed the query (with BGE prefix) and return top-k neighbors."""
    embedder = get_embedder()
    store = get_store()
    started = time.monotonic()

    query_vec = embedder.embed_query(payload.query)
    hits = store.search(payload.project_id, payload.namespace, query_vec, k=payload.k)

    return FaissSearchResponse(
        project_id=payload.project_id,
        namespace=payload.namespace,
        query=payload.query,
        k=payload.k,
        hits=[FaissHit(id=i, score=s) for i, s in hits],
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )


@router.get("/faiss/info")
def debug_faiss_info(project_id: int, namespace: str):
    from app.config import settings

    store = get_store()
    path = settings.faiss_dir / str(project_id) / f"{namespace}.faiss"
    return {
        "project_id": project_id,
        "namespace": namespace,
        "count": store.count(project_id, namespace),
        "path": str(path),
        "exists_on_disk": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


@router.delete("/faiss")
def debug_faiss_reset(project_id: int, namespace: str | None = None):
    """Wipe one namespace, or all namespaces for a project when namespace is omitted."""
    if project_id <= 0:
        raise HTTPException(400, "project_id must be > 0")
    get_store().reset(project_id, namespace)
    return {"project_id": project_id, "namespace": namespace, "reset": True}


# ── SSE event bus ─────────────────────────────────────────────────


class SSEPublishRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=128)
    type: str = Field(..., min_length=1, max_length=64)
    data: dict[str, Any] = Field(default_factory=dict)


class SSEPublishResponse(BaseModel):
    topic: str
    type: str
    seq: int
    timestamp: float


class SSEDemoRequest(BaseModel):
    topic: str = Field(default="demo", min_length=1, max_length=128)
    count: int = Field(default=5, ge=1, le=50)
    interval_seconds: float = Field(default=1.0, ge=0.05, le=10.0)


@router.post("/sse/publish", response_model=SSEPublishResponse)
def debug_sse_publish(payload: SSEPublishRequest):
    """Publish a single event to a topic. Anything subscribed to that topic
    will receive it immediately."""
    event = get_bus().publish(payload.topic, payload.type, payload.data)
    return SSEPublishResponse(
        topic=payload.topic, type=event.type, seq=event.seq, timestamp=event.timestamp,
    )


@router.post("/sse/demo")
async def debug_sse_demo(payload: SSEDemoRequest):
    """Spawn a background task that publishes ``count`` ``log`` events to the
    given topic at the given interval, then a terminal ``done`` event. Useful
    for testing SSE streaming without wiring an agent."""
    bus = get_bus()

    async def publish_loop():
        for i in range(payload.count):
            await asyncio.sleep(payload.interval_seconds)
            bus.publish(
                payload.topic,
                "log",
                {"message": f"event {i + 1} of {payload.count}", "step": i + 1},
            )
        bus.publish(payload.topic, "done", {"summary": "demo complete"})

    asyncio.create_task(publish_loop())
    return {
        "topic": payload.topic,
        "queued": payload.count,
        "interval_seconds": payload.interval_seconds,
        "note": (
            f"Subscribe at GET /api/_debug/sse/stream?topic={payload.topic} "
            "to watch events arrive."
        ),
    }


@router.get("/sse/stream")
async def debug_sse_stream(
    request: Request,
    topic: str,
    since_seq: int = 0,
):
    """Stream events from a topic as Server-Sent Events.

    Test from a shell with::

        curl -N 'http://localhost:8000/api/_debug/sse/stream?topic=demo'
    """
    return sse_for_topic(request, topic, since_seq=since_seq)


@router.get("/sse/topics")
def debug_sse_topics():
    """Snapshot of every live topic — last seq, history size, subscriber count."""
    return {"topics": get_bus().list_topics()}


# ── Structured LLM calls ──────────────────────────────────────────


class StructuredMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class StructuredRequest(BaseModel):
    """``schema`` is the JSON-schema dict; aliased so HTTP callers can send
    the natural key ``schema`` without colliding with Pydantic's own
    ``BaseModel.schema`` method on older versions."""

    model_config = {"populate_by_name": True}

    messages: list[StructuredMessage] = Field(..., min_length=1)
    json_schema: dict[str, Any] = Field(..., alias="schema")
    schema_name: str = Field(default="output", min_length=1, max_length=64)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=8192)


class StructuredResponse(BaseModel):
    parsed: Any
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@router.post("/llm/structured", response_model=StructuredResponse)
def debug_llm_structured(payload: StructuredRequest):
    """Round-trip a structured-output LLM call against the configured provider.

    Useful for verifying provider-specific JSON modes (Gemini ``response_schema``,
    OpenAI ``json_schema`` strict, compat ``json_object`` + parse-retry) before
    wiring the agent in step 3.
    """
    from app.db import SessionLocal
    from app.llm.base import ChatMessage
    from app.llm.factory import get_provider_from_db

    db = SessionLocal()
    try:
        try:
            provider = get_provider_from_db(db)
        except RuntimeError as e:
            raise HTTPException(400, str(e)) from e

        try:
            result = provider.chat_structured(
                messages=[
                    ChatMessage(role=m.role, content=m.content)
                    for m in payload.messages
                ],
                schema=payload.json_schema,
                schema_name=payload.schema_name,
                temperature=payload.temperature,
                max_output_tokens=payload.max_output_tokens,
            )
        except Exception as e:
            raise HTTPException(
                502,
                f"{type(e).__name__}: {str(e)[:500]}",
            ) from e

        return StructuredResponse(
            parsed=result.parsed,
            text=result.text,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    finally:
        db.close()


# ── Ingest parsers (PDF/DOCX added in step 3) ────────────────────


class ParseRequest(BaseModel):
    source_type: Literal["md", "paste"]
    content: str = Field(..., min_length=1)
    title: str | None = Field(default=None, max_length=255)


class ParseResponse(BaseModel):
    source_type: str
    parsed_md: str
    char_count: int
    line_count: int


@router.post("/parse", response_model=ParseResponse)
def debug_parse(payload: ParseRequest):
    """Run the markdown / paste parser and return canonical Markdown.

    Useful for verifying normalization (BOM strip, CRLF→LF, blank-line collapse,
    trailing-whitespace cleanup) before wiring up the full ingest service.
    PDF/DOCX testing happens through the upload endpoint — those formats are
    awkward to send as JSON.
    """
    if payload.source_type == "md":
        parsed = parse_markdown(payload.content.encode("utf-8"))
    else:
        parsed = parse_paste(payload.content, title=payload.title)

    return ParseResponse(
        source_type=payload.source_type,
        parsed_md=parsed,
        char_count=len(parsed),
        line_count=parsed.count("\n"),
    )


# ── Chunker ──────────────────────────────────────────────────────


class ChunkRequest(BaseModel):
    markdown: str = Field(..., min_length=1)
    target_size: int = Field(default=800, ge=50, le=8000)
    overlap: int = Field(default=100, ge=0, le=2000)


class ChunkOut(BaseModel):
    ordinal: int
    heading_path: str
    anchor: str
    text: str
    char_count: int


class ChunkResponse(BaseModel):
    total_chunks: int
    total_chars: int
    chunks: list[ChunkOut]


@router.post("/chunk", response_model=ChunkResponse)
def debug_chunk(payload: ChunkRequest):
    """Run the heading-aware chunker on canonical Markdown.

    Inspect outputs to verify:
    - Headings produce correct ``heading_path`` and ``anchor``
    - Big sections split with overlap near paragraph boundaries
    - Code-fenced ``#`` lines are NOT treated as headings
    """
    if payload.overlap >= payload.target_size:
        raise HTTPException(400, "overlap must be smaller than target_size")

    chunks = chunk_markdown(
        payload.markdown,
        target_size=payload.target_size,
        overlap=payload.overlap,
    )
    return ChunkResponse(
        total_chunks=len(chunks),
        total_chars=sum(c.char_count for c in chunks),
        chunks=[
            ChunkOut(
                ordinal=c.ordinal,
                heading_path=c.heading_path,
                anchor=c.anchor,
                text=c.text,
                char_count=c.char_count,
            )
            for c in chunks
        ],
    )


@router.post("/parse-file", response_model=ParseResponse)
async def debug_parse_file(
    source_type: Literal["pdf", "docx", "md"] = Form(...),
    file: UploadFile = File(...),
):
    """Multipart smoke test for the file-based parsers.

    PDF/DOCX bytes are awkward to send as JSON, so we accept multipart here.
    The full ingest service (step 5) handles the same job for real uploads;
    this endpoint just lets you verify a single parser end-to-end.
    """
    import tempfile
    from pathlib import Path

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "empty file")

    if source_type == "md":
        parsed = parse_markdown(contents)
    elif source_type in ("pdf", "docx"):
        suffix = ".pdf" if source_type == "pdf" else ".docx"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(contents)
            tmp_path = Path(f.name)
        try:
            parser = parse_pdf if source_type == "pdf" else parse_docx
            try:
                parsed = parser(tmp_path)
            except RuntimeError as e:
                raise HTTPException(400, str(e)) from e
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        raise HTTPException(400, f"Unsupported source_type: {source_type}")

    return ParseResponse(
        source_type=source_type,
        parsed_md=parsed,
        char_count=len(parsed),
        line_count=parsed.count("\n"),
    )
