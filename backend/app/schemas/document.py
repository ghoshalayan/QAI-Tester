"""Pydantic schemas for the documents router."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentKind = Literal["BRD", "FRD", "INSTRUCTIONS"]
DocumentStatus = Literal["pending", "parsing", "embedding", "parsed", "failed"]
DocumentSourceType = Literal["pdf", "docx", "md", "paste"]


# ── Requests ──────────────────────────────────────────────────────


class PasteRequest(BaseModel):
    kind: DocumentKind
    title: str | None = Field(default=None, max_length=255)
    content: str = Field(..., min_length=1, max_length=10_000_000)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=50)
    kind: DocumentKind | None = None  # filter to BRD or FRD only


# ── Responses ─────────────────────────────────────────────────────


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    kind: DocumentKind
    source_type: DocumentSourceType
    filename: str
    status: DocumentStatus
    error_message: str | None = None
    char_count: int
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class DocumentParsed(BaseModel):
    document_id: int
    parsed_md: str
    char_count: int


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    ordinal: int
    heading_path: str | None = None
    anchor: str | None = None
    text: str
    char_count: int


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    document_kind: DocumentKind
    document_filename: str
    heading_path: str | None = None
    anchor: str | None = None
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    k: int
    hits: list[SearchHit]
