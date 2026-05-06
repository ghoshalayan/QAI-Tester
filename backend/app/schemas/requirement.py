"""Pydantic schemas for the requirements router."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RequirementKind = Literal["FRD"]
RequirementStatus = Literal["proposed", "edited", "approved", "rejected"]
BulkAction = Literal["approve", "reject", "delete"]


# ── Read views ────────────────────────────────────────────────────


class RequirementRead(BaseModel):
    """Compact view used for list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    source_document_id: int | None = None
    source_chunk_ids: list[int]
    kind: RequirementKind
    code: str
    title: str
    body_md: str
    status: RequirementStatus
    confidence: float | None = None
    rationale: str | None = None
    embedding_id: int | None = None
    created_at: datetime
    updated_at: datetime
    reviewed_at: datetime | None = None


class SourceChunkRef(BaseModel):
    """Resolved source-chunk info for the detail/review view."""

    chunk_id: int
    document_id: int
    document_filename: str
    heading_path: str | None = None
    anchor: str | None = None
    text: str
    char_count: int
    ordinal: int


class RequirementDetail(RequirementRead):
    """Full review view — extends compact with resolved source context."""

    source_document_filename: str | None = None
    source_chunks: list[SourceChunkRef] = Field(default_factory=list)


# ── Write views ───────────────────────────────────────────────────


class RequirementUpdate(BaseModel):
    """Partial update.

    Status flow on save:
    - If ``status`` is provided explicitly, use it.
    - Otherwise, if ``title`` or ``body_md`` changed, status moves to ``edited``
      (so an approved item that gets edited returns to review).
    """

    title: str | None = Field(default=None, min_length=1, max_length=512)
    body_md: str | None = None
    rationale: str | None = None
    status: RequirementStatus | None = None


class BulkUpdateRequest(BaseModel):
    """Apply ``action`` to either a list of ids OR every row matching a status filter.

    At least one of ``requirement_ids`` or ``filter_status`` is required. If
    both are present, both filters apply (the row must be in the id list AND
    have the given status).
    """

    requirement_ids: list[int] | None = None
    filter_status: RequirementStatus | None = None
    action: BulkAction


class BulkUpdateResponse(BaseModel):
    affected: int
    affected_ids: list[int]
    action: BulkAction
