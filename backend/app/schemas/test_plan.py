"""Pydantic schemas for the TestPlan router."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.document import DocumentKind, DocumentStatus

PlanStatus = Literal["draft", "ready", "archived"]


# ── Credentials ────────────────────────────────────────────────────


class CredentialCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=64)
    username: str = Field(..., min_length=1, max_length=512)
    password: str = Field(..., min_length=1, max_length=512)
    # Phase 3 — TOTP secret. Accepts a base32 seed OR an
    # ``otpauth://...`` URI (the vault normalizes either form).
    # Empty / missing means "no TOTP for this credential — OTP
    # screens will fall back to HITL prompt".
    totp_secret: str | None = Field(default=None, max_length=512)
    url_pattern: str | None = Field(default=None, max_length=2048)
    username_selector_hint: str | None = Field(default=None, max_length=512)
    password_selector_hint: str | None = Field(default=None, max_length=512)
    notes: str | None = None


class CredentialUpdate(BaseModel):
    """Partial — empty/missing fields are preserved.

    Pass an empty/missing ``password`` to keep the existing one. Only a
    non-empty value replaces it.
    """

    label: str | None = Field(default=None, min_length=1, max_length=64)
    username: str | None = Field(default=None, min_length=1, max_length=512)
    password: str | None = Field(default=None, max_length=512)
    # Phase 3 — TOTP secret. Empty string explicitly clears any
    # stored seed (so the auth flow falls back to HITL). ``None``
    # leaves the existing value untouched.
    totp_secret: str | None = Field(default=None, max_length=512)
    url_pattern: str | None = Field(default=None, max_length=2048)
    username_selector_hint: str | None = Field(default=None, max_length=512)
    password_selector_hint: str | None = Field(default=None, max_length=512)
    notes: str | None = None


class CredentialRead(BaseModel):
    """Read view — never echoes the password OR the TOTP seed. The UI
    shows ``••••`` while ``password_set: true`` (and likewise for
    ``totp_set``)."""

    id: int
    plan_id: int
    label: str
    username: str
    password_set: bool
    # Phase 3 — TOTP indicator. UI renders an "OTP via TOTP enabled"
    # chip when true; agent will auto-generate codes from the
    # encrypted seed without prompting HITL.
    totp_set: bool = False
    url_pattern: str | None = None
    username_selector_hint: str | None = None
    password_selector_hint: str | None = None
    notes: str | None = None
    created_at: datetime
    updated_at: datetime


# ── Linked documents (read-only summary in plan detail) ───────────


class LinkedDocSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: int
    filename: str
    kind: DocumentKind
    status: DocumentStatus
    chunk_count: int


# ── Plans ─────────────────────────────────────────────────────────


class PlanCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    target_url: str = Field(..., min_length=1, max_length=2048)
    description: str | None = None
    scope: list[str] = Field(default_factory=list)
    status: PlanStatus = "draft"
    linked_document_ids: list[int] = Field(default_factory=list)
    credentials: list[CredentialCreate] = Field(default_factory=list)


class PlanUpdate(BaseModel):
    """Partial. ``linked_document_ids`` — when present — replaces the full set."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    target_url: str | None = Field(default=None, min_length=1, max_length=2048)
    description: str | None = None
    scope: list[str] | None = None
    status: PlanStatus | None = None
    linked_document_ids: list[int] | None = None


class PlanReadCompact(BaseModel):
    """Compact view for the plans list."""

    id: int
    project_id: int
    name: str
    target_url: str
    scope: list[str]
    status: PlanStatus
    credential_count: int
    linked_document_count: int
    created_at: datetime
    updated_at: datetime


class PlanReadDetail(BaseModel):
    """Full view used by the plan editor."""

    id: int
    project_id: int
    name: str
    target_url: str
    description: str | None
    scope: list[str]
    status: PlanStatus
    credentials: list[CredentialRead]
    linked_documents: list[LinkedDocSummary]
    created_at: datetime
    updated_at: datetime


# ── Heading suggestions (powers the scope dropdown) ───────────────


class HeadingSuggestionsResponse(BaseModel):
    """Top-level headings extracted from the chunks of given documents.

    The Plan editor's scope dropdown calls this with ``document_ids`` as the
    user (un)checks doc links — providing a curated suggestion list while
    still letting them type custom scope names.
    """

    suggestions: list[str]  # de-duplicated top-level heading names, sorted
    document_count: int     # how many docs contributed
    chunk_count: int        # total chunks scanned
