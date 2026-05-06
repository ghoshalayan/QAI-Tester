"""Pydantic schemas for the agent runs router."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AgentKind = Literal["brd_to_frd", "frd_to_tc", "execute", "reporter"]
AgentStatus = Literal[
    "queued", "running", "paused", "completed", "failed", "cancelled",
]


# ── Requests ──────────────────────────────────────────────────────


class BrdToFrdRunRequest(BaseModel):
    """Body for ``POST /agent-runs/brd-to-frd``."""

    source_document_ids: list[int] = Field(..., min_length=1)
    cap_chunks: int = Field(default=50, ge=1, le=200)


class FrdToTcRunRequest(BaseModel):
    """Body for ``POST /agent-runs/frd-to-tc``.

    The plan supplies scope + target_url + linked docs; the agent handles
    per-module retrieval. ``cap_per_module_*`` bounds the size of each LLM
    call's context.
    """

    plan_id: int = Field(..., gt=0)
    cap_per_module_frds: int = Field(default=15, ge=1, le=50)
    cap_per_module_chunks: int = Field(default=10, ge=0, le=50)


class ExecuteRunRequest(BaseModel):
    """Body for ``POST /agent-runs/execute``.

    The plan supplies ``target_url``, scope, and credentials. The executor
    walks the plan's TC tree, running every step where ``selectable_default``
    is True (or every id in ``selected_step_ids`` if that override is set).

    ``speed`` controls pacing — slow_mo, cursor glide steps, per-character
    type delay, network-idle wait timeout, and auto-retry count. Default
    ``"slow"`` because heavy-data sites need the longer settle window;
    ``"fast"`` skips the visible-typing animation and shortens timeouts.
    """

    plan_id: int = Field(..., gt=0)
    selected_step_ids: list[int] | None = Field(default=None, min_length=1)
    headless: bool = Field(default=False)
    speed: Literal["slow", "normal", "fast"] = Field(default="slow")


# ── Responses ─────────────────────────────────────────────────────


class AgentRunRead(BaseModel):
    """Full run row. Used both for list and detail."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    plan_id: int | None = None
    kind: AgentKind
    status: AgentStatus
    input_json: dict[str, Any] = Field(default_factory=dict)
    output_summary_json: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
