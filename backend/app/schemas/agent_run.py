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

    ``ai_assist`` enables the AI-driven correction layer that kicks in
    after auto-retries exhaust on a step. The LLM looks at the live page,
    proposes a fix (corrected target_hint / action_type / etc.), and the
    orchestrator runs the corrected step once. On success the row is
    promoted to ``passed`` and the correction is logged in
    ``details_json["ai_correction"]``. Defaults to ``True`` — gracefully
    no-ops when no LLM is configured in app_settings.

    ``auto_adjust`` controls what the AI does with its suggestion:

    - ``False`` (default): the suggestion is PROPOSED — the HITL modal
      pre-fills with it and the user accepts / edits / rejects. Vision
      escalation is skipped (the user is already in the loop).
    - ``True``: the AI silently auto-applies its suggestion. If the
      text-only fix still fails AND the provider supports vision, a
      second screenshot-attached call escalates. HITL only fires when
      both passes still leave the step failed.

    ``promote_fixes`` writes a successful correction back to the source
    ``tc_node`` so the next run starts with the fix baked in. Applies
    when an AI auto-applied fix passes OR when a HITL ``use_suggestion``
    / ``retry`` with overrides passes. Off by default — promoting a
    one-off fix can paper over a real test-case bug.
    """

    plan_id: int = Field(..., gt=0)
    selected_step_ids: list[int] | None = Field(default=None, min_length=1)
    headless: bool = Field(default=False)
    speed: Literal["slow", "normal", "fast"] = Field(default="slow")
    ai_assist: bool = Field(default=True)
    auto_adjust: bool = Field(default=False)
    promote_fixes: bool = Field(default=False)

    # Window geometry — set by the frontend from ``window.screen`` so the
    # headed Chromium tiles to fit the user's monitor and leaves the
    # right side free for the live presenter popup. All four are optional;
    # missing values fall through to ``browser_session`` defaults.
    # Headless launches ignore these.
    window_x: int | None = Field(default=None, ge=0, le=10_000)
    window_y: int | None = Field(default=None, ge=0, le=10_000)
    window_width: int | None = Field(default=None, ge=400, le=10_000)
    window_height: int | None = Field(default=None, ge=300, le=10_000)


class InterventionRequest(BaseModel):
    """Body for ``POST /agent-runs/{run_id}/intervention``.

    Sent by the user when an HITL modal pops on a step that survived
    auto-retry + AI assist. Choices:

    - ``retry`` — try the original step again as-is.
    - ``use_suggestion`` — try with the user's overrides (typically the
      AI's suggested target_hint, optionally edited inline in the modal).
    - ``skip`` — leave the step ``failed`` and continue to the next sibling.
    - ``stop`` — cancel the entire run.

    ``override_target_hint`` / ``override_action_type`` apply only to
    ``use_suggestion``. Empty / null means "don't override that field".

    ``apply_to_submodule`` remembers the choice and auto-applies it to
    every subsequent failure under the same submodule, so the user
    doesn't click 20 modals on a broken submodule.
    """

    step_id: int = Field(..., gt=0)
    choice: Literal["retry", "use_suggestion", "skip", "stop"]
    override_target_hint: str | None = Field(default=None, max_length=2048)
    override_action_type: str | None = Field(default=None, max_length=64)
    apply_to_submodule: bool = Field(default=False)


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
