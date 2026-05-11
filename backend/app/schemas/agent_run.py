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

    # Run mode:
    # - "scripted" (default): rigid step-walker; AI only intervenes on
    #   failure (improvise / recover / vision escalation).
    # - "agentic":  goal-oriented QA agent loop. One LLM-driven loop
    #   per submodule (= test case). The agent observes the page,
    #   picks tools, and self-verifies against derived success
    #   criteria. Requires an LLM provider configured in App Settings.
    # - "replay":   deterministic walk of each submodule's frozen
    #   path (captured the last time agentic mode passed cleanly on
    #   it). Zero LLM calls on the happy path; vision-LLM only fires
    #   for self-healing when a frozen step misses. Submodules
    #   without a frozen path fall through to agentic.
    mode: Literal["scripted", "agentic", "replay"] = Field(default="scripted")
    # Phase 6 — within ``mode='agentic'``, choose the action strategy:
    # - ``hybrid`` (default): existing DOM-first ladder with vision
    #   rescue. Cheaper & faster on most modern apps.
    # - ``vision_only``: every click / type goes through VL + pixel
    #   coordinates. DOM resolution bypassed entirely. ~3-5x more
    #   vision tokens, but works on apps where DOM is hopeless
    #   (heavy canvas, sealed shadow DOM, hostile rotating classes,
    #   SAP GUI for HTML in legacy frames). Ignored when ``mode``
    #   isn't ``agentic``.
    agent_strategy: Literal["hybrid", "vision_only"] = Field(
        default="hybrid",
    )

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
    - ``provide_text`` — Phase 4 / auth flow. Submitting an OTP, a
      manually-typed credential, a captcha solution, etc. ``text_value``
      carries the value; ``text_kind`` describes what kind of input it
      is (so the agent's auth flow knows how to use it). The agent's
      pending wait unblocks with the typed string and resumes the loop.
    - ``manual_solved`` — Phase 4. User clicks "I solved it in the
      browser, continue" for captcha / passkey / device-prompt cases
      where the value can't be typed. The agent retries the action
      that was blocked.

    ``override_target_hint`` / ``override_action_type`` apply only to
    ``use_suggestion``. Empty / null means "don't override that field".

    ``apply_to_submodule`` remembers the choice and auto-applies it to
    every subsequent failure under the same submodule, so the user
    doesn't click 20 modals on a broken submodule.
    """

    step_id: int = Field(..., gt=0)
    choice: Literal[
        "retry", "use_suggestion", "skip", "stop",
        "provide_text", "manual_solved",
    ]
    override_target_hint: str | None = Field(default=None, max_length=2048)
    override_action_type: str | None = Field(default=None, max_length=64)
    apply_to_submodule: bool = Field(default=False)
    # Phase 4 — typed HITL input.
    text_value: str | None = Field(default=None, max_length=512)
    # What kind of value ``text_value`` carries — agent's auth flow
    # branches on this. ``otp_code``, ``username``, ``password``,
    # ``captcha_text``, ``free_text``.
    text_kind: str | None = Field(default=None, max_length=32)
    # Optional second value for paired prompts (username + password
    # asked together when both are missing). Same encoding rules as
    # text_value.
    text_value_secondary: str | None = Field(default=None, max_length=512)


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
    # Cost tracking (migration 0017). Per-tier token columns
    # populated by the agent loop's cost context. Pre-feature runs
    # have all four at 0; the cost service treats those as
    # strong-tier aggregate.
    strong_input_tokens: int = 0
    strong_output_tokens: int = 0
    cheap_input_tokens: int = 0
    cheap_output_tokens: int = 0
    # Migration 0019 — cached portions of input (SUBSET of the
    # input totals above, not additive). 0 on legacy runs.
    strong_cached_input_tokens: int = 0
    cheap_cached_input_tokens: int = 0
    strong_model_snapshot: str | None = None
    cheap_model_snapshot: str | None = None
    # ``total_cost_usd`` is computed at read time by the runs router
    # (it joins the per-tier tokens against current AppSettings
    # pricing). NULL on the schema means "no pricing configured"
    # OR "no LLM activity on this run" — UI distinguishes.
    total_cost_usd: float | None = None
