"""Pydantic schemas for the run-report API.

The frontend calls ``GET /api/projects/{pid}/agent-runs/{run_id}/report``
to fetch a fully-aggregated tree of the run's results: run summary +
plan info + modules → submodules → step rows. The companion
``.../report.xlsx`` endpoint streams the same data as a workbook.

Module / submodule grouping is derived at query time by joining
``execution_steps`` to ``tc_nodes`` and walking the parent chain. We do
NOT persist a denormalized report — the run + step rows are the source
of truth, and re-aggregation is cheap.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ExecutionStepStatus = Literal[
    "pending", "running", "passed", "failed", "skipped", "blocked",
    "inconclusive",
]


class ReportAgentTurn(BaseModel):
    """One turn from the agentic-mode log — the agent's per-LLM-call
    breakdown of what it observed, decided, and did. Lifted from
    ``execution_steps.details_json["agent_log"]`` so the report UI can
    render the trail without re-fetching raw JSON.
    """

    turn: int
    tool: str
    args: dict = Field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    status: str = ""               # "ok" | "failed" | "blocked" | "stop"
    narration: str = ""
    error_message: str | None = None
    page_url: str = ""
    extracted_text: str = ""


class ReportSubGoal(BaseModel):
    """One ordered sub-goal the agent worked through.

    Phase A1 (legacy): sourced from ``details_json["goal"]["sub_goals"]`` —
    BRD-time text-derived hints.
    Phase A2 (current): sourced from ``details_json["sub_goals"]`` —
    VL-derived runtime sub-goals with reason + replan iteration.
    The report endpoint prefers (A2) when present, falls back to (A1).
    """

    id: str
    description: str
    status: str = "pending"
    completed_at_turn: int | None = None
    # Phase A — populated for VL-derived runtime sub-goals; absent
    # on the legacy BRD-time ones (model_config allows missing).
    success_criterion: str | None = None
    reason: str | None = None
    replan_iteration: int = 0
    started_at_turn: int | None = None
    ended_at_turn: int | None = None
    max_turns: int | None = None


AgentStatus = Literal[
    "queued", "running", "paused", "completed", "failed", "cancelled",
]


class ReportStepRead(BaseModel):
    """One leaf step row in the report."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    tc_node_id: int | None = None
    ordinal: int
    title: str
    action_type: str | None = None
    target_hint: str | None = None
    status: ExecutionStepStatus
    duration_ms: int | None = None
    screenshot_path: str | None = None
    error_message: str | None = None
    narration: str | None = None
    # Set when details_json["ai_correction"] is present AND the step
    # ended up passing — i.e. the AI's suggestion fixed it.
    ai_helped: bool = False
    # Set when AI assist used vision (text + screenshot) on its last try.
    ai_used_vision: bool = False

    # ── Agentic-mode fields (Phase C) ─────────────────────────────
    # Lifted from ``execution_steps.details_json`` when the run mode was
    # "agentic". ``mode`` is None for scripted runs.
    mode: Literal["scripted", "agentic"] | None = None
    halt_reason: str | None = None
    goal_description: str | None = None
    success_criteria: list[str] = Field(default_factory=list)
    sub_goals: list[ReportSubGoal] = Field(default_factory=list)
    agent_log: list[ReportAgentTurn] = Field(default_factory=list)
    # ── A4 visibility fields ──────────────────────────────────────
    # Divergence classification (A4.3). One of: passed_clean,
    # passed_with_help, test_case_outdated, feature_missing,
    # infra_issue, agent_drift, agent_gave_up, user_cancelled.
    # ``None`` for scripted runs.
    divergence_category: str | None = None
    divergence_summary: str | None = None
    # Counts of automatic interventions that ran on this row, so the
    # report can render "rescued by N fuzzy match(es) / N vision
    # search(es)" badges. Tells the user the value of the agentic
    # stack, not just its cost.
    fuzzy_rescues: int = 0
    vision_rescues: int = 0
    # A4.1a goal verification result (verdict + reasoning) when the
    # agent claimed completion. ``None`` if the agent didn't reach
    # mark_goal_complete.
    goal_verification: dict | None = None
    # Phase 11 — test-case dispute. When the agent flagged the test
    # step as provably wrong via ``flag_test_case_issue``, this dict
    # carries ``{issue_kind, evidence, suggested_fix, turn}``.
    # Submodule status is ``blocked`` in that case (not failed —
    # failure means the APP is broken; this means the TEST is).
    test_case_dispute: dict | None = None
    # Phase 14 — smart candidate selection record. When the agent's
    # target_hint matched 3+ visible elements and the vision LLM
    # picked the right one (skipping sponsored ads, etc.), this dict
    # carries ``{strategy, chosen_label, rejected_labels,
    # rejection_reasons, confidence, reasoning}``.
    smart_pick: dict | None = None
    # Phase 9 — semantic verify escalation result. When a literal
    # ``verify`` failed and the vision LLM escalation ruled, this
    # dict carries ``{verdict, reasoning, confidence,
    # visible_evidence}``.
    semantic_verify: dict | None = None
    # Production-α — AKB chunks recalled at submodule start (turn 1).
    # Each entry: ``{kind, content, confidence, tags, relevance}``.
    akb_recall: list[dict] = []
    # Plan-scoped WorldState snapshot at the moment this submodule
    # finished. None for legacy / non-agentic runs.
    world_state_snapshot: dict | None = None
    # Signal-voting trace: which of the goal's evidence_signals
    # matched at completion time. Shape:
    # ``{matched: int, total: int, traces: [{signal, matched, via}]}``.
    signal_voting: dict | None = None


class ReportSubmoduleRead(BaseModel):
    """Aggregate for one submodule (or "(none)" for orphan steps)."""

    title: str
    total: int
    passed: int
    failed: int
    blocked: int
    skipped: int
    # Agentic goals that halted before being verified — distinct from
    # failed; usually points at a test-case wording issue, not an app
    # bug. Default 0 for scripted runs / pre-Phase-C reports.
    inconclusive: int = 0
    pass_pct: float = Field(..., ge=0.0, le=100.0)
    fail_pct: float = Field(..., ge=0.0, le=100.0)
    # Short distinct error excerpts (first 200 chars each, deduped).
    # Empty list when the submodule had no failures.
    issues: list[str] = Field(default_factory=list)
    steps: list[ReportStepRead] = Field(default_factory=list)


class ReportModuleRead(BaseModel):
    """Aggregate for one module (or "(none)" for orphan submodules)."""

    title: str
    total: int
    passed: int
    failed: int
    blocked: int
    skipped: int
    inconclusive: int = 0
    pass_pct: float = Field(..., ge=0.0, le=100.0)
    fail_pct: float = Field(..., ge=0.0, le=100.0)
    submodules: list[ReportSubmoduleRead] = Field(default_factory=list)


class ReportRunSummary(BaseModel):
    """Run-level numbers lifted from ``agent_runs``."""

    id: int
    status: AgentStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    total_steps: int
    passed: int
    failed: int
    blocked: int
    skipped: int
    inconclusive: int = 0
    pass_pct: float = Field(..., ge=0.0, le=100.0)
    fail_pct: float = Field(..., ge=0.0, le=100.0)
    # AI-assist cost meter
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    ai_calls: int = 0
    ai_vision_calls: int = 0


class ReportPlanSummary(BaseModel):
    """Plan info — None when the plan has been deleted since."""

    id: int
    name: str
    target_url: str
    scope: list[str] = Field(default_factory=list)


class ReportRead(BaseModel):
    """Top-level shape returned by ``GET /agent-runs/{id}/report``."""

    run: ReportRunSummary
    plan: ReportPlanSummary | None = None
    modules: list[ReportModuleRead] = Field(default_factory=list)
    # Same URL the frontend can hit to download the xlsx (convenience).
    excel_download_url: str
