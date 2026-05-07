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
    agent_log: list[ReportAgentTurn] = Field(default_factory=list)


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
