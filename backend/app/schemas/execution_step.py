"""Pydantic schemas for execution steps + run summary.

The Runs tab fetches:
- ``GET /agent-runs/{run_id}``        → ``AgentRunRead`` (existing)
- ``GET /agent-runs/{run_id}/steps``  → ``list[ExecutionStepRead]``

The frontend overlays the step rows onto the live ``TcNodeTreeRead`` by
matching ``tc_node_id``, so module/submodule rows show derived pass/fail
counts without us having to persist them.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

ExecutionStepStatus = Literal[
    "pending", "running", "passed", "failed", "skipped", "blocked",
    "inconclusive",
]


class ExecutionStepRead(BaseModel):
    """Flat read of one ``execution_steps`` row."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    project_id: int
    plan_id: int | None = None
    tc_node_id: int | None = None

    # Snapshots (frozen at run-time)
    title_snapshot: str
    path_snapshot: str
    action_type_snapshot: str | None = None
    target_hint_snapshot: str | None = None
    expected_snapshot: str | None = None
    narrative_snapshot: str | None = None

    # Run-time state
    ordinal: int
    status: ExecutionStepStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None

    # Outputs
    screenshot_path: str | None = None
    narration: str | None = None
    error_message: str | None = None
    details_json: dict[str, Any] = {}

    created_at: datetime
    updated_at: datetime


class ExecutionRunSummary(BaseModel):
    """Shape stored on ``agent_runs.output_summary_json`` for ``execute`` runs.

    Not validated on read (the column is JSON), but useful as documentation
    and for the runner to type-check what it produces.
    """

    plan_id: int
    total_steps: int
    passed: int
    failed: int
    skipped: int
    blocked: int
    duration_ms: int | None = None
