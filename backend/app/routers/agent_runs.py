"""Agent runs router — start / list / detail / cancel / events.

Mounted at ``/api/projects/{project_id}/agent-runs``.

Route ordering note
-------------------
Literal paths (``/brd-to-frd``, ``/events``, ``""``) are declared before the
parametric ``/{run_id}`` so the router doesn't try to parse them as ints.
"""

from __future__ import annotations

import logging
import shutil

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.executor import chromium_installed
from app.models.agent_run import AgentRun
from app.models.document import Document
from app.models.execution_step import ExecutionStep
from app.models.project import Project
from app.models.tc_node import TcNode
from app.models.test_plan import TestPlan
from app.schemas.agent_run import (
    AgentKind,
    AgentRunRead,
    AgentStatus,
    BrdToFrdRunRequest,
    ExecuteRunRequest,
    FrdToTcRunRequest,
    InterventionRequest,
)
from app.schemas.execution_step import ExecutionStepRead
from app.schemas.report import ReportRead
from app.services.report_service import build_excel_workbook, build_run_report
from app.services.agent_run_service import (
    _has_pending_intervention,
    execute_brd_to_frd,
    execute_frd_to_tc,
    execute_run,
    provide_intervention,
    request_cancel,
    request_pause,
    request_resume,
    topic_for_project_agent_runs,
    topic_for_run,
)
from app.sse.response import sse_for_topic

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/projects/{project_id}/agent-runs",
    tags=["Agent Runs"],
)


def _require_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


def _require_run(
    db: Session, project_id: int, run_id: int,
) -> AgentRun:
    run = db.get(AgentRun, run_id)
    if not run or run.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Agent run not found")
    return run


# ── Start endpoints (one per agent kind; literal routes) ──────────


@router.post(
    "/brd-to-frd",
    response_model=AgentRunRead,
    status_code=status.HTTP_201_CREATED,
)
def start_brd_to_frd(
    project_id: int,
    payload: BrdToFrdRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Queue a BRD→FRD synthesis run.

    Returns 201 with the queued ``AgentRun`` row immediately. The runner
    runs on FastAPI's BackgroundTasks; subscribe to
    ``GET /agent-runs/{run.id}/events`` (SSE) to follow progress.
    """
    _require_project(db, project_id)

    # Pre-flight validation — fail fast with helpful 4xx codes before
    # spawning a background task. The orchestrator re-validates in detail.
    doc_ids = list(dict.fromkeys(payload.source_document_ids))
    docs = list(db.scalars(select(Document).where(Document.id.in_(doc_ids))))
    by_id = {d.id: d for d in docs}

    missing = [i for i in doc_ids if i not in by_id]
    if missing:
        raise HTTPException(404, f"Documents not found: {missing}")

    bad_project = [d.id for d in docs if d.project_id != project_id]
    if bad_project:
        raise HTTPException(
            403, f"Documents belong to a different project: {bad_project}",
        )

    bad_kind = [d.id for d in docs if d.kind != "BRD"]
    if bad_kind:
        raise HTTPException(
            400,
            f"Only BRD documents are valid input for BRD→FRD synthesis. "
            f"Wrong kind: {bad_kind}",
        )

    not_parsed = [d.id for d in docs if d.status != "parsed"]
    if not_parsed:
        raise HTTPException(
            409,
            f"Documents must be ingested before synthesis. Not parsed: "
            f"{not_parsed}",
        )

    run = AgentRun(
        project_id=project_id,
        kind="brd_to_frd",
        status="queued",
        input_json={
            "source_document_ids": doc_ids,
            "cap_chunks": payload.cap_chunks,
        },
        output_summary_json={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(execute_brd_to_frd, run.id)
    logger.info(
        "Queued BRD→FRD run %s for project %s (docs=%s, cap=%d)",
        run.id, project_id, doc_ids, payload.cap_chunks,
    )

    return run


@router.post(
    "/execute",
    response_model=AgentRunRead,
    status_code=status.HTTP_201_CREATED,
)
def start_execute(
    project_id: int,
    payload: ExecuteRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Queue an execution run against a plan.

    Pre-flight (fail fast with 4xx before spawning the background task):
    - Project + plan exist; plan belongs to project
    - Plan has a target_url
    - Chromium binary is downloaded — else 503 with the install command
    - If ``selected_step_ids`` provided, every id is a step in this plan
    - At least one step is actually selected (either via the override or
      via ``selectable_default``)

    Subscribe to ``GET /agent-runs/{run.id}/events`` (SSE) for the
    step_started / step_completed stream, plus ``GET /agent-runs/{run.id}/steps``
    for the current row snapshot.
    """
    _require_project(db, project_id)

    plan = db.get(TestPlan, payload.plan_id)
    if not plan:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Plan {payload.plan_id} not found",
        )
    if plan.project_id != project_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Plan {payload.plan_id} belongs to a different project",
        )
    if not (plan.target_url and plan.target_url.strip()):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Plan {plan.id} has no target_url — set one before running",
        )

    if not chromium_installed():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Chromium binary not installed. From v2/backend run: "
            "uv run playwright install chromium",
        )

    # Validate selected_step_ids belong to this plan + are actual steps
    if payload.selected_step_ids is not None:
        wanted_ids = list(dict.fromkeys(payload.selected_step_ids))
        rows = list(
            db.scalars(
                select(TcNode).where(TcNode.id.in_(wanted_ids)),
            ),
        )
        by_id = {n.id: n for n in rows}
        missing = [i for i in wanted_ids if i not in by_id]
        if missing:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Steps not found: {missing}",
            )
        wrong_plan = [
            n.id for n in rows if n.plan_id != plan.id
        ]
        if wrong_plan:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Steps belong to a different plan: {wrong_plan}",
            )
        wrong_kind = [n.id for n in rows if n.kind != "step"]
        if wrong_kind:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Only kind='step' nodes are runnable; got: {wrong_kind}",
            )
    else:
        # Cheap pre-flight count: are there ANY selectable steps to run?
        any_selected = db.scalar(
            select(TcNode.id)
            .where(
                TcNode.plan_id == plan.id,
                TcNode.kind == "step",
                TcNode.selectable_default.is_(True),
            )
            .limit(1),
        )
        if any_selected is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "No steps are selected for execution. Tick at least one step "
                "on the Test Cases tab, or pass selected_step_ids explicitly.",
            )

    run = AgentRun(
        project_id=project_id,
        plan_id=plan.id,
        kind="execute",
        status="queued",
        input_json={
            "plan_id": plan.id,
            "selected_step_ids": payload.selected_step_ids,
            "headless": payload.headless,
            "speed": payload.speed,
            "ai_assist": payload.ai_assist,
            "auto_adjust": payload.auto_adjust,
            "promote_fixes": payload.promote_fixes,
            "mode": payload.mode,
            "window_x": payload.window_x,
            "window_y": payload.window_y,
            "window_width": payload.window_width,
            "window_height": payload.window_height,
        },
        output_summary_json={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(execute_run, run.id)
    logger.info(
        "Queued execute run %s for project %s plan %s (headless=%s)",
        run.id, project_id, plan.id, payload.headless,
    )
    return run


@router.post(
    "/frd-to-tc",
    response_model=AgentRunRead,
    status_code=status.HTTP_201_CREATED,
)
def start_frd_to_tc(
    project_id: int,
    payload: FrdToTcRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Queue a FRD→TC synthesis run for a plan.

    The plan supplies scope + target_url + linked docs; the agent runs one
    LLM call per module in ``plan.scope`` (or one synthetic module if scope
    is empty). Returns 201 with the queued ``AgentRun`` row immediately.
    Subscribe to ``GET /agent-runs/{run.id}/events`` (SSE) to follow progress.
    """
    _require_project(db, project_id)

    plan = db.get(TestPlan, payload.plan_id)
    if not plan:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Plan {payload.plan_id} not found",
        )
    if plan.project_id != project_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Plan {payload.plan_id} belongs to a different project",
        )

    run = AgentRun(
        project_id=project_id,
        plan_id=plan.id,
        kind="frd_to_tc",
        status="queued",
        input_json={
            "plan_id": plan.id,
            "cap_per_module_frds": payload.cap_per_module_frds,
            "cap_per_module_chunks": payload.cap_per_module_chunks,
        },
        output_summary_json={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(execute_frd_to_tc, run.id)
    logger.info(
        "Queued FRD→TC run %s for project %s plan %s",
        run.id,
        project_id,
        plan.id,
    )

    return run


# ── Project-wide SSE stream (literal — declared before /{run_id}) ──


@router.get("/events")
async def stream_project_agent_run_events(
    request: Request,
    project_id: int,
    since_seq: int = 0,
):
    """Subscribe to live events for **all** agent runs in this project.

    The Requirements tab list view uses this to light up a new run card
    without polling. Honors ``Last-Event-ID`` for replay on reconnect.
    """
    return sse_for_topic(
        request,
        topic_for_project_agent_runs(project_id),
        since_seq=since_seq,
    )


# ── List ──────────────────────────────────────────────────────────


@router.get("", response_model=list[AgentRunRead])
def list_runs(
    project_id: int,
    kind: AgentKind | None = None,
    status_filter: AgentStatus | None = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
):
    """List all runs for the project, newest first.

    Optional filters: ``kind`` (e.g. ``brd_to_frd``), ``status``
    (e.g. ``running``, ``completed``).
    """
    _require_project(db, project_id)
    stmt = select(AgentRun).where(AgentRun.project_id == project_id)
    if kind is not None:
        stmt = stmt.where(AgentRun.kind == kind)
    if status_filter is not None:
        stmt = stmt.where(AgentRun.status == status_filter)
    stmt = stmt.order_by(AgentRun.created_at.desc())
    return list(db.scalars(stmt))


# ── Parametric routes ────────────────────────────────────────────


@router.get("/{run_id}", response_model=AgentRunRead)
def get_run(
    project_id: int, run_id: int, db: Session = Depends(get_db),
):
    return _require_run(db, project_id, run_id)


@router.get("/{run_id}/events")
async def stream_run_events(
    request: Request,
    project_id: int,
    run_id: int,
    since_seq: int = 0,
    db: Session = Depends(get_db),
):
    """Subscribe to live events for **one specific run**.

    The synthesis-progress card uses this. Closing the connection auto-frees
    the bus subscriber slot.
    """
    _require_run(db, project_id, run_id)
    return sse_for_topic(request, topic_for_run(run_id), since_seq=since_seq)


@router.post("/{run_id}/cancel", response_model=AgentRunRead)
def cancel_run(
    project_id: int, run_id: int, db: Session = Depends(get_db),
):
    """Mark a run for cancellation. Idempotent on terminal runs."""
    run = _require_run(db, project_id, run_id)

    if run.status in ("completed", "failed", "cancelled"):
        return run  # already terminal — no-op

    request_cancel(run.id)
    logger.info("Cancel requested for run %s (current status=%s)",
                run.id, run.status)
    return run


@router.delete(
    "/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_run(
    project_id: int, run_id: int, db: Session = Depends(get_db),
):
    """Delete a run and its artifacts.

    Refuses while the run is active (queued / running / paused) — the
    user must cancel first, otherwise the runner thread keeps writing
    rows that point at a deleted parent. Once terminal, the row delete
    cascades to ``execution_steps`` (CASCADE FK), and we also remove
    the run's screenshots directory under ``data/screenshots/{run_id}``
    so the disk doesn't grow unbounded.

    Idempotent: deleting a missing run returns 404 via ``_require_run``;
    re-deleting an already-deleted run returns 404.
    """
    run = _require_run(db, project_id, run_id)

    if run.status in ("queued", "running", "paused"):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Cannot delete an active run (status={run.status}). "
            f"Cancel it first via POST /{run_id}/cancel.",
        )

    # Best-effort screenshot cleanup. Failures are logged but don't
    # block the row delete — orphaned files on disk are recoverable
    # (a periodic janitor / manual rm), an orphaned DB row is not.
    try:
        run_dir = settings.screenshots_dir / str(run.id)
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(
            "Failed to remove screenshots dir for run %s: %s", run.id, e,
        )

    db.delete(run)
    db.commit()
    logger.info("Deleted run %s (project %s)", run_id, project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{run_id}/pause", response_model=AgentRunRead)
def pause_run(
    project_id: int, run_id: int, db: Session = Depends(get_db),
):
    """Signal a running run to pause at the next step boundary.

    Pause takes effect *between* steps (mid-step pause is deferred — see
    futurescope.md). The orchestrator flips ``run.status='paused'`` itself
    when it actually halts, so a freshly-pause-requested run still shows
    ``running`` until the current step finishes.

    No-op when:
    - the run is already paused, or
    - the run is in a terminal state (completed / failed / cancelled).

    Returns the current row so the caller can read the live state.
    """
    run = _require_run(db, project_id, run_id)

    if run.status in ("completed", "failed", "cancelled", "paused"):
        return run  # already terminal or already paused — no-op

    request_pause(run.id)
    logger.info("Pause requested for run %s (current status=%s)",
                run.id, run.status)
    return run


@router.post("/{run_id}/resume", response_model=AgentRunRead)
def resume_run(
    project_id: int, run_id: int, db: Session = Depends(get_db),
):
    """Wake a paused run.

    The pause vault sets the resume Event; the orchestrator unblocks at
    its checkpoint, flips ``run.status`` back to ``running``, and emits a
    ``resumed`` SSE event. Idempotent on already-running or terminal runs.
    """
    run = _require_run(db, project_id, run_id)

    if run.status != "paused":
        return run  # not paused — no-op

    request_resume(run.id)
    logger.info("Resume requested for run %s", run.id)
    return run


@router.post("/{run_id}/intervention", response_model=AgentRunRead)
def submit_intervention(
    project_id: int,
    run_id: int,
    payload: InterventionRequest,
    db: Session = Depends(get_db),
):
    """Resolve an HITL block on a step that survived auto-retry + AI assist.

    Validates that:
    - run exists and belongs to the project
    - run is not in a terminal state (a stuck run can still receive choices
      until it's cancelled)
    - the named ``ExecutionStep`` row belongs to this run
    - an intervention is currently pending in the vault for ``(run_id, step_id)``

    On the happy path, the orchestrator's ``wait_for_intervention`` call
    unblocks with the user's choice payload, applies the override (if
    ``use_suggestion``), and continues the loop. The status the run lands
    on after this is whatever the post-resolution attempt returned —
    ``passed`` if the override worked, ``failed`` if not, or ``cancelled``
    if the user picked ``stop``.

    Returns 409 when no waiter is pending — the step was already resolved
    or never blocked. The frontend treats that as a benign race and
    refetches step state.
    """
    run = _require_run(db, project_id, run_id)

    if run.status in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Run is in terminal state {run.status!r}; cannot accept intervention.",
        )

    # Confirm the step belongs to this run (cheap: we don't need the row,
    # just an existence check via the FK).
    from app.models.execution_step import ExecutionStep  # local — avoids cycle
    step = db.get(ExecutionStep, payload.step_id)
    if step is None or step.run_id != run_id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Step {payload.step_id} not found on run {run_id}",
        )

    if not _has_pending_intervention(run_id, payload.step_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No intervention is pending for this step. It was already "
            "resolved or never reached the HITL gate.",
        )

    choice_payload = {
        "choice": payload.choice,
        "override_target_hint": payload.override_target_hint,
        "override_action_type": payload.override_action_type,
        "apply_to_submodule": payload.apply_to_submodule,
    }
    delivered = provide_intervention(run.id, payload.step_id, choice_payload)
    if not delivered:
        # Race: the waiter exited (cancel) between our check and the
        # provide_intervention call. Same UX as not-pending.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Intervention waiter is no longer active.",
        )

    logger.info(
        "Intervention delivered for run %s step %s: %s%s",
        run.id,
        payload.step_id,
        payload.choice,
        " (apply_to_submodule)" if payload.apply_to_submodule else "",
    )
    return run


@router.get("/{run_id}/steps", response_model=list[ExecutionStepRead])
def list_run_steps(
    project_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    """Return the per-step rows for an ``execute`` run.

    Returns an empty list for non-execute runs so the frontend can call this
    blindly without branching on ``kind``. Rows are returned in execution
    order (``ordinal`` ascending), so the timeline UI renders top-to-bottom
    without sorting.
    """
    run = _require_run(db, project_id, run_id)
    if run.kind != "execute":
        return []
    stmt = (
        select(ExecutionStep)
        .where(ExecutionStep.run_id == run_id)
        .order_by(ExecutionStep.ordinal)
    )
    return list(db.scalars(stmt))


@router.get("/{run_id}/report", response_model=ReportRead)
def get_run_report(
    project_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    """Aggregated run report: run summary + plan info + module/submodule
    grouping with per-step rows.

    Only ``kind='execute'`` runs have meaningful reports. Returns 400 for
    other agent kinds so the frontend can render a clear "no report
    available" state. Runs that are still active (queued/running) DO get
    reports — the data is partial but useful for live observation.
    """
    run = _require_run(db, project_id, run_id)
    if run.kind != "execute":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Reports are only available for execute runs (got kind={run.kind!r})",
        )

    excel_url = (
        f"/api/projects/{project_id}/agent-runs/{run_id}/report.xlsx"
    )
    return build_run_report(db, run, excel_url=excel_url)


@router.get("/{run_id}/report.xlsx")
def get_run_report_xlsx(
    project_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    """Download the run report as an Excel workbook.

    Generated on demand from the same aggregate as the JSON endpoint —
    no caching. Two sheets: Summary (run/plan/AI cost) + Results (one
    row per step with module/submodule/title/status/issues).
    """
    run = _require_run(db, project_id, run_id)
    if run.kind != "execute":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Reports are only available for execute runs (got kind={run.kind!r})",
        )

    excel_url = (
        f"/api/projects/{project_id}/agent-runs/{run_id}/report.xlsx"
    )
    report = build_run_report(db, run, excel_url=excel_url)
    xlsx_bytes = build_excel_workbook(report)

    filename = f"qai-run-{run_id}-report.xlsx"
    return Response(
        content=xlsx_bytes,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            # No-cache so re-running invalidates the download
            "Cache-Control": "no-store",
        },
    )
