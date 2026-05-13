"""Phase W — recording event ingest + lifecycle endpoints.

Two router groups:

1. ``/api/recordings/{run_id}/events`` (no auth, project-agnostic) —
   the JS injected into the recording browser POSTs batched user
   events here. The endpoint MUST be accessible from the browser's
   origin (i.e. the chromium window that's recording). We accept
   any Origin so the JS doesn't trip CORS preflight on its own
   localhost.

2. ``/api/projects/{project_id}/agent-runs/start-recording``,
   ``/.../{run_id}/stop-recording`` — operator-driven lifecycle.
   These attach to the existing project-scoped agent-runs router so
   the frontend's existing api.startExecute / cancelAgentRun
   patterns apply.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.services.submodule_recording import (
    buffer_size,
    buffer_state,
    discard_buffer,
    push_events,
    set_active_submodule,
    signal_stop,
)

logger = logging.getLogger(__name__)


# ── Public ingest router (mounted at /api/recordings) ───────────


public_router = APIRouter(
    prefix="/api/recordings",
    tags=["Recording (ingest)"],
)


class RecordingEventBatch(BaseModel):
    events: list[dict[str, Any]] = Field(default_factory=list)


@public_router.post("/{run_id}/events")
async def post_events(
    run_id: int,
    request: Request,
    db: Session = Depends(get_db),  # noqa: ARG001 — kept for dep parity
):
    """Receive a batch of capture events from the injected JS.

    The JS POSTs JSON ``{events: [...]}`` with ``credentials: "omit"``.
    We accept the payload as-is and push into the per-run buffer.
    Returns the running total so the JS can log / debug.

    No auth — the run_id IS the (weak) capability token. In a local
    deployment this is fine; for a hosted version a short-lived
    signed token in the URL would replace this.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "expected JSON body")
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    events = body.get("events") or []
    if not isinstance(events, list):
        raise HTTPException(400, "events must be a list")
    # Trim each event to a sane size — defends against runaway DOM
    # text capture pumping huge strings into the buffer.
    trimmed: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        # Cap the largest fields.
        if isinstance(ev.get("value"), str) and len(ev["value"]) > 2000:
            ev["value"] = ev["value"][:2000]
        tgt = ev.get("target")
        if isinstance(tgt, dict):
            for k in ("text", "aria_label", "title", "placeholder"):
                if isinstance(tgt.get(k), str) and len(tgt[k]) > 240:
                    tgt[k] = tgt[k][:240]
        trimmed.append(ev)
    summary = push_events(run_id, trimmed)
    return {
        "received": len(trimmed),
        **summary,
    }


# ── Project-scoped lifecycle router (mounted at the agent-runs path)
#
# Re-uses the existing agent-runs prefix so the frontend's
# api.startExecute pattern carries.


lifecycle_router = APIRouter(
    prefix="/api/projects/{project_id}/agent-runs",
    tags=["Recording (lifecycle)"],
)


class StartRecordingRequest(BaseModel):
    """Phase W' — a recording session covers an entire MODULE.
    Submodule attribution happens live via set-active-submodule."""
    plan_id: int
    module_id: int


class SetActiveSubmoduleRequest(BaseModel):
    submodule_id: int


@lifecycle_router.post("/start-recording")
def start_recording(
    project_id: int,
    payload: StartRecordingRequest,
    db: Session = Depends(get_db),
):
    """Phase W' — start a per-MODULE recording session.

    Covers a whole module; the operator picks the active submodule
    live on the presenter (via ``set-active-submodule``). Creates an
    ``agent_runs(kind="record")`` row and queues the background task
    that launches the browser. Returns the run as ``AgentRunRead``.
    """
    from app.models.agent_run import AgentRun  # noqa: PLC0415
    from app.models.project import Project  # noqa: PLC0415
    from app.models.tc_node import TcNode  # noqa: PLC0415
    from app.models.test_plan import TestPlan  # noqa: PLC0415
    from app.schemas.agent_run import AgentRunRead  # noqa: PLC0415
    from app.services.agent_run_service import execute_recording  # noqa: PLC0415

    # Validate ownership chain.
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, f"project {project_id} not found")
    plan = db.get(TestPlan, payload.plan_id)
    if plan is None or plan.project_id != project_id:
        raise HTTPException(
            404, f"plan {payload.plan_id} not found on project {project_id}",
        )
    target_url = (plan.target_url or "").strip()
    if not target_url:
        raise HTTPException(
            400,
            "plan has no target_url; recording needs a URL to navigate to",
        )
    module = db.get(TcNode, payload.module_id)
    if (
        module is None
        or module.plan_id != plan.id
        or module.kind != "module"
    ):
        raise HTTPException(
            404,
            f"module {payload.module_id} not found on plan "
            f"{payload.plan_id}",
        )

    run = AgentRun(
        project_id=project_id,
        kind="record",
        status="queued",
        input_json={
            "plan_id": payload.plan_id,
            "module_id": payload.module_id,
        },
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Spawn in a daemon thread (FastAPI BackgroundTasks would tie
    # the task to the request lifetime; recordings outlive the
    # request).
    import threading  # noqa: PLC0415
    t = threading.Thread(
        target=execute_recording,
        args=(run.id,),
        daemon=True,
    )
    t.start()

    return AgentRunRead.model_validate(run)


@lifecycle_router.post("/{run_id}/active-submodule")
def set_recording_active_submodule(
    project_id: int,
    run_id: int,
    payload: SetActiveSubmoduleRequest,
    db: Session = Depends(get_db),
):
    """Phase W' — switch which submodule subsequent captured events
    attribute to. Called when the operator clicks "Start chunk" on
    the live presenter after picking a submodule from the searchable
    combobox.
    """
    from app.models.agent_run import AgentRun  # noqa: PLC0415
    from app.models.tc_node import TcNode  # noqa: PLC0415

    run = db.get(AgentRun, run_id)
    if run is None or run.project_id != project_id or run.kind != "record":
        raise HTTPException(
            404,
            f"recording run {run_id} not found on project {project_id}",
        )
    # Verify the submodule belongs to the module this run is recording.
    module_id = int((run.input_json or {}).get("module_id") or 0)
    sm = db.get(TcNode, payload.submodule_id)
    if (
        sm is None
        or sm.kind != "submodule"
        or (module_id and sm.parent_id != module_id)
    ):
        raise HTTPException(
            400,
            f"submodule {payload.submodule_id} is not under the "
            f"module {module_id} this recording is scoped to",
        )
    ok = set_active_submodule(run_id, payload.submodule_id)
    state = buffer_state(run_id)
    if not ok:
        raise HTTPException(
            409, "recording run not currently active",
        )
    return {
        "run_id": run_id,
        "active_submodule_id": payload.submodule_id,
        **state,
    }


@lifecycle_router.get("/{run_id}/recording-state")
def get_recording_state(
    project_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    """Phase W' — snapshot for the live presenter. Returns the
    active submodule + per-submodule event counts. Cheap; the
    presenter can poll every 2-3s while a recording is open.
    """
    from app.models.agent_run import AgentRun  # noqa: PLC0415

    run = db.get(AgentRun, run_id)
    if run is None or run.project_id != project_id or run.kind != "record":
        raise HTTPException(
            404,
            f"recording run {run_id} not found on project {project_id}",
        )
    state = buffer_state(run_id)
    return {
        "run_id": run_id,
        "module_id": int((run.input_json or {}).get("module_id") or 0),
        "status": run.status,
        **state,
    }


@lifecycle_router.post("/{run_id}/stop-recording")
def stop_recording(
    project_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    """Phase W — operator signals end of recording.

    Sets the per-run stop event so the background task closes the
    browser and persists the buffer to the submodule's frozen_path.
    Returns the buffered event count so the frontend can show a
    confirmation toast.
    """
    from app.models.agent_run import AgentRun  # noqa: PLC0415

    run = db.get(AgentRun, run_id)
    if run is None or run.project_id != project_id or run.kind != "record":
        raise HTTPException(
            404, f"recording run {run_id} not found on project {project_id}",
        )

    delivered = signal_stop(run_id)
    bsize = buffer_size(run_id)
    return {
        "run_id": run_id,
        "stop_delivered": delivered,
        "buffered_events": bsize,
        "note": (
            "Browser will close and the recording will be saved "
            "shortly. Watch the live feed for the recording_saved "
            "event."
            if delivered
            else "No active recording for this run id."
        ),
    }


@lifecycle_router.delete("/{run_id}/recording")
def discard_recording(
    project_id: int,
    run_id: int,
    db: Session = Depends(get_db),
):
    """Cancel an in-progress recording WITHOUT persisting."""
    from app.models.agent_run import AgentRun  # noqa: PLC0415
    from app.services.agent_run_service import request_cancel  # noqa: PLC0415

    run = db.get(AgentRun, run_id)
    if run is None or run.project_id != project_id or run.kind != "record":
        raise HTTPException(
            404, f"recording run {run_id} not found on project {project_id}",
        )
    request_cancel(run_id)
    discard_buffer(run_id)
    return {"run_id": run_id, "cancelled": True}
