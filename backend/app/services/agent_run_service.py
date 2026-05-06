"""Agent-run runtime — wraps a pure agent function in an ``AgentRun`` lifecycle.

Responsibilities
----------------
1. Manage the ``AgentRun`` row state machine
       queued → running → completed
                        ↘ failed
                        ↘ cancelled
2. Build the LLM provider from ``app_settings``.
3. Translate the orchestrator's ``emit_event(type, data)`` callback into
   ``bus.publish()`` on two topics:
   - ``agent_run:<id>``                — for the run-detail subscriber
   - ``project:<pid>:agent_runs``       — for project-wide list subscribers
4. Provide a process-wide cancel registry that the orchestrator's
   ``is_cancelled`` callback polls at safe checkpoints.

The runner functions below (``execute_brd_to_frd``) are the entry points the
agent-runs router schedules via FastAPI's ``BackgroundTasks``.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from app.agents import (
    AgentCancelled,
    execute_plan,
    synthesize_frd,
    synthesize_tc,
)
from app.db import SessionLocal
from app.executor import BrowserNotInstalledError
from app.llm.factory import get_provider_from_db
from app.models.agent_run import AgentRun
from app.sse.bus import get_bus

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Cancellation registry ────────────────────────────────────────

_cancelled_runs: set[int] = set()
_cancel_lock = threading.Lock()


def request_cancel(run_id: int) -> None:
    """Mark a run for cancellation.

    The runner polls this at safe checkpoints; cancellation takes effect
    *between phases* of the orchestrator (week 3 doesn't interrupt mid-LLM).
    Idempotent — calling on a non-running run is a no-op.

    Wakes both the pause-waiter and any pending intervention-waiters so
    a single cancel drains every blocking primitive at once. Otherwise
    a step blocked on HITL would hold the run for the indefinite wait.
    """
    with _cancel_lock:
        _cancelled_runs.add(run_id)
    # Wake any pause-waiter so it can observe the cancel and exit cleanly
    # instead of waiting on the resume event indefinitely.
    with _pause_lock:
        ev = _resume_events.get(run_id)
    if ev:
        ev.set()
    # Wake every intervention waiter for this run.
    _drop_interventions(run_id)


def _is_cancelled(run_id: int) -> bool:
    with _cancel_lock:
        return run_id in _cancelled_runs


def _drop_cancel(run_id: int) -> None:
    with _cancel_lock:
        _cancelled_runs.discard(run_id)


# ── Pause registry ───────────────────────────────────────────────

_paused_runs: set[int] = set()
_resume_events: dict[int, "threading.Event"] = {}
_pause_lock = threading.Lock()


def request_pause(run_id: int) -> None:
    """Signal a run to pause at the next safe checkpoint.

    Pause takes effect between steps (the orchestrator polls at the same
    boundary as cancellation — mid-step pause is deferred). Idempotent.
    Cancel-while-paused wakes the waiter via :func:`request_cancel`.
    """
    with _pause_lock:
        _paused_runs.add(run_id)
        if run_id not in _resume_events:
            _resume_events[run_id] = threading.Event()


def request_resume(run_id: int) -> None:
    """Resume a paused run. Idempotent on non-paused runs."""
    with _pause_lock:
        _paused_runs.discard(run_id)
        ev = _resume_events.pop(run_id, None)
    if ev:
        ev.set()


def _is_paused(run_id: int) -> bool:
    with _pause_lock:
        return run_id in _paused_runs


def _wait_until_resumed_or_cancelled(
    run_id: int, *, poll_interval_s: float = 0.5,
) -> bool:
    """Block until the run is resumed or cancelled.

    Returns True on resume, False on cancel. Polls every ``poll_interval_s``
    so a cancel arriving via :func:`request_cancel` (which sets the same
    event) wakes us promptly.
    """
    with _pause_lock:
        ev = _resume_events.get(run_id)
    if ev is None:
        return True  # not paused — nothing to wait for
    while True:
        if ev.wait(poll_interval_s):
            # Set by either request_resume (resumed) or request_cancel (cancelled)
            return not _is_cancelled(run_id)
        if _is_cancelled(run_id):
            return False


def _drop_pause(run_id: int) -> None:
    with _pause_lock:
        _paused_runs.discard(run_id)
        _resume_events.pop(run_id, None)


# ── HITL intervention vault ──────────────────────────────────────
#
# When auto-retry + AI assist both fail on a step, the orchestrator marks
# the row ``blocked`` and waits here for the user to pick: retry / use
# AI suggestion / skip / stop. Same Event+dict shape as the OTP vault
# captured in futurescope.md, but keyed by ``(run_id, step_id)`` since a
# single run can have multiple stuck steps.
#
# Cancel-while-blocked must wake every waiter for that run; we layer that
# into ``request_cancel`` below so a single cancel button drains pause +
# intervention + cancel together.

_intervention_events: dict[tuple[int, int], "threading.Event"] = {}
_intervention_responses: dict[tuple[int, int], dict] = {}
_intervention_lock = threading.Lock()


def request_intervention(
    run_id: int,
    step_id: int,
    *,
    poll_interval_s: float = 0.5,
) -> dict | None:
    """Block until the user submits an intervention or the run is cancelled.

    Returns the user's choice payload (a dict with ``choice`` plus any
    overrides) on submit. Returns ``None`` if the run was cancelled
    while we were waiting — callers should treat that as "skip this step
    and exit the loop".

    Indefinite wait per the user's spec (Q1: "wait until user takes
    action"). The poll interval just lets us re-check the cancel flag;
    cancellation also calls ``ev.set()`` directly so the wait wakes
    promptly without depending on the poll period.
    """
    key = (run_id, step_id)
    with _intervention_lock:
        ev = _intervention_events.setdefault(key, threading.Event())

    while True:
        if ev.wait(poll_interval_s):
            with _intervention_lock:
                response = _intervention_responses.pop(key, None)
                _intervention_events.pop(key, None)
            if _is_cancelled(run_id):
                return None
            return response
        if _is_cancelled(run_id):
            with _intervention_lock:
                _intervention_events.pop(key, None)
                _intervention_responses.pop(key, None)
            return None


def provide_intervention(
    run_id: int, step_id: int, choice: dict,
) -> bool:
    """Unblock a waiting intervention with the user's choice payload.

    Returns True if a waiter was found, False if the step isn't blocked
    (either never blocked, or already resolved). The endpoint surfaces
    False as a 409 so the UI can refetch state.
    """
    key = (run_id, step_id)
    with _intervention_lock:
        ev = _intervention_events.get(key)
        if ev is None:
            return False
        _intervention_responses[key] = dict(choice)
        ev.set()
    return True


def _has_pending_intervention(run_id: int, step_id: int) -> bool:
    """True if request_intervention is currently waiting on this step."""
    with _intervention_lock:
        return (run_id, step_id) in _intervention_events


def _drop_interventions(run_id: int) -> None:
    """Clean up any pending interventions for a run.

    Called on cancel and from the runtime's ``finally`` block. Sets the
    Event so any active waiter wakes up immediately; the waiter's own
    cleanup pops the keys.
    """
    with _intervention_lock:
        keys = [k for k in _intervention_events if k[0] == run_id]
        events = [_intervention_events[k] for k in keys]
    for ev in events:
        ev.set()


# ── SSE topic helpers ────────────────────────────────────────────


def topic_for_run(run_id: int) -> str:
    return f"agent_run:{run_id}"


def topic_for_project_agent_runs(project_id: int) -> str:
    return f"project:{project_id}:agent_runs"


def _emit_run_event(run: AgentRun, event_type: str, data: dict) -> None:
    """Publish to the per-run topic AND the per-project topic.

    The frontend's run-detail subscribes to per-run; the Requirements tab's
    list view subscribes to per-project so it lights up new runs without a
    refresh.
    """
    bus = get_bus()
    payload = {"run_id": run.id, "kind": run.kind, "status": run.status, **data}
    bus.publish(topic_for_run(run.id), event_type, payload)
    bus.publish(topic_for_project_agent_runs(run.project_id), event_type, payload)


def _mark_failed(db, run: AgentRun, message: str) -> None:
    logger.warning("Agent run %s failed: %s", run.id, message)
    run.status = "failed"
    run.completed_at = _utcnow()
    run.error_message = message[:2000]
    db.commit()
    _emit_run_event(run, "failed", {"error": message[:500]})


def _mark_cancelled(db, run: AgentRun, reason: str) -> None:
    logger.info("Agent run %s cancelled: %s", run.id, reason)
    run.status = "cancelled"
    run.completed_at = _utcnow()
    run.error_message = reason[:2000]
    db.commit()
    _emit_run_event(run, "cancelled", {"message": reason[:500]})


# ── Runners (one per agent kind) ─────────────────────────────────


def execute_brd_to_frd(run_id: int) -> None:
    """Background-task entry for a ``brd_to_frd`` agent run.

    Reads the ``AgentRun`` row, builds the provider, calls
    :func:`app.agents.brd_to_frd.synthesize_frd`, and records the outcome.
    """
    db = SessionLocal()
    try:
        run = db.get(AgentRun, run_id)
        if not run:
            logger.warning("execute_brd_to_frd: run %s not found", run_id)
            return

        if run.kind != "brd_to_frd":
            logger.error(
                "execute_brd_to_frd called for run %s with kind=%r",
                run.id,
                run.kind,
            )
            return

        # Reject runs that aren't in a startable state. Belt-and-braces: the
        # router only schedules from 'queued', but this guards against retries.
        if run.status != "queued":
            logger.warning(
                "execute_brd_to_frd: run %s is in status %r, not queued",
                run.id,
                run.status,
            )
            return

        # Honor a cancel that arrived before the runner picked it up
        if _is_cancelled(run.id):
            _mark_cancelled(db, run, "Cancelled before run started")
            return

        # ── Transition: queued → running ───────────────────────
        run.status = "running"
        run.started_at = _utcnow()
        db.commit()

        _emit_run_event(
            run,
            "started",
            {
                "input": run.input_json,
                "started_at": run.started_at.isoformat() if run.started_at else None,
            },
        )

        # ── Build provider ─────────────────────────────────────
        try:
            provider = get_provider_from_db(db)
        except RuntimeError as e:
            _mark_failed(db, run, f"LLM not configured: {e}")
            return

        # ── Pull params from input_json ────────────────────────
        input_data = run.input_json or {}
        source_doc_ids = input_data.get("source_document_ids", [])
        cap_chunks = input_data.get("cap_chunks", 50)

        if not isinstance(source_doc_ids, list) or not source_doc_ids:
            _mark_failed(
                db,
                run,
                "input.source_document_ids must be a non-empty list",
            )
            return

        # ── Run the orchestrator ───────────────────────────────
        try:
            result = synthesize_frd(
                db=db,
                provider=provider,
                project_id=run.project_id,
                source_document_ids=source_doc_ids,
                cap_chunks=int(cap_chunks),
                emit_event=lambda et, data: _emit_run_event(run, et, data),
                is_cancelled=lambda: _is_cancelled(run.id),
            )
        except AgentCancelled as e:
            _mark_cancelled(db, run, str(e))
            return
        except (ValueError, RuntimeError) as e:
            _mark_failed(db, run, f"{type(e).__name__}: {e}")
            return
        except Exception as e:
            logger.exception("Unexpected error in BRD→FRD run %s", run.id)
            _mark_failed(db, run, f"{type(e).__name__}: {e}")
            return

        # ── Transition: running → completed ────────────────────
        run.status = "completed"
        run.completed_at = _utcnow()
        run.output_summary_json = {
            "generated": result.generated_count,
            "requirement_ids": result.requirement_ids,
            "chunks_seen": result.chunks_seen,
            "truncated": result.truncated,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
        db.commit()
        _emit_run_event(run, "completed", run.output_summary_json)
        logger.info(
            "Agent run %s completed: %d FRDs from %d chunks",
            run.id,
            result.generated_count,
            result.chunks_seen,
        )

    finally:
        _drop_cancel(run_id)
        _drop_pause(run_id)
        _drop_interventions(run_id)
        db.close()


def execute_frd_to_tc(run_id: int) -> None:
    """Background-task entry for a ``frd_to_tc`` agent run.

    Reads the ``AgentRun`` row, builds the provider, calls
    :func:`app.agents.frd_to_tc.synthesize_tc`, and records the outcome.

    Same lifecycle pattern as :func:`execute_brd_to_frd` — pre-flight cancel
    check, transition queued→running, run orchestrator, mark terminal status.
    """
    db = SessionLocal()
    try:
        run = db.get(AgentRun, run_id)
        if not run:
            logger.warning("execute_frd_to_tc: run %s not found", run_id)
            return

        if run.kind != "frd_to_tc":
            logger.error(
                "execute_frd_to_tc called for run %s with kind=%r",
                run.id,
                run.kind,
            )
            return

        if run.status != "queued":
            logger.warning(
                "execute_frd_to_tc: run %s is in status %r, not queued",
                run.id,
                run.status,
            )
            return

        if _is_cancelled(run.id):
            _mark_cancelled(db, run, "Cancelled before run started")
            return

        # ── Transition: queued → running ───────────────────────
        run.status = "running"
        run.started_at = _utcnow()
        db.commit()

        _emit_run_event(
            run,
            "started",
            {
                "input": run.input_json,
                "started_at": run.started_at.isoformat() if run.started_at else None,
            },
        )

        # ── Build provider ─────────────────────────────────────
        try:
            provider = get_provider_from_db(db)
        except RuntimeError as e:
            _mark_failed(db, run, f"LLM not configured: {e}")
            return

        # ── Pull params from input_json ────────────────────────
        input_data = run.input_json or {}
        plan_id = input_data.get("plan_id")
        if not isinstance(plan_id, int) or plan_id <= 0:
            _mark_failed(
                db,
                run,
                "input.plan_id must be a positive integer",
            )
            return

        cap_frds = int(input_data.get("cap_per_module_frds", 15))
        cap_chunks = int(input_data.get("cap_per_module_chunks", 10))

        # ── Run the orchestrator ───────────────────────────────
        try:
            result = synthesize_tc(
                db=db,
                provider=provider,
                plan_id=plan_id,
                cap_per_module_frds=cap_frds,
                cap_per_module_chunks=cap_chunks,
                emit_event=lambda et, data: _emit_run_event(run, et, data),
                is_cancelled=lambda: _is_cancelled(run.id),
            )
        except AgentCancelled as e:
            _mark_cancelled(db, run, str(e))
            return
        except (ValueError, RuntimeError) as e:
            _mark_failed(db, run, f"{type(e).__name__}: {e}")
            return
        except Exception as e:
            logger.exception("Unexpected error in FRD→TC run %s", run.id)
            _mark_failed(db, run, f"{type(e).__name__}: {e}")
            return

        # ── Transition: running → completed ────────────────────
        run.status = "completed"
        run.completed_at = _utcnow()
        run.output_summary_json = {
            "plan_id": result.plan_id,
            "modules_requested": result.modules_requested,
            "modules_generated": result.modules_generated,
            "modules_skipped": result.modules_skipped,
            "nodes_total": result.nodes_total,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
        db.commit()
        _emit_run_event(run, "completed", run.output_summary_json)
        logger.info(
            "Agent run %s completed: %d module(s), %d nodes",
            run.id,
            result.modules_generated,
            result.nodes_total,
        )

    finally:
        _drop_cancel(run_id)
        _drop_pause(run_id)
        _drop_interventions(run_id)
        db.close()


def execute_run(run_id: int) -> None:
    """Background-task entry for an ``execute`` agent run.

    Calls :func:`app.agents.execute.execute_plan`, which drives a Playwright
    browser through the plan's selected steps, persisting per-step results
    as ``execution_steps`` rows.

    Differences vs. the LLM-driven runners:
    - No provider is built — the executor is browser-driven.
    - :class:`BrowserNotInstalledError` is caught separately so the run
      surfaces a single actionable error message ("run ``uv run playwright
      install chromium``") instead of a generic crash trace.
    - ``execute_plan`` itself owns the per-step lifecycle and its own
      cancellation polling, so this runner just translates outcomes.
    """
    db = SessionLocal()
    try:
        run = db.get(AgentRun, run_id)
        if not run:
            logger.warning("execute_run: run %s not found", run_id)
            return

        if run.kind != "execute":
            logger.error(
                "execute_run called for run %s with kind=%r",
                run.id,
                run.kind,
            )
            return

        if run.status != "queued":
            logger.warning(
                "execute_run: run %s is in status %r, not queued",
                run.id,
                run.status,
            )
            return

        if _is_cancelled(run.id):
            _mark_cancelled(db, run, "Cancelled before run started")
            return

        # ── Transition: queued → running ───────────────────────
        run.status = "running"
        run.started_at = _utcnow()
        db.commit()

        _emit_run_event(
            run,
            "started",
            {
                "input": run.input_json,
                "started_at": run.started_at.isoformat() if run.started_at else None,
            },
        )

        # ── Pull params from input_json ────────────────────────
        input_data = run.input_json or {}
        plan_id = input_data.get("plan_id")
        if not isinstance(plan_id, int) or plan_id <= 0:
            _mark_failed(
                db, run, "input.plan_id must be a positive integer",
            )
            return

        selected_step_ids = input_data.get("selected_step_ids")
        if selected_step_ids is not None and not (
            isinstance(selected_step_ids, list)
            and all(isinstance(i, int) and i > 0 for i in selected_step_ids)
        ):
            _mark_failed(
                db,
                run,
                "input.selected_step_ids must be omitted or a list of positive ints",
            )
            return

        headless = bool(input_data.get("headless", False))

        speed_raw = input_data.get("speed")
        if speed_raw is not None and not isinstance(speed_raw, str):
            _mark_failed(
                db, run, "input.speed must be omitted or a string",
            )
            return

        ai_assist = bool(input_data.get("ai_assist", True))
        auto_adjust = bool(input_data.get("auto_adjust", False))
        promote_fixes = bool(input_data.get("promote_fixes", False))

        # Build the LLM provider for AI assist on failure. Missing config is
        # NOT fatal — the run continues with ai_assist disabled. This keeps
        # the executor usable without an LLM (auto-retry only) and supports
        # the "tester ran offline" scenario.
        provider = None
        if ai_assist:
            try:
                provider = get_provider_from_db(db)
            except RuntimeError as e:
                logger.info(
                    "AI assist disabled for run %s — no LLM configured: %s",
                    run.id, e,
                )

        # ── Run the orchestrator ───────────────────────────────
        try:
            result = execute_plan(
                db=db,
                run_id=run.id,
                plan_id=plan_id,
                selected_step_ids=selected_step_ids,
                headless=headless,
                speed=speed_raw,
                provider=provider,
                ai_assist=ai_assist,
                auto_adjust=auto_adjust,
                promote_fixes=promote_fixes,
                emit_event=lambda et, data: _emit_run_event(run, et, data),
                is_cancelled=lambda: _is_cancelled(run.id),
                is_paused=lambda: _is_paused(run.id),
                wait_for_resume=lambda: _wait_until_resumed_or_cancelled(run.id),
                wait_for_intervention=lambda step_id: request_intervention(run.id, step_id),
            )
        except AgentCancelled as e:
            _mark_cancelled(db, run, str(e))
            return
        except BrowserNotInstalledError as e:
            _mark_failed(db, run, str(e))
            return
        except (ValueError, RuntimeError) as e:
            _mark_failed(db, run, f"{type(e).__name__}: {e}")
            return
        except Exception as e:
            logger.exception("Unexpected error in execute run %s", run.id)
            _mark_failed(db, run, f"{type(e).__name__}: {e}")
            return

        # ── Transition: running → completed ────────────────────
        run.status = "completed"
        run.completed_at = _utcnow()
        run.output_summary_json = {
            "plan_id": result.plan_id,
            "total_steps": result.total_steps,
            "passed": result.passed,
            "failed": result.failed,
            "skipped": result.skipped,
            "blocked": result.blocked,
            "duration_ms": result.duration_ms,
            # AI-assist cost meter — None when no AI call happened
            "llm_input_tokens": result.llm_input_tokens,
            "llm_output_tokens": result.llm_output_tokens,
            "ai_calls": result.ai_calls,
            "ai_vision_calls": result.ai_vision_calls,
        }
        db.commit()
        _emit_run_event(run, "completed", run.output_summary_json)
        logger.info(
            "Execute run %s completed: %d/%d passed (%dms)",
            run.id,
            result.passed,
            result.total_steps,
            result.duration_ms,
        )

    finally:
        _drop_cancel(run_id)
        _drop_pause(run_id)
        _drop_interventions(run_id)
        db.close()
