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

    Phase W — also wakes the recording stop event so a cancelled
    recording session closes its browser instead of hanging on the
    operator's "Stop" button.
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
    # Wake any active recording session waiter.
    try:
        from app.services.submodule_recording import signal_stop  # noqa: PLC0415
        signal_stop(run_id)
    except Exception:
        pass


def _is_cancelled(run_id: int) -> bool:
    with _cancel_lock:
        return run_id in _cancelled_runs


def _drop_cancel(run_id: int) -> None:
    with _cancel_lock:
        _cancelled_runs.discard(run_id)


# ── Force-pass registry ──────────────────────────────────────────
#
# Phase AA — operator force-pass shortcut (Ctrl+Shift+D on the live
# presenter). The operator has visually verified the test passed and
# is short-circuiting the remaining LLM/replay work. The run resolves
# as ``completed`` with every test case marked ``passed``, NOT as
# ``cancelled``. Distinct from request_cancel so reports + summaries
# show 100% pass rather than "Cancelled mid-loop".

_force_passed_runs: set[int] = set()
_force_pass_lock = threading.Lock()


def request_force_pass(run_id: int) -> None:
    """Mark a run as force-passed by the operator.

    Polled at the same safe checkpoints as cancellation. When the
    runner sees this flag, it stops at the current submodule, marks
    that submodule + every remaining submodule as ``passed``, and
    completes the run cleanly. The narration records that this was
    an operator decision (Ctrl+Shift+D), so reports/audits can tell
    "automation-passed" from "operator-asserted-passed".

    Idempotent. Also wakes pause / HITL waiters so the run can exit
    its current blocking primitive immediately.
    """
    with _force_pass_lock:
        _force_passed_runs.add(run_id)
    # Wake pause waiters so the loop can observe the flag.
    with _pause_lock:
        ev = _resume_events.get(run_id)
    if ev:
        ev.set()
    # Wake every HITL intervention waiter.
    _drop_interventions(run_id)


def _is_force_passed(run_id: int) -> bool:
    with _force_pass_lock:
        return run_id in _force_passed_runs


def _drop_force_pass(run_id: int) -> None:
    with _force_pass_lock:
        _force_passed_runs.discard(run_id)


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
# Phase 4 — typed HITL prompts. Tracks the OPEN prompt for a
# (run_id, step_id) so the live presenter can render the right
# input form (OTP code box, credentials pair, manual-solve resume
# button). Set by ``open_typed_prompt``; cleared on response
# delivery. Read by GET endpoint or surfaced via SSE event.
_open_prompts: dict[tuple[int, int], dict] = {}
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


def open_typed_prompt(
    run_id: int,
    step_id: int,
    *,
    kind: str,
    question: str,
    fields: list[dict] | None = None,
) -> None:
    """Phase 4 — record an open typed HITL prompt.

    Called by the agent's auth-flow orchestrator BEFORE blocking on
    ``request_intervention`` so the live presenter has enough info
    to render the right input form. ``kind`` is one of:
      - ``request_text`` (single free-form input — OTP, captcha solve)
      - ``request_credentials`` (paired username + password)
      - ``await_manual_solve`` (just a "I solved it, continue" button)

    ``fields`` lets the agent specify what to label each input field
    (e.g. ``[{name: "otp", label: "Enter the 6-digit code"}]``).
    Optional — when None the presenter uses sensible defaults from
    ``kind``.

    Pairs with ``close_typed_prompt`` (called by provide_intervention
    on response delivery).
    """
    key = (run_id, step_id)
    with _intervention_lock:
        _open_prompts[key] = {
            "kind": kind,
            "question": question[:400],
            "fields": list(fields or []),
        }


def get_open_prompt(run_id: int, step_id: int) -> dict | None:
    """Return the open typed prompt for a step, or None when there
    isn't one. Used by the GET endpoint that the popup polls on
    mount (in case it missed the SSE open-event)."""
    with _intervention_lock:
        prompt = _open_prompts.get((run_id, step_id))
        return dict(prompt) if prompt else None


def close_typed_prompt(run_id: int, step_id: int) -> None:
    with _intervention_lock:
        _open_prompts.pop((run_id, step_id), None)


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
        # Phase 4 — clear any open typed prompt so a stale form
        # doesn't render after the agent's already received the
        # value and moved on.
        _open_prompts.pop(key, None)
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
        # Phase 4 — also drop any open typed prompts for this run so
        # the popup-side form doesn't linger after a cancel.
        prompt_keys = [k for k in _open_prompts if k[0] == run_id]
        for k in prompt_keys:
            _open_prompts.pop(k, None)
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


def force_cancel(run_id: int, *, reason: str = "force-cancelled") -> dict:
    """Phase K.1 — strict stop.

    The cooperative cancel path (``request_cancel`` + poll at the next
    safe checkpoint) waits for the runner thread to notice the flag.
    If the thread is stuck inside an LLM call, browser navigation, or
    HITL wait, that can take minutes. Force-cancel bypasses the wait:

      1. Set the cancel flag (so the eventual checkpoint also halts).
      2. Wake every blocker (pause / intervention).
      3. WRITE THE DB ROW DIRECTLY to ``status='cancelled'`` — the
         system considers the run terminal from this instant.
      4. Emit ``cancelled`` on the SSE bus so the UI moves on.

    The orphan Python thread may keep running for a beat (we can't
    kill OS threads from inside Python). When it eventually reaches
    a checkpoint, it'll see ``is_cancelled=True`` and exit cleanly.
    Any DB writes it attempts afterwards target a row that's already
    terminal — harmless.

    Idempotent on terminal rows. Returns a small dict the API surfaces
    so the operator sees what happened.
    """
    request_cancel(run_id)

    db = SessionLocal()
    out: dict[str, object] = {
        "run_id": run_id,
        "force_cancel": True,
        "previous_status": None,
        "new_status": None,
    }
    try:
        run = db.get(AgentRun, run_id)
        if run is None:
            out["error"] = "run not found"
            return out
        out["previous_status"] = run.status
        if run.status in ("completed", "failed", "cancelled"):
            out["new_status"] = run.status
            out["note"] = "already terminal — no DB change"
            return out
        run.status = "cancelled"
        run.completed_at = _utcnow()
        run.error_message = (
            f"force-cancelled: {reason}"[:2000]
        )
        # Mark the worker side: stash a flag in output_summary_json
        # so reports can flag "DB cancelled but the runner thread may
        # have continued briefly".
        prev_summary = dict(run.output_summary_json or {})
        prev_summary["force_cancelled"] = True
        prev_summary["force_cancel_reason"] = reason[:500]
        run.output_summary_json = prev_summary
        db.commit()
        out["new_status"] = "cancelled"
        try:
            _emit_run_event(
                run, "cancelled",
                {"message": f"force-cancelled: {reason}"[:500]},
            )
        except Exception as e:
            logger.warning("emit force-cancel event failed: %s", e)
        logger.warning(
            "Run %s FORCE-CANCELLED (was %s): %s",
            run_id, out["previous_status"], reason,
        )
    except Exception as e:
        logger.exception("force_cancel DB write failed for run %s", run_id)
        out["error"] = str(e)[:200]
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()
    return out


def reap_orphaned_runs(*, stale_after_seconds: int = 60) -> dict:
    """Phase K.2 — reaper for runs left in non-terminal status after
    a process restart / crash. Marks them ``cancelled`` so they don't
    block the UI or duplicate-run guards forever.

    Definition of "orphaned": a run in ('queued', 'running', 'paused')
    whose ``started_at`` is older than ``stale_after_seconds`` AND
    whose ``completed_at`` is NULL. We don't try to distinguish
    "running and slow" from "dead" — at process boot, EVERY non-
    terminal row is orphaned by construction (no live thread exists
    for it). The threshold matters only when the reaper runs while
    other runs are legitimately active.
    """
    from datetime import timedelta  # noqa: PLC0415
    from sqlalchemy import select, or_ as _or  # noqa: PLC0415

    db = SessionLocal()
    cutoff = _utcnow() - timedelta(seconds=stale_after_seconds)
    reaped: list[int] = []
    try:
        rows = list(db.execute(
            select(AgentRun).where(
                AgentRun.status.in_(["queued", "running", "paused"]),
                AgentRun.completed_at.is_(None),
                _or(
                    AgentRun.started_at.is_(None),
                    AgentRun.started_at < cutoff,
                ),
            ),
        ).scalars())
        for r in rows:
            r.status = "cancelled"
            r.completed_at = _utcnow()
            prev_msg = r.error_message or ""
            r.error_message = (
                "orphaned: process restart / runner dead — auto-reaped"
                + (f" (prior msg: {prev_msg[:200]})" if prev_msg else "")
            )[:2000]
            prev_summary = dict(r.output_summary_json or {})
            prev_summary["reaped"] = True
            prev_summary["reap_reason"] = "process_restart"
            r.output_summary_json = prev_summary
            reaped.append(r.id)
        if reaped:
            db.commit()
            logger.warning(
                "Reaped %d orphaned agent run(s): %s",
                len(reaped), reaped,
            )
            # Emit cancelled events post-commit so subscribers see
            # the terminal transition.
            for r in rows:
                try:
                    _emit_run_event(
                        r, "cancelled",
                        {"message": "auto-reaped after process restart"},
                    )
                except Exception:
                    pass
    except Exception as e:
        logger.exception("reap_orphaned_runs failed")
        try:
            db.rollback()
        except Exception:
            pass
        return {"reaped": [], "error": str(e)[:200]}
    finally:
        db.close()
    return {"reaped": reaped, "count": len(reaped)}


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


def execute_recording(run_id: int) -> None:
    """Phase W — background-task entry for a ``record`` agent run.

    Opens a headed Playwright browser (start-maximized so the bottom
    of the page isn't hidden by the Windows taskbar), injects the
    capture JS, navigates to the plan's ``target_url``, and BLOCKS
    until a stop-signal arrives (operator clicks Stop on the live
    presenter). On stop, flushes the in-memory event buffer to the
    submodule's ``frozen_path``.

    Cancellation: the standard ``request_cancel`` / ``force_cancel``
    paths work — cancel sets the stop event AND marks the run row.
    """
    from app.executor.browser import (  # noqa: PLC0415
        browser_session, BrowserNotInstalledError,
    )
    from app.executor.recording_capture import (  # noqa: PLC0415
        build_capture_init_script,
    )
    from app.models.tc_node import TcNode  # noqa: PLC0415
    from app.models.test_plan import TestPlan  # noqa: PLC0415
    from app.services.submodule_recording import (  # noqa: PLC0415
        discard_buffer, finalize_to_submodule, init_buffer,
        register_stop_event,
    )
    from app.config import settings as _app_settings  # noqa: PLC0415

    db = SessionLocal()
    try:
        run = db.get(AgentRun, run_id)
        if not run:
            logger.warning("execute_recording: run %s not found", run_id)
            return
        if run.kind != "record":
            logger.error(
                "execute_recording called for run %s with kind=%r",
                run.id, run.kind,
            )
            return
        if run.status != "queued":
            logger.warning(
                "execute_recording: run %s in status %r, not queued",
                run.id, run.status,
            )
            return
        if _is_cancelled(run.id):
            _mark_cancelled(db, run, "Cancelled before recording started")
            return

        run.status = "running"
        run.started_at = _utcnow()
        db.commit()

        _emit_run_event(run, "started", {
            "input": run.input_json,
            "started_at": run.started_at.isoformat() if run.started_at else None,
        })

        input_data = run.input_json or {}
        plan_id = input_data.get("plan_id")
        module_id = input_data.get("module_id")
        if not isinstance(plan_id, int) or plan_id <= 0:
            _mark_failed(db, run, "input.plan_id must be a positive integer")
            return
        if not isinstance(module_id, int) or module_id <= 0:
            _mark_failed(
                db, run,
                "input.module_id must be a positive integer "
                "(recordings cover one module; submodule attribution "
                "happens live)",
            )
            return

        plan = db.get(TestPlan, plan_id)
        if plan is None:
            _mark_failed(db, run, f"plan {plan_id} not found")
            return
        target_url = (plan.target_url or "").strip()
        if not target_url:
            _mark_failed(
                db, run,
                "plan has no target_url; recording needs a URL to navigate to",
            )
            return

        module = db.get(TcNode, module_id)
        if (
            module is None
            or module.plan_id != plan.id
            or module.kind != "module"
        ):
            _mark_failed(
                db, run,
                f"module {module_id} not found on plan {plan_id}",
            )
            return

        # Initialise in-memory buffer + stop event BEFORE the JS
        # starts POSTing. Order matters — the JS injection runs as
        # part of the navigate, and the first /events POST can fire
        # within ~200ms.
        init_buffer(
            run.id,
            module_id=module.id,
            target_url=target_url,
        )
        stop_event = register_stop_event(run.id)

        # Backend origin for the JS to POST events to. Falls back to
        # localhost:<settings.api_port> in dev. If a reverse proxy is
        # in front, the operator must configure this in settings.
        backend_origin = getattr(
            _app_settings, "backend_origin_for_browser",
            "http://localhost:8000",
        )
        init_script = build_capture_init_script(run.id, backend_origin)

        try:
            with browser_session(
                headless=False,
                speed="slow",
                maximize=True,
            ) as page:
                # Inject the capture BEFORE navigating so the first
                # page load is recorded too.
                try:
                    page.context.add_init_script(init_script)
                except Exception as e:
                    logger.warning(
                        "recording: add_init_script failed: %s", e,
                    )

                try:
                    page.goto(target_url, timeout=45_000)
                except Exception as e:
                    logger.warning(
                        "recording: initial goto failed: %s", e,
                    )

                _emit_run_event(run, "recording_ready", {
                    "run_id": run.id,
                    "module_id": module.id,
                    "module_title": module.title or "",
                    "target_url": target_url,
                })

                # Block until Stop is signalled OR cancel fires.
                # Poll every 500ms so the operator's Stop click
                # propagates quickly.
                while True:
                    if stop_event.wait(0.5):
                        break
                    if _is_cancelled(run.id):
                        break

                # Capture the viewport for replay sanity check.
                try:
                    vp = page.viewport_size
                    if vp:
                        from app.services.submodule_recording import (  # noqa: PLC0415
                            _buffer_meta, _buffer_lock,
                        )
                        with _buffer_lock:
                            meta = _buffer_meta.get(run.id)
                            if meta:
                                meta["viewport"] = {
                                    "width": int(vp.get("width") or 1920),
                                    "height": int(vp.get("height") or 1040),
                                }
                except Exception:
                    pass
        except BrowserNotInstalledError as e:
            _mark_failed(db, run, str(e))
            discard_buffer(run.id)
            return
        except Exception as e:
            logger.exception("recording browser session failed")
            _mark_failed(db, run, f"{type(e).__name__}: {e}")
            discard_buffer(run.id)
            return

        # If cancelled, drop the buffer instead of persisting.
        if _is_cancelled(run.id):
            discard_buffer(run.id)
            _mark_cancelled(db, run, "Recording cancelled by user")
            return

        # Otherwise persist what we captured. Phase W' — the
        # finalize call now iterates every populated submodule
        # chunk and writes each to its respective frozen_path.
        summary = finalize_to_submodule(db, run_id=run.id)
        run.status = "completed"
        run.completed_at = _utcnow()
        run.output_summary_json = {
            "kind": "record",
            "module_id": module.id,
            "event_count": summary.get("event_count", 0),
            "saved": summary.get("saved", False),
            "submodules": summary.get("submodules", []),
            "reason": summary.get("reason", ""),
        }
        db.commit()
        _emit_run_event(run, "recording_saved", summary)
    finally:
        _drop_cancel(run_id)
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
        mode = input_data.get("mode", "scripted")
        if mode not in ("scripted", "agentic", "replay"):
            mode = "scripted"
        # Phase H — preflight selector. "auto" runs Scout + refine
        # when needed (default for agentic; off for scripted/replay
        # since those modes don't need a vision-grounded plan to
        # function). "force" always re-scouts. "skip" disables it.
        raw_preflight = input_data.get("preflight")
        if raw_preflight not in ("auto", "force", "skip"):
            raw_preflight = "auto" if mode == "agentic" else "skip"
        preflight = raw_preflight
        # Phase 6 — agent strategy. Only meaningful for ``agentic``.
        # Persist on the run row so the report can show which path
        # actually drove the run.
        agent_strategy = input_data.get("agent_strategy", "hybrid")
        if agent_strategy not in ("hybrid", "vision_only"):
            agent_strategy = "hybrid"
        try:
            run.agent_strategy = agent_strategy
            db.commit()
        except Exception:
            db.rollback()

        # Window geometry — frontend computes from screen.availWidth/Height
        # so the headed Chromium fits the user's monitor with the live
        # presenter popup on the right. None values fall through to
        # browser_session defaults.
        win_x = input_data.get("window_x")
        win_y = input_data.get("window_y")
        win_w = input_data.get("window_width")
        win_h = input_data.get("window_height")
        window_position = (
            (int(win_x), int(win_y))
            if isinstance(win_x, int) and isinstance(win_y, int)
            else None
        )
        window_size = (
            (int(win_w), int(win_h))
            if isinstance(win_w, int) and isinstance(win_h, int)
            else None
        )

        # Build the LLM provider for AI assist on failure. Missing config is
        # NOT fatal — the run continues with ai_assist disabled. This keeps
        # the executor usable without an LLM (auto-retry only) and supports
        # the "tester ran offline" scenario.
        provider = None
        cheap_provider = None
        if ai_assist:
            try:
                # Phase 1 — provider tiering. ``build_tier_pair`` returns
                # (strong, cheap_or_None) from app_settings. When the
                # user hasn't configured a ``cheap_model``, ``cheap`` is
                # None and the agent runs in single-model mode (legacy
                # behavior preserved). When configured, the cheap model
                # handles the volume of VL helper calls (search, on-
                # track, goal-verify, smart-pick, semantic-verify) with
                # the strong model as the escalation tier.
                from app.llm.router import build_tier_pair  # noqa: PLC0415

                provider, cheap_provider = build_tier_pair(db)
            except RuntimeError as e:
                logger.info(
                    "AI assist disabled for run %s — no LLM configured: %s",
                    run.id, e,
                )

        # ── Run the orchestrator ───────────────────────────────
        # Branch on mode: scripted = rigid step-walker (legacy);
        # agentic = goal-oriented QA agent loop per submodule.
        try:
            if mode == "agentic":
                from app.agents.qa_agent import run_qa_agent_for_plan

                if provider is None:
                    _mark_failed(
                        db, run,
                        "Agentic mode requires an LLM provider. "
                        "Configure one in App Settings or pick scripted mode.",
                    )
                    return

                result = run_qa_agent_for_plan(
                    db=db,
                    run_id=run.id,
                    plan_id=plan_id,
                    selected_step_ids=selected_step_ids,
                    headless=headless,
                    speed=speed_raw,
                    provider=provider,
                    cheap_provider=cheap_provider,
                    agent_strategy=agent_strategy,
                    preflight=preflight,
                    auto_adjust=auto_adjust,
                    promote_fixes=promote_fixes,
                    window_position=window_position,
                    window_size=window_size,
                    emit_event=lambda et, data: _emit_run_event(run, et, data),
                    # Phase AA — combine cancel + force-pass into the
                    # single ``is_cancelled`` signal that the inner
                    # loops (LLM turn loop, replay walker, etc.) all
                    # poll. That way force-pass exits the in-flight
                    # submodule's tight loop fast — same as cancel —
                    # without plumbing a new callback through every
                    # nested helper. The OUTER loop polls
                    # ``is_force_passed`` separately to pick the
                    # right finalisation branch (promote-to-passed
                    # vs skip-cancelled).
                    is_cancelled=lambda: (
                        _is_cancelled(run.id) or _is_force_passed(run.id)
                    ),
                    is_force_passed=lambda: _is_force_passed(run.id),
                    is_paused=lambda: _is_paused(run.id),
                    wait_for_resume=lambda: _wait_until_resumed_or_cancelled(run.id),
                    wait_for_intervention=lambda step_id: request_intervention(run.id, step_id),
                    open_typed_prompt=(
                        lambda *, run_id, step_id, kind, question, fields=None:
                        open_typed_prompt(
                            run_id, step_id,
                            kind=kind, question=question, fields=fields,
                        )
                    ),
                )
            elif mode == "replay":
                # Phase E: deterministic walk of frozen paths.
                # Submodules without a frozen path fall through to
                # agentic mode (handled inside the runner) — so
                # ``provider`` may be required for partial coverage
                # but isn't strictly required for fully-frozen plans.
                from app.agents.replay import run_replay_for_plan

                result = run_replay_for_plan(
                    db=db,
                    run_id=run.id,
                    plan_id=plan_id,
                    selected_step_ids=selected_step_ids,
                    headless=headless,
                    speed=speed_raw,
                    provider=provider,
                    self_heal_enabled=ai_assist,
                    auto_adjust=auto_adjust,
                    promote_fixes=promote_fixes,
                    window_position=window_position,
                    window_size=window_size,
                    emit_event=lambda et, data: _emit_run_event(run, et, data),
                    is_cancelled=lambda: _is_cancelled(run.id),
                    is_paused=lambda: _is_paused(run.id),
                    wait_for_resume=lambda: _wait_until_resumed_or_cancelled(run.id),
                    wait_for_intervention=lambda step_id: request_intervention(run.id, step_id),
                )
            else:
                result = execute_plan(
                    db=db,
                    run_id=run.id,
                    plan_id=plan_id,
                    selected_step_ids=selected_step_ids,
                    headless=headless,
                    speed=speed_raw,
                    provider=provider,
                    cheap_provider=cheap_provider,
                    preflight=preflight,
                    ai_assist=ai_assist,
                    auto_adjust=auto_adjust,
                    promote_fixes=promote_fixes,
                    window_position=window_position,
                    window_size=window_size,
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
            # Inconclusive is its own bucket — the run-progress card and
            # report charts render it distinctly (orange) so users can
            # tell "test case was unclear" from "the app failed".
            "inconclusive": result.inconclusive,
            "skipped": result.skipped,
            "blocked": result.blocked,
            "duration_ms": result.duration_ms,
            "mode": mode,
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
        _drop_force_pass(run_id)
        _drop_pause(run_id)
        _drop_interventions(run_id)
        db.close()
