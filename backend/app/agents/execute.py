"""Execute orchestrator (week 5) — drives a browser through the selected
steps of a plan's TC tree, recording per-step results.

Pipeline
--------
    load plan + tree → filter selected steps → pre-create rows (pending)
        ↓
    open browser → for each step:
        check cancel → mark running → execute_action → screenshot →
        mark passed/failed/blocked → emit step_completed
        ↓
    on cancel: mark remaining as skipped → close browser → done

Selection semantics
-------------------
- Default: a step runs iff its own ``selectable_default`` is True AND every
  ancestor (submodule, module) is also ``selectable_default=True``.
- Override: ``selected_step_ids`` skips the ancestor cut and runs exactly
  those steps. Used by "re-run failed steps" without flipping checkboxes.

Behavior on failure
-------------------
A failed step does NOT cut its siblings — the run continues so the user
sees the full picture. Only ``cancel`` stops the loop.

Screenshots
-----------
Saved to ``data/screenshots/<run_id>/step_<step_id>.png`` (relative path
stored on the row; the runtime exposes them via the ``/static/screenshots``
mount). Captured AFTER the action so the row reflects the state the action
produced.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.brd_to_frd import AgentCancelled  # reused exception
from app.config import settings
from app.executor import (
    ActionContext,
    BrowserNotInstalledError,
    browser_session,
    execute_action,
    get_speed_config,
    hide_narration,
    install_overlay,
    update_narration,
    wait_for_settled,
)
from app.models.agent_run import AgentRun
from app.models.execution_step import ExecutionStep
from app.models.tc_node import TcNode
from app.models.test_plan import TestPlan

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    plan_id: int
    total_steps: int
    passed: int
    failed: int
    skipped: int
    blocked: int
    duration_ms: int


# ── Helpers ───────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _check_cancel(
    is_cancelled: Callable[[], bool] | None, where: str,
) -> None:
    if is_cancelled and is_cancelled():
        raise AgentCancelled(f"Cancelled at: {where}")


def _emit(
    emit_event: Callable[[str, dict], None] | None,
    event_type: str,
    data: dict,
) -> None:
    if emit_event:
        try:
            emit_event(event_type, data)
        except Exception as e:
            logger.warning("emit_event raised, continuing: %s", e)


def _walk_steps_dfs(nodes: list[TcNode]) -> list[TcNode]:
    """Return all ``kind='step'`` leaves in DFS+ordinal order.

    Visits each parent before recursing into its ordinal-sorted children;
    yields only the leaves (steps).
    """
    children_by_parent: dict[int | None, list[TcNode]] = defaultdict(list)
    for n in nodes:
        children_by_parent[n.parent_id].append(n)
    for sibs in children_by_parent.values():
        sibs.sort(key=lambda n: n.ordinal)

    out: list[TcNode] = []
    stack = list(reversed(children_by_parent.get(None, [])))
    while stack:
        node = stack.pop()
        if node.kind == "step":
            out.append(node)
        # Push children in reverse so the first child pops first
        for child in reversed(children_by_parent.get(node.id, [])):
            stack.append(child)
    return out


def _select_target_steps(
    nodes: list[TcNode],
    *,
    selected_step_ids: list[int] | None,
) -> list[TcNode]:
    """Apply selection semantics; return steps in execution order.

    Ticked-step semantics: a step runs iff its own ``selectable_default``
    is True. Module/submodule selectable_default flags exist but are
    organizational only — when the user unticks a parent, the frontend
    cascades ``false`` to every descendant via bulk-update, so all the
    leaves are already excluded. Re-ticking a single step underneath a
    still-unticked parent must run that step (matches the user's
    intuition of "tick what I want to run"); an ancestor-cut would
    silently exclude it.
    """
    walk = _walk_steps_dfs(nodes)
    if selected_step_ids is not None:
        wanted = set(selected_step_ids)
        return [n for n in walk if n.id in wanted]
    return [n for n in walk if n.selectable_default]


# ── Row pre-creation ──────────────────────────────────────────────


def _create_pending_rows(
    db: Session,
    *,
    run_id: int,
    project_id: int,
    plan_id: int,
    steps: list[TcNode],
) -> list[ExecutionStep]:
    """Insert one ``execution_steps`` row per selected step in 'pending'.

    Snapshots are frozen here. The orchestrator updates each row in place
    as it runs through them.
    """
    rows: list[ExecutionStep] = []
    for ordinal, node in enumerate(steps):
        row = ExecutionStep(
            run_id=run_id,
            project_id=project_id,
            plan_id=plan_id,
            tc_node_id=node.id,
            title_snapshot=(node.title or "")[:512],
            path_snapshot=(node.path_cached or node.title or "")[:2048],
            action_type_snapshot=(node.action_type or None),
            target_hint_snapshot=node.target_hint,
            expected_snapshot=node.expected,
            narrative_snapshot=node.narrative,
            ordinal=ordinal,
            status="pending",
            details_json={},
        )
        db.add(row)
        rows.append(row)
    db.flush()
    return rows


# ── Per-step execution ────────────────────────────────────────────


def _build_action_context(
    db: Session,
    row: ExecutionStep,
    plan: TestPlan,
    speed_config,
) -> ActionContext:
    """Assemble the per-step context for the action dispatcher.

    ``data_needs`` is read live from the source tc_node so that fixing a
    `kind` typo on the node mid-run is honored on the next step. If the
    source node is gone (deleted between snapshot and execution), we treat
    the step as having no data needs.
    """
    data_needs: list[dict[str, Any]] = []
    if row.tc_node_id is not None:
        node = db.get(TcNode, row.tc_node_id)
        if node is not None and node.data_needs_json:
            data_needs = list(node.data_needs_json)

    return ActionContext(
        plan_target_url=plan.target_url or "",
        target_hint=row.target_hint_snapshot,
        narrative=row.narrative_snapshot,
        expected=row.expected_snapshot,
        data_needs=data_needs,
        speed_config=speed_config,
    )


def _execute_with_retry(
    page,
    row: ExecutionStep,
    ctx: ActionContext,
    speed_config,
    *,
    is_cancelled: Callable[[], bool] | None,
    emit_event: Callable[[str, dict], None] | None,
) -> tuple[Any, list[dict]]:
    """Run the action with auto-retry on failure.

    Retries up to ``speed_config.retry_count`` additional times on
    ``status == 'failed'``. Skips retry on ``passed`` (no need) and
    ``blocked`` (HITL territory — that's the AI-assist + intervention
    flow, not a flake). Cancellation between attempts breaks out cleanly.

    Backoff is exponential: ``backoff_ms × 2^(attempt-1)`` between attempts.
    Before each retry, ``wait_for_settled`` runs again so a slow XHR doesn't
    immediately re-trigger the same selector miss.

    Returns the final :class:`ActionResult` plus the per-attempt log
    (one dict per attempt: ``{attempt, status, narration}``) which the
    caller folds into ``details_json`` when there was more than one try.
    """
    attempts: list[dict] = []
    max_attempts = max(1, speed_config.retry_count + 1)
    last_result = None

    for attempt_idx in range(max_attempts):
        if attempt_idx > 0:
            if is_cancelled and is_cancelled():
                break
            backoff_ms = speed_config.retry_backoff_ms * (2 ** (attempt_idx - 1))
            time.sleep(backoff_ms / 1000.0)
            wait_for_settled(page, speed_config)
            _emit(emit_event, "step_retry", {
                "step_id": row.id,
                "attempt": attempt_idx + 1,
                "max_attempts": max_attempts,
                "backoff_ms": backoff_ms,
                "prior_error": (last_result.error_message if last_result else None),
            })

        try:
            result = execute_action(page, row.action_type_snapshot, ctx)
        except Exception as e:
            logger.exception(
                "Unhandled error in step %s attempt %d", row.id, attempt_idx + 1,
            )
            from app.executor import ActionResult as _AR
            result = _AR(
                status="failed",
                narration="unhandled error in dispatcher",
                error_message=f"{type(e).__name__}: {e}",
            )

        attempts.append({
            "attempt": attempt_idx + 1,
            "status": result.status,
            "narration": result.narration,
        })
        last_result = result

        # Don't retry on success or HITL block.
        if result.status in ("passed", "blocked"):
            break

    if last_result is None:
        from app.executor import ActionResult as _AR
        last_result = _AR(
            status="failed",
            narration="no attempts ran (cancelled before first try?)",
        )

    return last_result, attempts


def _take_screenshot(
    page, run_id: int, step_id: int,
) -> str | None:
    """Capture a PNG; return the path relative to ``screenshots_dir``.

    Returns None if the browser refused (e.g., target_closed after navigate
    error). We never let a screenshot failure cascade — it's diagnostic.
    """
    rel_path = f"{run_id}/step_{step_id}.png"
    abs_path = settings.screenshots_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        page.screenshot(path=str(abs_path), full_page=False)
        return rel_path
    except Exception as e:
        logger.warning(
            "screenshot failed for run %s step %s: %s", run_id, step_id, e,
        )
        return None


# ── Orchestrator entry point ──────────────────────────────────────


def execute_plan(
    db: Session,
    *,
    run_id: int,
    plan_id: int,
    selected_step_ids: list[int] | None = None,
    headless: bool = False,
    speed: str | None = None,
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    is_paused: Callable[[], bool] | None = None,
    wait_for_resume: Callable[[], bool] | None = None,
) -> ExecutionResult:
    """Run the executor agent against the plan.

    Args:
        db: SQLAlchemy session — caller commits/rolls back on failure.
        run_id: ``agent_runs.id`` for this execution. We write
            ``execution_steps`` rows referencing it and use it for the
            screenshot directory.
        plan_id: ``test_plans.id``.
        selected_step_ids: Optional override; if provided, run exactly
            those steps (skipping the ``selectable_default`` filter).
        headless: Whether to launch Chromium without a visible window.
        speed: Speed preset name — ``"slow"`` (default), ``"normal"``, or
            ``"fast"``. Controls slow_mo, cursor glide, type delay, and
            the network-idle timeout used by :func:`wait_for_settled`.
        emit_event: Optional SSE callback ``(event_type, data) -> None``.
        is_cancelled: Optional callback polled between steps.

    Returns:
        :class:`ExecutionResult` with summary counts + duration.

    Raises:
        ValueError: Plan missing, no steps selected, or plan has no target_url.
        BrowserNotInstalledError: Chromium binary not downloaded.
        AgentCancelled: ``is_cancelled()`` flipped to True at a safe boundary.
        RuntimeError: Anything else (browser launch, screenshot dir, etc.).
    """
    t0 = time.monotonic()
    speed_config = get_speed_config(speed)

    plan = db.get(TestPlan, plan_id)
    if not plan:
        raise ValueError(f"Plan {plan_id} not found")
    if not (plan.target_url and plan.target_url.strip()):
        raise ValueError(f"Plan {plan_id} has no target_url — cannot navigate")

    project_id = plan.project_id

    _emit(emit_event, "phase", {
        "phase": "loading_steps",
        "message": f"Loading TC tree for plan '{plan.name}'",
    })

    # Load all tc_nodes for the plan in tree-walk order
    stmt = (
        select(TcNode)
        .where(TcNode.plan_id == plan_id)
        .order_by(TcNode.depth, TcNode.parent_id, TcNode.ordinal)
    )
    all_nodes = list(db.scalars(stmt))

    selected = _select_target_steps(
        all_nodes, selected_step_ids=selected_step_ids,
    )
    if not selected:
        raise ValueError(
            "No steps selected for execution. Tick at least one step in the "
            "Test Cases tab, or pass selected_step_ids explicitly.",
        )

    rows = _create_pending_rows(
        db,
        run_id=run_id,
        project_id=project_id,
        plan_id=plan_id,
        steps=selected,
    )
    db.commit()

    _emit(emit_event, "phase", {
        "phase": "opening_browser",
        "message": (
            f"Launching {'headless' if headless else 'headed'} Chromium · "
            f"{len(rows)} steps queued · speed={speed or 'slow'}"
        ),
        "total": len(rows),
        "speed": speed or "slow",
    })

    counts = {"passed": 0, "failed": 0, "skipped": 0, "blocked": 0}
    cancelled = False

    try:
        with browser_session(headless=headless, speed=speed) as page:
            # Install the visible cursor + narration overlay. Always-on so
            # per-step screenshots inherit the cursor position + banner text
            # without any extra plumbing on the timeline side.
            install_overlay(page)

            for idx, row in enumerate(rows):
                if is_cancelled and is_cancelled():
                    cancelled = True
                    break

                # Pause checkpoint — block until resume or cancel.
                # The browser session stays open so the user can poke at
                # the page; on resume we pick up exactly where we stopped.
                if is_paused and is_paused():
                    run_record = db.get(AgentRun, run_id)
                    if run_record:
                        run_record.status = "paused"
                        db.commit()
                    _emit(emit_event, "paused", {
                        "step_id": row.id,
                        "ordinal": idx + 1,
                        "total": len(rows),
                        "status": "paused",
                    })
                    resumed = wait_for_resume() if wait_for_resume else True
                    if not resumed:
                        cancelled = True
                        break
                    if run_record:
                        run_record.status = "running"
                        db.commit()
                    _emit(emit_event, "resumed", {
                        "step_id": row.id,
                        "ordinal": idx + 1,
                        "total": len(rows),
                        "status": "running",
                    })

                # ── pending → running ─────────────────────────
                row.status = "running"
                row.started_at = _utcnow()
                db.commit()

                _emit(emit_event, "step_started", {
                    "step_id": row.id,
                    "tc_node_id": row.tc_node_id,
                    "ordinal": idx + 1,
                    "total": len(rows),
                    "title": row.title_snapshot,
                    "action_type": row.action_type_snapshot,
                })

                # Push the step's metadata into the on-page banner — neutral
                # blue ("about to do this") so the viewer sees the agent's
                # intent before it acts.
                update_narration(
                    page,
                    ordinal=idx + 1,
                    total=len(rows),
                    title=row.title_snapshot,
                    action_type=row.action_type_snapshot,
                    phase="about_to",
                )

                step_t0 = time.monotonic()
                ctx = _build_action_context(db, row, plan, speed_config)

                # Settle gate: wait for DOM + networkidle before acting.
                # On heavy-data sites the element may be a loading skeleton
                # that satisfies wait_for(visible) but isn't yet bound to
                # real data. wait_for_settled is best-effort — chatty SPAs
                # never reach networkidle and we proceed anyway.
                wait_for_settled(page, speed_config)

                result, attempt_log = _execute_with_retry(
                    page, row, ctx, speed_config,
                    is_cancelled=is_cancelled,
                    emit_event=emit_event,
                )
                result_status = result.status
                narration = result.narration
                error_msg = result.error_message
                details: dict[str, Any] = dict(result.details)
                if len(attempt_log) > 1:
                    details["attempts"] = attempt_log

                # Flip the banner to the outcome state BEFORE the screenshot
                # so the per-step PNG carries the green ✓ / red ✗ marker.
                _PHASE_BY_STATUS = {
                    "passed": "did",
                    "failed": "failed",
                    "blocked": "blocked",
                }
                update_narration(
                    page,
                    ordinal=idx + 1,
                    total=len(rows),
                    title=row.title_snapshot,
                    action_type=row.action_type_snapshot,
                    phase=_PHASE_BY_STATUS.get(result_status, "did"),
                )

                screenshot_path = _take_screenshot(page, run_id, row.id)

                row.status = result_status
                row.completed_at = _utcnow()
                row.duration_ms = int((time.monotonic() - step_t0) * 1000)
                row.narration = narration
                row.error_message = error_msg
                row.screenshot_path = screenshot_path
                row.details_json = details
                counts[result_status] = counts.get(result_status, 0) + 1
                db.commit()

                _emit(emit_event, "step_completed", {
                    "step_id": row.id,
                    "tc_node_id": row.tc_node_id,
                    "ordinal": idx + 1,
                    "total": len(rows),
                    "status": row.status,
                    "narration": row.narration,
                    "duration_ms": row.duration_ms,
                    "screenshot_path": row.screenshot_path,
                })

            # Drop the banner once the loop's done so the page is clean
            # if the user keeps watching after the browser closes.
            hide_narration(page)
    except BrowserNotInstalledError:
        # Bubble up unchanged — the runtime translates to a useful 4xx-style
        # failure message on the run row.
        raise
    finally:
        # If we exited the loop early via cancel, mark the leftover rows.
        if cancelled:
            now = _utcnow()
            for row in rows:
                if row.status in ("pending", "running"):
                    row.status = "skipped"
                    row.completed_at = now
                    row.narration = "run cancelled before this step"
                    counts["skipped"] = counts.get("skipped", 0) + 1
            db.commit()

    duration_ms = int((time.monotonic() - t0) * 1000)

    if cancelled:
        # Surface as a clean cancel through the runtime's path
        raise AgentCancelled(
            f"Run cancelled after {sum(counts.values())}/{len(rows)} steps",
        )

    _emit(emit_event, "done", {
        "plan_id": plan_id,
        "total_steps": len(rows),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "blocked": counts["blocked"],
        "duration_ms": duration_ms,
    })

    logger.info(
        "Execute run %s completed in %dms: %d passed, %d failed, %d blocked",
        run_id, duration_ms, counts["passed"], counts["failed"],
        counts["blocked"],
    )

    return ExecutionResult(
        plan_id=plan_id,
        total_steps=len(rows),
        passed=counts["passed"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        blocked=counts["blocked"],
        duration_ms=duration_ms,
    )
