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
from app.agents.page_intel import RecoverySuggestion, propose_recovery
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
from app.llm.base import LLMProvider
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
    # AI-assist token totals across every recovery + vision call in the
    # run. ``None`` when no AI calls were made (no LLM configured, or
    # ai_assist=False, or no step needed assist). The runner copies
    # these into ``agent_runs.output_summary_json`` at end-of-run.
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    ai_calls: int = 0
    ai_vision_calls: int = 0


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


# ── AI assist on failure ──────────────────────────────────────────


def _try_ai_correction(
    page,
    *,
    row: ExecutionStep,
    ctx: ActionContext,
    provider: LLMProvider,
    speed_config,
    prior_attempts: list[dict[str, Any]],
    original_error: str | None,
    emit_event: Callable[[str, dict], None] | None,
    include_screenshot: bool = False,
) -> dict[str, Any] | None:
    """Ask the LLM what to do about a failed step; run its suggestion once.

    Called only after :func:`_execute_with_retry` has already burned its
    retry budget. Single-shot — if the AI's correction also fails, we
    don't loop; HITL (M3) takes over from there.

    Returns a dict shaped like:
        {
            "status": "passed" | "failed",
            "narration": str,
            "error_message": str | None,
            "correction": {              # always present, used by M4 UI
                "action": "retry" | "replace" | "give_up",
                "reasoning": str,
                "confidence": float,
                "diff": {field: {old, new}, ...},  # only for "replace"
                "tokens_in": int | None,
                "tokens_out": int | None,
            },
        }
    Or ``None`` when the LLM call itself raised — caller leaves the
    original failure status in place.
    """
    _emit(emit_event, "ai_assist_started", {
        "step_id": row.id,
        "ordinal": row.ordinal + 1,
        "title": row.title_snapshot,
        "used_vision": include_screenshot,
    })

    try:
        suggestion = propose_recovery(
            provider, page,
            title=row.title_snapshot,
            target_hint=row.target_hint_snapshot,
            action_type=row.action_type_snapshot,
            narrative=row.narrative_snapshot,
            expected=row.expected_snapshot,
            error_message=original_error or "(no error message recorded)",
            prior_attempts=prior_attempts,
            include_screenshot=include_screenshot,
        )
    except Exception as e:
        logger.warning("AI assist call failed for step %s: %s", row.id, e)
        _emit(emit_event, "ai_assist_completed", {
            "step_id": row.id,
            "ordinal": row.ordinal + 1,
            "outcome": "llm_error",
            "error": str(e)[:300],
            "used_vision": include_screenshot,
        })
        return None

    correction: dict[str, Any] = {
        "action": suggestion.action,
        "reasoning": suggestion.reasoning,
        "confidence": suggestion.confidence,
        "tokens_in": suggestion.input_tokens,
        "tokens_out": suggestion.output_tokens,
        "used_vision": suggestion.used_vision,
        "diff": {},
    }

    if suggestion.action == "give_up":
        _emit(emit_event, "ai_assist_completed", {
            "step_id": row.id,
            "ordinal": row.ordinal + 1,
            "outcome": "give_up",
            "reasoning": suggestion.reasoning[:300],
            "used_vision": suggestion.used_vision,
        })
        return {
            "status": "failed",
            "narration": (
                f"AI gave up: {suggestion.reasoning[:200]}"
            ),
            "error_message": original_error,
            "correction": correction,
        }

    # Build the context the corrected attempt will use. For action='retry'
    # we re-use the original ctx unchanged. For 'replace' we substitute
    # only the fields the LLM explicitly proposed (empty string = no change).
    new_target_hint = ctx.target_hint
    new_action_type = row.action_type_snapshot
    new_expected = ctx.expected
    new_narrative = ctx.narrative

    if suggestion.action == "replace":
        if suggestion.new_target_hint:
            correction["diff"]["target_hint"] = {
                "old": ctx.target_hint,
                "new": suggestion.new_target_hint,
            }
            new_target_hint = suggestion.new_target_hint
        if suggestion.new_action_type:
            correction["diff"]["action_type"] = {
                "old": row.action_type_snapshot,
                "new": suggestion.new_action_type,
            }
            new_action_type = suggestion.new_action_type
        if suggestion.new_expected:
            correction["diff"]["expected"] = {
                "old": ctx.expected,
                "new": suggestion.new_expected,
            }
            new_expected = suggestion.new_expected
        if suggestion.new_narrative:
            correction["diff"]["narrative"] = {
                "old": ctx.narrative,
                "new": suggestion.new_narrative,
            }
            new_narrative = suggestion.new_narrative

        if not correction["diff"]:
            # LLM said "replace" but didn't actually propose any changes —
            # treat as a retry. Belt-and-braces against malformed responses.
            suggestion_action_effective = "retry"
        else:
            suggestion_action_effective = "replace"
    else:
        suggestion_action_effective = "retry"

    corrected_ctx = ActionContext(
        plan_target_url=ctx.plan_target_url,
        target_hint=new_target_hint,
        narrative=new_narrative,
        expected=new_expected,
        data_needs=list(ctx.data_needs),
        speed_config=ctx.speed_config,
    )

    # Update the on-page banner so the viewer sees the AI's intent.
    label_action = (new_action_type or row.action_type_snapshot or "step")
    update_narration(
        page,
        ordinal=row.ordinal + 1,
        total=row.ordinal + 1,  # we don't know total here; not critical
        title=f"AI: {suggestion.reasoning[:80]}",
        action_type=label_action,
        phase="about_to",
    )

    # Settle gate again — page may have changed since the failed attempt.
    wait_for_settled(page, speed_config)

    try:
        result = execute_action(page, new_action_type, corrected_ctx)
    except Exception as e:
        logger.exception(
            "Unhandled error in AI-corrected attempt for step %s", row.id,
        )
        result_status = "failed"
        result_narration = "AI-corrected attempt: dispatcher raised"
        result_error = f"{type(e).__name__}: {e}"
    else:
        result_status = result.status
        # Compose a narration that surfaces both the AI's reasoning and the
        # action outcome — UI uses this verbatim on the timeline row.
        prefix = (
            "AI replaced step → " if suggestion_action_effective == "replace"
            else "AI re-attempted → "
        )
        result_narration = prefix + result.narration
        result_error = result.error_message

    _emit(emit_event, "ai_assist_completed", {
        "step_id": row.id,
        "ordinal": row.ordinal + 1,
        "outcome": result_status,
        "action": suggestion_action_effective,
        "diff_keys": list(correction["diff"].keys()),
        "used_vision": suggestion.used_vision,
    })

    return {
        "status": result_status,
        "narration": result_narration,
        "error_message": result_error,
        "correction": correction,
    }


# ── Orchestrator entry point ──────────────────────────────────────


def execute_plan(
    db: Session,
    *,
    run_id: int,
    plan_id: int,
    selected_step_ids: list[int] | None = None,
    headless: bool = False,
    speed: str | None = None,
    provider: LLMProvider | None = None,
    ai_assist: bool = True,
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    is_paused: Callable[[], bool] | None = None,
    wait_for_resume: Callable[[], bool] | None = None,
    wait_for_intervention: Callable[[int], dict | None] | None = None,
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

    # AI-assist token accounting — accumulated per step across both the
    # text-only pass and the vision-escalation pass when either fires.
    # Surfaces on agent_runs.output_summary_json at end-of-run so the
    # frontend run header can render a cost meter.
    ai_input_tokens_total = 0
    ai_output_tokens_total = 0
    ai_calls = 0
    ai_vision_calls = 0

    # When the user picks an intervention with apply_to_submodule=True,
    # we cache the choice keyed by the failing step's parent (submodule)
    # id. Subsequent failures under that same submodule auto-apply the
    # cached choice without popping the modal again — matches the user's
    # "auto-skip the rest of this submodule" muscle memory.
    _auto_apply_by_submodule: dict[int, dict] = {}

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

                # AI assist on failure. Two-pass escalation:
                #   pass 1: text-only (accessibility tree)
                #   pass 2: vision (text + screenshot) — only when the
                #           provider supports it AND pass 1 didn't fix
                #           the step. Vision tokens are ~3-5× the cost of
                #           text-only, so we only spend them when the AX
                #           tree alone wasn't enough.
                # Outcome is recorded in details["ai_correction"]
                # regardless of pass/fail so the HITL modal can render
                # the diff + reasoning + which pass produced it.
                if (
                    result_status == "failed"
                    and provider is not None
                    and ai_assist
                    and not (is_cancelled and is_cancelled())
                ):
                    ai_outcome = _try_ai_correction(
                        page,
                        row=row,
                        ctx=ctx,
                        provider=provider,
                        speed_config=speed_config,
                        prior_attempts=attempt_log,
                        original_error=error_msg,
                        emit_event=emit_event,
                        include_screenshot=False,
                    )
                    if ai_outcome is not None:
                        ai_calls += 1
                        c = ai_outcome.get("correction") or {}
                        if isinstance(c.get("tokens_in"), int):
                            ai_input_tokens_total += c["tokens_in"]
                        if isinstance(c.get("tokens_out"), int):
                            ai_output_tokens_total += c["tokens_out"]

                    # Vision escalation: if the text-only suggestion
                    # didn't fix the step and the provider can see images,
                    # take another swing with the page screenshot
                    # attached. Tokens from BOTH passes accumulate so the
                    # cost meter reflects total LLM spend on this run.
                    if (
                        ai_outcome is not None
                        and ai_outcome["status"] == "failed"
                        and getattr(provider, "supports_vision", False)
                        and not (is_cancelled and is_cancelled())
                    ):
                        vision_outcome = _try_ai_correction(
                            page,
                            row=row,
                            ctx=ctx,
                            provider=provider,
                            speed_config=speed_config,
                            prior_attempts=attempt_log,
                            original_error=error_msg,
                            emit_event=emit_event,
                            include_screenshot=True,
                        )
                        if vision_outcome is not None:
                            ai_calls += 1
                            ai_vision_calls += 1
                            c = vision_outcome.get("correction") or {}
                            if isinstance(c.get("tokens_in"), int):
                                ai_input_tokens_total += c["tokens_in"]
                            if isinstance(c.get("tokens_out"), int):
                                ai_output_tokens_total += c["tokens_out"]
                            ai_outcome = vision_outcome

                    if ai_outcome is not None:
                        details["ai_correction"] = ai_outcome["correction"]
                        if ai_outcome["status"] == "passed":
                            # AI fixed it. Promote the row to passed and let
                            # the live narration reflect what changed.
                            result_status = "passed"
                            narration = ai_outcome["narration"]
                            error_msg = None
                        else:
                            # AI tried but failed too. Keep the row failed,
                            # but augment narration with the AI's attempt
                            # context so the user sees both errors.
                            narration = (
                                f"{narration} · {ai_outcome['narration']}"
                            )
                            if ai_outcome.get("error_message"):
                                error_msg = (
                                    f"{error_msg or ''} | "
                                    f"AI attempt: {ai_outcome['error_message']}"
                                ).strip(" |")

                # ── HITL intervention (after auto-retry + AI both failed) ──
                # Last-resort gate: ask the user what to do. Choices are
                # retry / use_suggestion (with optional overrides) / skip /
                # stop. apply_to_submodule remembers the choice for sibling
                # steps so the user isn't clicking through 20 modals on a
                # broken submodule.
                if (
                    result_status == "failed"
                    and wait_for_intervention is not None
                    and not (is_cancelled and is_cancelled())
                ):
                    submodule_id = None
                    if row.tc_node_id is not None:
                        node = db.get(TcNode, row.tc_node_id)
                        if node is not None:
                            submodule_id = node.parent_id

                    auto_choice = (
                        _auto_apply_by_submodule.get(submodule_id)
                        if submodule_id is not None else None
                    )

                    if auto_choice is not None:
                        choice = auto_choice
                        details["intervention"] = {
                            **choice, "auto_applied": True,
                        }
                        _emit(emit_event, "intervention_auto_applied", {
                            "step_id": row.id,
                            "submodule_id": submodule_id,
                            "choice": choice.get("choice"),
                        })
                    else:
                        # Mark the row blocked so the timeline shows the
                        # halt; the AGENT_RUN row stays 'running'. Take a
                        # screenshot now so the modal can show the failed
                        # state — the end-of-iteration screenshot will
                        # overwrite this with the post-resolution state.
                        row.status = "blocked"
                        db.commit()
                        snapshot = _take_screenshot(page, run_id, row.id)

                        _emit(emit_event, "needs_intervention", {
                            "step_id": row.id,
                            "ordinal": idx + 1,
                            "total": len(rows),
                            "title": row.title_snapshot,
                            "action_type": row.action_type_snapshot,
                            "target_hint": row.target_hint_snapshot,
                            "error_message": error_msg,
                            "ai_suggestion": details.get("ai_correction"),
                            "screenshot_path": snapshot,
                        })

                        choice = wait_for_intervention(row.id)
                        if choice is None:
                            # Cancelled while waiting — exit loop cleanly
                            cancelled = True
                            break

                        details["intervention"] = choice

                        if (choice.get("apply_to_submodule")
                                and submodule_id is not None):
                            _auto_apply_by_submodule[submodule_id] = choice

                        _emit(emit_event, "intervention_resolved", {
                            "step_id": row.id,
                            "choice": choice.get("choice"),
                        })

                    user_choice = choice.get("choice", "skip")
                    if user_choice == "stop":
                        cancelled = True
                        break
                    elif user_choice == "skip":
                        # Keep result_status='failed'; surface the user's
                        # call in the narration so it's not lost.
                        narration = (
                            f"{narration} · user skipped after retry+AI failed"
                        )
                    elif user_choice in ("retry", "use_suggestion"):
                        # Re-run the action once more, optionally with the
                        # user's overrides applied to ctx. Empty overrides
                        # mean "use the original value" — same contract as
                        # the AI suggestion's empty-string semantics.
                        new_target = (
                            choice.get("override_target_hint")
                            or row.target_hint_snapshot
                        )
                        new_action = (
                            choice.get("override_action_type")
                            or row.action_type_snapshot
                        )
                        retry_ctx = ActionContext(
                            plan_target_url=ctx.plan_target_url,
                            target_hint=new_target,
                            narrative=ctx.narrative,
                            expected=ctx.expected,
                            data_needs=list(ctx.data_needs),
                            speed_config=ctx.speed_config,
                        )
                        wait_for_settled(page, speed_config)
                        try:
                            retry_result = execute_action(
                                page, new_action, retry_ctx,
                            )
                        except Exception as e:
                            logger.exception(
                                "HITL retry dispatcher raise on step %s",
                                row.id,
                            )
                            from app.executor import ActionResult as _AR
                            retry_result = _AR(
                                status="failed",
                                narration="HITL retry: dispatcher raised",
                                error_message=f"{type(e).__name__}: {e}",
                            )
                        result_status = retry_result.status
                        prefix = (
                            "User override → "
                            if user_choice == "use_suggestion"
                            else "User retry → "
                        )
                        narration = prefix + retry_result.narration
                        error_msg = retry_result.error_message

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
        llm_input_tokens=(
            ai_input_tokens_total if ai_calls > 0 else None
        ),
        llm_output_tokens=(
            ai_output_tokens_total if ai_calls > 0 else None
        ),
        ai_calls=ai_calls,
        ai_vision_calls=ai_vision_calls,
    )
