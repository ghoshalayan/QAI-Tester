"""Phase E.2: deterministic replay runner.

Walks each submodule's frozen path step-by-step using Playwright
directly. No LLM calls on the happy path — replay-mode runs are
~5% of agentic-mode token cost and >95% reliable on stable apps
because we're literally re-running the same sequence that worked.

When a frozen step fails (selector changed because the app updated),
Phase E.3 self-healing kicks in: a single vision LLM call proposes
a new selector, the runner patches the frozen_path on the submodule
and continues. Only that one step pays an LLM cost; the rest of
the path stays deterministic.

Submodules without a frozen_path fall through to the agentic loop
in :mod:`qa_agent` — replay is incremental, not all-or-nothing.

The runner returns the same :class:`ExecutionResult` shape as the
scripted and agentic runners so :mod:`agent_run_service` can branch
on mode without knowing the implementation differs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.brd_to_frd import AgentCancelled
from app.agents.execute import ExecutionResult, _take_screenshot
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
from app.executor.selectors import SelectorNotFound, resolve
from app.llm.base import LLMProvider
from app.models.execution_step import ExecutionStep
from app.models.tc_node import TcNode
from app.models.test_plan import TestPlan

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _emit(
    emit_event: Callable[[str, dict], None] | None,
    event_type: str,
    data: dict,
) -> None:
    if emit_event:
        try:
            emit_event(event_type, data)
        except Exception as e:
            logger.warning("emit_event raised in replay: %s", e)


def _check_cancel(
    is_cancelled: Callable[[], bool] | None, where: str,
) -> None:
    if is_cancelled and is_cancelled():
        raise AgentCancelled(f"Cancelled at: {where}")


# ── Phase B Step 5 — segment success-criterion gate ───────────────


# Stopwords filtered out when extracting high-information tokens
# from a sub-goal's success_criterion. We don't want to "validate"
# a segment by finding the word "the" on the page.
_CRIT_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "and", "or", "in", "on", "at", "by",
    "for", "with", "as", "this", "that", "these", "those", "it",
    "its", "has", "have", "had", "should", "will", "can", "must",
    "may", "shows", "show", "shown", "visible", "displayed",
    "appears", "appear", "contains", "contain",
})


def _segment_criterion_visible(
    page: Any,
    criterion: str,
    *,
    min_token_len: int = 4,
    require_matches: int = 1,
) -> bool:
    """Best-effort DOM check that a sub-goal's success_criterion is
    observable on the current page.

    The criterion text is tokenized into high-information words
    (length >= ``min_token_len``, not in the stopword list) and we
    require at least ``require_matches`` of those tokens to appear
    in the page's visible text (case-insensitive substring).

    Heuristic by design — false-negatives (criterion met but our
    naive tokens didn't match) are preferred over false-positives
    (claiming success when nothing happened). False-negatives feed
    the segment to agentic recovery, which is the safe fallback.

    Returns True when the gate is satisfied OR when there's nothing
    to check (empty criterion / page closed / no high-info tokens).
    Failures inside this function never raise — replay treats a
    raised exception identically to "True" via the outer try block.
    """
    if not criterion or not criterion.strip():
        return True
    import re  # noqa: PLC0415

    tokens = [
        t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", criterion)
        if len(t) >= min_token_len and t.lower() not in _CRIT_STOPWORDS
    ]
    if not tokens:
        return True
    # Cap tokens to the first 6 distinct ones — long criteria
    # would otherwise require an unrealistic number of matches.
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        ordered.append(t)
        if len(ordered) >= 6:
            break
    try:
        page_text = page.evaluate(
            "() => (document.body && document.body.innerText) || ''"
        )
    except Exception:
        return True
    if not isinstance(page_text, str):
        return True
    haystack = page_text.lower()
    hits = sum(1 for t in ordered if t in haystack)
    return hits >= max(1, require_matches)


def _persist_self_heal_patches(
    db: Session,
    submodule: TcNode,
    frozen: dict[str, Any],
    steps: list[dict[str, Any]],
    self_healed_steps: list[int],
    submodule_run_id: int | None,
) -> None:
    """Write the in-memory ``steps`` (with self-heal patches applied)
    back to ``tc_nodes.frozen_path`` so the next replay starts from the
    healed prefix. No-op when nothing was healed.

    Called from BOTH the success path AND the early-return failure
    path — otherwise patches accumulated for steps 1..idx-1 are lost
    when step idx fails, and the next run pays the self-heal cost
    over again."""
    if not self_healed_steps:
        return
    patched_frozen = dict(frozen)
    patched_frozen["steps"] = steps
    patched_frozen["self_healed_at_run_id"] = submodule_run_id
    patched_frozen["self_healed_at"] = _utcnow().isoformat()
    sm_row = db.get(TcNode, submodule.id)
    if sm_row is not None:
        sm_row.frozen_path = patched_frozen
        db.commit()


# ── Step dispatch ─────────────────────────────────────────────────


def _dispatch_frozen_step(
    page,
    step: dict[str, Any],
    *,
    plan_target_url: str,
    speed_config,
) -> dict[str, Any]:
    """Run one frozen step. Same handler set as agentic mode, but the
    args come from the canonical path, not from a fresh LLM call.

    Returns ``{status, narration, error_message, working_selector}``.
    ``working_selector`` is the selector that resolved (the
    ``successful_selector`` from the frozen step if set, else the
    original ``args.target_hint``) — so the caller can patch the
    frozen path with a new selector when self-healing fires.
    """
    tool = step.get("tool")
    args = dict(step.get("args") or {})
    successful_selector = step.get("successful_selector")

    # Substitute the successful selector when one was captured at
    # freeze-time, so replay uses the WINNING form, not the
    # original test-case wording that may have been off.
    if successful_selector and "target_hint" in args:
        args["target_hint"] = successful_selector

    # Map frozen step → ActionContext shape used by the executor.
    target_hint = args.get("target_hint") or None
    value = args.get("value") or None
    url = args.get("url") or None
    expected = args.get("expected") or None
    duration_ms = int(args.get("duration_ms") or 0)

    # Build the ActionContext the same way the agentic loop does.
    if tool == "navigate":
        ctx = ActionContext(
            plan_target_url=plan_target_url,
            target_hint=url,
            narrative=None,
            expected=None,
            data_needs=[],
            speed_config=speed_config,
        )
    elif tool in ("type", "select"):
        ctx = ActionContext(
            plan_target_url=plan_target_url,
            target_hint=target_hint,
            narrative=None,
            expected=expected,
            data_needs=[],
            speed_config=speed_config,
            improvised_value=value,
        )
    elif tool == "wait" and duration_ms > 0:
        ctx = ActionContext(
            plan_target_url=plan_target_url,
            target_hint=None,
            narrative=f"wait {duration_ms}ms",
            expected=None,
            data_needs=[],
            speed_config=speed_config,
        )
    else:
        ctx = ActionContext(
            plan_target_url=plan_target_url,
            target_hint=target_hint,
            narrative=None,
            expected=expected,
            data_needs=[],
            speed_config=speed_config,
        )

    # Side actions that aren't dispatch_action handlers — replicate
    # the qa_agent logic for scroll / extract / dismiss_modal.
    if tool == "scroll":
        direction = (args.get("scroll_direction") or "down").lower()
        amount = int(args.get("scroll_amount") or 500)
        try:
            if direction == "down":
                page.mouse.wheel(0, amount)
            elif direction == "up":
                page.mouse.wheel(0, -amount)
            elif direction == "right":
                page.mouse.wheel(amount, 0)
            elif direction == "left":
                page.mouse.wheel(-amount, 0)
        except Exception as e:
            return {
                "status": "failed",
                "narration": "scroll failed",
                "error_message": f"{type(e).__name__}: {e}",
                "working_selector": None,
            }
        return {
            "status": "ok",
            "narration": f"replayed scroll {direction} {amount}px",
            "error_message": None,
            "working_selector": None,
        }

    if tool == "press_key":
        # Phase 0.10 — keyboard primitive replay. Same key, same focus
        # ordering as the agentic capture; the steps preceding this
        # in the frozen path are responsible for putting focus on the
        # right field (a click or a type), so we just dispatch the
        # key here.
        key = (args.get("key") or "").strip()
        if not key:
            return {
                "status": "failed",
                "narration": "press_key: missing 'key' arg in frozen step",
                "error_message": "frozen press_key without key",
                "working_selector": None,
            }
        try:
            page.keyboard.press(key)
        except Exception as e:
            return {
                "status": "failed",
                "narration": f"press_key {key!r} dispatch failed",
                "error_message": f"{type(e).__name__}: {e}",
                "working_selector": None,
            }
        return {
            "status": "ok",
            "narration": f"replayed press_key {key!r}",
            "error_message": None,
            "working_selector": None,
        }

    if tool == "go_back":
        try:
            response = page.go_back(
                wait_until="domcontentloaded", timeout=10_000,
            )
        except Exception as e:
            return {
                "status": "failed",
                "narration": "go_back failed",
                "error_message": f"{type(e).__name__}: {e}",
                "working_selector": None,
            }
        if response is None:
            return {
                "status": "failed",
                "narration": "go_back: history empty on replay",
                "error_message": "no previous page in history",
                "working_selector": None,
            }
        return {
            "status": "ok",
            "narration": f"replayed go_back to {page.url}",
            "error_message": None,
            "working_selector": None,
        }

    if tool == "dismiss_modal":
        # Use the same heuristic candidates as qa_agent.
        candidates = [
            "[aria-label='Close']",
            "[aria-label='close']",
            "button[aria-label*='close' i]",
            "[role='dialog'] button:has-text('Close')",
            ".modal-close",
        ]
        for sel in candidates:
            try:
                loc = page.locator(sel).first
                if loc.count() == 0:
                    continue
                if not loc.is_visible(timeout=500):
                    continue
                loc.click(timeout=1500)
                return {
                    "status": "ok",
                    "narration": f"replayed dismiss_modal via {sel!r}",
                    "error_message": None,
                    "working_selector": sel,
                }
            except Exception:
                continue
        try:
            page.keyboard.press("Escape")
            return {
                "status": "ok",
                "narration": "replayed dismiss_modal via Escape",
                "error_message": None,
                "working_selector": None,
            }
        except Exception as e:
            return {
                "status": "failed",
                "narration": "dismiss_modal: nothing close-able",
                "error_message": f"{type(e).__name__}: {e}",
                "working_selector": None,
            }

    if tool == "extract_text":
        if not target_hint:
            return {
                "status": "failed",
                "narration": "extract_text: no target",
                "error_message": "missing target_hint in frozen step",
                "working_selector": None,
            }
        try:
            resolved = resolve(page, target_hint)
        except SelectorNotFound as e:
            return {
                "status": "failed",
                "narration": f"extract_text: {target_hint!r} not visible",
                "error_message": str(e),
                "working_selector": target_hint,
                "details": {"failure_kind": "selector_not_found"},
            }
        try:
            text = resolved.locator.inner_text(timeout=5000)
        except Exception as e:
            return {
                "status": "failed",
                "narration": "extract_text: read failed",
                "error_message": f"{type(e).__name__}: {e}",
                "working_selector": target_hint,
            }
        return {
            "status": "ok",
            "narration": f"extracted {text[:120]!r}",
            "error_message": None,
            "working_selector": target_hint,
            "extracted_text": text[:1000],
        }

    # Standard action tools go through the executor dispatcher.
    try:
        result = execute_action(page, tool, ctx)
    except Exception as e:
        return {
            "status": "failed",
            "narration": "dispatcher raised on replay",
            "error_message": f"{type(e).__name__}: {e}",
            "working_selector": target_hint,
        }

    # Phase 0.10 — replay the type-and-submit pairing. When the
    # frozen step recorded ``submit: true`` AND the type itself
    # passed, fire Enter on the focused field so the form submits.
    # Same logic as the agentic loop's _execute_tool_call so a
    # frozen "type query and search" still replays as one step.
    submit_after = bool(args.get("submit")) and tool == "type"
    submit_warning: str | None = None
    if submit_after and result.status == "passed":
        try:
            page.keyboard.press("Enter")
        except Exception as e:
            submit_warning = f"Enter dispatch failed: {type(e).__name__}: {e}"

    base_narration = f"replayed {tool}: {result.narration}"
    if submit_after and result.status == "passed":
        base_narration = (
            f"{base_narration}; pressed Enter to submit"
            if submit_warning is None
            else f"{base_narration}; submit-Enter softly failed ({submit_warning})"
        )

    return {
        "status": "ok" if result.status == "passed" else (
            "blocked" if result.status == "blocked" else "failed"
        ),
        "narration": base_narration,
        "error_message": result.error_message,
        "working_selector": target_hint,
        "details": result.details,
    }


# ── Per-submodule replay ──────────────────────────────────────────


def _replay_submodule(
    page,
    submodule: TcNode,
    *,
    plan_target_url: str,
    speed_config,
    provider: LLMProvider | None,
    db: Session,
    emit_event: Callable[[str, dict], None] | None,
    is_cancelled: Callable[[], bool] | None,
    submodule_run_id: int | None,
    submodule_step_id: int | None,
    self_heal_enabled: bool,
) -> dict[str, Any]:
    """Replay one submodule's frozen path. Self-heals on miss when
    a vision-capable provider is available.

    Returns a dict shaped like the agentic ``AgentSubmoduleResult``
    so the caller can fold it into the row uniformly.
    """
    frozen = submodule.frozen_path
    # Phase B Step 4 — v2 (per-sub-goal segmented) frozen path.
    # Detect the new schema and flatten segments into a step list
    # where each step carries a ``_sub_goal_id`` annotation so the
    # replay loop can attribute outcomes per-sub-goal. On per-step
    # failure the loop skips the REST of the failing segment and
    # tries the next segment; sub-goals without a segment fall
    # through to agentic. The v1 (whole-submodule) path is preserved
    # below for back-compat with runs frozen before Phase B.
    segments_meta: list[dict[str, Any]] = []
    is_v2 = (
        isinstance(frozen, dict)
        and frozen.get("version") == 2
        and isinstance(frozen.get("segments"), list)
        and frozen.get("segments")  # non-empty
    )
    if is_v2:
        steps_v2: list[dict[str, Any]] = []
        for seg in frozen["segments"]:  # type: ignore[index]
            if not isinstance(seg, dict):
                continue
            sg_id = str(seg.get("sub_goal_id", ""))
            seg_steps = seg.get("steps") or []
            segments_meta.append({
                "sub_goal_id": sg_id,
                "description": seg.get("description", ""),
                "success_criterion": seg.get("success_criterion", ""),
                "step_count": len(seg_steps),
                "status": "pending",  # → done | failed | skipped
            })
            for s in seg_steps:
                if not isinstance(s, dict):
                    continue
                tagged = dict(s)
                tagged["_sub_goal_id"] = sg_id
                steps_v2.append(tagged)
        if not steps_v2:
            # v2 shape but every segment was empty — treat as
            # missing frozen path.
            return {
                "status": "blocked",
                "halt_reason": "no_frozen_path",
                "step_log": [],
                "self_healed_steps": [],
                "vision_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "narration": (
                    "Frozen path v2 had no replayable steps."
                ),
            }
        steps = steps_v2
    else:
        if not isinstance(frozen, dict) or not frozen.get("steps"):
            return {
                "status": "blocked",
                "halt_reason": "no_frozen_path",
                "step_log": [],
                "self_healed_steps": [],
                "vision_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "narration": (
                    "No frozen path available for this submodule. "
                    "Run agentic mode once to capture one."
                ),
            }
        steps = list(frozen.get("steps") or [])

    # Phase B Step 4 — track which sub-goal we're currently inside
    # so on failure we can skip the rest of the current segment.
    skip_until_sub_goal: str | None = None
    step_log: list[dict[str, Any]] = []
    self_healed_steps: list[int] = []
    vision_calls = 0
    total_input = 0
    total_output = 0

    # Lazy import for self-healing (avoids circular dep).
    from app.agents.page_intel import propose_search_action  # noqa: PLC0415

    provider_supports_vision = bool(
        provider and getattr(provider, "supports_vision", False),
    )

    # Phase B Step 4 — sub-goal entry tracking. Each time we see a
    # tagged step whose ``_sub_goal_id`` differs from the previous,
    # we emit a ``frozen_segment_started`` event. On a failure we
    # set ``skip_until_sub_goal`` so the loop drops the rest of the
    # current segment and resumes at the next sub-goal's first step.
    prev_sub_goal_id: str | None = None
    segments_by_id: dict[str, dict[str, Any]] = {
        s["sub_goal_id"]: s for s in segments_meta
    }
    for idx, step in enumerate(steps):
        _check_cancel(is_cancelled, f"frozen step {idx + 1}")

        # Phase B — sub-goal handling for v2 frozen paths.
        step_sub_goal_id = (
            step.get("_sub_goal_id") if is_v2 else None
        )
        if is_v2 and step_sub_goal_id:
            # Skip the rest of a failed segment.
            if (
                skip_until_sub_goal is not None
                and step_sub_goal_id == skip_until_sub_goal
            ):
                continue
            # Reached the next segment — clear the skip latch.
            if (
                skip_until_sub_goal is not None
                and step_sub_goal_id != skip_until_sub_goal
            ):
                skip_until_sub_goal = None
            # Emit segment-started on the first step of a sub-goal.
            if step_sub_goal_id != prev_sub_goal_id:
                seg = segments_by_id.get(step_sub_goal_id)
                seg_desc = seg["description"] if seg else ""
                _emit(emit_event, "frozen_segment_started", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "sub_goal_id": step_sub_goal_id,
                    "description": str(seg_desc)[:200],
                })
                if seg is not None:
                    seg["status"] = "in_progress"
                prev_sub_goal_id = step_sub_goal_id

        wait_for_settled(page, speed_config)

        update_narration(
            page,
            ordinal=idx + 1,
            total=len(steps),
            title=f"replay: {step.get('tool')}",
            action_type=step.get("tool") or "step",
            phase="about_to",
        )

        outcome = _dispatch_frozen_step(
            page, step,
            plan_target_url=plan_target_url,
            speed_config=speed_config,
        )

        # ── E.3: self-healing on a frozen-step miss ──────────────
        # When a step that used to work no longer resolves, ask
        # the vision LLM for a substitute. Only fires for
        # selector misses; not for other failure modes
        # (timeouts, dispatcher errors, etc.).
        healed = False
        outcome_details = outcome.get("details") or {}
        if (
            outcome["status"] == "failed"
            and self_heal_enabled
            and provider_supports_vision
            and provider is not None
            and step.get("tool") in ("click", "type", "select", "verify", "wait")
            and (
                outcome_details.get("failure_kind") == "selector_not_found"
                or "not visible" in (outcome.get("narration") or "").lower()
                or "no visible element" in (outcome.get("error_message") or "").lower()
            )
        ):
            target_hint = (
                outcome.get("working_selector")
                or step.get("args", {}).get("target_hint")
                or ""
            )
            if target_hint:
                _emit(emit_event, "frozen_step_self_healing", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "frozen_step_index": idx + 1,
                    "target_hint": target_hint,
                })
                try:
                    suggestion = propose_search_action(
                        provider, page,
                        target_hint=target_hint,
                        near_misses=None,
                    )
                except Exception as e:
                    logger.warning(
                        "self-heal LLM call failed: %s", e,
                    )
                    suggestion = None

                if suggestion is not None:
                    vision_calls += 1
                    if isinstance(suggestion.input_tokens, int):
                        total_input += suggestion.input_tokens
                    if isinstance(suggestion.output_tokens, int):
                        total_output += suggestion.output_tokens

                    # Apply the suggestion. For click_to_drill we
                    # try the new selector directly. For scroll /
                    # navigate / dismiss_modal, perform the side
                    # action then retry the original target.
                    if suggestion.action == "click_to_drill":
                        new_selector = suggestion.click_target_hint
                        if new_selector:
                            patched_step = dict(step)
                            patched_step.setdefault("args", {})
                            patched_step["args"] = {
                                **(patched_step.get("args") or {}),
                                "target_hint": new_selector,
                            }
                            patched_step["successful_selector"] = new_selector
                            retry = _dispatch_frozen_step(
                                page, patched_step,
                                plan_target_url=plan_target_url,
                                speed_config=speed_config,
                            )
                            if retry["status"] == "ok":
                                # Patch the frozen_path in-memory and
                                # persist so future runs use the new
                                # selector directly.
                                steps[idx] = patched_step
                                healed = True
                                outcome = {
                                    **retry,
                                    "narration": (
                                        f"SELF-HEALED: substituted "
                                        f"{new_selector!r} after "
                                        f"original missed. "
                                        f"{retry.get('narration') or ''}"
                                    ),
                                    "self_heal_action": (
                                        suggestion.action
                                    ),
                                }
                    elif suggestion.action == "scroll":
                        direction = (
                            suggestion.scroll_direction or "down"
                        )
                        amount = (
                            suggestion.scroll_amount_px or 800
                        )
                        try:
                            if direction == "down":
                                page.mouse.wheel(0, amount)
                            elif direction == "up":
                                page.mouse.wheel(0, -amount)
                        except Exception:
                            pass
                        retry = _dispatch_frozen_step(
                            page, step,
                            plan_target_url=plan_target_url,
                            speed_config=speed_config,
                        )
                        if retry["status"] == "ok":
                            healed = True
                            outcome = {
                                **retry,
                                "narration": (
                                    f"SELF-HEALED: scrolled "
                                    f"{direction} then succeeded. "
                                    f"{retry.get('narration') or ''}"
                                ),
                                "self_heal_action": "scroll",
                            }
                    # navigate / dismiss_modal / give_up: skip
                    # mid-replay; let the failure stand. Keep the
                    # path simple — agentic mode handles these
                    # cases on the next run.

                _emit(emit_event, "frozen_step_self_heal_completed", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "frozen_step_index": idx + 1,
                    "healed": healed,
                })

        if healed:
            self_healed_steps.append(idx + 1)

        phase = (
            "did" if outcome["status"] == "ok"
            else "blocked" if outcome["status"] == "blocked"
            else "failed"
        )
        update_narration(
            page,
            ordinal=idx + 1,
            total=len(steps),
            title=outcome.get("narration", "")[:80] or "replay",
            action_type=step.get("tool") or "step",
            phase=phase,
        )

        step_log.append({
            "frozen_step_index": idx + 1,
            "tool": step.get("tool"),
            "args": step.get("args"),
            "status": outcome["status"],
            "narration": outcome.get("narration"),
            "error_message": outcome.get("error_message"),
            "self_healed": healed,
            "self_heal_action": outcome.get("self_heal_action"),
        })

        _emit(emit_event, "frozen_step_completed", {
            "run_id": submodule_run_id,
            "step_id": submodule_step_id,
            "frozen_step_index": idx + 1,
            "total": len(steps),
            "tool": step.get("tool"),
            "status": outcome["status"],
            "self_healed": healed,
        })

        # Replay halts on the first hard failure — UNLESS this is a
        # v2 (per-sub-goal) frozen path, in which case we abandon
        # just the failing SEGMENT and try the next one. Sub-goals
        # without successful segments at the end will fall through
        # to an agentic recovery pass.
        if outcome["status"] == "failed":
            if is_v2 and step_sub_goal_id:
                seg = segments_by_id.get(step_sub_goal_id)
                if seg is not None:
                    seg["status"] = "failed"
                    seg["failure_reason"] = (
                        outcome.get("error_message")
                        or outcome.get("narration")
                        or "frozen step failed"
                    )[:300]
                _emit(emit_event, "frozen_segment_failed", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "sub_goal_id": step_sub_goal_id,
                    "frozen_step_index": idx + 1,
                    "reason": (
                        outcome.get("error_message")
                        or outcome.get("narration") or ""
                    )[:300],
                })
                # Skip the remaining steps that belong to this
                # sub-goal. Resumes at the next sub-goal's first
                # step.
                skip_until_sub_goal = step_sub_goal_id
                # DON'T return — continue the for-loop. Persist
                # any self-heal patches as we go.
                continue

            # Persist patches accumulated for steps 1..idx-1 even on
            # failure so the next run's healed prefix doesn't pay the
            # self-heal cost again. Only the failing step is lost.
            _persist_self_heal_patches(
                db, submodule, frozen, steps,
                self_healed_steps, submodule_run_id,
            )
            return {
                "status": "failed",
                "halt_reason": "frozen_step_failed",
                "step_log": step_log,
                "self_healed_steps": self_healed_steps,
                "vision_calls": vision_calls,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "narration": (
                    f"Frozen step {idx + 1}/{len(steps)} "
                    f"({step.get('tool')}) failed and could not "
                    f"self-heal: {outcome.get('error_message') or ''}"
                ),
                "failed_step_index": idx + 1,
            }

        # v2 — emit segment-complete on the LAST step of a sub-goal
        # (i.e. when the next step belongs to a different sub-goal,
        # or we've reached the end of the steps list).
        if is_v2 and step_sub_goal_id:
            next_sub_goal_id = (
                steps[idx + 1].get("_sub_goal_id")
                if idx + 1 < len(steps) else None
            )
            if next_sub_goal_id != step_sub_goal_id:
                seg = segments_by_id.get(step_sub_goal_id)
                # Phase B Step 5 — segment success_criterion gate.
                # Before marking the segment "done", run a cheap DOM
                # check that the criterion text fragments are visible
                # on the page. The check is best-effort heuristic:
                # tokenize the criterion into 2-3 high-information
                # phrases and require at least one to be visible.
                # Failure flags the segment as failed (skip-until-
                # next-sub-goal) so the agentic recovery picks it up.
                gate_ok = True
                if seg is not None and seg.get("status") != "failed":
                    crit_text = str(
                        seg.get("success_criterion", "")
                    ).strip()
                    if crit_text:
                        try:
                            gate_ok = _segment_criterion_visible(
                                page, crit_text,
                            )
                        except Exception:
                            gate_ok = True  # never fail-stop on gate
                if seg is not None and gate_ok and seg.get(
                    "status",
                ) != "failed":
                    seg["status"] = "done"
                    _emit(emit_event, "frozen_segment_completed", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "sub_goal_id": step_sub_goal_id,
                    })
                elif seg is not None and not gate_ok:
                    seg["status"] = "failed"
                    seg["failure_reason"] = (
                        f"success_criterion not observed on page: "
                        f"{str(seg.get('success_criterion', ''))[:120]}"
                    )
                    _emit(
                        emit_event, "frozen_segment_failed", {
                            "run_id": submodule_run_id,
                            "step_id": submodule_step_id,
                            "sub_goal_id": step_sub_goal_id,
                            "frozen_step_index": idx + 1,
                            "reason": (
                                "success_criterion check failed — "
                                "steps ran but the expected post-"
                                "state isn't visible on the page"
                            ),
                        },
                    )
                    # Skip-latch is moot here (this WAS the last step
                    # of the segment); the agentic recovery will
                    # pick up this sub-goal id from segments_meta.

    # All steps passed — persist any self-heal patches back to the
    # submodule so future runs use the patched selectors.
    _persist_self_heal_patches(
        db, submodule, frozen, steps,
        self_healed_steps, submodule_run_id,
    )

    # Phase B Step 4/5 — v2 partial-replay outcome. When some
    # segments failed, mark the submodule as partially-passed and
    # surface the failed sub-goals so the orchestrator can hand
    # the remaining work off to the agentic loop.
    if is_v2 and segments_meta:
        failed_segments = [
            s for s in segments_meta if s.get("status") == "failed"
        ]
        if failed_segments:
            return {
                "status": "partial",
                "halt_reason": "frozen_segments_partial",
                "step_log": step_log,
                "self_healed_steps": self_healed_steps,
                "vision_calls": vision_calls,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "narration": (
                    f"Replayed {len(steps)} frozen step(s) across "
                    f"{len(segments_meta)} segment(s); "
                    f"{len(failed_segments)} failed — "
                    "handing the rest to the agentic loop."
                ),
                "segments_meta": segments_meta,
                "failed_sub_goal_ids": [
                    s["sub_goal_id"] for s in failed_segments
                ],
            }
        return {
            "status": "passed",
            "halt_reason": "complete",
            "step_log": step_log,
            "self_healed_steps": self_healed_steps,
            "vision_calls": vision_calls,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "narration": (
                f"Replayed {len(steps)} frozen step(s) across "
                f"{len(segments_meta)} segment(s)"
                + (
                    f" — {len(self_healed_steps)} self-healed"
                    if self_healed_steps else ""
                )
            ),
            "segments_meta": segments_meta,
        }

    return {
        "status": "passed",
        "halt_reason": "complete",
        "step_log": step_log,
        "self_healed_steps": self_healed_steps,
        "vision_calls": vision_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "narration": (
            f"Replayed {len(steps)} frozen step(s)"
            + (
                f" — {len(self_healed_steps)} self-healed"
                if self_healed_steps else ""
            )
        ),
    }


# ── Plan-level entry point ────────────────────────────────────────


def run_replay_for_plan(
    db: Session,
    *,
    run_id: int,
    plan_id: int,
    selected_step_ids: list[int] | None = None,
    headless: bool = False,
    speed: str | None = None,
    provider: LLMProvider | None = None,
    self_heal_enabled: bool = True,
    auto_adjust: bool = False,  # noqa: ARG001 — signature parity
    promote_fixes: bool = False,  # noqa: ARG001
    window_position: tuple[int, int] | None = None,
    window_size: tuple[int, int] | None = None,
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    is_paused: Callable[[], bool] | None = None,  # noqa: ARG001
    wait_for_resume: Callable[[], bool] | None = None,  # noqa: ARG001
    wait_for_intervention: Callable[[int], dict | None] | None = None,  # noqa: ARG001
) -> ExecutionResult:
    """Plan-level replay runner.

    For each selected submodule:
    - If ``frozen_path`` is set → walk it deterministically.
      * On a step failure: try self-heal (vision LLM) once.
      * Persist self-heal patches back onto the frozen_path.
    - If no frozen_path → fall through to agentic mode for that
      submodule so coverage isn't all-or-nothing.

    Returns the same :class:`ExecutionResult` shape as scripted /
    agentic so :mod:`agent_run_service` branches without caring.
    """
    t0 = time.monotonic()
    speed_config = get_speed_config(speed)

    plan = db.get(TestPlan, plan_id)
    if not plan:
        raise ValueError(f"Plan {plan_id} not found")
    if not (plan.target_url and plan.target_url.strip()):
        raise ValueError(
            f"Plan {plan_id} has no target_url — cannot navigate",
        )
    project_id = plan.project_id

    _emit(emit_event, "phase", {
        "phase": "loading_steps",
        "message": f"Loading TC tree for plan '{plan.name}' (replay)",
    })

    # Reuse qa_agent's grouping logic so selected-step semantics match.
    from app.agents.qa_agent import _select_submodules_to_run  # noqa: PLC0415

    stmt = (
        select(TcNode)
        .where(TcNode.plan_id == plan_id)
        .order_by(TcNode.depth, TcNode.parent_id, TcNode.ordinal)
    )
    all_nodes = list(db.scalars(stmt))
    groups = _select_submodules_to_run(
        all_nodes, selected_step_ids=selected_step_ids,
    )
    if not groups:
        raise ValueError("No submodules selected for replay run.")

    # Pre-create one execution row per submodule (matches agentic mode).
    rows: list[ExecutionStep] = []
    for ordinal, (submodule, _steps) in enumerate(groups):
        row = ExecutionStep(
            run_id=run_id,
            project_id=project_id,
            plan_id=plan_id,
            tc_node_id=submodule.id,
            title_snapshot=(submodule.title or "")[:512],
            path_snapshot=(submodule.path_cached or submodule.title or "")[:2048],
            action_type_snapshot=None,
            target_hint_snapshot=None,
            expected_snapshot=None,
            narrative_snapshot=None,
            ordinal=ordinal,
            status="pending",
            details_json={"mode": "replay"},
        )
        db.add(row)
        rows.append(row)
    db.flush()
    db.commit()

    # Count how many submodules have a frozen path so the user can
    # see at a glance how much coverage replay can deliver before
    # the run starts.
    frozen_count = sum(
        1 for sm, _ in groups if isinstance(sm.frozen_path, dict)
    )
    _emit(emit_event, "phase", {
        "phase": "opening_browser",
        "message": (
            f"Replay: {frozen_count}/{len(groups)} submodules have a "
            f"frozen path; the rest will fall through to agentic mode."
        ),
        "total": len(rows),
        "speed": speed or "slow",
        "mode": "replay",
        "frozen_count": frozen_count,
    })

    counts = {
        "passed": 0, "failed": 0, "skipped": 0,
        "blocked": 0, "inconclusive": 0,
    }
    cancelled = False
    total_input_tokens = 0
    total_output_tokens = 0
    total_llm_calls = 0
    total_vision_calls = 0
    fallback_to_agentic_count = 0

    bs_kwargs: dict[str, Any] = {"headless": headless, "speed": speed}
    if window_position is not None:
        bs_kwargs["window_position"] = window_position
    if window_size is not None:
        bs_kwargs["window_size"] = window_size

    try:
        with browser_session(**bs_kwargs) as page:
            install_overlay(page)

            # Initial navigation — same as agentic mode.
            target_url = plan.target_url or ""
            if target_url:
                try:
                    page.goto(
                        target_url,
                        wait_until="domcontentloaded",
                        timeout=30_000,
                    )
                    wait_for_settled(page, speed_config)
                except Exception as e:
                    logger.warning(
                        "initial navigation failed: %s", e,
                    )

            for idx, ((submodule, steps_under), row) in enumerate(zip(groups, rows)):
                if is_cancelled and is_cancelled():
                    cancelled = True
                    break

                row.status = "running"
                row.started_at = _utcnow()
                db.commit()
                _emit(emit_event, "step_started", {
                    "step_id": row.id,
                    "tc_node_id": row.tc_node_id,
                    "ordinal": idx + 1,
                    "total": len(rows),
                    "title": row.title_snapshot,
                    "action_type": "replay" if isinstance(
                        submodule.frozen_path, dict,
                    ) else "agentic-fallback",
                })

                t_loop = time.monotonic()

                if isinstance(submodule.frozen_path, dict):
                    # Replay path — deterministic.
                    res = _replay_submodule(
                        page, submodule,
                        plan_target_url=plan.target_url or "",
                        speed_config=speed_config,
                        provider=provider,
                        db=db,
                        emit_event=emit_event,
                        is_cancelled=is_cancelled,
                        submodule_run_id=run_id,
                        submodule_step_id=row.id,
                        self_heal_enabled=self_heal_enabled,
                    )
                    total_input_tokens += res.get("input_tokens", 0)
                    total_output_tokens += res.get("output_tokens", 0)
                    total_vision_calls += res.get("vision_calls", 0)
                    if res.get("vision_calls"):
                        total_llm_calls += res["vision_calls"]

                    # Phase B Step 4 — partial replay → agentic handoff.
                    # When v2 frozen path partially completed (some
                    # segments succeeded, some failed), invoke the
                    # agentic loop on the SAME browser to recover the
                    # rest of the submodule. The fresh decomposer sees
                    # the (now-stale) frozen breakdown as a hint and
                    # plans a recovery path from the current page state.
                    if res.get("status") == "partial":
                        _emit(emit_event, "frozen_partial_handoff", {
                            "step_id": row.id,
                            "tc_node_id": row.tc_node_id,
                            "failed_sub_goal_ids": res.get(
                                "failed_sub_goal_ids", [],
                            ),
                        })
                        if provider is not None:
                            from app.agents.qa_agent import (  # noqa: PLC0415
                                run_agent_for_goal,
                            )
                            from app.agents.goal import (  # noqa: PLC0415
                                extract_goal,
                            )
                            try:
                                fb_goal = extract_goal(
                                    provider, submodule, steps_under,
                                )
                                fb_result = run_agent_for_goal(
                                    page, provider, fb_goal,
                                    plan_target_url=plan.target_url or "",
                                    speed_config=speed_config,
                                    emit_event=emit_event,
                                    is_cancelled=is_cancelled,
                                    submodule_run_id=run_id,
                                    submodule_step_id=row.id,
                                    db=db,
                                    plan=plan,
                                )
                                # Fold the agentic outcome over the
                                # partial replay result.
                                total_input_tokens += (
                                    fb_result.input_tokens or 0
                                )
                                total_output_tokens += (
                                    fb_result.output_tokens or 0
                                )
                                total_vision_calls += (
                                    fb_result.vision_calls or 0
                                )
                                total_llm_calls += (
                                    fb_result.llm_calls or 0
                                )
                                # Promote status from the agentic
                                # pass; capture the partial replay
                                # log into details_json for the report.
                                res = {
                                    **res,
                                    "status": fb_result.status,
                                    "halt_reason": fb_result.halt_reason,
                                    "narration": (
                                        f"Replay covered "
                                        f"{len(res.get('segments_meta') or []) - len(res.get('failed_sub_goal_ids') or [])} "
                                        f"of {len(res.get('segments_meta') or [])} segments; "
                                        f"agentic recovery: {fb_result.final_narration[:200]}"
                                    ),
                                    "agentic_recovery": {
                                        "halt_reason": fb_result.halt_reason,
                                        "turns": len(fb_result.turn_log),
                                        "sub_goals": list(
                                            getattr(
                                                fb_result, "sub_goals", [],
                                            ) or []
                                        ),
                                    },
                                }
                            except Exception as e:
                                logger.warning(
                                    "agentic recovery after partial "
                                    "replay failed: %s", e,
                                )
                                res["status"] = "failed"
                                res["halt_reason"] = (
                                    "agentic_recovery_error"
                                )

                    screenshot = _take_screenshot(page, run_id, row.id)
                    hide_narration(page)

                    row.status = res["status"]
                    row.completed_at = _utcnow()
                    row.duration_ms = int((time.monotonic() - t_loop) * 1000)
                    row.narration = res.get("narration", "")[:1024]
                    row.error_message = (
                        res.get("step_log", [])[-1].get("error_message")
                        if res["status"] == "failed"
                        and res.get("step_log") else None
                    )
                    row.screenshot_path = screenshot
                    row.details_json = {
                        "mode": "replay",
                        "frozen_path_run_id": (
                            submodule.frozen_path.get("frozen_at_run_id")
                            if isinstance(submodule.frozen_path, dict)
                            else None
                        ),
                        "halt_reason": res["halt_reason"],
                        "frozen_step_log": res["step_log"],
                        "self_healed_steps": res["self_healed_steps"],
                        "vision_calls": res["vision_calls"],
                        "input_tokens": res.get("input_tokens", 0),
                        "output_tokens": res.get("output_tokens", 0),
                        # Phase B — per-sub-goal segment status from v2
                        # replay; let the report timeline render
                        # frozen-vs-agentic per sub-goal.
                        "frozen_segments_meta": res.get(
                            "segments_meta", [],
                        ),
                        "agentic_recovery": res.get(
                            "agentic_recovery",
                        ),
                    }
                    # Phase B — clamp unknown statuses to "failed" for
                    # the counts dict (which only tracks the standard
                    # four). "partial" already promoted to an agentic
                    # outcome above; if we somehow still see it, treat
                    # as failed for accounting.
                    bucket = res["status"]
                    if bucket not in counts:
                        bucket = "failed"
                    counts[bucket] = counts.get(bucket, 0) + 1
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
                        "halt_reason": res["halt_reason"],
                        "mode": "replay",
                        "self_healed_count": len(res["self_healed_steps"]),
                    })
                else:
                    # No frozen path — fall through to agentic for
                    # this submodule. Single-row reuse keeps the
                    # cross-mode UX consistent.
                    fallback_to_agentic_count += 1
                    if provider is None:
                        row.status = "blocked"
                        row.completed_at = _utcnow()
                        row.duration_ms = int(
                            (time.monotonic() - t_loop) * 1000,
                        )
                        row.narration = (
                            "Replay needed agentic fallback but no "
                            "LLM provider is configured."
                        )
                        row.details_json = {
                            "mode": "replay",
                            "halt_reason": "agentic_fallback_no_llm",
                            "frozen_step_log": [],
                            "self_healed_steps": [],
                            "vision_calls": 0,
                        }
                        counts["blocked"] += 1
                        db.commit()
                        _emit(emit_event, "step_completed", {
                            "step_id": row.id,
                            "tc_node_id": row.tc_node_id,
                            "ordinal": idx + 1,
                            "total": len(rows),
                            "status": row.status,
                            "narration": row.narration,
                            "duration_ms": row.duration_ms,
                            "mode": "replay",
                        })
                        continue

                    # Lazy-import the agentic helper to avoid a
                    # circular dep with qa_agent.
                    from app.agents.qa_agent import (  # noqa: PLC0415
                        _build_frozen_path,
                        _categorize_divergence,
                        run_agent_for_goal,
                    )
                    from app.agents.goal import extract_goal  # noqa: PLC0415

                    try:
                        goal = extract_goal(
                            provider, submodule, steps_under,
                        )
                    except Exception as e:
                        logger.warning(
                            "agentic-fallback goal extraction failed "
                            "on submodule %s: %s",
                            submodule.id, e,
                        )
                        row.status = "inconclusive"
                        row.completed_at = _utcnow()
                        row.duration_ms = int(
                            (time.monotonic() - t_loop) * 1000,
                        )
                        row.narration = (
                            f"Goal extraction failed during "
                            f"agentic-fallback: {type(e).__name__}"
                        )
                        row.error_message = str(e)[:500]
                        counts["inconclusive"] += 1
                        db.commit()
                        continue

                    if isinstance(goal.input_tokens, int):
                        total_input_tokens += goal.input_tokens
                    if isinstance(goal.output_tokens, int):
                        total_output_tokens += goal.output_tokens
                    total_llm_calls += 1

                    agentic_result = run_agent_for_goal(
                        page, provider, goal,
                        plan_target_url=plan.target_url or "",
                        speed_config=speed_config,
                        emit_event=emit_event,
                        is_cancelled=is_cancelled,
                        submodule_run_id=run_id,
                        submodule_step_id=row.id,
                    )
                    total_input_tokens += agentic_result.input_tokens
                    total_output_tokens += agentic_result.output_tokens
                    total_llm_calls += agentic_result.llm_calls
                    total_vision_calls += agentic_result.vision_calls

                    screenshot = _take_screenshot(page, run_id, row.id)
                    hide_narration(page)

                    divergence = _categorize_divergence(
                        final_status=agentic_result.status,
                        halt_reason=agentic_result.halt_reason,
                        turn_log=agentic_result.turn_log,
                    )

                    row.status = agentic_result.status
                    row.completed_at = _utcnow()
                    row.duration_ms = int(
                        (time.monotonic() - t_loop) * 1000,
                    )
                    row.narration = agentic_result.final_narration[:1024]
                    row.error_message = agentic_result.error_message
                    row.screenshot_path = screenshot
                    row.details_json = {
                        "mode": "replay",
                        "submode": "agentic_fallback",
                        "goal": goal.to_dict(),
                        "halt_reason": agentic_result.halt_reason,
                        "divergence": divergence,
                        "agent_log": [
                            {
                                "turn": t.turn, "tool": t.tool,
                                "args": t.args, "reasoning": t.reasoning,
                                "confidence": t.confidence,
                                "status": t.status,
                                "narration": t.narration,
                                "error_message": t.error_message,
                                "page_url": t.page_url,
                                "extracted_text": t.extracted_text,
                                "search_log": t.search_log,
                            }
                            for t in agentic_result.turn_log
                        ],
                        "llm_calls": agentic_result.llm_calls,
                        "input_tokens": agentic_result.input_tokens,
                        "output_tokens": agentic_result.output_tokens,
                    }
                    counts[agentic_result.status] = counts.get(
                        agentic_result.status, 0,
                    ) + 1
                    db.commit()

                    # Same freezing rule as agentic mode — capture
                    # the now-working path so the NEXT replay run
                    # gets deterministic coverage on this submodule.
                    if agentic_result.status == "passed":
                        frozen = _build_frozen_path(
                            run_id=run_id,
                            goal=goal,
                            turn_log=agentic_result.turn_log,
                            agent_model=getattr(provider, "model", None),
                        )
                        if frozen:
                            sm_row = db.get(TcNode, submodule.id)
                            if sm_row is not None:
                                sm_row.frozen_path = frozen
                                db.commit()
                                _emit(emit_event, "frozen_path_captured", {
                                    "step_id": row.id,
                                    "tc_node_id": submodule.id,
                                    "step_count": len(frozen["steps"]),
                                    "during": "replay_fallback",
                                })

                    _emit(emit_event, "step_completed", {
                        "step_id": row.id,
                        "tc_node_id": row.tc_node_id,
                        "ordinal": idx + 1,
                        "total": len(rows),
                        "status": row.status,
                        "narration": row.narration,
                        "duration_ms": row.duration_ms,
                        "screenshot_path": row.screenshot_path,
                        "halt_reason": agentic_result.halt_reason,
                        "mode": "replay",
                        "submode": "agentic_fallback",
                    })
    except BrowserNotInstalledError:
        raise
    finally:
        if cancelled:
            now = _utcnow()
            for row in rows:
                if row.status in ("pending", "running"):
                    row.status = "skipped"
                    row.completed_at = now
                    row.narration = "run cancelled before this row"
                    counts["skipped"] = counts.get("skipped", 0) + 1
            db.commit()

    duration_ms = int((time.monotonic() - t0) * 1000)
    if cancelled:
        raise AgentCancelled(
            f"Replay run cancelled after "
            f"{sum(counts.values())}/{len(rows)} test cases",
        )

    _emit(emit_event, "done", {
        "plan_id": plan_id,
        "total_steps": len(rows),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "blocked": counts["blocked"],
        "inconclusive": counts.get("inconclusive", 0),
        "duration_ms": duration_ms,
        "mode": "replay",
        "fallback_to_agentic_count": fallback_to_agentic_count,
    })

    logger.info(
        "Replay run %s: %d passed, %d failed, %d agentic-fallback, "
        "%d vision-calls, %dms",
        run_id, counts["passed"], counts["failed"],
        fallback_to_agentic_count, total_vision_calls, duration_ms,
    )

    return ExecutionResult(
        plan_id=plan_id,
        total_steps=len(rows),
        passed=counts["passed"],
        failed=counts["failed"],
        inconclusive=counts.get("inconclusive", 0),
        skipped=counts["skipped"],
        blocked=counts["blocked"],
        duration_ms=duration_ms,
        llm_input_tokens=total_input_tokens if total_llm_calls > 0 else None,
        llm_output_tokens=total_output_tokens if total_llm_calls > 0 else None,
        ai_calls=total_llm_calls,
        ai_vision_calls=total_vision_calls,
    )
