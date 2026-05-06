"""Action dispatcher — one handler per ``action_type``.

The week-4 TC agent classifies every step into one of:

    navigate · click · type · select · verify · wait · submit · screenshot

This module turns a step (target_hint + narrative + expected + data_needs)
into a concrete page action and reports the outcome as an :class:`ActionResult`.

HITL deferral
-------------
Any step whose ``data_needs`` includes ``credentials`` or ``otp`` returns
``blocked`` immediately — week 6 wires the credential vault + OTP modal
that fill those in. The orchestrator (step 6) treats blocked the same as
failed for the purposes of cutting the branch, but records it distinctly
so the UI can light up an "intervention needed" affordance.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from playwright.sync_api import Page

from app.executor.selectors import SelectorNotFound, resolve

logger = logging.getLogger(__name__)

ActionStatus = Literal["passed", "failed", "blocked"]

_VALID_ACTIONS = (
    "navigate",
    "click",
    "type",
    "select",
    "verify",
    "wait",
    "submit",
    "screenshot",
)

# First quoted token in a narrative — `Type 'demo@example.com'` or
# `Click "Sign In"`. Single OR double quotes; first match wins.
_QUOTED_TEXT_RE = re.compile(r"""['"]([^'"]{1,500})['"]""")

# `5 seconds`, `500ms`, `2 sec`, `1.5s` — case-insensitive
_DURATION_RE = re.compile(
    r"""(\d+(?:\.\d+)?)\s*(milliseconds?|millis?|ms|seconds?|secs?|s)\b""",
    re.IGNORECASE,
)

# Bare URL in any field — http/https only; stops on whitespace or quotes
_URL_RE = re.compile(r"""https?://[^\s'"<>)]+""")

# Per-action defaults
_NAVIGATE_TIMEOUT_MS = 30_000
_WAIT_DEFAULT_MS = 1_000
_WAIT_MAX_MS = 30_000
_VERIFY_BODY_TIMEOUT_MS = 5_000


@dataclass
class ActionContext:
    """Per-step context the dispatcher needs.

    The orchestrator builds this from the step's tc_node + the plan's
    target_url before calling :func:`execute_action`.
    """

    plan_target_url: str
    target_hint: str | None
    narrative: str | None
    expected: str | None
    data_needs: list[dict[str, Any]]


@dataclass
class ActionResult:
    """Outcome the orchestrator records on the ``execution_steps`` row."""

    status: ActionStatus
    narration: str
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────


def _check_data_block(ctx: ActionContext) -> ActionResult | None:
    """If the step needs HITL data, short-circuit to ``blocked``."""
    for need in ctx.data_needs or ():
        kind = str(need.get("kind", "")).lower()
        if kind in ("credentials", "otp"):
            notes = need.get("notes") or ""
            return ActionResult(
                status="blocked",
                narration=f"Needs {kind} — HITL not wired yet (week 6).",
                details={"blocked_on": kind, "notes": notes[:200]},
            )
    return None


def _extract_text_payload(ctx: ActionContext) -> str | None:
    """Find the literal text to type/select for this step.

    Order of evidence:
    1. First quoted token in the narrative (LLM's most common pattern)
    2. ``data_needs[kind='data'].notes``

    Returns None if neither yields anything usable.
    """
    if ctx.narrative:
        m = _QUOTED_TEXT_RE.search(ctx.narrative)
        if m:
            return m.group(1)
    for need in ctx.data_needs or ():
        if str(need.get("kind", "")).lower() == "data":
            notes = (need.get("notes") or "").strip()
            if notes:
                return notes
    return None


def _extract_url(ctx: ActionContext) -> str | None:
    """Pull the navigation target URL out of the step + plan defaults."""
    for source in (ctx.target_hint, ctx.narrative):
        if source:
            m = _URL_RE.search(source)
            if m:
                return m.group(0).rstrip(".,;:!?")
    return ctx.plan_target_url or None


def _parse_duration_ms(narrative: str | None) -> int:
    """Parse `5 seconds` / `500ms` / `2 sec`; default 1s; capped at 30s."""
    if not narrative:
        return _WAIT_DEFAULT_MS
    m = _DURATION_RE.search(narrative)
    if not m:
        return _WAIT_DEFAULT_MS
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("ms") or unit.startswith("milli"):
        ms = int(value)
    else:
        ms = int(value * 1000)
    return max(0, min(ms, _WAIT_MAX_MS))


# ── Handlers ──────────────────────────────────────────────────────


def _do_navigate(page: Page, ctx: ActionContext) -> ActionResult:
    url = _extract_url(ctx)
    if not url:
        return ActionResult(
            status="failed",
            narration="navigate: no URL on step or plan",
            error_message="target_hint, narrative, and plan_target_url all empty",
        )
    try:
        response = page.goto(
            url, wait_until="domcontentloaded", timeout=_NAVIGATE_TIMEOUT_MS,
        )
    except Exception as e:
        return ActionResult(
            status="failed",
            narration=f"navigate to {url} failed",
            error_message=f"{type(e).__name__}: {e}",
            details={"url": url},
        )
    return ActionResult(
        status="passed",
        narration=f"navigated to {url}",
        details={
            "url": url,
            "http_status": response.status if response else None,
        },
    )


def _resolve_or_fail(
    page: Page, ctx: ActionContext, label: str,
) -> tuple[Any | None, ActionResult | None]:
    """Resolve target_hint or return a 'failed' ActionResult."""
    if not ctx.target_hint:
        return None, ActionResult(
            status="failed",
            narration=f"{label}: target_hint is required",
            error_message="No target_hint on this step",
        )
    try:
        return resolve(page, ctx.target_hint), None
    except SelectorNotFound as e:
        return None, ActionResult(
            status="failed",
            narration=f"{label}: target not visible",
            error_message=str(e),
            details={"target_hint": ctx.target_hint},
        )


def _do_click(page: Page, ctx: ActionContext) -> ActionResult:
    target, fail = _resolve_or_fail(page, ctx, "click")
    if fail:
        return fail
    try:
        target.locator.click()
    except Exception as e:
        return ActionResult(
            status="failed",
            narration=f"click failed: {ctx.target_hint!r}",
            error_message=f"{type(e).__name__}: {e}",
            details={"strategy": target.strategy},
        )
    return ActionResult(
        status="passed",
        narration=f"clicked {ctx.target_hint!r} (via {target.strategy})",
        details={"strategy": target.strategy, "attempts": target.attempts},
    )


def _do_type(page: Page, ctx: ActionContext) -> ActionResult:
    blocked = _check_data_block(ctx)
    if blocked:
        return blocked
    text = _extract_text_payload(ctx)
    if text is None:
        return ActionResult(
            status="failed",
            narration="type: cannot find text to enter",
            error_message=(
                "No quoted text in narrative and no 'data' need with notes"
            ),
        )
    target, fail = _resolve_or_fail(page, ctx, "type")
    if fail:
        return fail
    try:
        target.locator.fill(text)
    except Exception as e:
        return ActionResult(
            status="failed",
            narration=f"fill failed: {ctx.target_hint!r}",
            error_message=f"{type(e).__name__}: {e}",
            details={"strategy": target.strategy},
        )
    return ActionResult(
        status="passed",
        narration=f"typed into {ctx.target_hint!r} (via {target.strategy})",
        details={
            "strategy": target.strategy,
            "text_length": len(text),  # never log the text itself
        },
    )


def _do_select(page: Page, ctx: ActionContext) -> ActionResult:
    blocked = _check_data_block(ctx)
    if blocked:
        return blocked
    text = _extract_text_payload(ctx)
    if text is None:
        return ActionResult(
            status="failed",
            narration="select: cannot find option to choose",
            error_message=(
                "No quoted option in narrative and no 'data' need with notes"
            ),
        )
    target, fail = _resolve_or_fail(page, ctx, "select")
    if fail:
        return fail
    # Try by label first (visible text in the dropdown), fall back to value
    try:
        target.locator.select_option(label=text)
    except Exception as label_err:
        try:
            target.locator.select_option(value=text)
        except Exception as value_err:
            return ActionResult(
                status="failed",
                narration=f"select_option failed for {text!r}",
                error_message=(
                    f"label tried: {label_err!r}; value tried: {value_err!r}"
                ),
                details={"strategy": target.strategy, "option": text},
            )
    return ActionResult(
        status="passed",
        narration=f"selected {text!r} in {ctx.target_hint!r}",
        details={"strategy": target.strategy, "option": text},
    )


def _do_verify(page: Page, ctx: ActionContext) -> ActionResult:
    has_hint = bool(ctx.target_hint and ctx.target_hint.strip())
    has_expected = bool(ctx.expected and ctx.expected.strip())
    if not has_hint and not has_expected:
        return ActionResult(
            status="failed",
            narration="verify: needs target_hint and/or expected",
            error_message="Both fields empty — nothing to assert against",
        )

    parts: list[str] = []
    details: dict[str, Any] = {}

    if has_hint:
        try:
            target = resolve(page, ctx.target_hint)
        except SelectorNotFound as e:
            return ActionResult(
                status="failed",
                narration=f"verify: target not visible {ctx.target_hint!r}",
                error_message=str(e),
                details={"target_hint": ctx.target_hint},
            )
        parts.append(f"{ctx.target_hint!r} visible (via {target.strategy})")
        details["strategy"] = target.strategy

    if has_expected:
        expected = (ctx.expected or "").strip()
        try:
            body_text = page.locator("body").inner_text(
                timeout=_VERIFY_BODY_TIMEOUT_MS,
            )
        except Exception as e:
            return ActionResult(
                status="failed",
                narration="verify: could not read page text",
                error_message=f"{type(e).__name__}: {e}",
            )
        if expected.lower() not in body_text.lower():
            return ActionResult(
                status="failed",
                narration=f"verify: expected text not found",
                error_message=f"expected substring missing: {expected[:200]!r}",
                details={"expected": expected[:200], **details},
            )
        parts.append(f"expected text {expected[:60]!r} present")

    return ActionResult(
        status="passed",
        narration="verify: " + " · ".join(parts),
        details=details,
    )


def _do_wait(page: Page, ctx: ActionContext) -> ActionResult:
    # Selector-driven wait when target_hint is set; longer timeout than the
    # default selector waterfall — week 5's sole "waiting for X" primitive.
    if ctx.target_hint and ctx.target_hint.strip():
        try:
            target = resolve(page, ctx.target_hint, timeout_ms=15_000)
        except SelectorNotFound as e:
            return ActionResult(
                status="failed",
                narration=f"wait: {ctx.target_hint!r} never appeared",
                error_message=str(e),
            )
        return ActionResult(
            status="passed",
            narration=f"waited for {ctx.target_hint!r} (via {target.strategy})",
            details={"strategy": target.strategy},
        )

    duration_ms = _parse_duration_ms(ctx.narrative)
    page.wait_for_timeout(duration_ms)
    return ActionResult(
        status="passed",
        narration=f"waited {duration_ms}ms",
        details={"duration_ms": duration_ms},
    )


def _do_submit(page: Page, ctx: ActionContext) -> ActionResult:
    # In practice "submit" steps just click a submit button — the LLM uses
    # the semantic distinction, the page treats them identically.
    return _do_click(page, ctx)


def _do_screenshot(page: Page, ctx: ActionContext) -> ActionResult:
    # The orchestrator takes a screenshot after every step regardless;
    # the action-typed `screenshot` is a no-op success marker.
    return ActionResult(
        status="passed",
        narration="screenshot captured",
    )


_HANDLERS: dict[str, Callable[[Page, ActionContext], ActionResult]] = {
    "navigate": _do_navigate,
    "click": _do_click,
    "type": _do_type,
    "select": _do_select,
    "verify": _do_verify,
    "wait": _do_wait,
    "submit": _do_submit,
    "screenshot": _do_screenshot,
}


def execute_action(
    page: Page,
    action_type: str | None,
    ctx: ActionContext,
) -> ActionResult:
    """Dispatch on ``action_type`` and return the handler's result.

    Unknown / missing types fall back to ``verify`` — that's the safest
    default since it makes no destructive page changes. The orchestrator
    records the original action_type in the snapshot so this fallback is
    visible on the timeline.
    """
    key = (action_type or "verify").strip().lower()
    handler = _HANDLERS.get(key)
    if handler is None:
        return ActionResult(
            status="failed",
            narration=f"unknown action_type: {action_type!r}",
            error_message=(
                f"action_type {action_type!r} not in {sorted(_HANDLERS)}"
            ),
        )
    return handler(page, ctx)
