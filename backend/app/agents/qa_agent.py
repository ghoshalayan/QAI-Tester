"""QA agent — goal-oriented, tool-calling executor (Phase C).

The "human-like" alternative to ``execute.py``'s rigid step walker. Instead
of executing a fixed list of steps, the agent:

1. Receives a :class:`Goal` (description + observable success criteria
   + the original steps as hints) for ONE submodule.
2. Loops: ``observe(page) → think(goal, history) → act(tool) →
   verify(criteria)`` until the goal is achieved, halt-guards trip, or
   the agent itself decides to stop.
3. Returns a summary that the orchestrator persists as a single
   ``execution_steps`` row per submodule.

Tool palette
------------
The LLM picks ONE tool per turn, with arguments. Tools split into two:

- **Action tools** (mutate the page): ``navigate``, ``click``, ``type``,
  ``select``, ``verify``, ``wait``, ``scroll``, ``extract_text``,
  ``dismiss_modal``. The first six wrap ``executor.actions``; the last
  three are agent-only.
- **Meta tools** (terminate the loop): ``mark_goal_complete``,
  ``mark_goal_failed``, ``ask_human``.

Loop guards (anti-infinite-loop)
--------------------------------
- ``max_turns``         — hard cap (default 30)
- ``max_wallclock_s``   — hard cap (default 300s)
- ``max_input_tokens``  — budget kill-switch (default 80k)
- ``max_output_tokens`` — budget kill-switch (default 20k)
- Page-state hash unchanged for ``stall_threshold`` consecutive turns
  → halt with reason ``stall``
- Same action signature (``tool``, ``target_hint``, ``value``) seen
  ``signature_repeat_threshold`` times in the last ``signature_window``
  turns → halt with reason ``oscillation``
- Cancellation is honored at every turn boundary
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.brd_to_frd import AgentCancelled
from app.agents.execute import ExecutionResult, _take_screenshot
from app.agents.goal import Goal, extract_goal
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
from app.executor.overlay import highlight_target
from app.executor.selectors import SelectorNotFound, resolve
from app.llm.base import ChatMessage, LLMProvider
from app.models.agent_run import AgentRun
from app.models.execution_step import ExecutionStep
from app.models.tc_node import TcNode
from app.models.test_plan import TestPlan

logger = logging.getLogger(__name__)


# ── Tool catalog ───────────────────────────────────────────────────

ToolName = Literal[
    "navigate",
    "click",
    "type",
    "select",
    "verify",
    "wait",
    "scroll",
    "extract_text",
    "dismiss_modal",
    "mark_goal_complete",
    "mark_goal_failed",
    "ask_human",
]

ACTION_TOOLS: frozenset[str] = frozenset({
    "navigate", "click", "type", "select", "verify", "wait",
    "scroll", "extract_text", "dismiss_modal",
})
META_TOOLS: frozenset[str] = frozenset({
    "mark_goal_complete", "mark_goal_failed", "ask_human",
})


# Flat schema (OpenAI-strict + Gemini-friendly): every property required
# with a sensible empty default. The agent fills only the fields its
# chosen tool needs.
TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": list(ACTION_TOOLS) + list(META_TOOLS),
        },
        "target_hint": {"type": "string"},
        "value": {"type": "string"},
        "url": {"type": "string"},
        "expected": {"type": "string"},
        "duration_ms": {"type": "integer"},
        "scroll_direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right", ""],
        },
        "scroll_amount": {"type": "integer"},
        "question": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "tool", "target_hint", "value", "url", "expected",
        "duration_ms", "scroll_direction", "scroll_amount",
        "question", "reasoning", "confidence",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You are a senior QA tester driving a real browser to verify a test case.

The browser is ALREADY on the target site at run-start — you don't
need a `navigate` call to get there. Use `navigate` only if the goal
requires going to a DIFFERENT URL.

You are given:
- TARGET SITE: the URL the run is testing (informational; you start there).
- GOAL: what you must verify (one sentence + observable success criteria).
- HINTS: the original test-case steps. Treat as guidance, not contract.
  If a hint's selector is broken or the page changed, IMPROVISE around it.
- HISTORY: the last few turns (your action + what happened).
- OBSERVATION: a snapshot of the page right now (URL + interactive
  elements with role + accessible name).
- (Sometimes) SCREENSHOT: a PNG of the page, attached when the
  previous turn FAILED. Use it to see exact button labels, layout,
  modals/banners blocking your target, and any visual cue the AX tree
  alone might miss. When you see a screenshot, treat the AX tree's
  ``name`` field as a hint — the screenshot's actual text wins. If the
  screenshot shows a "Login" button where the test case said "Sign In",
  it's the same affordance — click "Login".

Each turn pick ONE tool and provide arguments. Think like a human
tester: small, deliberate steps; verify after each meaningful action.

ACTION TOOLS (mutate the page):
- navigate(url): go to url
- click(target_hint): click an element
- type(target_hint, value): type into a field
- select(target_hint, value): pick a dropdown option
- verify(target_hint?, expected?): assert visibility or text presence
- wait(duration_ms? OR target_hint?): wait for time or element
- scroll(scroll_direction, scroll_amount?): scroll the page
- extract_text(target_hint): read text out of an element (returned in
  the next observation as "extracted_text")
- dismiss_modal(): close any blocking modal (cookie banner, signup
  popup, etc.). Use this BEFORE retrying a click that fails because
  something is overlaying the page.

META TOOLS (terminate the loop):
- mark_goal_complete(reasoning): the goal is verifiably achieved.
  Only call this when AT LEAST ONE success criterion was checked
  via verify() or extract_text() and clearly held.
- mark_goal_failed(reasoning): the goal CANNOT be achieved here
  (page broken, feature missing, login expired). Different from
  the test being wrong — this is the APP failing.
- ask_human(question): you're stuck and need guidance. Use sparingly.

RULES:
- target_hint: a Playwright-resolvable hint. CSS selector, "text 'Sign In'",
  or "role=button[name='Sign In']". Prefer stable hints (data-testid,
  text, role) over fragile ones (nth-child).
- Don't repeat the same failing action 3 times — try a different approach.
- If you've tried for many turns with no progress, mark_goal_failed.
- Be concrete: copy product names, button labels VERBATIM from the
  observation. Don't fabricate.
- ALWAYS set: tool, reasoning (1 sentence), confidence (0.0-1.0).
  Set the args your tool needs and leave the rest as empty strings / 0.

Output JSON only.
"""


# ── Agent state + result ──────────────────────────────────────────


@dataclass
class ToolCall:
    """One LLM-decided turn — what the agent chose to do."""

    tool: str
    args: dict[str, Any]
    reasoning: str
    confidence: float
    input_tokens: int | None
    output_tokens: int | None


@dataclass
class TurnRecord:
    """Persisted log entry — folded into details_json["agent_log"]."""

    turn: int
    tool: str
    args: dict[str, Any]
    reasoning: str
    confidence: float
    status: str          # "ok" | "failed" | "blocked" | "stop"
    narration: str
    error_message: str | None = None
    page_url: str = ""
    extracted_text: str = ""


HaltReason = Literal[
    "complete", "agent_failed", "ask_human", "stall", "oscillation",
    "max_turns", "max_wallclock", "budget", "cancelled",
]


@dataclass
class AgentSubmoduleResult:
    """Outcome for one submodule's agent run."""

    submodule_id: int
    status: Literal["passed", "failed", "blocked", "inconclusive"]
    halt_reason: HaltReason
    turn_log: list[TurnRecord] = field(default_factory=list)
    final_narration: str = ""
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    # Subset of llm_calls that included a screenshot (vision pass).
    # Triggered after a failed action when the provider supports vision.
    vision_calls: int = 0
    duration_ms: int = 0
    final_screenshot: str | None = None


# ── Helpers ────────────────────────────────────────────────────────


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
            logger.warning("emit_event raised in qa_agent: %s", e)


def _signature(tool: str, args: dict[str, Any]) -> str:
    """Stable key for action-repeat detection."""
    return json.dumps(
        {
            "tool": tool,
            "target_hint": args.get("target_hint", ""),
            "value": args.get("value", ""),
            "url": args.get("url", ""),
        },
        sort_keys=True,
    )


def _capture_observation(page) -> dict[str, Any]:
    """Build the per-turn observation: URL + interactive-element summary.

    Reuses the page-summary JS from ``page_intel`` so the LLM sees the
    same shape it does for recovery / improvisation.
    """
    from app.agents.page_intel import _PAGE_SUMMARY_JS

    try:
        summary = page.evaluate(_PAGE_SUMMARY_JS)
    except Exception as e:
        logger.warning("observation evaluate failed: %s", e)
        summary = {"url": "(unknown)", "title": "", "items": []}
    return summary or {"url": "(unknown)", "title": "", "items": []}


def _hash_observation(obs: dict[str, Any]) -> str:
    """Stable digest for stall detection — URL + element fingerprint."""
    items = obs.get("items") or []
    fingerprint = [
        f"{i.get('role','')}:{(i.get('name','') or '')[:60]}" for i in items
    ]
    blob = obs.get("url", "") + "\n" + "\n".join(fingerprint)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]


def _format_observation_for_prompt(obs: dict[str, Any], max_items: int = 60) -> str:
    """Trim to the top N items so prompts stay bounded."""
    short = {
        "url": obs.get("url", ""),
        "title": obs.get("title", ""),
        "items": (obs.get("items") or [])[:max_items],
    }
    return json.dumps(short, indent=2, ensure_ascii=False)


def _format_history_for_prompt(turns: list[TurnRecord], window: int = 6) -> str:
    if not turns:
        return "(no turns yet — this is the first action)"
    recent = turns[-window:]
    out: list[str] = []
    for t in recent:
        out.append(
            f"  T{t.turn}: {t.tool}({_one_line_args(t.args)}) "
            f"→ {t.status}: {t.narration[:160]}",
        )
    return "\n".join(out)


def _one_line_args(args: dict[str, Any]) -> str:
    """Compact args repr — only non-empty fields."""
    keep = {
        k: v for k, v in args.items()
        if v not in (None, "", 0) and k not in ("reasoning", "confidence")
    }
    if not keep:
        return ""
    return json.dumps(keep, ensure_ascii=False)[:120]


def _format_goal_for_prompt(goal: Goal) -> str:
    crit_lines = "\n".join(
        f"  - {c}" for c in goal.success_criteria
    ) or "  (none specified — pick whatever observable signals fit)"
    hint_lines: list[str] = []
    for h in goal.hints[:20]:
        hint_lines.append(
            f"  {h.ordinal + 1}. [{h.action_type or '?'}] {h.title}"
            f" — target: {h.target_hint or '?'}",
        )
    hints_text = "\n".join(hint_lines) or "  (no hints — improvise)"
    return (
        f"GOAL: {goal.description}\n"
        f"PATH: {goal.path}\n"
        f"SUCCESS CRITERIA:\n{crit_lines}\n\n"
        f"HINTS (original test-case steps — guidance, not contract):\n{hints_text}"
    )


# ── Tool dispatch ─────────────────────────────────────────────────


def _tool_to_action_context(
    tool: str, args: dict[str, Any], plan_target_url: str, speed_config,
) -> ActionContext:
    """Map an agent tool call to an ActionContext for execute_action."""
    target = args.get("target_hint", "") or None
    value = args.get("value", "") or None
    url = args.get("url", "") or None
    expected = args.get("expected", "") or None
    duration = args.get("duration_ms") or 0

    if tool == "navigate":
        # navigate prefers url in target_hint to satisfy _extract_url
        return ActionContext(
            plan_target_url=plan_target_url,
            target_hint=url,
            narrative=None,
            expected=None,
            data_needs=[],
            speed_config=speed_config,
        )

    if tool in ("type", "select"):
        # value goes via improvised_value to bypass quoted-narrative parse
        return ActionContext(
            plan_target_url=plan_target_url,
            target_hint=target,
            narrative=None,
            expected=expected,
            data_needs=[],
            speed_config=speed_config,
            improvised_value=value,
        )

    if tool == "wait" and duration > 0:
        # wait by duration: encode in narrative to reuse _parse_duration_ms
        return ActionContext(
            plan_target_url=plan_target_url,
            target_hint=None,
            narrative=f"wait {int(duration)}ms",
            expected=None,
            data_needs=[],
            speed_config=speed_config,
        )

    return ActionContext(
        plan_target_url=plan_target_url,
        target_hint=target,
        narrative=None,
        expected=expected,
        data_needs=[],
        speed_config=speed_config,
    )


def _do_scroll(page, args: dict[str, Any]) -> tuple[str, str, str | None]:
    """Scroll handler — returns (status, narration, error)."""
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
        else:
            return "failed", f"unknown scroll direction: {direction}", direction
    except Exception as e:
        return "failed", "scroll failed", f"{type(e).__name__}: {e}"
    return "ok", f"scrolled {direction} by {amount}px", None


def _do_extract_text(
    page, args: dict[str, Any],
) -> tuple[str, str, str | None, str]:
    """Extract text — returns (status, narration, error, extracted_text)."""
    target = args.get("target_hint") or ""
    if not target:
        return "failed", "extract_text: target_hint required", None, ""
    try:
        resolved = resolve(page, target)
    except SelectorNotFound as e:
        return "failed", f"extract_text: target not visible {target!r}", str(e), ""
    try:
        text = resolved.locator.inner_text(timeout=5000)
    except Exception as e:
        return "failed", "extract_text: could not read text", f"{type(e).__name__}: {e}", ""
    return "ok", f"extracted from {target!r}: {text[:120]!r}", None, text[:1000]


def _do_dismiss_modal(page) -> tuple[str, str, str | None]:
    """Best-effort modal close. Tries common close-button selectors.

    A real human's first move when something blocks them — we mirror it.
    """
    candidates = [
        "[aria-label='Close']",
        "[aria-label='close']",
        "button[aria-label*='close' i]",
        "[role='dialog'] button:has-text('Close')",
        "[role='dialog'] button:has-text('No thanks')",
        "[role='dialog'] button:has-text('Decline')",
        "[role='dialog'] button:has-text('Reject')",
        "button:has-text('Accept all')",
        "button:has-text('Got it')",
        "button:has-text('×')",
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
            return "ok", f"dismissed modal via {sel!r}", None
        except Exception:
            continue
    # Last resort: press Escape — many modals listen for it
    try:
        page.keyboard.press("Escape")
        return "ok", "dismissed via Escape key", None
    except Exception as e:
        return "failed", "no modal close affordance found", f"{type(e).__name__}: {e}"


def _execute_tool_call(
    page,
    tool: str,
    args: dict[str, Any],
    *,
    plan_target_url: str,
    speed_config,
) -> dict[str, Any]:
    """Execute one action tool. Returns a dict with status / narration /
    error / extracted_text. Meta tools never come here.
    """
    if tool == "scroll":
        status, narration, error = _do_scroll(page, args)
        return {
            "status": status,
            "narration": narration,
            "error_message": error,
            "extracted_text": "",
        }
    if tool == "extract_text":
        status, narration, error, text = _do_extract_text(page, args)
        return {
            "status": status,
            "narration": narration,
            "error_message": error,
            "extracted_text": text,
        }
    if tool == "dismiss_modal":
        status, narration, error = _do_dismiss_modal(page)
        return {
            "status": status,
            "narration": narration,
            "error_message": error,
            "extracted_text": "",
        }

    # Wrapped action tools — go through the existing dispatcher.
    # Map agent tool name to the dispatcher's action_type.
    action_type = tool  # one-to-one
    ctx = _tool_to_action_context(tool, args, plan_target_url, speed_config)

    # Visual ring on resolvable targets so the on-page overlay shows
    # what the agent is acting on.
    if action_type == "verify" and ctx.target_hint:
        try:
            target = resolve(page, ctx.target_hint)
            highlight_target(page, target.locator, duration_ms=1500)
        except Exception:
            pass

    try:
        result = execute_action(page, action_type, ctx)
    except Exception as e:
        return {
            "status": "failed",
            "narration": "dispatcher raised",
            "error_message": f"{type(e).__name__}: {e}",
            "extracted_text": "",
        }

    return {
        "status": "ok" if result.status == "passed" else (
            "blocked" if result.status == "blocked" else "failed"
        ),
        "narration": result.narration,
        "error_message": result.error_message,
        "extracted_text": "",
    }


# ── The agent loop ────────────────────────────────────────────────


def run_agent_for_goal(
    page,
    provider: LLMProvider,
    goal: Goal,
    *,
    plan_target_url: str,
    speed_config,
    max_turns: int = 30,
    max_wallclock_s: int = 300,
    max_input_tokens: int = 80_000,
    max_output_tokens: int = 20_000,
    stall_threshold: int = 3,
    signature_repeat_threshold: int = 3,
    signature_window: int = 8,
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    submodule_run_id: int | None = None,
    submodule_step_id: int | None = None,
) -> AgentSubmoduleResult:
    """Drive ONE submodule to completion via the observe-think-act loop.

    Halt order (first to fire wins):
    1. ``mark_goal_complete``    → status=passed, halt=complete
    2. ``mark_goal_failed``      → status=failed, halt=agent_failed
    3. ``ask_human``             → status=blocked, halt=ask_human
    4. cancellation              → status=blocked, halt=cancelled
    5. token / time / turn caps  → status=inconclusive, halt=<cap_name>
    6. stall (no page change)    → status=inconclusive, halt=stall
    7. oscillation (action repeat) → status=inconclusive, halt=oscillation
    """
    t0 = time.monotonic()
    turn_log: list[TurnRecord] = []
    history_signatures: deque[str] = deque(maxlen=signature_window)
    obs_hashes: deque[str] = deque(maxlen=stall_threshold)
    total_input = 0
    total_output = 0
    llm_calls = 0
    vision_calls = 0
    # Vision-on-demand: when an action tool fails, we capture the page
    # screenshot and attach it to the NEXT turn's user message. This
    # gives the agent visual context (exact button labels, blocking
    # modals, layout cues) that the AX tree alone can miss — Atlas /
    # Comet-style behavior, but only paid for when text-only failed.
    pending_screenshot: bytes | None = None
    provider_supports_vision = bool(
        getattr(provider, "supports_vision", False),
    )

    halt_reason: HaltReason = "max_turns"
    final_status: Literal["passed", "failed", "blocked", "inconclusive"] = (
        "inconclusive"
    )
    final_narration = ""
    final_error: str | None = None

    for turn_idx in range(1, max_turns + 1):
        # ── Cancellation check ─────────────────────────────────────
        if is_cancelled and is_cancelled():
            halt_reason = "cancelled"
            final_status = "blocked"
            final_narration = "Cancelled mid-loop"
            break

        # ── Budget caps ───────────────────────────────────────────
        if total_input > max_input_tokens or total_output > max_output_tokens:
            halt_reason = "budget"
            final_status = "inconclusive"
            final_narration = (
                f"Token budget exceeded "
                f"(in={total_input}/{max_input_tokens}, "
                f"out={total_output}/{max_output_tokens})"
            )
            break

        if (time.monotonic() - t0) > max_wallclock_s:
            halt_reason = "max_wallclock"
            final_status = "inconclusive"
            final_narration = f"Wall-clock budget exceeded ({max_wallclock_s}s)"
            break

        # ── Observe ────────────────────────────────────────────────
        wait_for_settled(page, speed_config)
        observation = _capture_observation(page)
        obs_hash = _hash_observation(observation)

        # Stall guard: if last `stall_threshold` observations are
        # identical AND we already took at least one action (so we're
        # not just sitting on the initial page), halt.
        obs_hashes.append(obs_hash)
        if (
            len(obs_hashes) == stall_threshold
            and len(set(obs_hashes)) == 1
            and turn_idx > stall_threshold
        ):
            halt_reason = "stall"
            final_status = "inconclusive"
            final_narration = (
                f"Page unchanged for {stall_threshold} consecutive turns"
            )
            break

        # ── Think (LLM call) ───────────────────────────────────────
        # Vision-on-demand: if the previous turn failed AND the provider
        # can see images, attach the screenshot we captured then. The
        # agent will use it to recover from selectors that look right in
        # the AX tree but visually don't match (different label, hidden
        # by overlay, button moved, etc.).
        attach_screenshot = (
            pending_screenshot is not None and provider_supports_vision
        )
        vision_note = ""
        if attach_screenshot:
            vision_note = (
                "\nSCREENSHOT ATTACHED: the previous turn failed. The "
                "page's PNG is attached to this message — use it to read "
                "the actual visible labels, see what's blocking, and "
                "pick a more reliable target. The screenshot is the "
                "ground truth; if it disagrees with the AX tree, trust "
                "the screenshot.\n"
            )

        user_prompt = (
            f"TARGET SITE (the app under test): "
            f"{plan_target_url or '(none configured)'}\n"
            f"The browser was already navigated here at run-start; you "
            f"should normally NOT need to navigate again unless the goal "
            f"itself requires going elsewhere.\n"
            f"{vision_note}\n"
            f"{_format_goal_for_prompt(goal)}\n\n"
            f"HISTORY (last few turns):\n{_format_history_for_prompt(turn_log)}\n\n"
            f"CURRENT OBSERVATION:\n{_format_observation_for_prompt(observation)}\n\n"
            f"This is turn {turn_idx}/{max_turns}. Pick ONE tool."
        )

        try:
            user_msg = ChatMessage(
                role="user",
                content=user_prompt,
                image=pending_screenshot if attach_screenshot else None,
            )
            llm_result = provider.chat_structured(
                messages=[
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    user_msg,
                ],
                schema=TOOL_CALL_SCHEMA,
                schema_name="qa_tool_call",
                temperature=0.3,
                max_output_tokens=1024,
            )
            if attach_screenshot:
                vision_calls += 1
            # Consume the screenshot — only attached once per failure.
            pending_screenshot = None
        except Exception as e:
            halt_reason = "agent_failed"
            final_status = "inconclusive"
            err_str = f"{type(e).__name__}: {str(e)[:300]}"
            # Bake the cause into the narration too — the live panel
            # only shows ``narration``; otherwise users see "LLM call
            # failed mid-loop" with no clue what actually happened
            # (rate limit, schema rejection, network, etc).
            final_narration = f"LLM call failed mid-loop — {err_str}"
            final_error = err_str
            logger.warning("agent LLM error: %s", e, exc_info=True)
            break

        llm_calls += 1
        if isinstance(llm_result.input_tokens, int):
            total_input += llm_result.input_tokens
        if isinstance(llm_result.output_tokens, int):
            total_output += llm_result.output_tokens

        parsed = llm_result.parsed
        if not isinstance(parsed, dict) or not parsed.get("tool"):
            halt_reason = "agent_failed"
            final_status = "inconclusive"
            final_narration = "Agent returned malformed tool call"
            final_error = f"parsed={type(parsed).__name__}"
            break

        tool = str(parsed.get("tool"))
        reasoning = str(parsed.get("reasoning", "")).strip()
        confidence = float(parsed.get("confidence", 0.0))
        args = {
            k: v for k, v in parsed.items()
            if k not in ("tool", "reasoning", "confidence")
        }

        _emit(emit_event, "agent_thought", {
            "run_id": submodule_run_id,
            "step_id": submodule_step_id,
            "turn": turn_idx,
            "tool": tool,
            "reasoning": reasoning[:500],
            "confidence": confidence,
        })

        # Update on-page banner so the visible browser shows the agent's
        # intent (matches scripted-mode UX).
        update_narration(
            page,
            ordinal=turn_idx,
            total=max_turns,
            title=reasoning[:80] or f"agent: {tool}",
            action_type=tool,
            phase="about_to",
        )

        # ── Meta tool branch: terminating decisions ───────────────
        if tool == "mark_goal_complete":
            halt_reason = "complete"
            final_status = "passed"
            final_narration = (
                reasoning or "Agent marked goal complete"
            )[:500]
            turn_log.append(TurnRecord(
                turn=turn_idx, tool=tool, args=args, reasoning=reasoning,
                confidence=confidence, status="stop",
                narration=final_narration, page_url=observation.get("url", ""),
            ))
            break

        if tool == "mark_goal_failed":
            halt_reason = "agent_failed"
            final_status = "failed"
            final_narration = (
                reasoning or "Agent marked goal failed"
            )[:500]
            turn_log.append(TurnRecord(
                turn=turn_idx, tool=tool, args=args, reasoning=reasoning,
                confidence=confidence, status="stop",
                narration=final_narration, page_url=observation.get("url", ""),
            ))
            break

        if tool == "ask_human":
            halt_reason = "ask_human"
            final_status = "blocked"
            final_narration = (
                args.get("question") or "Agent asked for help"
            )[:500]
            turn_log.append(TurnRecord(
                turn=turn_idx, tool=tool, args=args, reasoning=reasoning,
                confidence=confidence, status="blocked",
                narration=final_narration, page_url=observation.get("url", ""),
            ))
            break

        # ── Action tool: oscillation guard ─────────────────────────
        sig = _signature(tool, args)
        history_signatures.append(sig)
        sig_count = Counter(history_signatures)[sig]
        if sig_count >= signature_repeat_threshold:
            halt_reason = "oscillation"
            final_status = "inconclusive"
            final_narration = (
                f"Action {tool}({_one_line_args(args)}) repeated "
                f"{sig_count}x in last {signature_window} turns"
            )
            turn_log.append(TurnRecord(
                turn=turn_idx, tool=tool, args=args, reasoning=reasoning,
                confidence=confidence, status="failed",
                narration=final_narration,
                page_url=observation.get("url", ""),
            ))
            break

        # ── Act ────────────────────────────────────────────────────
        outcome = _execute_tool_call(
            page, tool, args,
            plan_target_url=plan_target_url,
            speed_config=speed_config,
        )

        # Update banner phase to reflect outcome
        phase = (
            "did" if outcome["status"] == "ok"
            else "blocked" if outcome["status"] == "blocked"
            else "failed"
        )
        update_narration(
            page,
            ordinal=turn_idx,
            total=max_turns,
            title=outcome["narration"][:80] or f"agent: {tool}",
            action_type=tool,
            phase=phase,
        )

        rec = TurnRecord(
            turn=turn_idx,
            tool=tool,
            args=args,
            reasoning=reasoning,
            confidence=confidence,
            status=outcome["status"],
            narration=outcome["narration"][:500],
            error_message=outcome.get("error_message"),
            page_url=observation.get("url", ""),
            extracted_text=outcome.get("extracted_text", ""),
        )
        turn_log.append(rec)

        # Vision-on-demand: when an action tool fails AND the provider
        # supports vision, capture a screenshot now so the NEXT turn's
        # observation includes visual context. We don't take screenshots
        # on success — keeps token cost zero on the happy path.
        if outcome["status"] == "failed" and provider_supports_vision:
            try:
                pending_screenshot = page.screenshot(full_page=False)
            except Exception as e:
                logger.debug(
                    "post-failure screenshot capture failed: %s", e,
                )
                pending_screenshot = None

        _emit(emit_event, "agent_acted", {
            "run_id": submodule_run_id,
            "step_id": submodule_step_id,
            "turn": turn_idx,
            "tool": tool,
            "status": outcome["status"],
            "narration": rec.narration,
            "error": rec.error_message,
            "vision_pending": pending_screenshot is not None,
        })

    else:
        # for-else: loop ran to max_turns without a break
        halt_reason = "max_turns"
        final_status = "inconclusive"
        final_narration = f"Hit max_turns={max_turns} without resolution"

    duration_ms = int((time.monotonic() - t0) * 1000)
    return AgentSubmoduleResult(
        submodule_id=goal.submodule_id,
        status=final_status,
        halt_reason=halt_reason,
        turn_log=turn_log,
        final_narration=final_narration,
        error_message=final_error,
        input_tokens=total_input,
        output_tokens=total_output,
        llm_calls=llm_calls,
        vision_calls=vision_calls,
        duration_ms=duration_ms,
    )


# ── Plan-level entry point ────────────────────────────────────────


def _select_submodules_to_run(
    nodes: list[TcNode],
    *,
    selected_step_ids: list[int] | None,
) -> list[tuple[TcNode, list[TcNode]]]:
    """Group steps under their submodule parent, keeping only those a
    scripted run would have included.

    Returns ``[(submodule, [steps...]), ...]`` in execution order.

    A submodule runs iff it has at least one step that:
    - is in ``selected_step_ids`` (when given), OR
    - has ``selectable_default=True`` (when not).

    Modules-with-direct-child-steps (no submodule layer) are also
    treated as goal-bearing units — we pretend the module IS the
    submodule for purposes of the agent.
    """
    by_id = {n.id: n for n in nodes}
    children_by_parent: dict[int | None, list[TcNode]] = {}
    for n in nodes:
        children_by_parent.setdefault(n.parent_id, []).append(n)
    for sibs in children_by_parent.values():
        sibs.sort(key=lambda n: n.ordinal)

    wanted_steps = (
        set(selected_step_ids) if selected_step_ids is not None else None
    )

    groups: list[tuple[TcNode, list[TcNode]]] = []
    for n in sorted(nodes, key=lambda x: (x.depth, x.parent_id or 0, x.ordinal)):
        if n.kind != "step":
            continue
        is_target = (
            n.id in wanted_steps if wanted_steps is not None
            else n.selectable_default
        )
        if not is_target:
            continue
        # Walk up to find the goal-bearing ancestor (submodule, or the
        # closest module if there's no submodule).
        cur = n
        owner: TcNode | None = None
        while cur.parent_id is not None:
            parent = by_id.get(cur.parent_id)
            if parent is None:
                break
            if parent.kind == "submodule":
                owner = parent
                break
            cur = parent
        if owner is None:
            # No submodule ancestor — fall back to the topmost module.
            cur = n
            while cur.parent_id is not None:
                parent = by_id.get(cur.parent_id)
                if parent is None:
                    break
                cur = parent
            if cur.kind in ("module", "submodule"):
                owner = cur
        if owner is None:
            continue

        # Append to the group, dedup'd by submodule id.
        existing = next(
            (g for g in groups if g[0].id == owner.id), None,
        )
        if existing is None:
            groups.append((owner, [n]))
        else:
            existing[1].append(n)

    # Sort groups by their owner's path in tree order
    groups.sort(key=lambda g: (g[0].depth, g[0].parent_id or 0, g[0].ordinal))
    return groups


def run_qa_agent_for_plan(
    db: Session,
    *,
    run_id: int,
    plan_id: int,
    selected_step_ids: list[int] | None = None,
    headless: bool = False,
    speed: str | None = None,
    provider: LLMProvider | None = None,
    auto_adjust: bool = False,  # noqa: ARG001 — accepted for parity with execute_plan
    promote_fixes: bool = False,  # noqa: ARG001
    window_position: tuple[int, int] | None = None,
    window_size: tuple[int, int] | None = None,
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    is_paused: Callable[[], bool] | None = None,  # noqa: ARG001
    wait_for_resume: Callable[[], bool] | None = None,  # noqa: ARG001
    wait_for_intervention: Callable[[int], dict | None] | None = None,  # noqa: ARG001
    max_turns_per_goal: int = 30,
    max_wallclock_s_per_goal: int = 300,
) -> ExecutionResult:
    """Run the agentic executor: one agent-loop per submodule.

    Mirrors ``execute_plan``'s contract (same return type, same emit
    semantics) so ``agent_run_service`` can branch on mode without
    knowing the implementation differs.

    The pause / intervention plumbing is accepted for signature parity
    but not yet wired into the agent loop — the agent's internal
    halt-and-replan flow handles most of what HITL did in scripted
    mode. (Pause + intervention support for the agent is Phase D.)
    """
    if provider is None:
        raise ValueError(
            "Agentic mode requires an LLM provider — configure one in "
            "App Settings or run with mode='scripted'.",
        )

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
        "message": f"Loading TC tree for plan '{plan.name}' (agentic mode)",
    })

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
        raise ValueError(
            "No submodules selected. In agentic mode the agent runs at "
            "the test-case (submodule) level — tick at least one step "
            "under a submodule.",
        )

    # Pre-create one row per submodule.
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
            details_json={"mode": "agentic"},
        )
        db.add(row)
        rows.append(row)
    db.flush()
    db.commit()

    _emit(emit_event, "phase", {
        "phase": "opening_browser",
        "message": (
            f"Launching {'headless' if headless else 'headed'} Chromium · "
            f"{len(rows)} test case{'' if len(rows) == 1 else 's'} (agentic)"
        ),
        "total": len(rows),
        "speed": speed or "slow",
        "mode": "agentic",
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

    bs_kwargs: dict[str, Any] = {"headless": headless, "speed": speed}
    if window_position is not None:
        bs_kwargs["window_position"] = window_position
    if window_size is not None:
        bs_kwargs["window_size"] = window_size

    try:
        with browser_session(**bs_kwargs) as page:
            install_overlay(page)

            # Bootstrap navigation: scripted runs typically have an
            # explicit ``navigate`` step authored as the first action,
            # so the browser doesn't sit on about:blank when the loop
            # starts. The agent has no such authored step — without
            # this pre-nav it would burn its first few turns staring
            # at an empty page or guessing the URL. We do it ONCE for
            # the whole run, before any submodule loop, so the agent
            # observes the real app from turn 1. Failures here are
            # logged but non-fatal — the agent can still navigate
            # itself with the navigate tool if needed.
            target_url = plan.target_url or ""
            if target_url:
                _emit(emit_event, "phase", {
                    "phase": "initial_navigation",
                    "message": f"Navigating to {target_url}",
                    "target_url": target_url,
                })
                try:
                    page.goto(
                        target_url,
                        wait_until="domcontentloaded",
                        timeout=30_000,
                    )
                    wait_for_settled(page, speed_config)
                except Exception as e:
                    logger.warning(
                        "initial navigation to %s failed; agent will "
                        "have to navigate itself: %s",
                        target_url, e,
                    )
                    _emit(emit_event, "phase", {
                        "phase": "initial_navigation_failed",
                        "message": (
                            f"Couldn't open {target_url}: {type(e).__name__}. "
                            "Agent will try its own navigate tool."
                        ),
                    })

            for idx, ((submodule, steps), row) in enumerate(zip(groups, rows)):
                if is_cancelled and is_cancelled():
                    cancelled = True
                    break

                # Extract goal — single LLM call per submodule.
                _emit(emit_event, "agent_goal_extracting", {
                    "step_id": row.id,
                    "submodule_id": submodule.id,
                    "ordinal": idx + 1,
                    "total": len(rows),
                })
                try:
                    goal = extract_goal(provider, submodule, steps)
                except Exception as e:
                    logger.warning(
                        "Goal extraction failed for submodule %s: %s",
                        submodule.id, e,
                    )
                    row.status = "inconclusive"
                    row.completed_at = _utcnow()
                    row.narration = (
                        f"Goal extraction failed: {type(e).__name__}"
                    )
                    row.error_message = str(e)[:500]
                    counts["inconclusive"] += 1
                    db.commit()
                    _emit(emit_event, "step_completed", {
                        "step_id": row.id,
                        "tc_node_id": row.tc_node_id,
                        "ordinal": idx + 1,
                        "total": len(rows),
                        "status": row.status,
                        "narration": row.narration,
                    })
                    continue

                if isinstance(goal.input_tokens, int):
                    total_input_tokens += goal.input_tokens
                if isinstance(goal.output_tokens, int):
                    total_output_tokens += goal.output_tokens
                total_llm_calls += 1

                _emit(emit_event, "agent_goal_ready", {
                    "step_id": row.id,
                    "submodule_id": submodule.id,
                    "ordinal": idx + 1,
                    "total": len(rows),
                    "description": goal.description,
                    "criteria_count": len(goal.success_criteria),
                })

                # Mark row running + emit step_started.
                row.status = "running"
                row.started_at = _utcnow()
                row.details_json = {
                    "mode": "agentic",
                    "goal": goal.to_dict(),
                }
                db.commit()
                _emit(emit_event, "step_started", {
                    "step_id": row.id,
                    "tc_node_id": row.tc_node_id,
                    "ordinal": idx + 1,
                    "total": len(rows),
                    "title": row.title_snapshot,
                    "action_type": "agent",
                })

                # Run the loop.
                t_loop = time.monotonic()
                result = run_agent_for_goal(
                    page, provider, goal,
                    plan_target_url=plan.target_url or "",
                    speed_config=speed_config,
                    max_turns=max_turns_per_goal,
                    max_wallclock_s=max_wallclock_s_per_goal,
                    emit_event=emit_event,
                    is_cancelled=is_cancelled,
                    submodule_run_id=run_id,
                    submodule_step_id=row.id,
                )

                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens
                total_llm_calls += result.llm_calls
                total_vision_calls += result.vision_calls

                screenshot = _take_screenshot(page, run_id, row.id)
                hide_narration(page)

                row.status = result.status
                row.completed_at = _utcnow()
                row.duration_ms = int((time.monotonic() - t_loop) * 1000)
                row.narration = result.final_narration[:1024]
                row.error_message = result.error_message
                row.screenshot_path = screenshot
                row.details_json = {
                    "mode": "agentic",
                    "goal": goal.to_dict(),
                    "halt_reason": result.halt_reason,
                    "agent_log": [
                        {
                            "turn": t.turn,
                            "tool": t.tool,
                            "args": t.args,
                            "reasoning": t.reasoning,
                            "confidence": t.confidence,
                            "status": t.status,
                            "narration": t.narration,
                            "error_message": t.error_message,
                            "page_url": t.page_url,
                            "extracted_text": t.extracted_text,
                        }
                        for t in result.turn_log
                    ],
                    "llm_calls": result.llm_calls,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
                counts[result.status] = counts.get(result.status, 0) + 1
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
                    "halt_reason": result.halt_reason,
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
                    row.narration = "run cancelled before this test case"
                    counts["skipped"] = counts.get("skipped", 0) + 1
            db.commit()

    duration_ms = int((time.monotonic() - t0) * 1000)

    if cancelled:
        raise AgentCancelled(
            f"Agentic run cancelled after "
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
        "mode": "agentic",
    })

    logger.info(
        "Agentic run %s completed in %dms: %d passed, %d failed, "
        "%d inconclusive, %d blocked",
        run_id, duration_ms,
        counts["passed"], counts["failed"],
        counts.get("inconclusive", 0), counts["blocked"],
    )

    return ExecutionResult(
        plan_id=plan_id,
        total_steps=len(rows),
        passed=counts["passed"],
        # Roll inconclusive into ``failed`` for the legacy summary so the
        # cost meter / report card don't break; the per-row status is
        # what the report drills into anyway.
        failed=counts["failed"] + counts.get("inconclusive", 0),
        skipped=counts["skipped"],
        blocked=counts["blocked"],
        duration_ms=duration_ms,
        llm_input_tokens=total_input_tokens if total_llm_calls > 0 else None,
        llm_output_tokens=total_output_tokens if total_llm_calls > 0 else None,
        ai_calls=total_llm_calls,
        ai_vision_calls=total_vision_calls,
    )
