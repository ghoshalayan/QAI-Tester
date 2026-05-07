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
        # Page memory — agent produces a 1-2 line "what's on this
        # page right now" summary alongside the tool call. We cache it
        # keyed by URL so future turns on the same page can read the
        # memory instead of re-scraping the AX tree (Atlas/Comet
        # pattern). Piggybacks on the existing tool call so it costs
        # ~30 output tokens, not a separate LLM round-trip.
        "page_memory_note": {"type": "string"},
        # Sub-goal tracking — the agent declares which decomposed
        # sub-goal it's working on this turn, and (when applicable)
        # which sub-goal this turn's action just completed. Both are
        # short stable ids (e.g. "sg1") matching ``Goal.sub_goals[].id``.
        # Empty string when not applicable.
        "current_sub_goal_id": {"type": "string"},
        "sub_goal_completed_id": {"type": "string"},
    },
    "required": [
        "tool", "target_hint", "value", "url", "expected",
        "duration_ms", "scroll_direction", "scroll_amount",
        "question", "reasoning", "confidence",
        "page_memory_note",
        "current_sub_goal_id", "sub_goal_completed_id",
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
  HARD RULE — you MAY ONLY call this if your HISTORY contains at
  least one successful verify() or extract_text() call that confirms
  one of the SUCCESS CRITERIA. If you haven't verified anything yet,
  call verify() FIRST, then mark_goal_complete on the next turn.
  Don't claim a goal complete just because the page "looks right" —
  verify the criteria explicitly.
- mark_goal_failed(reasoning): the goal CANNOT be achieved here
  (page broken, feature missing, login expired). Different from
  the test being wrong — this is the APP failing.
- ask_human(question): you're stuck and need guidance. Use sparingly.

RULES:
- target_hint: a Playwright-resolvable hint. CSS selector, "text 'Sign In'",
  or "role=button[name='Sign In']". Prefer stable hints (data-testid,
  text, role) over fragile ones (nth-child).
- If WHAT CHANGED SINCE LAST TURN says "PAGE UNCHANGED", your previous
  action had NO visible effect. Do NOT repeat it. Try a fundamentally
  different approach: dismiss_modal first, scroll, change selector,
  use the screenshot if attached.
- Don't repeat the same failing action 3 times — try a different approach.
- If you've tried for many turns with no progress, mark_goal_failed.
- Be concrete: copy product names, button labels VERBATIM from the
  observation. Don't fabricate.
- ALWAYS set: tool, reasoning (1 sentence), confidence (0.0-1.0).
  Set the args your tool needs and leave the rest as empty strings / 0.

PAGE MEMORY:
- ``page_memory_note``: a 1-2 sentence summary of what's on the page
  RIGHT NOW (what kind of page, key affordances, key labels, blocking
  modals). NOT what you're about to do — that's ``reasoning``.
- This is cached by URL across turns. Future turns on the same URL
  see your memory note instead of the full element list, so write
  enough that future-you can act without re-scanning.
- Good example: "Product detail page for iPhone 13. 'Add to cart'
  button (data-testid=add-cart) top-right. Price $999, in-stock badge."
- Bad example (too vague): "homepage". (Useless on revisit.)
- Set to "" (empty string) ONLY when you've already memorized this
  exact URL on a previous turn AND nothing visibly changed.

SUB-GOALS:
- The goal is decomposed into ordered SUB-GOALS (you'll see them in
  the GOAL block). Work through them sequentially.
- ``current_sub_goal_id``: the id of the sub-goal you're advancing
  THIS turn (e.g. "sg2"). Always set this when sub-goals exist —
  it's how the UI shows what you're doing. Empty string only when
  there are NO sub-goals defined.
- ``sub_goal_completed_id``: ONLY set this on a turn whose action
  verifiably advanced past the named sub-goal. Examples:
    * You just clicked "Add to cart" and the next observation
      shows the cart count went up → set this to the sub-goal id
      for "click add to cart".
    * Your verify() succeeded against the criterion that proves a
      sub-goal → set the corresponding sub-goal id.
  Do NOT mark a sub-goal complete just because you took an action
  toward it — only when the OUTCOME confirms it.
- Don't call ``mark_goal_complete`` until ALL sub-goals are done.
  If you skip a sub-goal because the page architecture differs
  from the test case's assumption, that's fine — pick the next
  one and proceed.

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
    # A4.1b: when a target_hint missed and the orchestrator ran the
    # vision-search helper, the side-actions taken (scroll / navigate
    # / dismiss / drill) and their LLM cost get folded in here so the
    # report timeline shows the recovery path.
    search_log: dict[str, Any] | None = None


HaltReason = Literal[
    "complete", "agent_failed", "ask_human", "stall", "oscillation",
    "max_turns", "max_wallclock", "budget", "cancelled",
]


# A4.3: actionable categorization of why a row didn't pass cleanly.
# Lifted into ``details_json["divergence"]`` and surfaced as a chip on
# the report row + a category-specific recommendation in the panel.
DivergenceCategory = Literal[
    "passed_clean",          # row passed, no fixes needed
    "passed_with_help",      # row passed but only after fuzzy / vision
    "test_case_outdated",    # near-misses found high-similar elements
    "feature_missing",       # no near-misses, target genuinely absent
    "infra_issue",           # page errors, network timeout, browser crash
    "agent_drift",           # agent stalled / wandered; goal recoverable
    "agent_gave_up",         # agent voluntarily marked failed
    "user_cancelled",
]


def _build_frozen_path(
    *,
    run_id: int,
    goal: Goal,
    turn_log: list[TurnRecord],
    agent_model: str | None,
) -> dict[str, Any] | None:
    """Phase E.1: serialize the agent's working tool sequence so future
    runs can replay it deterministically.

    Discards meta turns (mark_goal_complete / mark_goal_failed /
    ask_human) and any turn whose status wasn't ``ok``. Captures the
    SUCCESSFUL selector that resolved (post-fuzzy / post-vision-search)
    when available, so replay uses the strongest target the system
    found, not the original test-case wording that may have been off.

    Returns the frozen-path dict, or ``None`` if there's nothing
    worth freezing (e.g., agent reached mark_goal_complete in 1 turn
    without any action — replay would be trivial).
    """
    steps: list[dict[str, Any]] = []
    for t in turn_log:
        if t.status != "ok":
            continue
        if t.tool not in (
            "navigate", "click", "type", "select", "verify", "wait",
            "scroll", "extract_text", "dismiss_modal",
        ):
            continue
        # Strip empty / zero args before persisting — keeps the frozen
        # JSON compact and makes the replay step contract obvious.
        slim_args = {
            k: v for k, v in (t.args or {}).items()
            if v not in ("", 0, None, False)
        }
        # If the agent's resolver substituted a selector (fuzzy or
        # vision-search), the substitution sits in the narration.
        # Capture it as ``successful_selector`` so replay can use the
        # *winning* form rather than the original test-case wording.
        successful_selector: str | None = None
        if "fuzzy matched" in (t.narration or "").lower():
            # Format: "...fuzzy matched 'Add to Cart' (score 0.9)"
            import re as _re  # noqa: PLC0415
            m = _re.search(
                r"fuzzy matched ['\"]([^'\"]+)['\"]",
                t.narration or "",
            )
            if m:
                successful_selector = m.group(1)
        steps.append({
            "turn": t.turn,
            "tool": t.tool,
            "args": slim_args,
            "successful_selector": successful_selector,
            "page_url_after": t.page_url,
        })
    if not steps:
        return None
    return {
        "version": 1,
        "frozen_at_run_id": run_id,
        "frozen_at": _utcnow().isoformat(),
        "agent_model": agent_model,
        "goal_description": goal.description,
        "success_criteria": list(goal.success_criteria),
        "steps": steps,
    }


def _categorize_divergence(
    *,
    final_status: str,
    halt_reason: str,
    turn_log: list[TurnRecord],
) -> dict[str, Any]:
    """Classify why a submodule ended the way it did.

    Decision tree (first match wins):
    - status passed + no fuzzy/vision rescues  → ``passed_clean``
    - status passed +    fuzzy/vision rescues  → ``passed_with_help``
    - cancelled                                → ``user_cancelled``
    - halt=agent_failed                        → ``agent_gave_up``
    - any turn had a real-page-error           → ``infra_issue``
    - search log shows near-misses 0.30-0.60   → ``test_case_outdated``
    - search log shows 0 near-misses           → ``feature_missing``
    - default                                  → ``agent_drift``

    Returns a dict with the category, a short one-line summary, and
    counts of the interventions that did/didn't work — so the report's
    recommendation panel can be specific (e.g. "5 fuzzy substitutions
    rescued this test — consider updating the test case wording").
    """
    fuzzy_rescues = 0
    vision_rescues = 0
    near_miss_max = 0.0
    near_miss_observed = False
    infra_signals = 0

    for t in turn_log:
        # Fuzzy-rescue signal: action narration explicitly mentions
        # "fuzzy matched" (added by selectors.py / actions.py).
        if "fuzzy matched" in (t.narration or "").lower():
            fuzzy_rescues += 1
        # Vision-search rescue signal: search_log halted=completed.
        sl = t.search_log
        if isinstance(sl, dict):
            kind = sl.get("kind") or "search"
            if kind == "search" and sl.get("halted") == "completed":
                vision_rescues += 1
            # Look at near-miss scores for divergence categorization.
            for action in sl.get("actions") or []:
                if not isinstance(action, dict):
                    continue
            # Any pre-search miss saw near-misses > 0?
            # We track via the helper's own near_misses list — but
            # we don't carry it forward; fall back to the agent_log
            # narration heuristic via "near-miss" mention.
        # Infra signal heuristic — error_message contains classic
        # network / browser-side errors, NOT mere selector misses.
        err = (t.error_message or "").lower()
        if any(s in err for s in (
            "net::err", "timeout", "navigation", "target_closed",
            "browser has been closed", "context was destroyed",
        )):
            infra_signals += 1
        # Capture if the narration / error mentions near-miss
        # candidates from selectors.py's enriched failure message,
        # and pull out the highest score quoted in it. Format example:
        # "Closest candidates on page: button:'Buy now' (0.42), ..."
        err_lower = (t.error_message or "").lower()
        if "closest candidates" in err_lower:
            near_miss_observed = True
            import re as _re  # noqa: PLC0415
            for m in _re.finditer(
                r"\((\d\.\d+)\)", t.error_message or "",
            ):
                try:
                    near_miss_max = max(near_miss_max, float(m.group(1)))
                except ValueError:
                    pass

    summary = ""
    if final_status == "passed":
        if fuzzy_rescues == 0 and vision_rescues == 0:
            category: DivergenceCategory = "passed_clean"
            summary = "Clean pass — no recovery interventions needed."
        else:
            category = "passed_with_help"
            parts: list[str] = []
            if fuzzy_rescues:
                parts.append(
                    f"{fuzzy_rescues} fuzzy match"
                    f"{'es' if fuzzy_rescues != 1 else ''}",
                )
            if vision_rescues:
                parts.append(
                    f"{vision_rescues} vision-guided search"
                    f"{'es' if vision_rescues != 1 else ''}",
                )
            summary = (
                f"Passed only after {' + '.join(parts)} — review "
                "test-case wording to remove the friction."
            )
    elif halt_reason == "cancelled":
        category = "user_cancelled"
        summary = "User cancelled the run."
    elif halt_reason == "agent_failed":
        category = "agent_gave_up"
        summary = (
            "Agent voluntarily marked the goal failed. The page may "
            "genuinely lack the feature, or the test case's "
            "preconditions weren't met."
        )
    elif infra_signals >= 2:
        category = "infra_issue"
        summary = (
            f"{infra_signals} infrastructure-style errors during the "
            "run (timeouts / navigation / closed contexts). Re-run "
            "before suspecting the test or the app."
        )
    elif near_miss_observed and near_miss_max >= 0.30:
        category = "test_case_outdated"
        summary = (
            "Page contains elements similar to the test-case targets "
            "but not similar enough to substitute automatically. "
            "Likely the test case's wording is stale — update the "
            "target_hints to match what the page actually shows."
        )
    elif near_miss_observed:
        category = "feature_missing"
        summary = (
            "Target elements weren't found AND nothing on the page "
            "is similar. The feature this test case exercises may "
            "not exist on the target app, or the test case is in "
            "the wrong scope."
        )
    else:
        category = "agent_drift"
        summary = (
            "Agent ran out of turns / stalled / oscillated without "
            "a clear page-side reason. Try re-running, or split "
            "the test case into smaller sub-goals."
        )

    return {
        "category": category,
        "summary": summary,
        "fuzzy_rescues": fuzzy_rescues,
        "vision_rescues": vision_rescues,
        "infra_signals": infra_signals,
    }


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


def _diff_observations(
    prev: dict[str, Any] | None, curr: dict[str, Any],
) -> str:
    """Build a 1-3 line diff that tells the agent what just changed.

    Without this, the agent can't tell whether its last action made
    progress — leading to the "click button → page didn't change → click
    same button again" stall pattern. The diff highlights:

    - URL changes (strongest signal of progress)
    - Number of new / removed interactive elements
    - Up to 3 example new elements (so the agent has something
      concrete to act on next turn)

    Returns empty string for the first turn (no prior observation).
    """
    if prev is None:
        return ""

    prev_url = prev.get("url", "")
    curr_url = curr.get("url", "")

    prev_keys = {
        f"{i.get('role','')}::{(i.get('name','') or '')[:60]}"
        for i in (prev.get("items") or [])
    }
    curr_items = curr.get("items") or []
    curr_keys = {
        f"{i.get('role','')}::{(i.get('name','') or '')[:60]}"
        for i in curr_items
    }
    added_keys = curr_keys - prev_keys
    removed_keys = prev_keys - curr_keys

    parts: list[str] = []
    if prev_url != curr_url:
        parts.append(f"URL changed: {prev_url} → {curr_url}")
    elif not added_keys and not removed_keys:
        # The single most-useful diagnostic: page is unchanged. The
        # agent should NOT repeat its last action.
        parts.append(
            "PAGE UNCHANGED since your last action — your last action "
            "had no visible effect. Pick a DIFFERENT approach.",
        )
    else:
        if added_keys:
            sample = sorted(added_keys)[:3]
            parts.append(
                f"+{len(added_keys)} new element(s); examples: {sample}",
            )
        if removed_keys:
            parts.append(f"-{len(removed_keys)} element(s) removed")

    return "\n".join(parts)


def _format_observation_for_prompt(obs: dict[str, Any], max_items: int = 60) -> str:
    """Trim to the top N items so prompts stay bounded."""
    short = {
        "url": obs.get("url", ""),
        "title": obs.get("title", ""),
        "items": (obs.get("items") or [])[:max_items],
    }
    return json.dumps(short, indent=2, ensure_ascii=False)


def _format_page_memory_for_prompt(
    memory: dict[str, dict[str, Any]],
    current_url: str,
    *,
    max_entries: int = 10,
) -> str:
    """Render the page-memory cache as a compact block for the prompt.

    Each entry is one line: ``- /path (T<turn>): <note>``. The current
    URL gets a ``← YOU ARE HERE`` marker so the agent can immediately
    read its own prior summary instead of re-parsing the AX tree.

    Returns "" when memory is empty so the caller can skip the block.
    """
    if not memory:
        return ""
    # Sort by turn (most recent first) and cap to avoid prompt bloat.
    items = sorted(
        memory.items(),
        key=lambda kv: kv[1].get("turn", 0),
        reverse=True,
    )[:max_entries]
    lines: list[str] = []
    for url, entry in items:
        # Show only the path part — most apps share a host across all
        # pages, so the host is just noise.
        try:
            from urllib.parse import urlparse
            path = urlparse(url).path or url
        except Exception:
            path = url
        marker = "  ← YOU ARE HERE" if url == current_url else ""
        note = (entry.get("note") or "")[:240]
        lines.append(
            f"- {path} (T{entry.get('turn', '?')}): {note}{marker}",
        )
    return "\n".join(lines)


def _store_page_memory(
    memory: dict[str, dict[str, Any]],
    *,
    url: str,
    note: str,
    turn: int,
    cap: int = 30,
) -> None:
    """Insert / update a memory entry. LRU-evicts the oldest by turn
    when the cap is exceeded.

    Empty notes are ignored — the agent's contract is to set "" only
    when there's nothing new to record, so we keep the existing entry.
    """
    if not url or not note.strip():
        return
    memory[url] = {"note": note.strip()[:240], "turn": turn}
    if len(memory) > cap:
        # Evict the oldest by turn number.
        oldest_url = min(
            memory.keys(),
            key=lambda u: memory[u].get("turn", 0),
        )
        memory.pop(oldest_url, None)


def _format_history_for_prompt(
    turns: list[TurnRecord],
    *,
    verbose_tail: int = 3,
    compact_window: int = 8,
) -> str:
    """Compressed history.

    The most recent ``verbose_tail`` turns get the full narration; older
    turns in the window collapse to a one-liner with just the tool +
    args + status. Anything older than ``compact_window`` is dropped.

    Saves ~30-50% on history tokens vs sending six verbose turns —
    older turns rarely justify the extra context, while the most
    recent few are what the agent actually reasons against.
    """
    if not turns:
        return "(no turns yet — this is the first action)"
    window = turns[-compact_window:]
    if len(window) <= verbose_tail:
        verbose = window
        compact: list[TurnRecord] = []
    else:
        verbose = window[-verbose_tail:]
        compact = window[:-verbose_tail]
    out: list[str] = []
    for t in compact:
        out.append(f"  T{t.turn}: {t.tool}({_one_line_args(t.args)}) → {t.status}")
    for t in verbose:
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


_SUB_GOAL_ICON = {
    "pending": "☐",
    "in_progress": "▶",
    "done": "✓",
    "failed": "✗",
    "skipped": "⊘",
}


def _format_goal_for_prompt(goal: Goal, *, turn_idx: int = 1) -> str:
    """Render the goal block.

    Hints fade by turn:
    - T1: full hints (action, title, target).
    - T2-T3: just titles (no targets — agent should be acting on
      observation by now, not following selectors blindly).
    - T4+: no hints at all (the goal description + criteria are the
      contract; hints have served their purpose).

    Sub-goals are always shown (they're how the agent stays oriented
    on multi-step flows). Each line gets a status icon so the agent
    can see at a glance which one to advance next.
    """
    crit_lines = "\n".join(
        f"  - {c}" for c in goal.success_criteria
    ) or "  (none specified — pick whatever observable signals fit)"

    if turn_idx <= 1:
        hint_lines = [
            f"  {h.ordinal + 1}. [{h.action_type or '?'}] {h.title}"
            f" — target: {h.target_hint or '?'}"
            for h in goal.hints[:20]
        ]
        hints_block = (
            "HINTS (original test-case steps — guidance, not contract):\n"
            + ("\n".join(hint_lines) or "  (no hints — improvise)")
        )
    elif turn_idx <= 3:
        hint_lines = [
            f"  {h.ordinal + 1}. {h.title}" for h in goal.hints[:20]
        ]
        hints_block = (
            "HINT TITLES (titles only — you should be acting on the "
            "observation now, not following selectors blindly):\n"
            + ("\n".join(hint_lines) or "  (no hints)")
        )
    else:
        hints_block = ""  # T4+: drop hints entirely

    sub_goal_block = ""
    if goal.sub_goals:
        sg_lines = [
            f"  {_SUB_GOAL_ICON.get(sg.status, '?')} [{sg.id}] "
            f"{sg.description}"
            for sg in goal.sub_goals
        ]
        sub_goal_block = (
            "SUB-GOALS (work through these in order; set "
            "``current_sub_goal_id`` each turn):\n"
            + "\n".join(sg_lines)
        )

    blocks = [
        f"GOAL: {goal.description}",
        f"PATH: {goal.path}",
        f"SUCCESS CRITERIA:\n{crit_lines}",
    ]
    if sub_goal_block:
        blocks.append(sub_goal_block)
    if hints_block:
        blocks.append(hints_block)
    return "\n\n".join(blocks)


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


def _vision_search_for_target(
    page,
    provider: LLMProvider,
    *,
    target_hint: str,
    max_attempts: int = 3,
    emit_event: Callable[[str, dict], None] | None = None,
    submodule_run_id: int | None = None,
    submodule_step_id: int | None = None,
) -> dict[str, Any]:
    """Vision-guided target search (Phase A4.1b).

    Loops up to ``max_attempts`` times: each iteration the LLM looks
    at a fresh screenshot + the target_hint and the resolver's
    near-miss list, and proposes ONE side-action — scroll, click a
    drill-in card, navigate, or dismiss a modal — that should bring
    the target into the page. Dispatches the action, then the caller
    retries the original tool against the now-different page state.

    Returns a summary dict the caller can fold into ``details_json``:
        {
          "attempted": int,
          "actions": [{action, reasoning, confidence, ...}],
          "input_tokens": int,
          "output_tokens": int,
          "halted": "give_up" | "max_attempts" | "completed",
        }

    Token cost is bounded: max_attempts × (vision-LLM call + side
    action). Default 3 = ~3 vision calls per truly-stuck target.
    Cheap vs the cost of an entirely failed test case.
    """
    # Lazy-import to avoid circular dep with page_intel.
    from app.agents.page_intel import (  # noqa: PLC0415
        propose_search_action, SearchSuggestion,
    )
    from app.executor.selectors import (  # noqa: PLC0415
        _capture_ax_tree, _similarity, FUZZY_NEAR_MISS_THRESHOLD,
    )

    actions_log: list[dict[str, Any]] = []
    total_in = 0
    total_out = 0
    halt_reason = "max_attempts"

    if not getattr(provider, "supports_vision", False):
        # No vision capability → can't run search. Caller falls
        # through to the normal failure path.
        return {
            "attempted": 0,
            "actions": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "halted": "no_vision",
        }

    for attempt in range(1, max_attempts + 1):
        # Build near-miss list from the CURRENT AX tree — fresh each
        # iteration since the page may have changed via the previous
        # search step (scroll / dismiss / etc.).
        items = _capture_ax_tree(page)
        scored = []
        for item in items:
            name = item.get("name") or ""
            if not name:
                continue
            score = _similarity(target_hint, name)
            if score >= FUZZY_NEAR_MISS_THRESHOLD:
                scored.append((score, item))
        scored.sort(key=lambda t: t[0], reverse=True)
        near_misses = [
            {
                "role": item.get("role"),
                "name": item.get("name"),
                "score": round(score, 2),
            }
            for score, item in scored[:5]
        ]

        _emit(emit_event, "agent_searching", {
            "run_id": submodule_run_id,
            "step_id": submodule_step_id,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "target_hint": target_hint,
            "near_misses": near_misses,
        })

        try:
            suggestion: SearchSuggestion = propose_search_action(
                provider, page,
                target_hint=target_hint,
                near_misses=near_misses,
            )
        except Exception as e:
            logger.warning("vision search call failed: %s", e)
            actions_log.append({
                "attempt": attempt,
                "action": "llm_error",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })
            halt_reason = "llm_error"
            break

        if isinstance(suggestion.input_tokens, int):
            total_in += suggestion.input_tokens
        if isinstance(suggestion.output_tokens, int):
            total_out += suggestion.output_tokens

        action_log: dict[str, Any] = {
            "attempt": attempt,
            "action": suggestion.action,
            "reasoning": suggestion.reasoning[:300],
            "confidence": suggestion.confidence,
        }

        # Dispatch the suggested side-action.
        if suggestion.action == "give_up":
            actions_log.append(action_log)
            halt_reason = "give_up"
            break

        if suggestion.action == "scroll":
            direction = (suggestion.scroll_direction or "down").lower()
            amount = suggestion.scroll_amount_px or 800
            try:
                if direction == "down":
                    page.mouse.wheel(0, amount)
                elif direction == "up":
                    page.mouse.wheel(0, -amount)
                elif direction == "right":
                    page.mouse.wheel(amount, 0)
                elif direction == "left":
                    page.mouse.wheel(-amount, 0)
                action_log["dispatched"] = (
                    f"scrolled {direction} {amount}px"
                )
            except Exception as e:
                action_log["dispatched_error"] = (
                    f"{type(e).__name__}: {e}"
                )

        elif suggestion.action == "click_to_drill":
            click_hint = suggestion.click_target_hint
            try:
                target = resolve(page, click_hint, timeout_ms=2_000)
                target.locator.click()
                action_log["dispatched"] = f"clicked {click_hint!r}"
            except Exception as e:
                action_log["dispatched_error"] = (
                    f"click failed: {type(e).__name__}: {e}"
                )

        elif suggestion.action == "navigate":
            url = suggestion.navigate_url
            try:
                page.goto(
                    url, wait_until="domcontentloaded", timeout=15_000,
                )
                action_log["dispatched"] = f"navigated to {url}"
            except Exception as e:
                action_log["dispatched_error"] = (
                    f"navigate failed: {type(e).__name__}: {e}"
                )

        elif suggestion.action == "dismiss_modal":
            status, narration, error = _do_dismiss_modal(page)
            action_log["dispatched"] = f"dismiss_modal: {status}"
            if error:
                action_log["dispatched_error"] = error

        actions_log.append(action_log)

        # Probe the original target — did the side-action bring it
        # in? If yes, we're done; the caller will re-resolve cleanly.
        try:
            resolve(page, target_hint, timeout_ms=1_500)
            halt_reason = "completed"
            _emit(emit_event, "agent_search_completed", {
                "run_id": submodule_run_id,
                "step_id": submodule_step_id,
                "attempts_used": attempt,
                "halt": "completed",
            })
            break
        except Exception:
            # Still missing — continue searching.
            continue
    else:
        halt_reason = "max_attempts"
        _emit(emit_event, "agent_search_completed", {
            "run_id": submodule_run_id,
            "step_id": submodule_step_id,
            "attempts_used": max_attempts,
            "halt": "max_attempts",
        })

    return {
        "attempted": len(actions_log),
        "actions": actions_log,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "halted": halt_reason,
    }


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
    # A4.1c: every N turns, run a vision LLM "is the agent on track?"
    # check. Cheap (~1 vision call per N turns) and catches the
    # wandering pattern that other guards miss only after many wasted
    # turns. Set to 0 to disable.
    on_track_interval: int = 5,
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
    # Track previous observation so we can give the agent an explicit
    # "what changed" diff each turn — by far the biggest cause of the
    # "agent stalls clicking the same button" pattern is the agent not
    # noticing the previous click did nothing.
    prev_observation: dict[str, Any] | None = None
    # Vision-on-demand: when an action tool fails, we capture the page
    # screenshot and attach it to the NEXT turn's user message. This
    # gives the agent visual context (exact button labels, blocking
    # modals, layout cues) that the AX tree alone can miss — Atlas /
    # Comet-style behavior, but only paid for when text-only failed.
    pending_screenshot: bytes | None = None
    provider_supports_vision = bool(
        getattr(provider, "supports_vision", False),
    )
    # Page memory — persistent within this submodule's loop, keyed by
    # URL. Each entry is the agent's own 1-2 line "what's on this
    # page" note from a previous turn. Future turns on the same URL
    # read the cached note instead of re-parsing the AX tree, the
    # Atlas/Comet site-map pattern. Cuts observation tokens by ~80%
    # on multi-page flows where the agent revisits pages.
    page_memory: dict[str, dict[str, Any]] = {}

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

        # Page-state diff — explicit "what changed" feedback so the
        # agent doesn't repeat its last action when it had no effect.
        diff_text = _diff_observations(prev_observation, observation)
        diff_block = (
            f"\nWHAT CHANGED SINCE LAST TURN:\n{diff_text}\n"
            if diff_text else ""
        )

        # A4.1c: mid-flow vision check. Every ``on_track_interval``
        # turns, ask a vision LLM whether the agent is still making
        # progress against the goal. Catches the "wandering / wrong
        # page / repeating broken click" patterns that the
        # deterministic guards (stall, oscillation) only detect AFTER
        # they've already cost several turns. Skipped when:
        #  - provider can't see images
        #  - it's still the first few turns (nothing to assess yet)
        #  - all sub-goals are already done (no point checking)
        on_track_block = ""
        unverified_sub_goals = (
            bool(goal.sub_goals)
            and any(
                sg.status not in ("done", "skipped")
                for sg in goal.sub_goals
            )
        )
        should_check_on_track = (
            provider_supports_vision
            and on_track_interval > 0
            and turn_idx >= on_track_interval
            and turn_idx % on_track_interval == 0
            and unverified_sub_goals
        )
        if should_check_on_track:
            try:
                from app.agents.page_intel import (  # noqa: PLC0415
                    check_on_track,
                )
                sg_summary = "\n".join(
                    f"  {_SUB_GOAL_ICON.get(sg.status, '?')} "
                    f"[{sg.id}] {sg.description}"
                    for sg in goal.sub_goals
                )
                recent_summary = _format_history_for_prompt(turn_log)
                on_track = check_on_track(
                    provider, page,
                    goal_description=goal.description,
                    sub_goal_summary=sg_summary,
                    recent_turns_summary=recent_summary,
                )
            except Exception as e:
                logger.debug("on-track check skipped: %s", e)
                on_track = None

            if on_track is not None:
                if isinstance(on_track.input_tokens, int):
                    total_input += on_track.input_tokens
                if isinstance(on_track.output_tokens, int):
                    total_output += on_track.output_tokens
                llm_calls += 1
                vision_calls += 1
                _emit(emit_event, "agent_on_track_check", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "turn": turn_idx,
                    "on_track": on_track.on_track,
                    "suggestion": on_track.suggestion[:200],
                    "reasoning": on_track.reasoning[:200],
                    "confidence": on_track.confidence,
                })
                # Only inject a warning when the LLM thinks the agent
                # is wandering. Confirming "on track" each time would
                # just add noise to the prompt.
                if not on_track.on_track and on_track.suggestion:
                    on_track_block = (
                        f"\n⚠ MID-FLOW CHECK (vision): you appear off-"
                        f"track. {on_track.reasoning[:200]} "
                        f"Suggestion: {on_track.suggestion[:200]}\n"
                    )

        # Page memory — show prior site-map notes the agent has
        # captured. The current URL gets a "YOU ARE HERE" marker so
        # the agent can act from memory directly when it has visited
        # this page before.
        current_url = observation.get("url", "")
        memory_text = _format_page_memory_for_prompt(
            page_memory, current_url,
        )
        memory_block = (
            f"\nPAGE MEMORY (your prior site-map notes; reuse instead "
            f"of re-scanning when possible):\n{memory_text}\n"
            if memory_text else ""
        )

        # If the agent has already memorized THIS URL, send a TRIMMED
        # observation (just URL + title + element_count) — the memory
        # note + the diff block (which names any new/removed elements)
        # give the agent enough to act, without paying for the full
        # 60-item AX tree every turn. This is the biggest token win.
        already_memorized = (
            current_url in page_memory and turn_idx > 1
        )
        if already_memorized:
            obs_block = (
                "CURRENT OBSERVATION (compressed — you already "
                "memorized this URL; full element list omitted):\n"
                + json.dumps(
                    {
                        "url": current_url,
                        "title": observation.get("title", ""),
                        "element_count": len(observation.get("items") or []),
                    },
                    indent=2,
                )
            )
        else:
            obs_block = (
                "CURRENT OBSERVATION:\n"
                + _format_observation_for_prompt(observation)
            )

        user_prompt = (
            f"TARGET SITE (the app under test): "
            f"{plan_target_url or '(none configured)'}\n"
            f"The browser was already navigated here at run-start; you "
            f"should normally NOT need to navigate again unless the goal "
            f"itself requires going elsewhere.\n"
            f"{vision_note}"
            f"{diff_block}"
            f"{on_track_block}"
            f"{memory_block}\n"
            f"{_format_goal_for_prompt(goal, turn_idx=turn_idx)}\n\n"
            f"HISTORY (last few turns):\n{_format_history_for_prompt(turn_log)}\n\n"
            f"{obs_block}\n\n"
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
        # Capture the page-memory note BEFORE we filter it out of
        # ``args``. The agent emits this alongside every tool call.
        memory_note = str(parsed.get("page_memory_note", "")).strip()
        if memory_note:
            _store_page_memory(
                page_memory,
                url=observation.get("url", ""),
                note=memory_note,
                turn=turn_idx,
            )
        # Sub-goal tracking: the agent declares which sub-goal it's
        # working on this turn, and which (if any) just completed.
        # Apply the status changes onto goal.sub_goals so the next
        # turn's prompt renders the up-to-date checklist.
        current_sg_id = str(parsed.get("current_sub_goal_id", "")).strip()
        completed_sg_id = str(parsed.get("sub_goal_completed_id", "")).strip()
        if goal.sub_goals:
            for sg in goal.sub_goals:
                if sg.id == current_sg_id and sg.status == "pending":
                    sg.status = "in_progress"
            if completed_sg_id:
                for sg in goal.sub_goals:
                    if sg.id == completed_sg_id and sg.status != "done":
                        sg.status = "done"
                        sg.completed_at_turn = turn_idx
                        _emit(emit_event, "sub_goal_progress", {
                            "run_id": submodule_run_id,
                            "step_id": submodule_step_id,
                            "sub_goal_id": sg.id,
                            "description": sg.description,
                            "status": "done",
                            "turn": turn_idx,
                            "remaining": sum(
                                1 for s in goal.sub_goals
                                if s.status not in ("done", "skipped")
                            ),
                            "total": len(goal.sub_goals),
                        })
        args = {
            k: v for k, v in parsed.items()
            if k not in (
                "tool", "reasoning", "confidence", "page_memory_note",
                "current_sub_goal_id", "sub_goal_completed_id",
            )
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
            # Soft-guard #1: did the agent actually verify anything?
            verified = any(
                t.tool in ("verify", "extract_text") and t.status == "ok"
                for t in turn_log
            )
            # Soft-guard #2: when sub-goals exist, were enough of
            # them closed out? Allow the last sub-goal to be implicit
            # (some agents call mark_goal_complete with the final
            # sub-goal still "in_progress" because the verify itself
            # closed it). So the bar is "≥ 80% done OR ≥ all-but-1".
            sub_goal_completion_ok = True
            if goal.sub_goals:
                done = sum(
                    1 for sg in goal.sub_goals if sg.status == "done"
                )
                total = len(goal.sub_goals)
                pct = done / total if total else 1.0
                sub_goal_completion_ok = (
                    pct >= 0.80 or done >= total - 1
                )

            # A4.1a: vision-grounded verdict. Only run when the
            # deterministic guards already say "OK" — otherwise we'd
            # double-fail and waste the vision call on something we'd
            # downgrade to inconclusive anyway.
            verification_record: dict[str, Any] | None = None

            if not verified or not sub_goal_completion_ok:
                halt_reason = "complete"
                final_status = "inconclusive"
                reasons = []
                if not verified:
                    reasons.append("no successful verify/extract_text")
                if not sub_goal_completion_ok:
                    done = sum(
                        1 for sg in goal.sub_goals if sg.status == "done"
                    )
                    total = len(goal.sub_goals)
                    reasons.append(
                        f"only {done}/{total} sub-goals closed"
                    )
                final_narration = (
                    "Agent marked complete but flagged inconclusive: "
                    f"{', '.join(reasons)}. Reasoning: {reasoning[:200]}"
                )
                logger.info(
                    "agent claim-complete soft-guard tripped on "
                    "submodule %s: %s",
                    goal.submodule_id, ", ".join(reasons),
                )
            else:
                # Both deterministic guards passed. Now run the
                # screenshot ground-truth check via vision LLM.
                halt_reason = "complete"
                final_status = "passed"
                final_narration = (
                    reasoning or "Agent marked goal complete"
                )[:500]

                if provider_supports_vision:
                    _emit(emit_event, "agent_verifying", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "goal_description": goal.description[:200],
                    })
                    try:
                        from app.agents.page_intel import (  # noqa: PLC0415
                            verify_goal_via_screenshot,
                        )
                        verdict = verify_goal_via_screenshot(
                            provider, page,
                            goal_description=goal.description,
                            success_criteria=list(goal.success_criteria),
                        )
                    except Exception as e:
                        logger.warning(
                            "goal verification call failed on "
                            "submodule %s: %s", goal.submodule_id, e,
                        )
                        verdict = None

                    if verdict is not None:
                        # Token + call accounting
                        if isinstance(verdict.input_tokens, int):
                            total_input += verdict.input_tokens
                        if isinstance(verdict.output_tokens, int):
                            total_output += verdict.output_tokens
                        llm_calls += 1
                        vision_calls += 1

                        verification_record = {
                            "verdict": verdict.verdict,
                            "reasoning": verdict.reasoning[:500],
                            "confidence": verdict.confidence,
                            "criteria_met": list(verdict.criteria_met),
                            "criteria_missed": list(verdict.criteria_missed),
                            "input_tokens": verdict.input_tokens,
                            "output_tokens": verdict.output_tokens,
                        }

                        _emit(emit_event, "agent_verified", {
                            "run_id": submodule_run_id,
                            "step_id": submodule_step_id,
                            "verdict": verdict.verdict,
                            "reasoning": verdict.reasoning[:300],
                            "confidence": verdict.confidence,
                        })

                        if verdict.verdict == "fail":
                            # Hard contradiction with the screenshot —
                            # downgrade. The agent's claim is wrong.
                            final_status = "inconclusive"
                            final_narration = (
                                "Vision check FAILED — page does not "
                                "show the goal achieved. "
                                f"{verdict.reasoning[:200]}"
                            )
                        elif verdict.verdict == "partial":
                            # Some criteria met, others not. Agent
                            # was over-optimistic; mark inconclusive
                            # so the user reviews.
                            final_status = "inconclusive"
                            final_narration = (
                                f"Vision check PARTIAL — "
                                f"{len(verdict.criteria_met)}/"
                                f"{len(goal.success_criteria) or 1} "
                                f"criteria met. {verdict.reasoning[:160]}"
                            )
                        else:
                            # verdict == "pass": agent's claim confirmed.
                            # Augment the narration so the user knows
                            # the verification ran and agreed.
                            final_narration = (
                                f"Goal confirmed by vision check. "
                                f"{verdict.reasoning[:200]}"
                            )

            stop_record = TurnRecord(
                turn=turn_idx, tool=tool, args=args, reasoning=reasoning,
                confidence=confidence, status="stop",
                narration=final_narration, page_url=observation.get("url", ""),
            )
            if verification_record is not None:
                # Reuse the search_log slot pattern: persist the
                # verification block on the stop record so the
                # report's per-turn surface can render it.
                stop_record.search_log = {
                    "kind": "goal_verification",
                    **verification_record,
                }
            turn_log.append(stop_record)
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

        # ── A4.1b: vision-guided target search on a missed selector ─
        # When a target-bound action just failed because the selector
        # missed, ask the vision LLM for a concrete next step (scroll /
        # navigate / dismiss-modal / click-to-drill), dispatch it, and
        # retry the original action ONCE. Caps at 3 search attempts so
        # a confused LLM can't loop. Skipped when:
        # - tool isn't target-bound (navigate / wait-by-duration / etc.)
        # - provider can't see images
        # - all sub-goals are already done (no point searching for a
        #   target the agent's chasing past completion)
        target_bound = tool in ("click", "type", "select", "verify", "wait")
        miss_due_to_selector = (
            outcome["status"] == "failed"
            and isinstance(outcome.get("error_message"), str)
            and (
                "target not visible" in (outcome.get("narration") or "").lower()
                or "no visible element" in (outcome.get("error_message") or "").lower()
            )
        )
        sub_goals_done = (
            bool(goal.sub_goals)
            and all(
                sg.status in ("done", "skipped") for sg in goal.sub_goals
            )
        )
        if (
            target_bound
            and miss_due_to_selector
            and not sub_goals_done
            and provider_supports_vision
            and args.get("target_hint")
        ):
            search_result = _vision_search_for_target(
                page,
                provider,
                target_hint=str(args.get("target_hint", "")),
                max_attempts=3,
                emit_event=emit_event,
                submodule_run_id=submodule_run_id,
                submodule_step_id=submodule_step_id,
            )
            # Roll vision search tokens into the run-level cost meter.
            if isinstance(search_result.get("input_tokens"), int):
                total_input += search_result["input_tokens"]
            if isinstance(search_result.get("output_tokens"), int):
                total_output += search_result["output_tokens"]
            # Each attempt was its own LLM call; count vision calls
            # for the run summary (used by the live panel).
            attempts_made = int(search_result.get("attempted") or 0)
            vision_calls += attempts_made
            llm_calls += attempts_made

            # When the search succeeded (the original target_hint
            # resolves now), retry the original action ONCE.
            if search_result.get("halted") == "completed":
                retry_outcome = _execute_tool_call(
                    page, tool, args,
                    plan_target_url=plan_target_url,
                    speed_config=speed_config,
                )
                # Annotate the retry with the search trail so the
                # timeline/report show the recovery path.
                retry_outcome["search_log"] = search_result
                retry_outcome["narration"] = (
                    f"{retry_outcome.get('narration') or tool} "
                    f"(after vision-guided search: "
                    f"{attempts_made} attempt"
                    f"{'' if attempts_made == 1 else 's'})"
                )
                outcome = retry_outcome
            else:
                # Search couldn't bring the target in view. Keep the
                # original failed outcome but enrich with the search
                # trail so the user sees what was tried.
                outcome["search_log"] = search_result
                outcome["narration"] = (
                    f"{outcome.get('narration') or tool}"
                    f" — vision search halted: "
                    f"{search_result.get('halted')}"
                )

                # ── LAST-RESORT: pixel-coordinate click ───────────
                # Operator / Computer-Use pattern. When the DOM
                # chain is exhausted (literal → fuzzy → vision-
                # guided search) and the agent still can't reach
                # the target, ask the vision LLM for raw pixel
                # coordinates and click them directly via
                # page.mouse.click(). Bypasses the DOM entirely —
                # the only fallback that works for canvas-rendered
                # widgets, sealed shadow DOM, cross-origin iframes
                # and similar elements that selectors physically
                # can't reach.
                # Only fires for click/type (the actions where
                # coordinate-based dispatch is meaningful).
                if (
                    tool in ("click", "type")
                    and args.get("target_hint")
                ):
                    try:
                        from app.agents.page_intel import (  # noqa: PLC0415
                            propose_click_coordinates,
                        )
                        coords = propose_click_coordinates(
                            provider, page,
                            target_hint=str(args.get("target_hint", "")),
                        )
                    except Exception as e:
                        logger.warning(
                            "coordinate-click LLM call failed: %s", e,
                        )
                        coords = None

                    if coords is not None:
                        if isinstance(coords.input_tokens, int):
                            total_input += coords.input_tokens
                        if isinstance(coords.output_tokens, int):
                            total_output += coords.output_tokens
                        llm_calls += 1
                        vision_calls += 1

                        coord_record: dict[str, Any] = {
                            "x": coords.x,
                            "y": coords.y,
                            "label_visible": coords.label_visible,
                            "reasoning": coords.reasoning[:300],
                            "confidence": coords.confidence,
                            "tool": tool,
                            "applied": False,
                        }

                        _emit(emit_event, "coordinate_click_proposed", {
                            "run_id": submodule_run_id,
                            "step_id": submodule_step_id,
                            "x": coords.x,
                            "y": coords.y,
                            "label_visible": coords.label_visible[:120],
                            "confidence": coords.confidence,
                        })

                        # Confidence gate — below 0.6 the LLM said
                        # it's guessing. Don't click random pixels.
                        if coords.confidence >= 0.6:
                            try:
                                page.mouse.click(coords.x, coords.y)
                                # If this was meant to be a 'type'
                                # action, send the value to the
                                # newly-focused element.
                                if tool == "type" and args.get("value"):
                                    typed_value = str(args.get("value", ""))
                                    delay = (
                                        speed_config.type_delay_ms
                                        if hasattr(speed_config, "type_delay_ms")
                                        else 0
                                    )
                                    if delay > 0:
                                        page.keyboard.type(
                                            typed_value, delay=delay,
                                        )
                                    else:
                                        page.keyboard.type(typed_value)
                                coord_record["applied"] = True
                                outcome = {
                                    "status": "ok",
                                    "narration": (
                                        f"COORDINATE {tool.upper()} at "
                                        f"({coords.x}, {coords.y}) on "
                                        f"{coords.label_visible[:80]!r} "
                                        f"(confidence "
                                        f"{coords.confidence:.2f}). "
                                        f"DOM resolution failed; vision "
                                        f"LLM pointed at pixels."
                                    ),
                                    "error_message": None,
                                    "extracted_text": "",
                                    "search_log": {
                                        **search_result,
                                        "coordinate_click": coord_record,
                                    },
                                }
                            except Exception as e:
                                coord_record["dispatched_error"] = (
                                    f"{type(e).__name__}: {e}"
                                )
                                outcome["search_log"] = {
                                    **search_result,
                                    "coordinate_click": coord_record,
                                }
                        else:
                            # Low-confidence — record but don't act.
                            outcome["search_log"] = {
                                **search_result,
                                "coordinate_click": coord_record,
                            }
                            outcome["narration"] = (
                                f"{outcome.get('narration') or tool}"
                                f" — coord click skipped: confidence "
                                f"{coords.confidence:.2f} < 0.60"
                            )

                        _emit(emit_event, "coordinate_click_completed", {
                            "run_id": submodule_run_id,
                            "step_id": submodule_step_id,
                            "applied": coord_record["applied"],
                            "status": outcome["status"],
                        })

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
            search_log=outcome.get("search_log"),
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

        # Snapshot for next turn's diff. Only update at end of a
        # successful loop iteration so the diff compares against the
        # last "settled" page state, not a transient mid-action one.
        prev_observation = observation

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

                # A4.3: classify why this row landed where it did so
                # the report can recommend test_case_outdated /
                # feature_missing / infra_issue / agent_drift instead
                # of a generic inconclusive.
                divergence = _categorize_divergence(
                    final_status=result.status,
                    halt_reason=result.halt_reason,
                    turn_log=result.turn_log,
                )

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
                    "divergence": divergence,
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
                            "search_log": t.search_log,
                        }
                        for t in result.turn_log
                    ],
                    "llm_calls": result.llm_calls,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                }
                counts[result.status] = counts.get(result.status, 0) + 1
                db.commit()

                # ── Phase E.1: freeze the working path ────────────
                # If this submodule passed the deterministic guards,
                # the soft-guards, AND vision verification (when
                # present) said "pass", serialize the agent's tool
                # sequence onto the submodule TcNode. Replay-mode
                # runs walk this list deterministically. Skip when
                # vision said partial/fail or when verification
                # didn't run — we don't want to canonicalize a path
                # we're not sure actually worked.
                vision_verdict = None
                for t in result.turn_log:
                    sl = t.search_log
                    if (
                        isinstance(sl, dict)
                        and sl.get("kind") == "goal_verification"
                    ):
                        vision_verdict = sl.get("verdict")
                        break
                # Freeze when either (a) vision said pass, OR
                # (b) vision didn't run (no provider vision support)
                # but everything else passed. Don't freeze on
                # partial/fail.
                should_freeze = (
                    result.status == "passed"
                    and vision_verdict in (None, "pass")
                )
                if should_freeze:
                    frozen = _build_frozen_path(
                        run_id=run_id,
                        goal=goal,
                        turn_log=result.turn_log,
                        agent_model=getattr(provider, "model", None),
                    )
                    if frozen and submodule.id is not None:
                        # Reload the submodule node freshly — the
                        # one in scope is still valid since Tc nodes
                        # don't get deleted mid-run, but explicit is
                        # safer.
                        sm_row = db.get(TcNode, submodule.id)
                        if sm_row is not None:
                            sm_row.frozen_path = frozen
                            db.commit()
                            _emit(emit_event, "frozen_path_captured", {
                                "step_id": row.id,
                                "tc_node_id": submodule.id,
                                "step_count": len(frozen["steps"]),
                                "agent_model": frozen.get("agent_model"),
                            })
                            logger.info(
                                "froze path for submodule %s "
                                "(%d steps) from run %s",
                                submodule.id, len(frozen["steps"]),
                                run_id,
                            )

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
                    "divergence_category": divergence["category"],
                    "fuzzy_rescues": divergence["fuzzy_rescues"],
                    "vision_rescues": divergence["vision_rescues"],
                    "frozen": should_freeze,
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
        failed=counts["failed"],
        # Inconclusive is its own bucket — NOT failed. A halted-before-
        # verification goal usually points at a test-case wording issue
        # or a missing precondition, not an app bug. Surfacing it
        # separately is what lets the report recommend "review the test"
        # vs "file a bug".
        inconclusive=counts.get("inconclusive", 0),
        skipped=counts["skipped"],
        blocked=counts["blocked"],
        duration_ms=duration_ms,
        llm_input_tokens=total_input_tokens if total_llm_calls > 0 else None,
        llm_output_tokens=total_output_tokens if total_llm_calls > 0 else None,
        ai_calls=total_llm_calls,
        ai_vision_calls=total_vision_calls,
    )
