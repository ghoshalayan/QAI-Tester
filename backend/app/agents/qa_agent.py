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
from app.agents.execute import (
    ExecutionResult, _capture_screenshot_meta, _take_screenshot,
)
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
    # Phase 0.10 — palette completion. press_key was missing entirely
    # (agent had no way to press Enter / Escape / Tab); type now
    # accepts a `submit` boolean so "type query and search" is one
    # tool call; go_back exposes the browser-history primitive that
    # complex flows (graph-like navigation, return-to-search-results)
    # need without re-navigating to a remembered URL.
    "press_key", "go_back",
    # Phase F — bundled form-fill meta-tool. Enumerates the visible
    # form's fields, classifies each, fills with the appropriate
    # per-widget strategy, observes inline aria-invalid validation
    # errors, retries the offending fields, then clicks submit. The
    # agent invokes this with ``form_fields`` (a string-encoded JSON
    # array of {label, value, required}) + ``form_submit_label``
    # (default "Save"). One turn replaces the 6-10 turns the agent
    # would otherwise spend filling fields one at a time.
    "fill_form",
})
META_TOOLS: frozenset[str] = frozenset({
    "mark_goal_complete", "mark_goal_failed", "ask_human",
    # Phase 11 — test-case dispute. Agent flags a step as provably
    # wrong (selector dead, action impossible, precondition not met).
    # Logged to the report; submodule status flips to ``blocked``
    # with the dispute attached. Frozen path is suppressed for the
    # disputed run. STRICT v1 — agent should ONLY use this when the
    # test step is physically impossible to follow, not just hard.
    "flag_test_case_issue",
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
        # Phase 0.10 — keyboard primitive. ``key`` carries the value
        # for the press_key tool AND can be set on type() to fire an
        # extra key after typing (most commonly Enter, equivalent to
        # `submit: true`). Empty string when not used.
        "key": {
            "type": "string",
            "enum": [
                "", "Enter", "Tab", "Escape",
                "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
                "Backspace", "Delete", "Home", "End",
                "PageUp", "PageDown", "Space",
            ],
        },
        # Phase 0.10 — type-and-submit. When true on a ``type`` call,
        # the executor presses Enter on the same field after typing.
        # Most form fields submit on Enter; folding "type+submit"
        # into one tool call avoids the agent's separate "now find
        # the submit button" dance.
        "submit": {"type": "boolean"},
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
        # Phase B.1 — sub-goal SKIP. The agent marks a sub-goal as
        # NOT-APPLICABLE when the page's current state makes the
        # sub-goal physically impossible (e.g. "click remove on
        # cart item" when cart is empty; "verify post-login dashboard"
        # when login was already done in a prior submodule). Goal
        # completion accepts all-done-OR-skipped sub-goals so a
        # correctly-judged "already satisfied" run isn't trapped
        # in inconclusive. ``skip_sub_goal_reason`` is mandatory
        # when ``skip_sub_goal_id`` is set — surfaces in the report
        # so the user sees WHY the sub-goal was skipped.
        "skip_sub_goal_id": {"type": "string"},
        "skip_sub_goal_reason": {"type": "string"},
        # Phase 11 — test-case dispute. Set ONLY when tool=
        # ``flag_test_case_issue``. Carries the structured payload
        # the agent uses to claim the test step is provably wrong.
        # Empty strings on every other turn.
        "issue_kind": {
            "type": "string",
            "enum": [
                "",
                "wrong_selector",
                "missing_step",
                "impossible_action",
                "misleading_description",
                "precondition_failed",
            ],
        },
        "issue_evidence": {"type": "string"},
        "issue_suggested_fix": {"type": "string"},
        # Phase F — fill_form payload. JSON-encoded array of
        # ``{label, value, required, role_hint?}`` for the routine
        # to fill in order. Empty string when tool != "fill_form".
        # JSON string (not nested object) because OpenAI's strict
        # mode doesn't allow arbitrary array shapes inside one
        # property without bloating the top-level schema.
        "form_fields": {"type": "string"},
        # Submit button label the routine fuzzy-matches. Default
        # "Save"; pass "Create" / "Submit" / "Confirm" / "" (no
        # submit, just fill).
        "form_submit_label": {"type": "string"},
    },
    "required": [
        "tool", "target_hint", "value", "url", "expected",
        "duration_ms", "scroll_direction", "scroll_amount",
        "question", "reasoning", "confidence",
        "page_memory_note",
        "current_sub_goal_id", "sub_goal_completed_id",
        "key", "submit",
        "skip_sub_goal_id", "skip_sub_goal_reason",
        "issue_kind", "issue_evidence", "issue_suggested_fix",
        "form_fields", "form_submit_label",
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
- type(target_hint, value, submit?): type into a field. Set submit=true
  when typing into a SEARCH FIELD or SINGLE-INPUT FORM that should
  submit on Enter — this types AND presses Enter in one go. Cheaper
  and more reliable than typing then hunting for a submit button
  (Amazon, Google, most search bars have unlabelled submit icons).
  Leave submit=false when the form has explicit submit/save buttons
  (checkout forms, multi-field registration, etc.).
- press_key(key): press one key — Enter, Tab, Escape, Arrow*, Backspace,
  Delete, Home, End, PageUp, PageDown, Space. Whichever element has
  focus receives it. Use this when the previous action left a field
  focused and you need to dispatch a key without a separate click.
- select(target_hint, value): pick a dropdown option
- verify(target_hint?, expected?): assert visibility or text presence.
  ``expected`` should be a SHORT, CONCRETE token you'd literally find
  on the page (e.g. "Cart", "Sign out", "Order placed", "₹"). Do NOT
  pass a long sentence describing the goal — the literal substring
  check skips natural-language descriptions and the call gets routed
  to semantic verify. Use ``mark_goal_complete`` for goal-level
  semantic claims; use ``verify`` for spot-checks on visible text.
- wait(duration_ms? OR target_hint?): wait for time or element
- scroll(scroll_direction, scroll_amount?): scroll the page
- extract_text(target_hint): read text out of an element (returned in
  the next observation as "extracted_text")
- dismiss_modal(): close any blocking modal (cookie banner, signup
  popup, etc.). Use this BEFORE retrying a click that fails because
  something is overlaying the page.
- go_back(): browser back. Use for cascade flows like
  "search → product → back to results → another product" — cheaper
  than re-navigating to a remembered URL on heavy SPAs.
- fill_form(form_fields, form_submit_label): bundled multi-field fill.
  USE THIS instead of individual type/select/click calls whenever
  a form has more than 2 fields to set. The routine enumerates the
  open form/drawer/modal, classifies each input (textbox, textarea,
  native_select, custom_combobox, checkbox, radio, date, file),
  fills each with the right strategy, watches for inline
  aria-invalid validation errors, retries the offending fields
  (up to 2x with the same value), then clicks the submit button.
  ``form_fields`` is a JSON STRING of an array:
    [{"label": "First Name", "value": "Alice", "required": true},
     {"label": "Email Address", "value": "alice@acme.com", "required": true},
     {"label": "Phone Number", "value": "9999999999"},
     {"label": "Send notifications", "value": "true",
      "role_hint": "checkbox"},
     {"label": "Role", "value": "QA Tester Role",
      "role_hint": "custom_combobox"}]
  ``form_submit_label`` is the submit button's visible text
  ("Save" / "Create" / "Submit"); pass "" to fill only.
  Labels are fuzzy-matched against placeholders / aria-label / the
  wrapping <label>. Don't pass non-existent fields — they'll be
  marked as miss in the result and surfaced as failures.

  COMPOUND WIDGET role_hints (use these when the App Map flagged the
  flow with [permission tree] or [paginated resource table]):

  - role_hint="permission_tree": Treat the whole tree as ONE field.
    ``value`` is one of: "all" / "none" / "only:A,B,C" /
    "all_except:X,Y". Example for "grant all permissions":
       {"label": "Permissions", "value": "all",
        "role_hint": "permission_tree"}
    The routine clicks Expand All if present, enumerates every leaf
    checkbox, and toggles only the ones whose state differs from
    desired. Do NOT emit one form_fields entry per leaf — the routine
    handles atomicity.

  CRITICAL — DO NOT click() permission-tree checkboxes one-by-one.
  When you see a tree-shaped permission control (parent rows with
  expand chevrons + child checkboxes), the ONLY correct tool is
  fill_form with role_hint="permission_tree". Individual click()
  calls on each checkbox will:
    (a) burn 5-15 turns of LLM cost ticking checkboxes manually,
    (b) usually run out of turns before reaching Save,
    (c) leave most permissions unchecked, so Save fails validation
        ("at least one permission required") and the role is never
        created.
  If you're tempted to click a permission checkbox, STOP and emit
  a single fill_form turn with the whole tree as one field instead.

  HARD RULE — fill_form MANDATE for create-flow drawers.
  When the CURRENT SUB-GOAL description starts with "fill_form:"
  OR mentions "fill the <X> drawer / form" OR you see an open
  drawer with 3+ visible form fields on the page:
  → The ONLY correct tool is fill_form.
  → Bundle every visible required field into a SINGLE form_fields
    array in ONE turn.
  → Include permission_tree / paginated_resource_table fields in
    the SAME fill_form call (as additional entries with their
    role_hint set), not as separate sub-goals.
  → Set form_submit_label to the visible Save / Create / Submit
    button text so the routine clicks it after filling.
  Sequential click() + type() + click() across multiple turns
  costs 6-12 turns minimum, frequently runs out before Save, and
  leaves the record un-persisted. ONE fill_form turn does it
  atomically with validation-error retry.

  ENTITY REUSE — read KNOWN STATE before inventing values.
  Before you fill any combobox / dropdown / role-picker field,
  scan the KNOWN STATE block above for "recently created entities"
  of a matching kind. If KNOWN STATE shows
  ``role='QA Auto Role-616023'`` and the form has a Role field,
  the fill_form value for Role is EXACTLY ``'QA Auto Role-616023'``
  — not a made-up name, not "the previously created role". Use
  the verbatim identity string from KNOWN STATE.

  - role_hint="paginated_resource_table": Treat the whole table as
    ONE field. ``value`` is one of:
      "all:read,update"      → tick column-master checkboxes (or
                                walk rows if no masters) for every
                                row across every page.
      "specific:CH-0001:read,update;CH-0002:read"
                              → tick only the named rows; pagination
                                walked automatically.
      "none"                  → untick everything currently visible.
    Example for "grant read access on every chainage":
       {"label": "Resource Access Control",
        "value": "all:read",
        "role_hint": "paginated_resource_table"}

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
- flag_test_case_issue(issue_kind, issue_evidence, issue_suggested_fix):
  flag a test step as PROVABLY WRONG. Submodule is marked
  ``blocked`` (not failed — failure means the APP is broken; this
  means the TEST is broken). STRICT criteria — only use when:
    * issue_kind="wrong_selector" — the target_hint is dead and
      the page has a clearly equivalent element with a different
      label/selector. ``issue_suggested_fix`` should name it.
    * issue_kind="missing_step" — the test skipped a required
      page (e.g. variant-selector dialog before "Add to cart").
    * issue_kind="impossible_action" — the action can't physically
      happen on this page (clicking "Proceed to checkout" on an
      empty cart, "remove item" with no items, etc.).
    * issue_kind="misleading_description" — the test's narrative
      describes a different page or flow than the actual app.
    * issue_kind="precondition_failed" — a prior submodule was
      supposed to set up state this submodule needs (cart having
      items; user logged in; etc.) and that state is absent.
  ALWAYS provide ``issue_evidence`` (1-2 verbatim phrases from
  the page that prove the dispute). DO NOT use this tool just
  because something is hard to find — that's what the search /
  fuzzy / vision tools are for. This tool is for "the test case
  itself is wrong, not just a poor selector hint".

RULES:
- KNOWN ABOUT THIS APP: when present, this block contains BRD
  excerpts, scout-walk notes, pattern rules, and prior dispute
  outcomes the system has learned. Read it FIRST — it tells you
  what's expected, where things live, and known gotchas. Higher-
  confidence chunks are more reliable.
- GOAL CONTRACT: when present, this block carries:
    * Preconditions — state that must hold BEFORE you act. If a
      precondition is marked NOT MET, the test case's assumed
      starting state is wrong: call ``flag_test_case_issue`` with
      ``issue_kind=precondition_failed`` and explain. Do NOT try
      to set up the precondition yourself; that's a different
      submodule's job.
    * Postconditions — state that must hold AFTER for the goal
      to count as passed. The agent loop applies these to
      WorldState on success.
    * Evidence signals — observable cues that prove the post-
      conditions. Treat them as a checklist; you don't need to
      verify all of them, just enough to be CONFIDENT (≥ majority).
      Use ``verify`` with SHORT concrete tokens for each signal.
    * Alternative paths — when the obvious flow is blocked, try
      these.
- WORLD STATE: the run's plan-scoped memory. Don't restate it; use
  it to skip work the previous submodule already did (e.g., don't
  re-login when ``logged_in_as`` is set).
- target_hint: a Playwright-resolvable hint. CSS selector, "text 'Sign In'",
  or "role=button[name='Sign In']". Prefer stable hints (data-testid,
  text, role) over fragile ones (nth-child).
- SEARCH/SINGLE-FIELD FORMS: prefer ``type(..., submit=true)`` over
  ``type`` followed by ``click("Search button")``. The submit button
  on most search bars has no readable text (just a magnifying glass
  icon) and the click hunt usually fails. Pressing Enter is what a
  human actually does there.
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
- ``skip_sub_goal_id`` + ``skip_sub_goal_reason``: when a sub-goal
  is PHYSICALLY IMPOSSIBLE on the current page state, skip it.
  The completion gate accepts skipped sub-goals as closed.
  When to skip:
    * Sub-goal "click remove on cart item" but cart is already
      empty → skip with reason "cart is already empty; nothing
      to remove".
    * Sub-goal "verify post-login dashboard" but login was
      already done by an earlier submodule → skip with reason
      "already logged in; dashboard verified earlier".
    * Sub-goal "select variant" but the product has only one
      variant → skip with reason "no variant selector — single
      SKU product".
  ALWAYS provide ``skip_sub_goal_reason``; the user reads it in
  the report. Do NOT skip a sub-goal just because it's hard or
  the test case wording is ambiguous — that's not "impossible",
  that's "the agent gave up", which is a different failure mode.
- Call ``mark_goal_complete`` when ALL sub-goals are either
  ``done`` or ``skipped`` AND the goal's success criteria are
  observably met.

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
    # Set True for any turn whose successful execution depended on a
    # human typing a value (credentials, OTP, captcha, passkey
    # resume) into the live HITL popup. The freeze gate skips runs
    # that contain ANY such turn — a path that only passed because a
    # human filled in a one-time secret is NOT a deterministic
    # replay candidate. See ``_build_frozen_path`` callers.
    manual_intervention_used: bool = False


HaltReason = Literal[
    "complete", "agent_failed", "ask_human", "stall", "oscillation",
    "max_turns", "max_wallclock", "budget", "cancelled",
    # Phase 11 — agent flagged a test step as provably wrong via
    # ``flag_test_case_issue``. Submodule status flips to ``blocked``
    # with the dispute attached. Frozen path is suppressed.
    "test_case_disputed",
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
            # Phase 0.10 — keyboard / history primitives are
            # deterministic and replayable. press_key fires the same
            # key against whatever has focus (which is determined by
            # prior steps in the path); go_back walks history one
            # entry. type-with-submit is captured by carrying the
            # ``submit`` arg through ``slim_args`` below — replay
            # dispatches both the type and the Enter press.
            "press_key", "go_back",
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


def _build_frozen_path_segments(
    *,
    run_id: int,
    goal: Goal,
    turn_log: list[TurnRecord],
    runtime_sub_goals: list[Any],
    agent_model: str | None,
) -> dict[str, Any] | None:
    """Phase B Step 1/2 — build per-sub-goal frozen segments.

    Same idea as :func:`_build_frozen_path` but the output groups
    successful steps by which sub-goal each turn was working on.
    Replay (Phase B Step 4) walks segment-by-segment: for any
    sub-goal that has a segment with status="done", replay walks
    the steps deterministically; sub-goals without a segment fall
    through to the agentic loop with the goal narrowed to that
    sub-goal.

    Output shape::

        {
          "version": 2,
          "frozen_at_run_id": ...,
          "frozen_at": ...,
          "agent_model": ...,
          "goal_description": ...,
          "success_criteria": [...],
          "segments": [
            {
              "sub_goal_id": "sg1",
              "description": "...",
              "success_criterion": "...",
              "max_turns": 6,
              "steps": [...same shape as v1 steps...],
              "status": "done",  # only "done" sub-goals are frozen
            },
            ...
          ],
        }

    Returns None when no sub-goal has any frozen-worthy steps.
    Sub-goals whose status was anything OTHER than "done" are
    deliberately omitted — only proven flows get frozen.

    Important: the FIRST non-done sub-goal in the list breaks the
    chain. We DO emit later done sub-goals (so the report shows
    the whole timeline), but replay treats them as "partial" and
    falls through to agentic the moment it hits a gap. Sub-goals
    from a replan (id contains "r1", "r2") are SKIPPED entirely —
    those flows came from a hot-path correction and aren't a
    deterministic replay candidate.
    """
    if not runtime_sub_goals:
        return None

    # Group turns by their declared current_sub_goal_id. Turns
    # without one fall into the LAST in-progress sub-goal.
    by_sg: dict[str, list[TurnRecord]] = {}
    cur_sg_id: str | None = None
    for t in turn_log:
        declared = (t.args or {}).get("current_sub_goal_id") or ""
        if declared and isinstance(declared, str):
            cur_sg_id = declared
        if cur_sg_id is None:
            continue
        if t.status != "ok":
            continue
        by_sg.setdefault(cur_sg_id, []).append(t)

    segments: list[dict[str, Any]] = []
    for rsg in runtime_sub_goals:
        # Skip replanned sub-goals — id pattern "...r1", "...r2"
        # marks the sub-goal as having been re-decomposed mid-run,
        # which is incompatible with deterministic replay.
        if "r" in rsg.id and rsg.id.split("r")[-1].isdigit():
            continue
        if rsg.status != "done":
            continue
        sg_turns = by_sg.get(rsg.id, [])
        if not sg_turns:
            continue
        seg_steps: list[dict[str, Any]] = []
        for t in sg_turns:
            if t.tool not in (
                "navigate", "click", "type", "select", "verify",
                "wait", "scroll", "extract_text", "dismiss_modal",
                "press_key", "go_back",
            ):
                continue
            slim_args = {
                k: v for k, v in (t.args or {}).items()
                if v not in ("", 0, None, False)
                and k not in (
                    # Drop sub-goal-tracking fields from frozen
                    # args — they're meta, not replay primitives.
                    "current_sub_goal_id", "sub_goal_completed_id",
                    "skip_sub_goal_id", "skip_sub_goal_reason",
                    "issue_kind", "issue_evidence",
                    "issue_suggested_fix", "page_memory_note",
                    "reasoning", "confidence",
                )
            }
            successful_selector: str | None = None
            if "fuzzy matched" in (t.narration or "").lower():
                import re as _re  # noqa: PLC0415
                m = _re.search(
                    r"fuzzy matched ['\"]([^'\"]+)['\"]",
                    t.narration or "",
                )
                if m:
                    successful_selector = m.group(1)
            seg_steps.append({
                "turn": t.turn,
                "tool": t.tool,
                "args": slim_args,
                "successful_selector": successful_selector,
                "page_url_after": t.page_url,
            })
        if not seg_steps:
            continue
        segments.append({
            "sub_goal_id": rsg.id,
            "description": rsg.description,
            "success_criterion": rsg.success_criterion,
            "max_turns": rsg.max_turns,
            "steps": seg_steps,
            "status": "done",
        })

    if not segments:
        return None

    return {
        "version": 2,
        "frozen_at_run_id": run_id,
        "frozen_at": _utcnow().isoformat(),
        "agent_model": agent_model,
        "goal_description": goal.description,
        "success_criteria": list(goal.success_criteria),
        "segments": segments,
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
    # True when the auth-flow pre-run rescued a credentials / OTP /
    # captcha / passkey screen using a human-typed value. The freeze
    # gate must skip such runs even when ``turn_log`` itself contains
    # no flagged turn — the secret was typed BEFORE any agent turn ran.
    manual_intervention_used: bool = False
    # Phase A — VL-derived sub-goals (RuntimeSubGoal.to_dict()) folded
    # into ``details_json["sub_goals"]`` so the report renders a
    # per-sub-goal pass/fail/skip timeline under each submodule row.
    # Empty when decomposition was disabled or failed (legacy behavior).
    sub_goals: list[dict[str, Any]] = field(default_factory=list)
    # Phase A — number of replan iterations the submodule used. 0 =
    # first decomposition stuck through to the end (good); higher =
    # the agent had to re-decompose after sub-goal failures. Capped
    # by ``plan.max_replans_per_submodule``.
    replans_used: int = 0


# ── Helpers ────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Phase C.3 — TcVersion → TcNode materialization ────────────────


def _materialize_tcnodes_from_version(
    db: "Session",
    *,
    plan_id: int,
    version_id: int,
) -> list[TcNode]:
    """Build transient TcNode objects from a TcVersion's snapshot
    tree so the rest of the agent (which walks ``TcNode.parent_id``)
    keeps working without changes.

    Rules:
    - ``.id`` is set to ``original_tc_node_id`` when one exists (so
      freeze paths + AKB recall hit the live row). For ``added``
      snapshots with no live counterpart, ``.id`` is a synthetic
      negative integer derived from the snapshot id so collisions
      with real PKs are impossible.
    - ``.parent_id`` is rewired to the parent's ``.id`` (which may
      itself be original or synthetic). This keeps tree-walk code
      intact.
    - These objects are NOT added to the session and NOT persisted.
      Mutating ``.frozen_path`` on a synthetic node is a no-op.
    """
    from app.models.tc_version import TcNodeSnapshot  # noqa: PLC0415
    from sqlalchemy import select as _select  # noqa: PLC0415

    snaps = list(db.execute(
        _select(TcNodeSnapshot)
        .where(TcNodeSnapshot.tc_version_id == version_id)
        .order_by(
            TcNodeSnapshot.depth,
            TcNodeSnapshot.parent_snapshot_id,
            TcNodeSnapshot.ordinal,
        ),
    ).scalars())

    snap_id_to_node_id: dict[int, int] = {}
    out: list[TcNode] = []
    for s in snaps:
        # Choose a stable id for this snapshot's TcNode shadow.
        if s.original_tc_node_id is not None:
            node_id = int(s.original_tc_node_id)
        else:
            node_id = -int(s.id)  # synthetic, won't collide
        snap_id_to_node_id[int(s.id)] = node_id

        parent_node_id: int | None = None
        if s.parent_snapshot_id is not None:
            parent_node_id = snap_id_to_node_id.get(
                int(s.parent_snapshot_id),
            )

        node = TcNode(
            id=node_id,
            project_id=0,  # unused at this layer
            plan_id=plan_id,
            parent_id=parent_node_id,
            kind=s.kind,
            ordinal=s.ordinal,
            depth=s.depth,
            path_cached=s.path_cached,
            title=s.title,
            description_md=s.description_md,
            action_type=s.action_type,
            target_hint=s.target_hint,
            narrative=s.narrative,
            expected=s.expected,
            data_needs_json=s.data_needs_json,
            selectable_default=s.selectable_default,
            status="draft",
            source_requirement_ids=[],
        )
        # ``frozen_path`` lives on the LIVE TcNode keyed by original
        # id — load it lazily so replay / freeze still finds it for
        # rows that have a live counterpart.
        if s.original_tc_node_id is not None:
            live = db.get(TcNode, int(s.original_tc_node_id))
            if live is not None and live.frozen_path is not None:
                node.frozen_path = live.frozen_path
        out.append(node)
    return out


# ── Phase A.6 Step 4 — verify gate helper ─────────────────────────


# Field names that mark a value as the entity's "identity" (the name
# that should appear in the list after creation). When the agent
# typed into one of these earlier in the submodule, the verify gate
# uses that value as the needle for the "is the new row visible?"
# check. Matched case-insensitively against the resolved target_hint.
_IDENTITY_FIELD_HINTS: tuple[str, ...] = (
    "name", "title", "label", "display", "first name",
    "email", "username", "project", "role", "user", "id",
)


def _last_typed_entity_value(
    turn_log: list["TurnRecord"],
) -> str:
    """Scan back through the submodule's turn log for the most recent
    ``type`` action whose ``target_hint`` looks identity-like.

    Returns the typed value, or ``""`` when nothing identity-shaped
    was typed yet. Used by the verify-in-list gate (Step 4) to know
    which name the agent SHOULD see in the list after creation.

    Matches LATEST first so multi-field forms work: agent types
    First Name, Last Name, Email — verify uses Email (last
    identity-like field typed) which is typically the unique key.
    """
    for t in reversed(turn_log):
        if t.tool != "type":
            continue
        value = (t.args.get("value") or "").strip()
        if not value or len(value) < 2:
            continue
        hint = (t.args.get("target_hint") or "").lower()
        if any(h in hint for h in _IDENTITY_FIELD_HINTS):
            return value
    # Fallback: the most recent typed value of >= 3 chars regardless
    # of target. Many forms have only one obvious text input ("role
    # name", "tag name") that doesn't match the heuristic list.
    for t in reversed(turn_log):
        if t.tool != "type":
            continue
        value = (t.args.get("value") or "").strip()
        if len(value) >= 3:
            return value
    return ""


# Cheap heuristics that catch the common login/auth screen patterns
# without paying for a VL call. Used as the gate before invoking
# ``auth_flow.run_auth_loop`` — once inside, the loop's own VL
# classifier is the source of truth, so a false-positive here just
# costs one extra detect_auth_fields call (auth_loop returns quickly
# with kind="success"/"unknown" + low confidence and we fall through).
_LOGIN_URL_HINTS: tuple[str, ...] = (
    "login", "signin", "sign-in", "sign_in",
    "auth", "authenticate", "session",
    "logon", "log-on", "log_on", "log-in", "log_in",
    "/oauth", "/sso", "/saml",
)


def _looks_like_login_page(page: Any) -> bool:
    """Cheap pre-check: does this page LOOK like a login/auth screen?

    True when EITHER:
    - URL contains a login-ish substring (``/login``, ``/signin``,
      ``/oauth``, ``/sso``, etc.), OR
    - The DOM exposes at least one ``input[type=password]`` —
      strongest single signal, works across SPA shells / sealed
      shadow DOM where the URL is generic.

    Failures (page closed, eval throws) return False — auth flow
    is skipped and the agent's main loop takes over.
    """
    try:
        url = (page.url or "").lower()
    except Exception:
        url = ""
    if any(h in url for h in _LOGIN_URL_HINTS):
        return True
    try:
        has_password = page.evaluate(
            "() => !!document.querySelector('input[type=\"password\"]')",
        )
        if bool(has_password):
            return True
    except Exception:
        pass
    return False


def _json_safe(value: Any) -> Any:
    """Recursively replace JSON-incompatible values (bytes, sets, etc.)
    in a dict/list tree with serializable placeholders.

    Why this exists
    ---------------
    ``details_json`` is a SQLAlchemy ``JSON`` column — anything we put
    in it gets serialized via ``json.dumps`` on commit. A single rogue
    ``bytes`` value (e.g. a screenshot leaking through ``search_log``)
    raises ``TypeError`` mid-flush and rolls back the entire run's
    transaction, losing other completed steps' updates too.

    This sanitizer is the seatbelt: walk the tree right before commit
    and swap any ``bytes`` for ``"<bytes:N>"``, sets/tuples for lists.
    The structure stays intact for the report; only the offending
    value is replaced. Cheap defence against future regressions
    where someone adds a new field carrying binary data without
    realizing it'll be persisted.
    """
    if isinstance(value, bytes | bytearray):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    return value


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
    # Skip sidecar keys like ``_graph`` — they're tracking metadata,
    # not page memory entries.
    page_entries = {
        k: v for k, v in memory.items()
        if not k.startswith("_") and isinstance(v, dict)
        and "note" in v
    }
    if not page_entries:
        return ""
    items = sorted(
        page_entries.items(),
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


def _format_page_graph_for_prompt(
    page_graph: dict[str, Any],
    current_url: str,
    *,
    max_edges: int = 10,
) -> str:
    """Phase 12 — render the navigation graph as a compact prompt
    block. Lets the agent reason about how to back-navigate ("I was
    on /search-results, then clicked into /product/X — to go back
    I can use go_back or navigate to the recorded URL").

    Returns "" when the graph is empty so the caller can skip the
    block. Edges are de-duplicated and capped — the LLM doesn't
    need a full history, just the recent path.
    """
    if not page_graph:
        return ""
    edges = page_graph.get("edges") or []
    if not edges:
        return ""
    # Show last N edges. De-dup adjacent identical (from, to)
    # transitions — they bloat without adding info.
    recent: list[dict[str, Any]] = []
    seen_pair: tuple[str, str] | None = None
    for edge in edges[-max_edges * 2:]:
        pair = (edge.get("from", ""), edge.get("to", ""))
        if pair == seen_pair:
            continue
        recent.append(edge)
        seen_pair = pair
        if len(recent) >= max_edges:
            break
    if not recent:
        return ""
    try:
        from urllib.parse import urlparse  # noqa: PLC0415
    except Exception:
        urlparse = None  # type: ignore[assignment]

    def _path(url: str) -> str:
        if urlparse:
            try:
                p = urlparse(url).path or url
                return p[:80]
            except Exception:
                pass
        return url[:80]

    lines: list[str] = []
    for edge in recent:
        from_p = _path(str(edge.get("from", "")))
        to_p = _path(str(edge.get("to", "")))
        tool = edge.get("tool") or "?"
        marker = "  ← YOU ARE HERE" if edge.get("to") == current_url else ""
        lines.append(
            f"- T{edge.get('turn', '?')}: {from_p} -> {to_p} (via {tool}){marker}",
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


def _format_failed_approaches_block(
    failed_approaches: list[dict[str, str]],
) -> str:
    """Phase O.3 — render the recent failed-approach memory.

    When the agent's prior attempts in THIS submodule have produced
    actionable failures (click on a target that didn't resolve, type
    into a field that wasn't accepted, fill_form with a validation
    error), surface them so the planner explicitly does NOT retry the
    same approach. Empty list → empty string so the prompt skips it.
    """
    if not failed_approaches:
        return ""
    lines: list[str] = [
        "\nFAILED APPROACHES SO FAR (do NOT retry these; pick a "
        "different target / value / tool):",
    ]
    for fa in failed_approaches[-6:]:
        bits: list[str] = [
            f"  - T{fa.get('turn', '?')}",
            f"{fa.get('tool', '?')}",
        ]
        tgt = (fa.get('target') or '').strip()
        if tgt:
            bits.append(f"target={tgt!r}")
        val = (fa.get('value') or '').strip()
        if val:
            bits.append(f"value={val!r}")
        reason = (fa.get('reason') or '').strip()
        line = " ".join(bits)
        if reason:
            line += f" → {reason[:120]}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


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
) -> tuple[str, str, str | None, str, dict[str, Any]]:
    """Extract text — returns (status, narration, error, extracted_text, details).

    The ``details`` slot carries ``failure_kind`` for typed dispatch on
    the failure side: orchestrators downstream branch on
    ``selector_not_found`` to decide whether to run vision search /
    coord-click rescues without string-matching the narration.
    """
    target = args.get("target_hint") or ""
    if not target:
        return "failed", "extract_text: target_hint required", None, "", {}
    try:
        resolved = resolve(page, target)
    except SelectorNotFound as e:
        return (
            "failed",
            f"extract_text: target not visible {target!r}",
            str(e),
            "",
            {"target_hint": target, "failure_kind": "selector_not_found"},
        )
    try:
        text = resolved.locator.inner_text(timeout=5000)
    except Exception as e:
        return (
            "failed",
            "extract_text: could not read text",
            f"{type(e).__name__}: {e}",
            "",
            {},
        )
    return "ok", f"extracted from {target!r}: {text[:120]!r}", None, text[:1000], {}


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


_VALID_KEYS: frozenset[str] = frozenset({
    "Enter", "Tab", "Escape",
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "Backspace", "Delete", "Home", "End",
    "PageUp", "PageDown", "Space",
})


def _evaluate_evidence_signals(
    page,
    signals: list[str],
) -> tuple[int, int, list[dict[str, Any]]]:
    """α.7 — multi-evidence signal voting for verify.

    Tries each signal in two passes:
    1. Treat as a Playwright-resolvable target (selectors, text
       markers, role lookups) — call the existing resolver.
    2. Fall back to substring match against ``body.inner_text()``.

    Returns ``(matched, total, per_signal_traces)``. The orchestrator
    converts this into a verify outcome: pass when matched ≥
    majority (ceil(total/2)); inconclusive when 1 ≤ matched <
    majority; fail when 0 match.

    Single-signal legacy verifies skip this helper entirely — the
    classic literal substring check still runs in actions.py.
    """
    traces: list[dict[str, Any]] = []
    matched = 0
    if not signals:
        return 0, 0, traces

    body_text_cache: str | None = None

    def _body_text() -> str:
        nonlocal body_text_cache
        if body_text_cache is None:
            try:
                body_text_cache = page.locator("body").inner_text(
                    timeout=4_000,
                )
            except Exception:
                body_text_cache = ""
        return body_text_cache

    for sig in signals:
        s = (sig or "").strip()
        if not s:
            continue
        match_via = ""
        ok = False
        # Try as a Playwright-resolvable hint first.
        try:
            from app.executor.selectors import resolve  # noqa: PLC0415

            try:
                resolve(page, s, timeout_ms=1_500)
                ok = True
                match_via = "resolver"
            except Exception:
                ok = False
        except Exception:
            pass
        # Fall back to substring match against body text.
        if not ok:
            txt = _body_text().lower()
            # Strip quotes the LLM may have added around signal phrases.
            stripped = s.strip("\"' ")
            # Treat as natural-language signal: look for key tokens.
            # First try the literal phrase, then individual words >3 chars.
            if stripped and stripped.lower() in txt:
                ok = True
                match_via = "substring"
            else:
                tokens = [
                    t for t in stripped.lower().split()
                    if len(t) > 3
                ]
                if tokens and all(t in txt for t in tokens):
                    ok = True
                    match_via = "tokens"
        if ok:
            matched += 1
        traces.append({
            "signal": s[:120],
            "matched": ok,
            "via": match_via or "none",
        })
    return matched, len([s for s in signals if s and s.strip()]), traces


def _do_press_key(
    page, args: dict[str, Any],
) -> tuple[str, str, str | None]:
    """Phase 0.10 — keyboard primitive.

    Submits a single keystroke against whatever element currently has
    focus. Most useful for ``Enter`` to submit a form after typing
    (Amazon's search button is hard to locate by selector); ``Escape``
    to bail a stuck modal that ``dismiss_modal`` couldn't close;
    ``Tab`` to advance focus into a hidden field; arrow keys to step
    through a date picker / autocomplete.

    The caller is expected to have given the field focus already (via
    a click or via the typed-into field still being active). We do
    NOT take focus here — that's the agent's job in the prior turn.

    Returns ``(status, narration, error)`` matching the other ``_do_*``
    helpers' shape.
    """
    key = (args.get("key") or "").strip()
    if not key:
        return (
            "failed",
            "press_key: 'key' arg is required",
            "no key specified",
        )
    if key not in _VALID_KEYS:
        return (
            "failed",
            f"press_key: unsupported key {key!r}",
            f"key must be one of {sorted(_VALID_KEYS)}",
        )
    try:
        page.keyboard.press(key)
    except Exception as e:
        return (
            "failed",
            f"press_key {key!r} dispatch failed",
            f"{type(e).__name__}: {e}",
        )
    return "ok", f"pressed {key!r}", None


def _do_go_back(page) -> tuple[str, str, str | None]:
    """Phase 0.10 — browser back primitive.

    Wraps ``page.go_back()`` so cascade flows (e.g. "search → product
    → back to results → another product") don't have to round-trip
    through a remembered URL with ``navigate``. Materially cheaper
    on heavy SPAs that re-execute their full bootstrap on a fresh
    navigate but keep the prior page hot in history.

    Waits for the post-back navigation to settle so the next turn's
    observation reflects the restored page, not a transient.
    """
    try:
        response = page.go_back(wait_until="domcontentloaded", timeout=10_000)
    except Exception as e:
        return (
            "failed",
            "go_back: browser refused (no history?)",
            f"{type(e).__name__}: {e}",
        )
    if response is None:
        # Playwright returns None when there's no entry to go back to.
        return (
            "failed",
            "go_back: no previous page in history",
            "history empty",
        )
    return "ok", f"navigated back to {page.url}", None


def _persist_module_bundle(
    db: "Session",
    pairs: list[tuple[Any, dict[str, Any]]],
    *,
    plan_target_url: str,
    run_id: int,
    agent_model: str | None,
) -> None:
    """γ.2 — write a cross-submodule frozen-flow bundle.

    Walks the (submodule, per-submodule-frozen-path) pairs and groups
    them by parent module (TcNode.parent_id). For each module that has
    ALL of its child submodules passing in this contiguous streak, we
    write a single ``frozen_path`` to the MODULE node containing the
    concatenated step list. Replay's plan-level dispatcher prefers the
    module bundle when present (one continuous walk vs. submodule-by-
    submodule re-orchestration).

    Submodules whose parent isn't fully covered fall through to the
    existing per-submodule ``frozen_path`` — the bundle is additive,
    never destructive.
    """
    if not pairs:
        return
    from app.models.tc_node import TcNode  # noqa: PLC0415

    # Group by parent_id — same module = same bundle.
    by_parent: dict[int, list[tuple[Any, dict[str, Any]]]] = {}
    for sm, frozen in pairs:
        parent_id = getattr(sm, "parent_id", None)
        if parent_id is None:
            continue
        by_parent.setdefault(parent_id, []).append((sm, frozen))

    for parent_id, group in by_parent.items():
        parent = db.get(TcNode, parent_id)
        if parent is None:
            continue
        # Did THIS run cover every direct child submodule of the
        # module? If not, don't claim the module is fully frozen —
        # leave per-submodule freezes in place.
        children = list(parent.children or [])
        sub_children = [
            c for c in children
            if (c.kind or "").lower() == "submodule"
        ]
        covered_ids = {sm.id for sm, _ in group}
        if not sub_children or not all(
            c.id in covered_ids for c in sub_children
        ):
            logger.debug(
                "module bundle skipped for parent %s: %d/%d covered",
                parent_id, len(covered_ids), len(sub_children),
            )
            continue

        # Concatenate steps in submodule-ordinal order.
        ordered = sorted(
            group,
            key=lambda t: getattr(t[0], "ordinal", 0),
        )
        bundle_steps: list[dict[str, Any]] = []
        for sm, frozen in ordered:
            steps = (frozen or {}).get("steps") or []
            for s in steps:
                if isinstance(s, dict):
                    s2 = dict(s)
                    s2["_from_submodule_id"] = sm.id
                    bundle_steps.append(s2)
        if not bundle_steps:
            continue
        bundle = {
            "kind": "module_bundle",
            "steps": bundle_steps,
            "submodule_ids": [sm.id for sm, _ in ordered],
            "agent_model": agent_model,
            "source_run_id": run_id,
            "target_url": plan_target_url,
        }
        try:
            parent.frozen_path = bundle
            db.commit()
            logger.info(
                "Module bundle frozen on parent %s (%d steps from "
                "%d submodules) from run %s",
                parent_id, len(bundle_steps),
                len(ordered), run_id,
            )
        except Exception as e:
            logger.warning(
                "module bundle persist failed for parent %s: %s",
                parent_id, e,
            )
            try:
                db.rollback()
            except Exception:
                pass


def _vision_only_dispatch(
    *,
    page,
    tool: str,
    args: dict[str, Any],
    provider: LLMProvider,
    cheap_provider: LLMProvider | None,
    emit_event: Callable[[str, dict], None] | None,
    on_escalate: Any,
    submodule_run_id: int | None,
    submodule_step_id: int | None,
) -> dict[str, Any] | None:
    """Phase 6 — vision-only action dispatch.

    Bypasses the DOM resolver entirely. The vision LLM looks at the
    full-resolution screenshot and returns pixel coordinates; we
    dispatch via ``page.mouse.click(x, y)`` (and for ``type``,
    follow with ``page.keyboard.type(value)`` plus an Enter when
    the agent set ``submit: true``).

    Returns a dict with ``outcome`` (the standard outcome shape
    callers expect) plus token counts. Returns ``None`` when the
    LLM call failed or returned low confidence — caller falls
    through to the regular DOM dispatch as a safety net (vision-
    only doesn't mean "no fallback ever"; it means "VL coords first,
    DOM only when the agent's vision call genuinely couldn't
    decide").
    """
    target_hint = str(args.get("target_hint") or "").strip()
    if not target_hint:
        return None
    try:
        from app.agents.page_intel import (  # noqa: PLC0415
            propose_click_coordinates,
        )
        coords = propose_click_coordinates(
            provider, page,
            target_hint=target_hint,
        )
    except Exception as e:
        logger.warning("vision-only coord LLM call failed: %s", e)
        return None
    if coords.confidence < 0.6:
        # Low confidence → fall back to DOM. The strict-only mode
        # would refuse here, but for v1 we trade purity for
        # robustness on ambiguous screens.
        logger.info(
            "vision-only confidence %.2f < 0.6 — falling through to DOM",
            coords.confidence,
        )
        return None

    _emit(emit_event, "vision_only_action", {
        "run_id": submodule_run_id,
        "step_id": submodule_step_id,
        "tool": tool,
        "x": coords.x,
        "y": coords.y,
        "label_visible": coords.label_visible[:120],
        "confidence": coords.confidence,
    })

    # Dispatch.
    try:
        if tool == "click":
            page.mouse.click(coords.x, coords.y)
            outcome = {
                "status": "ok",
                "narration": (
                    f"vision-only click at ({coords.x},{coords.y}) — "
                    f"{coords.label_visible[:80]}"
                ),
                "error_message": None,
                "extracted_text": "",
                "details": {"strategy": "vision_only_coords"},
            }
        elif tool == "type":
            value = str(args.get("value") or "")
            page.mouse.click(coords.x, coords.y)
            # Clear any pre-existing value FIRST so a retry replaces
            # rather than stacks (Select-All + Delete). Without this,
            # typing into an already-filled field positions the new
            # text at the cursor and produces garbled "abcabc"
            # interleaved strings on validation-error retries.
            from app.executor.actions import (  # noqa: PLC0415
                clear_focused_field,
            )
            clear_focused_field(page)
            try:
                page.keyboard.type(value, delay=20)
            except Exception:
                page.keyboard.type(value)
            if bool(args.get("submit")):
                page.keyboard.press("Enter")
            outcome = {
                "status": "ok",
                "narration": (
                    f"vision-only type at ({coords.x},{coords.y}) — "
                    f"{len(value)} chars"
                    + (" + Enter" if args.get("submit") else "")
                ),
                "error_message": None,
                "extracted_text": "",
                "details": {"strategy": "vision_only_coords"},
            }
        else:
            return None
    except Exception as e:
        outcome = {
            "status": "failed",
            "narration": (
                f"vision-only {tool} dispatch failed at "
                f"({coords.x},{coords.y})"
            ),
            "error_message": f"{type(e).__name__}: {e}",
            "extracted_text": "",
            "details": {"strategy": "vision_only_coords"},
        }

    # ``cheap_provider`` and ``on_escalate`` are threaded through for
    # the future tier-aware coord dispatch path; unused here today.
    del cheap_provider, on_escalate
    return {
        "outcome": outcome,
        "input_tokens": coords.input_tokens,
        "output_tokens": coords.output_tokens,
    }


def _maybe_run_smart_pick(
    *,
    page,
    provider: LLMProvider,
    cheap_provider: LLMProvider | None,
    goal: Goal,
    tool: str,
    args: dict[str, Any],
    emit_event: Callable[[str, dict], None] | None,
    on_escalate: Any,
    submodule_run_id: int | None,
    submodule_step_id: int | None,
    # Phase J.5 — DOM ambiguity threshold. Default 3 keeps the
    # historical hybrid behavior. Vision-only callers pass 2 so the
    # DOM-aware tie-breaker fires for 2-candidate cases too (where
    # vision_only's pure-pixel proposal might pick the wrong one of
    # two visually similar Save / Save-As buttons).
    min_matches: int = 3,
) -> dict[str, Any] | None:
    """Phase 14 — smart candidate selection.

    Probes whether ``args.target_hint`` is ambiguous (3+ visible
    matches). If yes, runs the vision LLM to pick the right one
    among them (or scroll / give up) and applies the result:

    - ``selector`` → mutates ``args.target_hint`` to the picked
      selector. Caller's downstream dispatcher then resolves and
      clicks normally. Search-log captures the substitution.
    - ``coords``  → dispatches the click directly via
      ``page.mouse.click(x, y)`` and returns a ``preempt_outcome``
      that the caller substitutes for the regular dispatcher.
    - ``scroll``  → dispatches the scroll, returns a "skip the
      click this turn" preempt outcome so the next turn sees the
      scrolled page.
    - ``none``    → returns ``None`` to leave normal dispatch alone.

    Returns ``None`` when smart-pick is not applicable (no ambiguity,
    no vision, helper raised), or a record dict when it ran. The
    record carries ``input_tokens``, ``output_tokens``, and
    optionally ``preempt_outcome`` (when the helper itself dispatched
    the action and the caller should NOT call _execute_tool_call).
    """
    target_hint = (args.get("target_hint") or "").strip()
    if not target_hint:
        return None

    # Cheap probe: count locator matches. Locator-builder mirrors the
    # shape ``selectors.resolve`` accepts so the count reflects the
    # same population the resolver would draw from.
    try:
        locator = page.locator(target_hint)
        match_count = locator.count()
    except Exception as e:
        # Build/probe failed — let normal resolve() raise downstream.
        logger.debug(
            "smart-pick ambiguity probe failed for %r: %s",
            target_hint, e,
        )
        return None

    # Threshold: ``min_matches`` (default 3) visible matches qualifies
    # as ambiguous. Below that, the existing fuzzy / vision-search
    # ladder is fine. Vision-only callers pass min_matches=2 (Phase
    # J.5) so the DOM-grounded tie-breaker fires for 2-candidate
    # cases too — that's exactly where pure-pixel vision_only mode
    # tends to pick the wrong one of two similarly-styled buttons.
    if match_count < min_matches:
        return None

    _emit(emit_event, "smart_pick_started", {
        "run_id": submodule_run_id,
        "step_id": submodule_step_id,
        "target_hint": target_hint,
        "match_count": match_count,
        "tool": tool,
    })

    # Build the criteria block from the goal's success criteria +
    # sub-goal context. Empty list is fine — helper will pick by
    # general fitness.
    criteria: list[str] = list(goal.success_criteria or [])

    # Build a small candidate pre-filter from the AX tree so the
    # LLM doesn't have to re-derive it from the screenshot alone.
    visible_candidates: list[dict[str, Any]] = []
    try:
        from app.executor.selectors import (  # noqa: PLC0415
            _capture_ax_tree,
        )
        ax = _capture_ax_tree(page)
        # Crude pre-filter: items whose name contains the literal
        # token of the target_hint (the agent's hint after stripping
        # quotes / "text " prefix). Keeps the LLM focused.
        token = target_hint
        for prefix in ("text '", 'text "', "text "):
            if token.lower().startswith(prefix):
                token = token[len(prefix):].rstrip("'\"").strip()
                break
        token_lower = token.lower()
        for item in ax[:60]:
            name = (item.get("name") or "")
            if token_lower and token_lower in name.lower():
                visible_candidates.append({
                    "role": item.get("role"),
                    "name": name,
                    "selector_hint": item.get("selector_hint"),
                })
    except Exception as e:
        logger.debug("smart-pick AX pre-filter skipped: %s", e)

    # Build target description from the agent's reasoning + sub-goal.
    current_sg = next(
        (sg for sg in goal.sub_goals
         if sg.id == args.get("current_sub_goal_id")),
        None,
    )
    target_description = (
        (args.get("target_hint") or "")
        + (f" — sub-goal: {current_sg.description}" if current_sg else "")
        + (f" — goal: {goal.description}" if goal.description else "")
    )

    from app.agents.page_intel import (  # noqa: PLC0415
        capture_screenshot_for_vision, propose_smart_candidate,
    )
    try:
        screenshot = capture_screenshot_for_vision(page, downscale=False)
    except Exception as e:
        logger.warning("smart-pick screenshot capture failed: %s", e)
        return None

    try:
        pick = propose_smart_candidate(
            provider, page,
            target_description=target_description,
            criteria=criteria,
            visible_candidates=visible_candidates or None,
            screenshot_bytes=screenshot,
            cheap_provider=cheap_provider,
            on_escalate=on_escalate,
        )
    except Exception as e:
        logger.warning("smart-pick LLM call failed: %s", e)
        _emit(emit_event, "smart_pick_completed", {
            "run_id": submodule_run_id,
            "step_id": submodule_step_id,
            "outcome": "llm_error",
            "error": str(e)[:200],
        })
        return None

    _emit(emit_event, "smart_pick_completed", {
        "run_id": submodule_run_id,
        "step_id": submodule_step_id,
        "strategy": pick.strategy,
        "chosen_label": pick.chosen_label[:160],
        "rejected_count": len(pick.rejected_labels),
        "confidence": pick.confidence,
    })

    record: dict[str, Any] = {
        "input_tokens": pick.input_tokens,
        "output_tokens": pick.output_tokens,
        "strategy": pick.strategy,
        "chosen_label": pick.chosen_label,
        "rejected_labels": list(pick.rejected_labels),
        "rejection_reasons": list(pick.rejection_reasons),
        "reasoning": pick.reasoning,
        "confidence": pick.confidence,
    }

    if pick.strategy == "selector" and pick.selector:
        # Patch args in place — downstream dispatcher resolves the
        # picked selector cleanly. Keep the original hint visible
        # for telemetry.
        record["original_target_hint"] = target_hint
        args["target_hint"] = pick.selector
        return record

    if pick.strategy == "coords" and pick.x > 0 and pick.y > 0:
        # Direct dispatch via mouse — skip the resolver entirely.
        # Caller substitutes preempt_outcome for the dispatcher's
        # would-be result.
        try:
            page.mouse.click(pick.x, pick.y)
            preempt = {
                "status": "ok",
                "narration": (
                    f"smart-pick clicked at ({pick.x},{pick.y}) — "
                    f"{pick.chosen_label[:60]}"
                ),
                "error_message": None,
                "extracted_text": "",
                "details": {"smart_pick_strategy": "coords"},
            }
        except Exception as e:
            preempt = {
                "status": "failed",
                "narration": (
                    f"smart-pick coord click failed at "
                    f"({pick.x},{pick.y})"
                ),
                "error_message": f"{type(e).__name__}: {e}",
                "extracted_text": "",
                "details": {"smart_pick_strategy": "coords"},
            }
        record["preempt_outcome"] = preempt
        return record

    if pick.strategy == "scroll" and pick.scroll_direction:
        # Scroll the requested direction; skip the click for now —
        # the next turn re-evaluates against the scrolled page.
        amount = pick.scroll_amount_px or 600
        try:
            if pick.scroll_direction == "down":
                page.mouse.wheel(0, amount)
            elif pick.scroll_direction == "up":
                page.mouse.wheel(0, -amount)
            elif pick.scroll_direction == "right":
                page.mouse.wheel(amount, 0)
            elif pick.scroll_direction == "left":
                page.mouse.wheel(-amount, 0)
            preempt = {
                "status": "ok",
                "narration": (
                    f"smart-pick scrolled {pick.scroll_direction} {amount}px "
                    f"(no visible candidate matched criteria)"
                ),
                "error_message": None,
                "extracted_text": "",
                "details": {
                    "smart_pick_strategy": "scroll",
                    "scroll_skip_click": True,
                },
            }
        except Exception as e:
            preempt = {
                "status": "failed",
                "narration": "smart-pick scroll dispatch failed",
                "error_message": f"{type(e).__name__}: {e}",
                "extracted_text": "",
                "details": {"smart_pick_strategy": "scroll"},
            }
        record["preempt_outcome"] = preempt
        return record

    # strategy == "none" or malformed → fall through to normal
    # dispatch. The caller's existing fuzzy + vision-search rescue
    # path can still try, OR the dispute tool (Phase 11) can flag it.
    return record


def _vision_search_for_target(
    page,
    provider: LLMProvider,
    *,
    target_hint: str,
    max_attempts: int = 3,
    emit_event: Callable[[str, dict], None] | None = None,
    submodule_run_id: int | None = None,
    submodule_step_id: int | None = None,
    cheap_provider: LLMProvider | None = None,
    on_escalate: Any = None,
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
    # Surface the LAST iteration's near_misses + screenshot bytes to
    # the caller so the coord-click rescue (which runs on the same
    # page state, after this function exhausts) doesn't have to
    # recompute the AX tree OR re-screenshot the page.
    last_near_misses: list[dict[str, Any]] = []
    last_screenshot: bytes | None = None

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
        last_near_misses = near_misses

        # Snapshot the page ONCE per attempt, pass it down so the
        # vision LLM and any rescue rung that runs after this
        # function exhausts (coord-click) reuses the same bytes.
        # On the LAST attempt, capture full_page so the LLM can see
        # off-viewport content (the typical "user said scroll but
        # the target is way down the page" scenario).
        try:
            full_page = (attempt == max_attempts)
            from app.agents.page_intel import (  # noqa: PLC0415
                capture_screenshot_for_vision,
            )
            attempt_screenshot = capture_screenshot_for_vision(
                page, full_page=full_page,
            )
            last_screenshot = attempt_screenshot
        except Exception as e:
            logger.debug("vision-search screenshot failed: %s", e)
            attempt_screenshot = None

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
                screenshot_bytes=attempt_screenshot,
                cheap_provider=cheap_provider,
                on_escalate=on_escalate,
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
        "last_near_misses": last_near_misses,
        "last_screenshot": last_screenshot,
    }


# ── Diag.4 — inter-submodule state reset ─────────────────────────


_RESET_DRAWER_JS = r"""
() => {
  // Dismiss any visible drawer / modal / dialog. Strategy:
  //   1. Find every visible role=dialog / .MuiDrawer-paper /
  //      heuristic-detected fixed-position drawer.
  //   2. For each, click the most likely "close" affordance —
  //      aria-label="Close", button with × / Close text, or
  //      role=button positioned at the top-right corner.
  //   3. As a fallback, hide the element so it can't intercept
  //      subsequent clicks (last-resort; the DOM teardown happens
  //      later when the agent navigates away).
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden' &&
           parseFloat(cs.opacity) > 0.05;
  };
  const drawerSel = [
    '[role=dialog]', '[role=alertdialog]',
    '[aria-modal="true"]',
    '.MuiDialog-paper', '.MuiDrawer-paper', '.MuiModal-root',
    '[class*="Drawer"]', '[class*="drawer"]',
    '[class*="Modal"]', '[class*="modal"]',
  ].join(',');
  const drawers = [...document.querySelectorAll(drawerSel)]
    .filter(VISIBLE)
    .filter(el => {
      const r = el.getBoundingClientRect();
      return r.width >= 240 && r.height >= 240;
    });
  let closed = 0;
  for (const dr of drawers) {
    // Try a labelled Close button first.
    const closeBtn =
      dr.querySelector('[aria-label="Close"]') ||
      dr.querySelector('[aria-label="close"]') ||
      dr.querySelector('button[title="Close"]') ||
      // × character inside a button
      [...dr.querySelectorAll('button, [role=button]')].find(b => {
        const t = (b.innerText || b.textContent || '').trim();
        return t === '×' || t === 'X' || t === 'x' ||
               t.toLowerCase() === 'close' ||
               t.toLowerCase() === 'cancel';
      });
    if (closeBtn) {
      try {
        closeBtn.click();
        closed += 1;
        continue;
      } catch (e) { /* fall through */ }
    }
    // Top-right icon button (no label, just an SVG / icon-class).
    const dr_rect = dr.getBoundingClientRect();
    const corner = [...dr.querySelectorAll('button, [role=button]')]
      .filter(VISIBLE)
      .find(b => {
        const r = b.getBoundingClientRect();
        return r.right >= dr_rect.right - 60
            && r.top <= dr_rect.top + 60
            && r.width <= 60 && r.height <= 60;
      });
    if (corner) {
      try { corner.click(); closed += 1; continue; } catch (e) {}
    }
  }
  return closed;
};
"""


def _reset_inter_submodule_state(page) -> None:
    """Diag.4 — clean the page between submodules.

    Sequence:
      1. Press Escape (closes most MUI / shadcn / AntD dialogs cheap).
      2. Run a JS pass that finds every visible drawer and clicks
         its Close affordance (aria-label="Close" / × / top-right
         icon button).
      3. Repeat Escape once more for stacked drawers.
      4. Scroll the page to the top so the next observation starts
         at a known baseline.

    All steps are best-effort; failures are logged at DEBUG and never
    raise. If a drawer refuses to close, the agent loop will see it
    on the next observation and can plan around it (or fail cleanly
    via the action-no-effect circuit-breaker).
    """
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(120)
    except Exception:
        pass
    try:
        closed = page.evaluate(_RESET_DRAWER_JS)
        if isinstance(closed, int) and closed > 0:
            page.wait_for_timeout(180)
    except Exception:
        pass
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(80)
    except Exception:
        pass
    try:
        page.evaluate("window.scrollTo(0, 0)")
    except Exception:
        pass


# ── Phase O.2 — pre-action existence check ────────────────────────


_TARGET_PROBE_JS = r"""
(needle) => {
  // Cheap DOM probe: does any visible interactive element have a
  // label / aria-label / placeholder / text content that contains
  // ``needle`` (case-insensitive, first 60 chars)? Also returns the
  // 8 closest-looking visible labels so the planner-feedback prompt
  // can show the agent what IS on the page.
  const want = (needle || '').trim().toLowerCase();
  if (!want) return { exists: false, similar: [] };
  const SEL = [
    'button', '[role=button]', 'a[href]',
    'input', 'textarea', 'select',
    '[role=combobox]', '[role=listbox]', '[role=checkbox]',
    '[role=radio]', '[role=textbox]', '[role=tab]',
    '[role=menuitem]', '[role=heading]', '[role=link]',
    '[role=switch]', '[role=row]',
  ].join(',');
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden' &&
           parseFloat(cs.opacity) > 0.05;
  };
  const labelOf = (el) => {
    let v = el.getAttribute('aria-label') ||
            el.getAttribute('title') ||
            el.getAttribute('placeholder') ||
            el.getAttribute('name') || '';
    if (!v) v = (el.innerText || el.textContent || '').trim();
    return String(v || '').trim().slice(0, 120);
  };
  const labels = [];
  let found = false;
  for (const el of document.querySelectorAll(SEL)) {
    if (!VISIBLE(el)) continue;
    const lab = labelOf(el);
    if (!lab) continue;
    const ll = lab.toLowerCase();
    if (ll === want || ll.includes(want) || want.includes(ll)) {
      found = true;
      // Keep scanning to populate similar; the agent can use it for
      // adjacent label hints even on a match.
    }
    if (labels.length < 30 && !labels.includes(lab)) labels.push(lab);
    if (found && labels.length >= 30) break;
  }
  // Pick the 8 labels with greatest token overlap to `want` for the
  // "did you mean" hint when not found.
  const tokens = (want.match(/\w+/g) || []);
  function score(lab) {
    const ll = lab.toLowerCase();
    let s = 0;
    for (const t of tokens) if (t && ll.includes(t)) s += 1;
    return s;
  }
  const similar = labels
    .map(l => [score(l), l])
    .sort((a, b) => b[0] - a[0])
    .slice(0, 8)
    .map(x => x[1]);
  return { exists: found, similar };
};
"""


def _check_target_exists(
    page,
    target_hint: str,
) -> tuple[bool, list[str]]:
    """Phase O.2 — does the target_hint refer to anything visible on
    the current page?

    Heuristic, fast, no LLM. Returns ``(exists, similar)``:
      - ``exists=True`` when at least one visible interactive element
        has a label containing the target_hint (or vice versa).
      - ``similar`` is the top-8 visible-element labels ranked by
        token overlap with the target_hint, so the planner-feedback
        prompt can show "you said 'Save'; visible options are
        'Save User', 'Cancel', '+ Add New User', …".

    Returns ``(True, [])`` for any target_hint that LOOKS like a CSS
    or attribute selector (starts with '#', '.', '[', or contains
    '>' / ':has(' / '/' / 'xpath=') — we don't try to fuzz-resolve
    those; the DOM resolver downstream handles them.
    """
    hint = (target_hint or "").strip()
    if not hint:
        return (True, [])
    # Skip selector-shaped hints.
    if (
        hint.startswith(("#", ".", "[", "(", ":", "/"))
        or "xpath=" in hint
        or " > " in hint
        or "[role=" in hint
    ):
        return (True, [])
    try:
        raw = page.evaluate(_TARGET_PROBE_JS, hint[:120])
    except Exception:
        return (True, [])
    if not isinstance(raw, dict):
        return (True, [])
    return (
        bool(raw.get("exists")),
        [str(s)[:120] for s in (raw.get("similar") or [])],
    )


# ── Phase O.1 — sub-goal verification gate ────────────────────────


def _verify_subgoal_criterion(
    page,
    *,
    criterion: str,
    observation: dict[str, Any],
) -> tuple[bool, str]:
    """Phase O.1 — deterministic check that a sub-goal's
    ``success_criterion`` is actually observable on the current page.

    Returns ``(ok, reason)``. ``ok=True`` means the criterion's
    deterministic signal matched; the runtime allows the sub-goal to
    transition to ``done``. ``ok=False`` means the agent claimed
    success but the page disagrees; the runtime should refuse the
    close and surface a feedback signal so the next turn fixes it.

    Strategy — only the patterns we can verify deterministically:

      - URL match: ``"URL contains X"`` / ``"current URL is X"``
        → compare against ``observation["url"]``.
      - Visible text: any QUOTED token ("...") in the criterion must
        appear in visible page text (case-insensitive).
      - Drawer state: ``"drawer is visible"`` / ``"drawer is closed"``
        → query DOM for an open drawer.
      - Toast: ``"toast says X"`` → check the page for ``X`` text.

    When NONE of the deterministic patterns match the criterion (e.g.
    "permissions are selected appropriately"), we return ``(True,
    "no deterministic signal")`` — the verification gate doesn't fire,
    legacy behavior. The agent's own claim stands.
    """
    import re  # noqa: PLC0415

    text = (criterion or "").strip()
    if not text:
        return (True, "empty criterion — gate skipped")
    low = text.lower()

    # URL match.
    m = re.search(r"url(?:\s+contains)?\s+['\"]?([^'\"]+?)['\"]?(?:\s|$)", low)
    if m and ("url" in low):
        needle = m.group(1).strip().strip(".,;:!?")
        cur_url = (observation.get("url") or "").lower()
        if needle and needle not in cur_url:
            return (
                False,
                f"URL {cur_url!r} does not contain {needle!r}",
            )
        return (True, f"URL contains {needle!r}")

    # Drawer state.
    if "drawer" in low or "modal" in low or "dialog" in low:
        wants_open = (
            "visible" in low or "open" in low or "appears" in low
            or "shown" in low
        )
        wants_closed = (
            "closed" in low or "dismissed" in low
            or "disappears" in low or "hidden" in low
        )
        if wants_open or wants_closed:
            try:
                is_open = bool(page.evaluate(
                    "(() => {const s=['[role=dialog]','[role=alertdialog]',"
                    "'.MuiDialog-paper','.MuiDrawer-paper','[class*=\"Drawer\"]',"
                    "'[class*=\"drawer\"]','[class*=\"Modal\"]','[class*=\"modal\"]']"
                    ".join(',');return [...document.querySelectorAll(s)]"
                    ".some(el => {const r=el.getBoundingClientRect();"
                    "if(r.width<100||r.height<100)return false;"
                    "const cs=getComputedStyle(el);"
                    "return cs.display!=='none'&&cs.visibility!=='hidden';});})()",
                ))
            except Exception:
                is_open = None
            if is_open is None:
                return (True, "drawer probe failed — gate skipped")
            if wants_open and not is_open:
                return (False, "criterion expects open drawer; none visible")
            if wants_closed and is_open:
                return (False, "criterion expects closed drawer; still open")
            return (True, "drawer state matches criterion")

    # Quoted-token presence — the criterion mentions specific text
    # the page must show. Multiple quotes are OR (ANY must appear).
    quoted = re.findall(r"['\"]([^'\"]{2,80})['\"]", text)
    if quoted:
        try:
            visible_text = page.evaluate(
                "() => (document.body && document.body.innerText) || ''",
            )
        except Exception:
            visible_text = ""
        visible_text = (visible_text or "").lower()
        for q in quoted:
            if q.strip().lower() in visible_text:
                return (True, f"page contains {q!r}")
        return (
            False,
            f"none of {quoted!r} found in visible page text",
        )

    # Toast / banner / alert (un-quoted form).
    if "toast" in low or "snackbar" in low or "alert" in low:
        try:
            visible_text = page.evaluate(
                "() => (document.body && document.body.innerText) || ''",
            )
        except Exception:
            visible_text = ""
        # Strip the trigger word and check the remainder.
        tail = re.sub(
            r"\b(toast|snackbar|alert|notification|banner)\s+(?:says|shows|reads|appears)?\s*",
            "",
            low,
        ).strip().strip("'\".")
        if tail and tail not in (visible_text or "").lower():
            return (
                False,
                f"expected toast text {tail!r} not visible",
            )
        return (True, "toast keyword satisfied")

    return (True, "no deterministic signal — gate skipped")


# ── Phase N — HITL → direct action dispatch ──────────────────────


_HITL_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": [
                "click", "type", "select", "scroll", "navigate",
                "wait", "verify", "dismiss_modal", "extract_text",
                "fill_form", "go_back",
                "give_up",  # special: user input doesn't map to an action
            ],
        },
        "target_hint": {"type": "string"},
        "value": {"type": "string"},
        "x": {"type": ["integer", "null"]},
        "y": {"type": ["integer", "null"]},
        "form_fields": {"type": "string"},
        "form_submit_label": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
        },
    },
    "required": [
        "tool", "target_hint", "value", "x", "y",
        "form_fields", "form_submit_label",
        "reasoning", "confidence",
    ],
    "additionalProperties": False,
}


_HITL_INTERPRETER_SYSTEM_PROMPT = (
    "You are an action interpreter. The QA agent stalled. A HUMAN just "
    "looked at the screen and gave instructions on how to proceed. Your "
    "ONE job: translate the human's instruction into ONE tool call that "
    "the agent runtime will execute IMMEDIATELY.\n\n"
    "Inputs you receive:\n"
    "  - The current page screenshot.\n"
    "  - The human's typed instruction.\n"
    "  - Optionally a second image showing the human's hand-drawn "
    "marks on the screenshot (e.g. a box around the element to click). "
    "When the drawing shows a clear bounding box, return tool=click "
    "with x/y at the box's center.\n\n"
    "RULES:\n"
    "1. Map the instruction to EXACTLY ONE tool call. Never two.\n"
    "2. Trust the human. If they say 'click Save', dispatch "
    "click(target_hint=\"Save\"). Don't second-guess.\n"
    "3. If the drawing shows a clear region, prefer click(x, y) at "
    "the region's center over click(target_hint=...) — the drawing "
    "is the most reliable signal.\n"
    "4. For form-fill instructions ('fill First Name=Alice, "
    "Last Name=Smith, then Save'), emit tool=fill_form with "
    "form_fields as a JSON array string and form_submit_label set.\n"
    "5. Use tool='give_up' ONLY when the instruction is purely "
    "advisory ('be careful', 'note that this is a tricky page') and "
    "doesn't name an actionable next step.\n"
    "6. confidence: 0.9+ when the instruction names an exact element; "
    "0.6-0.8 when interpretation is needed; <0.5 if the instruction "
    "is vague.\n\n"
    "Output STRICT JSON. Empty strings for unused fields. Use null "
    "for x/y when not applicable."
)


def _hitl_rule_based_parser(text: str) -> dict[str, Any] | None:
    """Phase R — deterministic, zero-LLM HITL parser.

    Tried BEFORE the LLM interpreter so the emergency path doesn't
    depend on a network round-trip. Covers the high-value patterns:

      "click Save"           → click(target_hint="Save")
      "click 800, 400"       → click at coords (800, 400)
      "click at 800,400"     → click at coords (800, 400)
      "type Alice"           → type with value="Alice" (target inferred
                                by the runtime's last-focused-field
                                heuristic)
      "type 'Alice' into First Name"
                             → type(target_hint="First Name",
                                    value="Alice")
      "scroll down"          → scroll(scroll_direction="down")
      "skip"                 → give_up (lets the planner re-plan)
      "save" / "submit"      → click(target_hint=word)

    Returns ``None`` when no rule matches (LLM fallback fires).
    """
    import re  # noqa: PLC0415

    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()

    # click X, Y coords
    m = re.match(
        r"^click\s+(?:at\s+)?\(?\s*(\d{1,5})\s*,\s*(\d{1,5})\s*\)?",
        low,
    )
    if m:
        return {
            "tool": "click",
            "target_hint": "",
            "value": "",
            "x": int(m.group(1)),
            "y": int(m.group(2)),
            "form_fields": "",
            "form_submit_label": "",
            "reasoning": "HITL rule: explicit coords",
            "confidence": 1.0,
        }

    # type 'X' into Y
    m = re.match(
        r"^type\s+['\"]([^'\"]+)['\"]\s+(?:in(?:to)?|to)\s+(.+)$",
        t, flags=re.IGNORECASE,
    )
    if m:
        return {
            "tool": "type",
            "target_hint": m.group(2).strip(),
            "value": m.group(1),
            "x": None,
            "y": None,
            "form_fields": "",
            "form_submit_label": "",
            "reasoning": "HITL rule: type quoted into target",
            "confidence": 0.95,
        }
    m = re.match(
        r"^type\s+(.+)$",
        t, flags=re.IGNORECASE,
    )
    if m:
        return {
            "tool": "type",
            "target_hint": "",
            "value": m.group(1).strip(),
            "x": None,
            "y": None,
            "form_fields": "",
            "form_submit_label": "",
            "reasoning": "HITL rule: type value",
            "confidence": 0.85,
        }

    # click X / press X
    m = re.match(
        r"^(?:click|press|tap)\s+(?:on\s+|the\s+)?(.+)$",
        t, flags=re.IGNORECASE,
    )
    if m:
        return {
            "tool": "click",
            "target_hint": m.group(1).strip().strip(".'\""),
            "value": "",
            "x": None,
            "y": None,
            "form_fields": "",
            "form_submit_label": "",
            "reasoning": "HITL rule: click named target",
            "confidence": 0.95,
        }

    # scroll
    if low in ("scroll", "scroll down", "down", "page down"):
        return {
            "tool": "scroll",
            "target_hint": "",
            "value": "down",
            "x": None,
            "y": None,
            "form_fields": "",
            "form_submit_label": "",
            "reasoning": "HITL rule: scroll down",
            "confidence": 0.9,
        }
    if low in ("scroll up", "up", "page up"):
        return {
            "tool": "scroll",
            "target_hint": "",
            "value": "up",
            "x": None,
            "y": None,
            "form_fields": "",
            "form_submit_label": "",
            "reasoning": "HITL rule: scroll up",
            "confidence": 0.9,
        }

    # single-word save / submit / cancel / etc → click that label
    if re.match(r"^[a-z][a-z\-\s]{1,30}$", low):
        if any(
            kw == low or kw in low.split()
            for kw in (
                "save", "submit", "create", "confirm",
                "apply", "next", "cancel", "back", "ok",
            )
        ):
            return {
                "tool": "click",
                "target_hint": t,
                "value": "",
                "x": None,
                "y": None,
                "form_fields": "",
                "form_submit_label": "",
                "reasoning": "HITL rule: bare action word → click",
                "confidence": 0.85,
            }

    return None


def _interpret_hitl_as_action(
    *,
    pending_hitl: dict[str, Any],
    screenshot: bytes | None,
    provider: LLMProvider,
    cheap_provider: LLMProvider | None,
) -> dict[str, Any] | None:
    """Phase N — translate a human's HITL submission into ONE tool call.

    Returns a tool-call dict in the shape the agent's runtime expects
    (``{tool, target_hint, value, ...}``). Returns ``None`` when:
      - HITL text is empty AND no drawing is present
      - The interpreter LLM call fails / produces tool='give_up'
      - The result's confidence < 0.4 (too unsure to dispatch directly)

    On failure, the caller falls back to the existing
    "stash-in-page_memory + let planner handle it" path so HITL never
    HARDS — worst case is parity with the old behavior.
    """
    from app.llm.base import ChatMessage  # noqa: PLC0415
    import base64 as _b64  # noqa: PLC0415

    text = (pending_hitl.get("text") or "").strip()
    drawing_b64 = (pending_hitl.get("drawing_b64") or "").strip()
    if not text and not drawing_b64:
        return None

    # Phase R — rule-based fast path. ZERO LLM call when the user's
    # text matches a simple pattern (click X / type X / coords / save
    # / scroll). Triggered BEFORE the LLM interpreter so the
    # emergency path is instant even when OpenAI is slow / down.
    # Skipped when the user attached a drawing — we want the LLM to
    # see the drawing in that case.
    if text and not drawing_b64:
        rule_parsed = _hitl_rule_based_parser(text)
        if rule_parsed is not None:
            tool = str(rule_parsed.get("tool") or "")
            return {
                "tool": tool,
                "target_hint": str(rule_parsed.get("target_hint") or ""),
                "value": str(rule_parsed.get("value") or ""),
                "reasoning": str(rule_parsed.get("reasoning") or "")[:400],
                "confidence": float(rule_parsed.get("confidence", 0.9)),
                "page_memory_note": "",
                "current_sub_goal_id": (
                    pending_hitl.get("sub_goal_id") or ""
                ),
                "sub_goal_completed_id": "",
                "skip_sub_goal_id": "",
                "skip_sub_goal_reason": "",
                "issue_kind": "",
                "issue_evidence": "",
                "issue_suggested_fix": "",
                "form_fields": str(rule_parsed.get("form_fields") or ""),
                "form_submit_label": str(
                    rule_parsed.get("form_submit_label") or "",
                ),
                "url": "",
                "expected": "",
                "duration_ms": 0,
                "scroll_direction": str(
                    rule_parsed.get("value") if tool == "scroll"
                    else ""
                ),
                "scroll_amount": 0,
                "question": "",
                "key": "",
                "submit": False,
                "_hitl_coord_x": rule_parsed.get("x"),
                "_hitl_coord_y": rule_parsed.get("y"),
                "_via_hitl": True,
                "_hitl_tokens_in": 0,
                "_hitl_tokens_out": 0,
            }

    # Build messages: screenshot + optional drawing + the text.
    parts: list[str] = []
    if text:
        parts.append(f"HUMAN INSTRUCTION:\n  \"{text}\"\n")
    else:
        parts.append(
            "HUMAN INSTRUCTION: (no text — interpret the drawing "
            "as the action target)\n"
        )
    parts.append(
        "Translate this into ONE tool call the agent will execute "
        "immediately. The agent is stalled and waiting for you to "
        "decide.\n"
    )
    user_text = "".join(parts)

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=_HITL_INTERPRETER_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=user_text,
            image=screenshot,
        ),
    ]
    # Optional second image — the user's drawing.
    if drawing_b64:
        try:
            drawing_bytes = _b64.b64decode(drawing_b64)
            messages.append(
                ChatMessage(
                    role="user",
                    content="(The human's drawing on the screenshot — "
                    "use any box / arrow to locate the target.)",
                    image=drawing_bytes,
                ),
            )
        except Exception:
            pass

    # Prefer cheap-tier; the interpretation is a small structured task.
    interpreter = cheap_provider or provider
    try:
        result = interpreter.chat_structured(
            messages=messages,
            schema=_HITL_TOOL_SCHEMA,
            schema_name="hitl_tool_call",
            temperature=0.1,
            max_output_tokens=400,
        )
    except Exception as e:
        logger.warning("HITL interpreter LLM call failed: %s", e)
        return None
    parsed = result.parsed
    if not isinstance(parsed, dict):
        return None
    tool = str(parsed.get("tool") or "").strip()
    if not tool or tool == "give_up":
        return None
    try:
        conf = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if conf < 0.4:
        logger.info(
            "HITL interpreter confidence %.2f < 0.4 — falling through "
            "to planner with stashed guidance", conf,
        )
        return None

    # Translate into the planner's tool-call shape.
    out: dict[str, Any] = {
        "tool": tool,
        "target_hint": str(parsed.get("target_hint") or ""),
        "value": str(parsed.get("value") or ""),
        "reasoning": str(parsed.get("reasoning") or "")[:400],
        "confidence": conf,
        "page_memory_note": "",
        "current_sub_goal_id": pending_hitl.get("sub_goal_id") or "",
        "sub_goal_completed_id": "",
        "skip_sub_goal_id": "",
        "skip_sub_goal_reason": "",
        "issue_kind": "",
        "issue_evidence": "",
        "issue_suggested_fix": "",
        "form_fields": str(parsed.get("form_fields") or ""),
        "form_submit_label": str(parsed.get("form_submit_label") or ""),
        "url": "",
        "expected": "",
        "duration_ms": 0,
        "scroll_direction": "",
        "scroll_amount": 0,
        "question": "",
        "key": "",
        "submit": False,
        # Phase N marker — downstream dispatchers can prefer coord
        # click when these are non-null (skips smart-pick + DOM
        # resolution for the user's exact target).
        "_hitl_coord_x": parsed.get("x"),
        "_hitl_coord_y": parsed.get("y"),
        "_via_hitl": True,
        "_hitl_tokens_in": result.input_tokens or 0,
        "_hitl_tokens_out": result.output_tokens or 0,
    }
    return out


def _dispatch_fill_form(
    page,
    *,
    args: dict[str, Any],
    emit_event: Callable[[str, dict], None] | None = None,
    submodule_run_id: int | None = None,
    submodule_step_id: int | None = None,
    turn_idx: int | None = None,
) -> dict[str, Any]:
    """Phase F — dispatch the bundled fill_form routine.

    Parses ``args["form_fields"]`` (JSON string of field objects) +
    ``args["form_submit_label"]``, invokes
    :func:`form_fill.run_form_fill`, and translates the result into
    the standard outcome dict the turn loop expects.

    Failure modes:
    - empty/invalid JSON in form_fields → status="failed" with a
      clear narration ("malformed form_fields payload")
    - all fields missed (no DOM match) → status="failed"
    - submit returned validation errors → status="failed" so the
      next-turn prompt sees the FORM SIGNAL and the agent can plan
      a recovery (most likely a manual per-field type)
    - filled + submit ok → status="ok"
    """
    import json as _json  # noqa: PLC0415
    from app.executor.form_fill import (  # noqa: PLC0415
        FormField, run_form_fill,
    )

    raw_fields = args.get("form_fields") or ""
    raw_submit = args.get("form_submit_label")
    submit_label = (
        raw_submit if isinstance(raw_submit, str) else "Save"
    )
    try:
        parsed = _json.loads(raw_fields) if raw_fields else []
    except Exception as e:
        return {
            "status": "failed",
            "narration": (
                f"fill_form: malformed JSON in form_fields "
                f"({type(e).__name__})"
            ),
            "error_message": (
                "form_fields must be a JSON-encoded array of "
                "{label, value, required?, role_hint?}"
            ),
            "extracted_text": "",
            "details": {"fill_form": "parse_error"},
        }
    if not isinstance(parsed, list) or not parsed:
        return {
            "status": "failed",
            "narration": "fill_form: form_fields is empty",
            "error_message": "no fields to fill",
            "extracted_text": "",
            "details": {"fill_form": "empty"},
        }

    fields: list[FormField] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", "")).strip()
        value = str(entry.get("value", ""))
        if not label:
            continue
        role_hint = entry.get("role_hint")
        fields.append(FormField(
            label=label,
            value=value,
            required=bool(entry.get("required", False)),
            role_hint=role_hint if isinstance(role_hint, str) and role_hint else None,  # type: ignore[arg-type]
        ))
    if not fields:
        return {
            "status": "failed",
            "narration": "fill_form: no usable field entries",
            "error_message": "all entries missing label",
            "extracted_text": "",
            "details": {"fill_form": "no_usable_entries"},
        }

    # Decorate the form_fill events with run/step/turn context so
    # the live feed can scope them.
    def _decorated_emit(t: str, d: dict) -> None:
        if emit_event is None:
            return
        try:
            payload = dict(d)
            payload.setdefault("run_id", submodule_run_id)
            payload.setdefault("step_id", submodule_step_id)
            payload.setdefault("turn", turn_idx)
            emit_event(t, payload)
        except Exception:
            pass

    result = run_form_fill(
        page,
        fields=fields,
        submit_label=submit_label,
        emit_event=_decorated_emit,
    )

    filled = result.filled_count
    misses = result.miss_count
    total = len(result.fields)
    seconds = result.total_seconds

    # Status policy:
    #   - any miss on a required field → failed
    #   - submit validation_error → failed
    #   - everything filled + submit ok / skipped → ok
    required_miss = any(
        o.status == "miss"
        and any(
            ff.label == o.label and ff.required for ff in fields
        )
        for o in result.fields
    )
    if result.submit_status == "validation_error":
        status = "failed"
    elif required_miss:
        status = "failed"
    elif result.submit_status in ("ok", "no_submit"):
        status = "ok"
    elif result.submit_status == "error":
        status = "failed"
    else:
        status = "ok"

    narration = (
        f"fill_form: {filled}/{total} fields set"
        + (f", {misses} missed" if misses else "")
        + (
            f"; submit={result.submit_status}"
            f"{f' ({result.submit_message[:120]})' if result.submit_message else ''}"
        )
        + f" — {seconds}s"
    )
    error_message = None
    if status == "failed":
        bits: list[str] = []
        if required_miss:
            bits.append(
                "required field(s) could not be filled: "
                + ", ".join(
                    o.label for o in result.fields
                    if o.status == "miss"
                )
            )
        if result.submit_status == "validation_error":
            bits.append(
                "validation errors on: "
                + ", ".join(result.validation_fields[:6])
            )
        if result.submit_status == "error":
            bits.append(f"submit error: {result.submit_message}")
        error_message = " | ".join(bits) or "fill_form failed"

    return {
        "status": status,
        "narration": narration,
        "error_message": error_message,
        "extracted_text": "",
        "details": {
            "fill_form": {
                "total": total,
                "filled": filled,
                "miss": misses,
                "submit_status": result.submit_status,
                "validation_fields": result.validation_fields,
                "seconds": seconds,
                "fields": [
                    {
                        "label": o.label,
                        "role": o.role,
                        "status": o.status,
                        "attempts": o.attempts,
                        "error": o.error[:200] if o.error else "",
                    }
                    for o in result.fields
                ],
            },
        },
    }


# Phase J.4 — entity-creation detector. Examines the goal text and
# the fill_form payload to decide whether THIS submission produced
# a new persistent entity (role, user, project, chainage, …) and,
# if so, records it on WorldState so subsequent submodules can
# REUSE it instead of re-creating.
_ENTITY_KIND_HINTS: tuple[tuple[str, str], ...] = (
    # Order matters — longer / more specific phrases first so
    # "user role" doesn't match before "role".
    ("chainage", "chainage"),
    ("user", "user"),
    ("role", "role"),
    ("project", "project"),
    ("client", "client"),
    ("vendor", "vendor"),
    ("organisation", "organisation"),
    ("organization", "organisation"),
    ("group", "group"),
    ("team", "team"),
    ("policy", "policy"),
    ("permission", "permission"),
    ("contract", "contract"),
    ("template", "template"),
    ("workflow", "workflow"),
    ("schedule", "schedule"),
    ("task", "task"),
)

_CREATE_VERB_HINTS: tuple[str, ...] = (
    "create", "add", "register", "make", "new ", "+create",
    "+ create", "+add", "+ add",
)


def _record_entity_from_fill_form(
    *,
    world_state: dict[str, Any] | None,
    goal_text: str,
    args: dict[str, Any],
    page_url: str,
) -> None:
    """Post-fill_form hook: write to ``world_state.entities_created``
    when the goal text indicates a create flow + we can identify the
    entity from the submitted fields. Safe no-op when:
      - ``world_state`` is None (legacy run)
      - the goal text doesn't include a create verb
      - no entity kind hint matches the goal
      - the form_fields payload doesn't yield an identity
    """
    if not isinstance(world_state, dict):
        return
    text = (goal_text or "").lower()
    if not any(v in text for v in _CREATE_VERB_HINTS):
        return
    kind: str | None = None
    for needle, normalized in _ENTITY_KIND_HINTS:
        if needle in text:
            kind = normalized
            break
    if kind is None:
        return

    # Parse the form_fields payload (JSON string) and pick the most
    # likely identity. Order of preference:
    #   1. A field whose label matches "name" / "title" / "id" / "code"
    #      (case-insensitive substring match).
    #   2. The first non-empty textbox value.
    import json as _json  # noqa: PLC0415

    raw_fields = args.get("form_fields") or ""
    try:
        parsed = _json.loads(raw_fields) if raw_fields else []
    except Exception:
        return
    if not isinstance(parsed, list) or not parsed:
        return

    identity: str | None = None
    identity_label = ""
    preferred_keys = ("name", "title", "identifier", "id", "code", "username")
    for k in preferred_keys:
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label", "")).strip()
            val = str(entry.get("value", "")).strip()
            if not label or not val:
                continue
            if k in label.lower():
                identity = val
                identity_label = label
                break
        if identity is not None:
            break
    if identity is None:
        # Fallback: first non-empty textbox-style value (skip the
        # compound-widget DSL values).
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            role_hint = str(entry.get("role_hint") or "")
            if role_hint in (
                "permission_tree", "paginated_resource_table",
                "checkbox", "radio",
            ):
                continue
            val = str(entry.get("value", "")).strip()
            if val:
                identity = val
                identity_label = str(entry.get("label", "")).strip()
                break
    if not identity:
        return

    try:
        from app.services.world_state import (  # noqa: PLC0415
            record_entity_created,
        )
        record_entity_created(
            world_state,
            kind=kind,
            identity=identity,
            url=page_url,
        )
        logger.info(
            "WorldState: recorded %s=%r (field=%r)",
            kind, identity, identity_label,
        )
    except Exception as e:
        logger.debug(
            "record_entity_created failed: %s", e,
        )


def _execute_tool_call(
    page,
    tool: str,
    args: dict[str, Any],
    *,
    plan_target_url: str,
    speed_config,
    emit_event: Callable[[str, dict], None] | None = None,
    submodule_run_id: int | None = None,
    submodule_step_id: int | None = None,
    turn_idx: int | None = None,
) -> dict[str, Any]:
    """Execute one action tool. Returns a dict with status / narration /
    error / extracted_text. Meta tools never come here.

    ``emit_event`` + ``submodule_run_id`` / ``submodule_step_id`` are
    optional; the bundled ``fill_form`` routine uses them to push
    per-field progress events into the live feed. Other tool branches
    don't need them.
    """
    if tool == "scroll":
        status, narration, error = _do_scroll(page, args)
        return {
            "status": status,
            "narration": narration,
            "error_message": error,
            "extracted_text": "",
            "details": {},
        }
    if tool == "extract_text":
        status, narration, error, text, details = _do_extract_text(page, args)
        return {
            "status": status,
            "narration": narration,
            "error_message": error,
            "extracted_text": text,
            "details": details,
        }
    if tool == "dismiss_modal":
        status, narration, error = _do_dismiss_modal(page)
        return {
            "status": status,
            "narration": narration,
            "error_message": error,
            "extracted_text": "",
            "details": {},
        }
    if tool == "press_key":
        status, narration, error = _do_press_key(page, args)
        return {
            "status": status,
            "narration": narration,
            "error_message": error,
            "extracted_text": "",
            "details": {"key": args.get("key", "")},
        }
    if tool == "go_back":
        status, narration, error = _do_go_back(page)
        return {
            "status": status,
            "narration": narration,
            "error_message": error,
            "extracted_text": "",
            "details": {},
        }

    # Phase F — bundled fill_form. Intercepts BEFORE the generic
    # dispatcher because it isn't a single primitive — it's an
    # orchestrated routine with enumerate / per-widget fill /
    # validation retry / submit.
    if tool == "fill_form":
        return _dispatch_fill_form(
            page,
            args=args,
            emit_event=emit_event,
            submodule_run_id=submodule_run_id,
            submodule_step_id=submodule_step_id,
            turn_idx=turn_idx,
        )

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

    # Phase A.6 Step 2 — pre-click drawer scroll. When the agent is
    # about to click a submit-like button (Save / Create / Submit /
    # Confirm), proactively scroll the active drawer to the bottom
    # so the button is in view. No-op when no drawer exists. Cheap
    # (one JS evaluate), eliminates the common "filled the form but
    # never found Save because it was below the fold" failure mode.
    if action_type == "click" and ctx.target_hint:
        hint_lower = (ctx.target_hint or "").lower()
        if any(s in hint_lower for s in (
            "save", "create", "submit", "confirm", "update",
            "add ", "add ", "publish", "apply",
        )):
            try:
                from app.executor.actions import (  # noqa: PLC0415
                    scroll_drawer_to_bottom,
                )
                scroll_drawer_to_bottom(page)
            except Exception:
                pass
            # Phase A.6 Step 3 — empty-required-field safety net.
            # Scan the visible form for [required] / [aria-required]
            # fields that are still empty. If any → DON'T dispatch
            # the submit; return an outcome that tells the agent
            # which fields to fill. Catches "scout missed a field"
            # AND "agent skipped a field" in one shot.
            try:
                from app.agents.form_signals import (  # noqa: PLC0415
                    find_empty_required_fields,
                )
                missing = find_empty_required_fields(page)
            except Exception:
                missing = []
            if missing:
                missing_str = [
                    f"{m.label} ({m.role})" for m in missing[:10]
                ]
                return {
                    "status": "failed",
                    "narration": (
                        "Pre-submit check blocked the click: "
                        f"{len(missing_str)} required field(s) still "
                        "empty — " + ", ".join(missing_str[:6])
                    ),
                    "error_message": (
                        "REQUIRED FIELDS EMPTY: "
                        + ", ".join(missing_str)
                        + " — fill these first, then resubmit. Do "
                        "NOT mark the sub-goal done."
                    ),
                    "extracted_text": "",
                    "details": {
                        "pre_submit_check": "blocked",
                        "missing_required": missing_str,
                    },
                }

    try:
        result = execute_action(page, action_type, ctx)
    except Exception as e:
        return {
            "status": "failed",
            "narration": "dispatcher raised",
            "error_message": f"{type(e).__name__}: {e}",
            "extracted_text": "",
            "details": {},
        }

    # Phase 0.10 — type-and-submit. When the agent set ``submit: true``
    # on a ``type`` call AND the type itself succeeded, fire Enter on
    # the same field so the form actually submits. Folding the press
    # into the same tool call avoids the agent's "now find the
    # submit button" misadventure (Amazon's submit button is hard
    # to locate by text). A failed Enter is a soft warning, not a
    # type failure — the typed value did make it into the field.
    submit_after = bool(args.get("submit")) and tool == "type"
    submit_warning: str | None = None
    if submit_after and result.status == "passed":
        try:
            page.keyboard.press("Enter")
        except Exception as e:
            submit_warning = f"Enter dispatch failed: {type(e).__name__}: {e}"

    narration = result.narration
    if submit_after and result.status == "passed":
        narration = (
            f"{narration}; pressed Enter to submit"
            if submit_warning is None
            else f"{narration}; submit-Enter softly failed ({submit_warning})"
        )

    return {
        "status": "ok" if result.status == "passed" else (
            "blocked" if result.status == "blocked" else "failed"
        ),
        "narration": narration,
        "error_message": result.error_message,
        "extracted_text": "",
        "details": result.details,
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
    # Plan-scoped page memory: when the orchestrator passes one in,
    # observations on URLs visited by EARLIER submodules can be
    # served from the cached note instead of re-parsing the AX tree.
    # When None, a fresh local dict is used (legacy single-submodule
    # behavior preserved). Mutated in place — caller keeps the dict
    # alive across submodules.
    page_memory: dict[str, dict[str, Any]] | None = None,
    # Phase 1 — provider tiering. ``provider`` is the STRONG model
    # (always used for the per-turn planner call). ``cheap_provider``
    # is the cheap-tier model used as the first-attempt for VL
    # helpers (vision-search, on-track, goal-verify, smart-pick,
    # semantic-verify). When None, every helper routes to ``provider``
    # and tiering is disabled — legacy single-model behavior.
    cheap_provider: LLMProvider | None = None,
    # Phase 6 — agent strategy. ``hybrid`` (default) keeps the
    # DOM-first ladder. ``vision_only`` routes click / type through
    # VL+coords directly, bypassing DOM resolution. Used for apps
    # where DOM resolution can't reach (heavy canvas, sealed shadow
    # DOM, etc.).
    agent_strategy: str = "hybrid",
    # Production-α.5/6 — plan-scoped WorldState + target_url for
    # AKB lookup. WorldState is mutated in place; the orchestrator
    # persists it across submodule boundaries. AKB context is read
    # ONCE at submodule start and rendered into the prompt block
    # so the agent has business / app knowledge available.
    world_state: dict[str, Any] | None = None,
    db: "Session | None" = None,
    # Auth-flow plumbing — when these are wired AND the page looks like
    # a login screen at submodule start, ``run_auth_loop`` runs once
    # before the agent's main turn loop. ``plan`` carries the vault
    # credentials; ``open_typed_prompt`` opens the HITL popup when the
    # vault misses or the OTP needs human entry; ``request_intervention``
    # blocks until the popup answers. All four are optional — the agent
    # falls back to its own ``ask_human`` tool when they're missing.
    plan: "TestPlan | None" = None,
    open_typed_prompt: Callable[..., None] | None = None,
    request_intervention: Callable[[int], dict | None] | None = None,
    # Phase A — vision-driven sub-goal layer. When ``plan`` is given,
    # we call ``decompose_goal`` once at submodule start using the
    # current screen, replacing the BRD-time text-derived sub-goals
    # with VL-derived ones anchored to actual UI. Replan budget comes
    # from ``plan.max_replans_per_submodule`` (default 2).
    # ``som_enabled`` toggles Set-of-Mark annotation on screenshots
    # sent to ALL helper VL calls; defaults from ``app_settings``.
    som_enabled: bool = True,
    # Phase A.6 Step 6 — plan-wide submodule summary so the
    # reconciliation pass (which runs ONCE after first-time scout)
    # can compare the AppMap against EVERY submodule, not just the
    # current one. List of ``{submodule_id, title, description}``.
    # Passed in by ``run_qa_agent_for_plan``; None disables
    # reconciliation.
    plan_submodules_summary: list[dict[str, Any]] | None = None,
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

    # α.5 — local WorldState alias. Default to empty dict when caller
    # didn't supply one (legacy / single-submodule-test usage).
    if world_state is None:
        world_state = {}

    # α.6 — query AKB for chunks relevant to this submodule's goal.
    # Best-effort: AKB is empty on a fresh app or when ingestion
    # was skipped. The block ends up in ``akb_block`` for the prompt.
    #
    # β.2 — pattern-pack autoload. If the AKB has zero pattern_rule
    # chunks for this target, we run the pack detector once. Cheap
    # (URL/DOM heuristics + dedup'd writes); skipped on subsequent
    # submodules because the second query sees the rules are
    # already there.
    akb_chunks: list[Any] = []
    if db is not None:
        try:
            from app.services.akb import query_akb  # noqa: PLC0415

            # β.2 — try pack autoload via URL hint first; the DOM-
            # signature path runs after the page is loaded (in the
            # turn-1 query below). One call → idempotent — re-runs
            # are no-ops thanks to AKB write-chunk dedup.
            try:
                from app.agents.patterns import (  # noqa: PLC0415
                    autoload_pack,
                )
                autoload_pack(db, target_url=plan_target_url, page=page)
            except Exception as e:
                logger.debug(
                    "pattern pack autoload skipped: %s", e,
                )

            query_text = (
                f"{goal.description}\n"
                + " | ".join(goal.success_criteria[:5])
            )
            akb_chunks = query_akb(
                db,
                target_url=plan_target_url,
                query=query_text,
                k=6,
            )
            if akb_chunks:
                _emit(emit_event, "akb_recall", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "count": len(akb_chunks),
                    "kinds": sorted({c.kind for c in akb_chunks}),
                })
        except Exception as e:
            logger.debug("AKB query skipped: %s", e)
            akb_chunks = []

    # α.5 — assert preconditions. When the goal carries explicit
    # preconditions and the WorldState shows them violated, the
    # agent is informed in its prompt; the deterministic checker
    # ALSO produces a list of unsatisfied preconditions surfaced
    # in the live feed. The agent is encouraged (via system prompt)
    # to flag_test_case_issue with kind=precondition_failed when
    # the violation is structural (e.g., cart empty for a "remove
    # items" submodule).
    from app.services.world_state import (  # noqa: PLC0415
        check_preconditions, format_for_prompt as _ws_format,
    )
    pre_ok, unmet_preconditions = check_preconditions(
        world_state, goal.preconditions,
    )
    if not pre_ok:
        _emit(emit_event, "preconditions_unmet", {
            "run_id": submodule_run_id,
            "step_id": submodule_step_id,
            "unmet": unmet_preconditions[:5],
        })
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
    # Page memory — keyed by URL, each entry is the agent's own 1-2
    # line "what's on this page" note from a previous turn. Future
    # turns on the same URL read the cached note instead of re-
    # parsing the AX tree (Atlas/Comet site-map pattern). Cuts
    # observation tokens by ~80% on multi-page flows where the
    # agent revisits pages.
    #
    # When the orchestrator passes a dict in, it's shared across
    # submodules — entries written during submodule N stay readable
    # for submodule N+1, so e.g. a login screen catalogued during
    # the auth submodule isn't re-scraped during a follow-up flow.
    # Default: a fresh local dict (legacy single-submodule scope).
    if page_memory is None:
        page_memory = {}

    # Phase 12 — page graph. Edge-list of URL transitions observed
    # during this submodule. Each edge is keyed by ``from_url`` and
    # carries the agent's tool that caused the transition (navigate,
    # click, go_back, etc.) plus the destination URL. Future turns
    # use this to back-navigate cheaply (look up "how did I get to
    # page X from Y?") without re-deriving the path. The structure
    # is intentionally tiny — keys/values are URLs and short tool
    # names, never full content. Survives the submodule via
    # plan-scoped page_memory's "_graph" sidecar.
    graph_key = "_graph"
    if graph_key not in page_memory:
        page_memory[graph_key] = {"edges": [], "visited_urls": []}
    page_graph = page_memory[graph_key]

    # Phase 1 — escalation emitter. Wires the router's on_escalate
    # callback to the live-feed event stream so the user can watch
    # tier transitions in real time. Closure captures emit_event +
    # the run/step context so the emit signature matches what the
    # presenter expects.
    def _emit_escalation(role_name: str, from_model: str, reason: str) -> None:
        _emit(emit_event, "llm_escalated", {
            "run_id": submodule_run_id,
            "step_id": submodule_step_id,
            "role": role_name,
            "from_model": from_model,
            "reason": (reason or "")[:200],
        })

    halt_reason: HaltReason = "max_turns"
    final_status: Literal["passed", "failed", "blocked", "inconclusive"] = (
        "inconclusive"
    )
    final_narration = ""
    final_error: str | None = None

    # ── Auth-flow pre-run ─────────────────────────────────────────────
    # Before the main observe-think-act loop, if the page looks like a
    # login / OTP / captcha screen AND we have the HITL channel + a
    # plan (for vault credentials), let the dedicated auth orchestrator
    # drive the page to a logged-in state. This bypasses the agent's
    # DOM-first ladder for the entire auth surface — coord-typing only,
    # clear-before-type, error-driven retry, OTP via TOTP-or-HITL.
    #
    # Skipped on subsequent submodules (the page is already logged in,
    # detect_auth_fields returns kind="success" or "unknown" with low
    # confidence in <1 iteration, so this stays cheap even when called
    # speculatively). When skipped or unsuccessful, the agent's main
    # loop takes over — auth-flow failure is non-fatal.
    auth_used_manual_intervention = False
    # Phase F.1 — HITL defensive counters. ONE overlay per submodule;
    # any further stall after the user already gave guidance halts
    # the submodule cleanly with halt_reason="hitl_exhausted". Avoids
    # the "user submits, page doesn't move, HITL re-opens, user
    # confused, run looks frozen" loop.
    hitl_attempts_this_submodule = 0
    HITL_MAX_PER_SUBMODULE = 1
    # Track how many turns since the last HITL submission. If 3
    # consecutive post-HITL turns pick no-op tools (wait/verify/
    # extract_text) with no page change, we halt with
    # halt_reason="planner_no_op_after_hitl" rather than letting the
    # operator think the run is frozen.
    turns_since_hitl_consumed: int | None = None
    POST_HITL_NOOP_WINDOW = 3
    _POST_HITL_NOOP_TOOLS = frozenset({
        "wait", "verify", "extract_text",
    })
    post_hitl_noop_streak = 0
    # Phase I.1 — universal "action with no observable effect" guard.
    # The existing post-HITL detector only catches passive tools
    # (wait / verify / extract_text); it misses the worst freeze mode
    # where the agent fires action tools (click / type / fill_form)
    # whose status="ok" but the page hash didn't advance — e.g. the
    # agent clicks the form title "Create Role" thinking it's the
    # Save button, every turn, forever. Triggers a clean halt after
    # ``ACTION_NO_EFFECT_THRESHOLD`` consecutive such turns.
    ACTION_NO_EFFECT_THRESHOLD = 4
    action_no_effect_streak = 0
    last_obs_hash_after_action: str | None = None
    _ACTION_TOOLS_FOR_PROGRESS = frozenset({
        "click", "type", "select", "fill_form", "scroll",
        "navigate", "go_back", "dismiss_modal",
    })
    # Phase I.1 — tighter post-HITL obs-hash gate. The agent's planner
    # can pick action tools repeatedly after HITL and still NOT move
    # the page (clicking the wrong thing because of the same wrong
    # belief). If ``POST_HITL_OBS_UNCHANGED_LIMIT`` consecutive turns
    # post-HITL produce the same observation hash, halt — regardless
    # of which tool was picked.
    POST_HITL_OBS_UNCHANGED_LIMIT = 3
    post_hitl_unchanged_turns = 0
    obs_hash_at_hitl_submission: str | None = None
    # Phase O.3 — confidence + mistake memory.
    # The planner emits a self-reported confidence per turn. When 2
    # consecutive non-HITL turns produce confidence < LOW_CONFIDENCE
    # AND no observable progress (no sub-goal closed since the streak
    # started), halt for HITL escalation INSTEAD of burning the rest
    # of max_turns. The mistake memory captures (target_hint →
    # failure_reason) so the next planner turn sees a "don't retry"
    # block and picks a different approach.
    LOW_CONFIDENCE = 0.5
    low_confidence_streak = 0
    sub_goals_closed_at_streak_start = 0
    failed_approaches: list[dict[str, str]] = []
    MAX_FAILED_APPROACHES_REMEMBERED = 8
    if (
        plan is not None
        and request_intervention is not None
        and open_typed_prompt is not None
        and _looks_like_login_page(page)
    ):
        try:
            from app.agents.auth_flow import (  # noqa: PLC0415
                run_auth_loop,
            )
            auth_result = run_auth_loop(
                page,
                plan=plan,
                provider=provider,
                cheap_provider=cheap_provider,
                submodule_run_id=submodule_run_id,
                submodule_step_id=submodule_step_id,
                emit_event=emit_event,
                on_escalate=_emit_escalation,
                open_typed_prompt=open_typed_prompt,
                request_intervention=request_intervention,
                is_cancelled=is_cancelled,
            )
            # Fold auth-flow LLM cost into the submodule's totals so
            # the cost meter and per-step drilldown stay accurate.
            if auth_result.input_tokens:
                total_input += auth_result.input_tokens
            if auth_result.output_tokens:
                total_output += auth_result.output_tokens
            if auth_result.vision_calls:
                vision_calls += auth_result.vision_calls
                llm_calls += auth_result.vision_calls
            if auth_result.manual_intervention_used:
                auth_used_manual_intervention = True
            _emit(emit_event, "auth_flow_completed", {
                "run_id": submodule_run_id,
                "step_id": submodule_step_id,
                "status": auth_result.status,
                "iterations": auth_result.iterations,
                "screens_seen": auth_result.screens_seen,
                "manual_intervention_used": (
                    auth_result.manual_intervention_used
                ),
                "error_message": auth_result.error_message,
            })
            # Phase E — when auth succeeded, record the identity onto
            # WorldState so subsequent submodules see "auth_status:
            # logged_in" and don't waste turns trying to re-auth.
            if auth_result.status == "ok" and plan is not None:
                try:
                    from app.services.world_state import (  # noqa: PLC0415
                        record_auth_success,
                    )
                    # Pick the credential we actually used. Same
                    # resolution as auth_flow._pick_credential.
                    from app.agents.auth_flow import (  # noqa: PLC0415
                        _pick_credential,
                    )
                    cred = _pick_credential(plan, page.url if page else "")
                    if cred is not None:
                        from app.security.vault import (  # noqa: PLC0415
                            VaultError, read_credential,
                        )
                        try:
                            cred_plain = read_credential(cred)
                            record_auth_success(
                                world_state,
                                username=cred_plain.username,
                            )
                        except VaultError:
                            # Vault decrypt failed — fall back to the
                            # label so we still record SOMETHING.
                            record_auth_success(
                                world_state,
                                username=cred.label or "unknown",
                            )
                except Exception as e:
                    logger.debug(
                        "WorldState auth update skipped: %s", e,
                    )
            if auth_result.status == "cancelled":
                halt_reason = "cancelled"
                final_status = "blocked"
                final_narration = "Cancelled during auth flow"
                # Fall through to return below (turn loop won't run
                # because turn_idx never gets bound; jump via early
                # return).
                duration_ms = int((time.monotonic() - t0) * 1000)
                return AgentSubmoduleResult(
                    submodule_id=goal.submodule_id,
                    status=final_status,
                    halt_reason=halt_reason,
                    turn_log=turn_log,
                    final_narration=final_narration,
                    error_message=auth_result.error_message,
                    input_tokens=total_input,
                    output_tokens=total_output,
                    llm_calls=llm_calls,
                    vision_calls=vision_calls,
                    duration_ms=duration_ms,
                    manual_intervention_used=auth_used_manual_intervention,
                )
        except Exception as e:
            logger.warning(
                "auth_flow pre-run failed (%s); falling back to main "
                "agent loop", e,
            )

    # ── Phase A.5: AppMap (mindmap) load + first-time scout ──────────
    # Try to load a previously-saved AppMap for this target_url. When
    # absent (first run against this app), inline-scout the post-auth
    # surface to build one. The map is then passed to the decomposer
    # as ground truth — real button labels, real form fields, real
    # navigation. Subsequent submodules in the same run reuse the same
    # map; subsequent RUNS against the same target_url reuse it via
    # the AKB persistence layer.
    app_map_for_decomposer: "Any | None" = None
    if plan is not None and db is not None:
        try:
            from app.agents.app_map import (  # noqa: PLC0415
                load_app_map, save_app_map, consolidate_app_map,
            )
            from app.agents.authenticated_scout import (  # noqa: PLC0415
                run_authenticated_scout,
            )
            target = plan_target_url or (
                getattr(plan, "target_url", "") or ""
            )
            existing_map = load_app_map(db, target_url=target)
            if existing_map is not None:
                app_map_for_decomposer = existing_map
                _emit(emit_event, "app_map_loaded", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "modules": len(existing_map.modules),
                    "create_flows": len(existing_map.create_flows),
                    "pages_scouted": existing_map.pages_scouted,
                })
            else:
                # First-time scouting for this target_url. Inline so
                # the decomposer sees the map on this submodule; the
                # UX overhead is ~30-90s on the FIRST submodule only.
                # Skipped when we don't have a logged-in browser
                # (no auth_flow ran or page is still on a login URL).
                page_looks_logged_in = not _looks_like_login_page(page)
                if page_looks_logged_in:
                    _emit(emit_event, "app_map_scout_started", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "target_url": target,
                    })
                    scout = run_authenticated_scout(
                        page,
                        target_url=target,
                        depth="deep",
                        emit_event=emit_event,
                        is_cancelled=is_cancelled,
                        submodule_run_id=submodule_run_id,
                    )
                    if scout.error_message:
                        logger.info(
                            "auth scout partial: %s", scout.error_message,
                        )
                    if scout.pages:
                        new_map, in_tok, out_tok = consolidate_app_map(
                            provider,
                            scout_result=scout,
                            cheap_provider=cheap_provider,
                            on_escalate=_emit_escalation,
                        )
                        if in_tok:
                            total_input += in_tok
                        if out_tok:
                            total_output += out_tok
                        try:
                            save_app_map(
                                db,
                                target_url=target,
                                app_map=new_map,
                                source_run_id=submodule_run_id,
                            )
                        except Exception as e:
                            logger.warning(
                                "app_map save failed (non-fatal): %s", e,
                            )
                        app_map_for_decomposer = new_map
                        _emit(emit_event, "app_map_built", {
                            "run_id": submodule_run_id,
                            "step_id": submodule_step_id,
                            "modules": len(new_map.modules),
                            "create_flows": len(new_map.create_flows),
                            "pages_scouted": new_map.pages_scouted,
                        })
                        # Phase A.6 Step 6 — plan ↔ AppMap reconciliation.
                        # ONE VL call that compares every submodule
                        # against the freshly-built map and tags each
                        # as ok / uncertain / mismatch / missing.
                        # User sees the report BEFORE execution
                        # progresses past submodule 1.
                        if plan_submodules_summary:
                            try:
                                from app.agents.app_map import (  # noqa: PLC0415
                                    reconcile_plan_with_map,
                                )
                                rows, rin, rout = reconcile_plan_with_map(
                                    provider,
                                    app_map=new_map,
                                    submodules=plan_submodules_summary,
                                    cheap_provider=cheap_provider,
                                    on_escalate=_emit_escalation,
                                )
                                if rin:
                                    total_input += rin
                                if rout:
                                    total_output += rout
                                if rows:
                                    by_status: dict[str, int] = {}
                                    for r in rows:
                                        by_status[r.status] = (
                                            by_status.get(r.status, 0) + 1
                                        )
                                    _emit(emit_event, "plan_reconciled", {
                                        "run_id": submodule_run_id,
                                        "step_id": submodule_step_id,
                                        "counts": by_status,
                                        "rows": [
                                            {
                                                "submodule_id": r.submodule_id,
                                                "title": r.title,
                                                "status": r.status,
                                                "reason": r.reason,
                                                "matched_module": r.matched_module,
                                                "matched_create_flow": (
                                                    r.matched_create_flow
                                                ),
                                            }
                                            for r in rows
                                        ],
                                    })
                            except Exception as e:
                                logger.debug(
                                    "reconciliation skipped: %s", e,
                                )
        except Exception as e:
            logger.warning(
                "AppMap load / scout failed (%s); decomposer falls "
                "back to no-map mode", e,
            )

    # ── Phase A: vision-driven sub-goal decomposition ────────────────
    # Replace the BRD-time text-derived sub-goals on ``goal.sub_goals``
    # with VL-derived ones anchored to the ACTUAL UI. The agent's
    # existing planner already consumes ``goal.sub_goals`` and the
    # tool schema has ``current_sub_goal_id`` / ``sub_goal_completed_id``
    # fields — so we don't need to change the inner loop. We just swap
    # the list and track the runtime metadata (replan iteration,
    # failure reason) in a parallel ``runtime_sub_goals`` list that
    # gets persisted into ``details_json``.
    from app.agents.goal import SubGoal as _StaticSubGoal  # noqa: PLC0415
    from app.agents.sub_goals import (  # noqa: PLC0415
        RuntimeSubGoal, decompose_goal, replan_sub_goals,
    )
    runtime_sub_goals: list[RuntimeSubGoal] = []
    replans_used = 0
    max_replans = 2
    if plan is not None:
        try:
            max_replans = int(
                getattr(plan, "max_replans_per_submodule", 2),
            )
        except (TypeError, ValueError):
            max_replans = 2
    max_replans = max(0, min(5, max_replans))

    if plan is not None:
        # Capture the screen now so the decomposer sees what the agent
        # will see on turn 1. Single VL call; cost is logged into the
        # submodule's totals like any other helper call.
        try:
            from app.agents.page_intel import (  # noqa: PLC0415
                capture_screenshot_for_vision,
            )
            decomp_shot = capture_screenshot_for_vision(page)
        except Exception as e:
            logger.debug("decomposer screenshot capture failed: %s", e)
            decomp_shot = None

        akb_text = ""
        try:
            akb_text = "\n".join(
                f"- {getattr(c, 'text', '')[:240]}" for c in akb_chunks[:5]
            )
        except Exception:
            akb_text = ""

        # Phase B Step 3 — load frozen v2 segments as decomposer hints.
        # When the submodule has a prior-run frozen path with the
        # ``segments`` shape, we hand it to the decomposer as
        # "re-emit this breakdown if the screen still supports it".
        # The replay walker (Step 4) checks the resulting fresh
        # sub-goals against the frozen segments and walks the
        # proven steps deterministically when they line up.
        frozen_hint: list[dict[str, Any]] | None = None
        try:
            tc_row = None
            if db is not None:
                tc_row = db.get(TcNode, goal.submodule_id)
            existing_frozen = (
                getattr(tc_row, "frozen_path", None) if tc_row else None
            )
            if (
                isinstance(existing_frozen, dict)
                and existing_frozen.get("version") == 2
                and isinstance(existing_frozen.get("segments"), list)
            ):
                frozen_hint = [
                    {
                        "description": seg.get("description", ""),
                        "success_criterion": seg.get(
                            "success_criterion", "",
                        ),
                        "max_turns": seg.get("max_turns", 6),
                    }
                    for seg in existing_frozen["segments"]
                    if isinstance(seg, dict)
                ]
        except Exception as e:
            logger.debug("frozen-hint load skipped: %s", e)
            frozen_hint = None

        decomp = decompose_goal(
            provider,
            goal_description=goal.description,
            goal_success_criteria=list(goal.success_criteria or []),
            screenshot_bytes=decomp_shot,
            akb_block=akb_text,
            app_map=app_map_for_decomposer,
            frozen_sub_goals_hint=frozen_hint,
            world_state=world_state,
            cheap_provider=cheap_provider,
            on_escalate=_emit_escalation,
        )
        if decomp.input_tokens:
            total_input += decomp.input_tokens
        if decomp.output_tokens:
            total_output += decomp.output_tokens
        if decomp.sub_goals:
            # Phase A.5 — hardcoded create→verify guarantor. Catches
            # cases where the decomposer forgot rule 7g (verify-in-list
            # for create flows). Deterministic; ~free.
            from app.agents.sub_goals import (  # noqa: PLC0415
                ensure_create_verify_pattern,
            )
            augmented, appended_verify = ensure_create_verify_pattern(
                decomp.sub_goals,
                goal_description=goal.description,
                app_map=app_map_for_decomposer,
            )
            runtime_sub_goals = augmented
            if appended_verify:
                _emit(emit_event, "sub_goal_verify_appended", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "reason": "decomposer omitted verify-in-list step",
                })
            # Mirror into the static SubGoal shape the existing planner
            # / on-track / freeze code expects.
            goal.sub_goals = [
                _StaticSubGoal(
                    id=rsg.id,
                    description=rsg.description,
                    status="pending",
                    completed_at_turn=None,
                )
                for rsg in runtime_sub_goals
            ]
            _emit(emit_event, "sub_goals_decomposed", {
                "run_id": submodule_run_id,
                "step_id": submodule_step_id,
                "count": len(runtime_sub_goals),
                "sub_goals": [
                    {"id": rsg.id, "description": rsg.description[:200]}
                    for rsg in runtime_sub_goals
                ],
            })
        elif decomp.error_message:
            logger.info(
                "sub-goal decomposition skipped: %s — falling back to "
                "BRD-time sub-goals", decomp.error_message,
            )

    def _mirror_runtime_status() -> None:
        """Copy goal.sub_goals[].status into runtime_sub_goals[] and
        emit transition events for any newly completed / failed /
        skipped sub-goal since the last call."""
        for i, sg in enumerate(goal.sub_goals or []):
            if i >= len(runtime_sub_goals):
                continue
            rsg = runtime_sub_goals[i]
            new_status = sg.status
            if rsg.status == new_status:
                continue
            old = rsg.status
            rsg.status = new_status  # type: ignore[assignment]
            if new_status == "in_progress":
                rsg.started_at_turn = sg.completed_at_turn or 0
                _emit(emit_event, "sub_goal_started", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "id": rsg.id,
                    "description": rsg.description[:200],
                })
            elif new_status in ("done", "failed", "skipped"):
                rsg.ended_at_turn = sg.completed_at_turn
                _emit(emit_event, f"sub_goal_{new_status}", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "id": rsg.id,
                    "description": rsg.description[:200],
                    "reason": rsg.reason,
                    "from_status": old,
                })

    for turn_idx in range(1, max_turns + 1):
        # Phase F.1 — per-turn heartbeat so the live feed never
        # appears dead even on long-running observation / planner
        # calls. Cheap event; no payload beyond turn index.
        _emit(emit_event, "agent_turn_starting", {
            "run_id": submodule_run_id,
            "step_id": submodule_step_id,
            "turn": turn_idx,
            "max_turns": max_turns,
        })

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

        # Phase E — keep WorldState's current_url in sync each turn
        # so the decomposer + planner see live location without an
        # extra page round-trip. Cheap; no LLM.
        try:
            from app.services.world_state import (  # noqa: PLC0415
                record_current_page,
            )
            record_current_page(
                world_state,
                url=observation.get("url"),
            )
        except Exception:
            pass

        # Phase I.1 — action-no-effect guard. Fires BEFORE the regular
        # stall guard so we halt the burning-LLM-loop much earlier
        # than the stall_threshold lets the existing path catch.
        # Trigger: the previous turn picked an action tool, returned
        # status="ok", but the observation hash didn't advance — i.e.
        # the agent thinks it's clicking the right thing but the page
        # doesn't agree.
        if turn_log:
            _last_t = turn_log[-1]
            _was_action = _last_t.tool in _ACTION_TOOLS_FOR_PROGRESS
            _ok = (_last_t.status == "ok")
            if _was_action and _ok:
                if (
                    last_obs_hash_after_action is not None
                    and obs_hash == last_obs_hash_after_action
                ):
                    action_no_effect_streak += 1
                else:
                    action_no_effect_streak = 0
                last_obs_hash_after_action = obs_hash
        if action_no_effect_streak >= ACTION_NO_EFFECT_THRESHOLD:
            _emit(emit_event, "agent_action_no_effect_halt", {
                "run_id": submodule_run_id,
                "step_id": submodule_step_id,
                "turn": turn_idx,
                "streak": action_no_effect_streak,
                "last_tool": (
                    turn_log[-1].tool if turn_log else "(none)"
                ),
                "last_target": (
                    (turn_log[-1].args.get("target_hint") or "")[:120]
                    if turn_log else ""
                ),
            })
            halt_reason = "agent_failed"
            final_status = "blocked"
            final_narration = (
                f"Agent fired {action_no_effect_streak} consecutive "
                "actions with no observable page change — halting "
                "to avoid burning the LLM budget. Likely cause: the "
                "planner is clicking a non-interactive label (e.g. "
                "form title) thinking it's an action button."
            )
            final_error = "actions_no_effect"
            break

        # Phase I.1 — tighter post-HITL obs-hash gate. Independent of
        # the no-op-tool detector below; this one catches "agent picks
        # actions after HITL but page doesn't move".
        if obs_hash_at_hitl_submission is not None:
            if obs_hash == obs_hash_at_hitl_submission:
                post_hitl_unchanged_turns += 1
            else:
                obs_hash_at_hitl_submission = None
                post_hitl_unchanged_turns = 0
            if post_hitl_unchanged_turns >= POST_HITL_OBS_UNCHANGED_LIMIT:
                _emit(emit_event, "post_hitl_no_progress_halt", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "turn": turn_idx,
                    "unchanged_turns": post_hitl_unchanged_turns,
                })
                halt_reason = "agent_failed"
                final_status = "blocked"
                final_narration = (
                    f"After HITL guidance, page state has not moved "
                    f"for {post_hitl_unchanged_turns} consecutive "
                    "turns — halting. The human's input did not "
                    "produce observable progress."
                )
                final_error = "post_hitl_no_progress"
                break

        # Stall guard: if last `stall_threshold` observations are
        # identical AND we already took at least one action (so we're
        # not just sitting on the initial page), halt.
        obs_hashes.append(obs_hash)
        if (
            len(obs_hashes) == stall_threshold
            and len(set(obs_hashes)) == 1
            and turn_idx > stall_threshold
        ):
            # Phase A — before giving up on stall, try replanning the
            # remaining sub-goals from the current screen. Bounded by
            # ``max_replans`` from the plan. The next observation hash
            # also gets reset (the replan resets the cooldown).
            replanned = False
            if (
                runtime_sub_goals
                and plan is not None
                and replans_used < max_replans
            ):
                # Find the current sub-goal (first non-done / non-skipped)
                cur_idx = next(
                    (i for i, rsg in enumerate(runtime_sub_goals)
                     if rsg.status not in ("done", "skipped")),
                    None,
                )
                if cur_idx is not None:
                    failed_rsg = runtime_sub_goals[cur_idx]
                    failed_rsg.status = "failed"
                    failed_rsg.reason = (
                        f"page unchanged for {stall_threshold} turns"
                    )
                    failed_rsg.ended_at_turn = turn_idx
                    _emit(emit_event, "sub_goal_failed", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "id": failed_rsg.id,
                        "description": failed_rsg.description[:200],
                        "reason": failed_rsg.reason,
                    })
                    try:
                        from app.agents.page_intel import (  # noqa: PLC0415
                            capture_screenshot_for_vision as _cap,
                        )
                        replan_shot = _cap(page)
                    except Exception:
                        replan_shot = None

                    _emit(emit_event, "sub_goal_replan_started", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "iteration": replans_used + 1,
                        "max_replans": max_replans,
                        "after_sub_goal": failed_rsg.id,
                    })
                    rp = replan_sub_goals(
                        provider,
                        goal_description=goal.description,
                        completed_sub_goals=[
                            rsg for rsg in runtime_sub_goals[:cur_idx]
                            if rsg.status == "done"
                        ],
                        failed_sub_goal=failed_rsg,
                        failure_reason=failed_rsg.reason,
                        screenshot_bytes=replan_shot,
                        cheap_provider=cheap_provider,
                        on_escalate=_emit_escalation,
                        replan_iteration=replans_used + 1,
                        app_map=app_map_for_decomposer,
                        world_state=world_state,
                    )
                    if rp.input_tokens:
                        total_input += rp.input_tokens
                    if rp.output_tokens:
                        total_output += rp.output_tokens
                    if rp.sub_goals:
                        replans_used += 1
                        # Keep the completed ones, replace the rest.
                        keep = runtime_sub_goals[:cur_idx]
                        # Renumber the new ones so ids are unique within
                        # the submodule (sg<N>r<i>).
                        new_runtime: list[RuntimeSubGoal] = []
                        for j, new_sg in enumerate(rp.sub_goals, start=1):
                            new_sg.id = f"sg{cur_idx + j}r{replans_used}"
                            new_runtime.append(new_sg)
                        runtime_sub_goals = keep + new_runtime
                        goal.sub_goals = [
                            _StaticSubGoal(
                                id=rsg.id,
                                description=rsg.description,
                                status=rsg.status if rsg.status == "done" else "pending",
                                completed_at_turn=rsg.ended_at_turn,
                            )
                            for rsg in runtime_sub_goals
                        ]
                        obs_hashes.clear()
                        replanned = True
                        _emit(emit_event, "sub_goals_decomposed", {
                            "run_id": submodule_run_id,
                            "step_id": submodule_step_id,
                            "count": len(new_runtime),
                            "replan_iteration": replans_used,
                            "sub_goals": [
                                {"id": rsg.id, "description": rsg.description[:200]}
                                for rsg in new_runtime
                            ],
                        })

            if not replanned:
                # Phase A.6 Step 0 — when replans are exhausted (or
                # were disabled by plan setting), try the in-test-
                # browser HITL overlay before giving up. The user
                # draws on the screenshot + types guidance → we feed
                # it back as a user_guidance observation and reset
                # the stall counter. ONE overlay attempt per
                # submodule's stall path; if the user skips / times
                # out / further stall occurs after the guidance, we
                # then halt with stall.
                # Phase F.1 — use the counter, not a derived flag from
                # final_error (which gets cleared on successful submit).
                # Caps HITL to ``HITL_MAX_PER_SUBMODULE`` attempts per
                # submodule so a user-submits-page-doesnt-move loop
                # doesn't re-trigger the overlay forever.
                hitl_attempted_already = (
                    hitl_attempts_this_submodule >= HITL_MAX_PER_SUBMODULE
                )
                if (
                    not hitl_attempted_already
                    and runtime_sub_goals
                    and request_intervention is not None
                    and open_typed_prompt is not None
                ):
                    try:
                        from app.executor.hitl_overlay import (  # noqa: PLC0415
                            open_and_wait as _open_hitl_overlay,
                        )
                        from app.agents.page_intel import (  # noqa: PLC0415
                            capture_screenshot_for_vision as _cap_shot,
                        )
                        cur_rsg = next(
                            (rsg for rsg in runtime_sub_goals
                             if rsg.status not in ("done", "skipped")),
                            None,
                        )
                        # Build a "what I tried" summary from the
                        # last 3 turns so the user sees context.
                        tail = turn_log[-3:] if turn_log else []
                        tried_lines = []
                        for t in tail:
                            tried_lines.append(
                                f"T{t.turn} · {t.tool}({(t.args.get('target_hint','') or t.args.get('value',''))[:60]}) "
                                f"→ {t.status}"
                                + (f": {t.error_message[:120]}"
                                   if t.error_message else "")
                            )
                        tried_summary = "\n".join(tried_lines) or "(no turn history)"
                        try:
                            stuck_shot = _cap_shot(page)
                        except Exception:
                            stuck_shot = b""
                        # F.1 — count this attempt before the
                        # blocking call so cancellation mid-wait
                        # still decrements correctly.
                        hitl_attempts_this_submodule += 1
                        _emit(emit_event, "hitl_overlay_opened", {
                            "run_id": submodule_run_id,
                            "step_id": submodule_step_id,
                            "sub_goal": (
                                cur_rsg.description[:200]
                                if cur_rsg else "(unknown)"
                            ),
                            "replans_used": replans_used,
                            "attempt": hitl_attempts_this_submodule,
                            "max_attempts": HITL_MAX_PER_SUBMODULE,
                        })
                        response = _open_hitl_overlay(
                            page,
                            sub_goal_description=(
                                cur_rsg.description if cur_rsg else
                                "Agent is stuck"
                            ),
                            tried_summary=tried_summary,
                            screenshot_png=stuck_shot,
                            idle_skip_seconds=15,
                        )
                        _emit(emit_event, "hitl_overlay_submitted", {
                            "run_id": submodule_run_id,
                            "step_id": submodule_step_id,
                            "status": response.get("status", "error"),
                            "text_preview": (
                                str(response.get("text", ""))[:160]
                            ),
                        })
                        if response.get("status") == "submitted":
                            # Inject the user's guidance into the
                            # next turn as a special observation.
                            user_text = (response.get("text") or "").strip()
                            drawing_b64 = response.get("drawing_b64") or ""
                            pending_user_guidance = {
                                "text": user_text,
                                "drawing_b64": drawing_b64,
                                "sub_goal_id": (
                                    cur_rsg.id if cur_rsg else None
                                ),
                            }
                            # Stash on the page_memory sidecar so the
                            # next turn's prompt-builder finds it
                            # (similar to the diff/screenshot side-
                            # channels already used).
                            page_memory.setdefault("_pending_hitl", []).append(
                                pending_user_guidance,
                            )
                            # Mark the sub-goal we WERE on as
                            # in-progress again, reset obs_hashes,
                            # consume one of the "replan budget"
                            # slots for telemetry, and continue.
                            if cur_rsg is not None:
                                cur_rsg.status = "pending"
                                cur_rsg.reason = (
                                    "human guidance received via "
                                    "HITL overlay; retrying"
                                )
                            obs_hashes.clear()
                            final_error = None  # clear the stall marker
                            # Mark in the result that HITL was used
                            # so freeze gate skips this run.
                            auth_used_manual_intervention = True
                            # Phase F.1 — start the no-op streak
                            # tracker from this turn. If the next
                            # POST_HITL_NOOP_WINDOW turns all pick
                            # wait/verify/extract_text with no page
                            # change, the run halts explicitly with
                            # halt_reason="planner_no_op_after_hitl"
                            # instead of looking frozen.
                            turns_since_hitl_consumed = 0
                            post_hitl_noop_streak = 0
                            # Phase I.1 — capture obs hash at HITL
                            # submission so the obs-hash gate above
                            # can detect "page didn't move post-HITL"
                            # regardless of which tools the agent picks.
                            obs_hash_at_hitl_submission = obs_hash
                            post_hitl_unchanged_turns = 0
                            # Also reset the action-no-effect streak —
                            # the human's input is a fresh start; we
                            # don't want pre-HITL stuckness to count.
                            action_no_effect_streak = 0
                            last_obs_hash_after_action = None
                            # Phase C.1 — wallclock grace. The user
                            # might have spent 10-15s reading + drawing
                            # + typing inside the overlay. Without
                            # this grace the next turn's wallclock
                            # guard can fire immediately ("max_wallclock")
                            # and silently halt the run before the
                            # planner even sees the guidance. Push
                            # the cap out by ``HITL_GRACE_SECONDS``
                            # one-shot per HITL submission.
                            HITL_GRACE_SECONDS = 60
                            max_wallclock_s += HITL_GRACE_SECONDS
                            logger.info(
                                "HITL guidance accepted; "
                                "max_wallclock extended by %ds "
                                "(now %ds)",
                                HITL_GRACE_SECONDS, max_wallclock_s,
                            )
                            # Skip the halt — continue the turn loop.
                            continue
                        elif response.get("status") in (
                            "skipped", "idle_timeout",
                        ):
                            # Phase J.3 — idle/skip escalates the
                            # whole submodule, not just the current
                            # sub-goal. The old behavior ("mark this
                            # sub-goal skipped, advance to the next")
                            # was a re-entry trap: the page hadn't
                            # moved, so the next sub-goal also
                            # couldn't make progress, stall triggered
                            # again, and the user saw the same screen
                            # hang again. Idle = user is gone or
                            # doesn't know; do NOT silently try the
                            # rest of the plan on a screen they
                            # already gave up on.
                            skip_reason = (
                                "user skipped via HITL overlay"
                                if response.get("status") == "skipped"
                                else "auto-skipped after 15s idle"
                            )
                            # Mark the current AND remaining sub-goals
                            # as skipped so the report shows the
                            # whole submodule was abandoned at this
                            # point.
                            for _rsg in runtime_sub_goals:
                                if _rsg.status in ("pending", "in_progress"):
                                    _rsg.status = "skipped"
                                    _rsg.reason = skip_reason
                                    _rsg.ended_at_turn = turn_idx
                            _emit(emit_event, "submodule_abandoned_via_hitl", {
                                "run_id": submodule_run_id,
                                "step_id": submodule_step_id,
                                "reason": skip_reason,
                                "remaining_sub_goals": sum(
                                    1 for r in runtime_sub_goals
                                    if r.status == "skipped"
                                ),
                            })
                            halt_reason = "agent_failed"
                            final_status = "blocked"
                            final_narration = (
                                "Submodule abandoned after HITL "
                                f"{skip_reason}. The screen had not "
                                "moved before HITL fired and no "
                                "guidance was provided; advancing "
                                "to the next submodule instead of "
                                "spinning on the same state."
                            )
                            final_error = "hitl_submodule_abandoned"
                            break
                        else:
                            # error / no-response — fall through to
                            # the stall halt path below.
                            final_error = "hitl_overlay_error"
                    except Exception as e:
                        logger.warning(
                            "HITL overlay invocation failed: %s — "
                            "halting with stall", e,
                        )
                        final_error = f"hitl_overlay_error: {e!s}"[:200]

                halt_reason = "stall"
                final_status = "inconclusive"
                final_narration = (
                    f"Page unchanged for {stall_threshold} consecutive "
                    f"turns; replans={replans_used}/{max_replans}; "
                    "HITL attempted="
                    + ("yes" if final_error and "hitl_overlay" in (
                        final_error or ""
                    ) else "no")
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

        # Phase A.6 Step 1 — pop any pending FORM SIGNAL (toast /
        # inline error from the previous submit) into this turn's
        # prompt so the agent reads "after your Save click, a toast
        # said 'Display Name is required'" before picking the next
        # tool. Consumed once so we don't keep nagging.
        form_signal_block = ""
        pending_signals = page_memory.get("_pending_signals") or []
        if pending_signals:
            sig = pending_signals.pop(0)
            kind_l = str(sig.get("kind") or "")
            msg = str(sig.get("message") or "")[:300]
            fields = sig.get("fields") or []
            kind_label = {
                "toast_error": "ERROR TOAST",
                "toast_warning": "WARNING TOAST",
                "toast_info": "INFO TOAST",
                "toast_success": "SUCCESS TOAST",
                "inline_error": "INLINE FORM ERROR",
                "validation_error": "VALIDATION ERROR",
            }.get(kind_l, "FORM SIGNAL")
            fields_str = (
                f" (fields: {', '.join(fields[:5])})"
                if fields else ""
            )
            form_signal_block = (
                f"\nFORM SIGNAL after your previous submit "
                f"({kind_label}){fields_str}:\n"
                f"  \"{msg}\"\n"
                "Treat this as authoritative — the app rejected or "
                "confirmed the submit. If it's an error, fix the "
                "called-out fields then resubmit. If it's a success "
                "toast, advance to the next sub-goal.\n"
            )

        # Phase A.6 Step 0 — pop any pending HITL guidance from the
        # user's overlay submission into THIS turn's prompt. The
        # guidance is consumed once (popped) so subsequent turns
        # aren't repeatedly told the same thing — if the user wants
        # to guide again, the overlay opens again on the next stall.
        hitl_guidance_block = ""
        # Phase N — capture the popped HITL so the planner-bypass
        # downstream can read it. Set to the dict that was popped or
        # None when no HITL is pending this turn.
        pending_hitl_consumed_this_turn: dict[str, Any] | None = None
        pending_hitl_queue = page_memory.get("_pending_hitl") or []
        if pending_hitl_queue:
            g = pending_hitl_queue.pop(0)
            pending_hitl_consumed_this_turn = g
            txt = (g.get("text") or "").strip()
            if txt:
                hitl_guidance_block = (
                    "\nUSER GUIDANCE (from a human watching the run — "
                    "treat as authoritative; previous attempts stalled "
                    "and the user manually pointed out the next step):\n"
                    f"  {txt}\n"
                )
            # Note: we keep `g["drawing_b64"]` for telemetry but
            # don't re-attach the drawing as an image here — vision-
            # on-demand will surface it next turn if the action
            # fails. (Avoids ballooning every subsequent turn's
            # image payload.)
            if g.get("drawing_b64"):
                hitl_guidance_block += (
                    "  (The user also drew on the screenshot to point "
                    "at the correct element; if you struggle, look at "
                    "the attached screenshot for marks.)\n"
                )
            # Phase C.1 — emit a visible event so the operator KNOWS
            # the guidance reached the planner's prompt for this turn.
            # Closes the "I submitted but nothing happened" feedback
            # gap. The next agent_acted event shows what the planner
            # decided to do with the guidance.
            _emit(emit_event, "hitl_overlay_consumed", {
                "run_id": submodule_run_id,
                "step_id": submodule_step_id,
                "turn": turn_idx,
                "sub_goal_id": g.get("sub_goal_id"),
                "guidance_preview": txt[:240],
                "has_drawing": bool(g.get("drawing_b64")),
            })

        # A4.1c: mid-flow vision check. Every ``on_track_interval``
        # turns, ask a vision LLM whether the agent is still making
        # progress against the goal. Catches the "wandering / wrong
        # page / repeating broken click" patterns that the
        # deterministic guards (stall, oscillation) only detect AFTER
        # they've already cost several turns. Skipped when:
        #  - provider can't see images
        #  - it's still the first few turns (nothing to assess yet)
        #  - all sub-goals are already done (no point checking)
        #
        # Phase B.6 — smart gating. Even at the cadence interval,
        # SKIP the check when the agent is clearly making healthy
        # progress (a sub-goal closed in the last K turns AND the
        # observation hash has been changing). The on-track check
        # is meant to catch "stuck without realising it" — when the
        # checklist is moving and the page is moving, the call adds
        # cost without value. The deterministic stall guard still
        # catches bona fide stuck states.
        on_track_block = ""
        unverified_sub_goals = (
            bool(goal.sub_goals)
            and any(
                sg.status not in ("done", "skipped")
                for sg in goal.sub_goals
            )
        )
        # Healthy-progress signals.
        recent_window = max(3, on_track_interval - 1)
        sub_goal_advanced_recently = any(
            sg.completed_at_turn is not None
            and (turn_idx - sg.completed_at_turn) <= recent_window
            for sg in goal.sub_goals
        )
        page_state_moving = len(set(obs_hashes)) >= 2
        healthy_progress = (
            sub_goal_advanced_recently or page_state_moving
        )
        should_check_on_track = (
            provider_supports_vision
            and on_track_interval > 0
            and turn_idx >= on_track_interval
            and turn_idx % on_track_interval == 0
            and unverified_sub_goals
            and not healthy_progress  # B.6 — skip when moving
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
                    cheap_provider=cheap_provider,
                    on_escalate=_emit_escalation,
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

        # Phase 12 — page graph block. Shows recent URL transitions
        # so the agent can back-navigate via go_back() or by knowing
        # the URL it came from. Especially useful for cascade flows
        # ("search -> product -> back to results -> another product").
        graph_text = _format_page_graph_for_prompt(
            page_graph, current_url,
        )
        graph_block = (
            f"\nNAV GRAPH (recent URL transitions in this submodule; "
            f"use go_back or navigate(url) to return):\n{graph_text}\n"
            if graph_text else ""
        )

        # α.5/6 — WorldState + AKB + preconditions/postconditions/signals
        # blocks. Built once per turn so the agent always sees its
        # plan-level state, app knowledge, and goal contract.
        ws_text = _ws_format(world_state)
        ws_block = (
            f"\nWORLD STATE (carried across submodules in this run; "
            f"use to assert preconditions and update on success):\n"
            f"{ws_text}\n"
            if ws_text else ""
        )

        akb_block = ""
        if akb_chunks and turn_idx == 1:
            # Show AKB ONLY on turn 1 — it's submodule-level context,
            # not per-turn. Subsequent turns use page memory + graph.
            akb_lines: list[str] = [
                "\nKNOWN ABOUT THIS APP (retrieved from BRD / scout / "
                "patterns / past disputes):",
            ]
            for c in akb_chunks[:6]:
                tag_str = (
                    f" [{', '.join(c.tags)}]" if c.tags else ""
                )
                akb_lines.append(
                    f"  - ({c.kind}{tag_str}, conf={c.confidence:.2f}) "
                    f"{c.content[:400]}"
                )
            akb_block = "\n".join(akb_lines) + "\n"

        contract_block = ""
        if goal.preconditions or goal.postconditions or goal.evidence_signals:
            contract_lines: list[str] = ["\nGOAL CONTRACT:"]
            if goal.preconditions:
                contract_lines.append("  Preconditions (must hold BEFORE acting):")
                for p in goal.preconditions:
                    marker = ""
                    if p in unmet_preconditions:
                        marker = "  ⚠ NOT MET — consider flag_test_case_issue(precondition_failed)"
                    contract_lines.append(f"    - {p}{marker}")
            if goal.postconditions:
                contract_lines.append("  Postconditions (must hold AFTER for goal to pass):")
                for p in goal.postconditions:
                    contract_lines.append(f"    - {p}")
            if goal.evidence_signals:
                contract_lines.append(
                    "  Evidence signals (verify N-of-M to confirm; "
                    "majority match = goal achieved):",
                )
                for s in goal.evidence_signals:
                    contract_lines.append(f"    - {s}")
            if goal.alternative_paths:
                contract_lines.append(
                    "  Alternative paths (when primary flow blocked):",
                )
                for ap in goal.alternative_paths:
                    contract_lines.append(f"    - {ap}")
            contract_block = "\n".join(contract_lines) + "\n"

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
            f"{akb_block}"     # α.6 — app knowledge (turn 1 only)
            f"{contract_block}" # α.4 — pre/post/signals/alt-paths
            f"{ws_block}"      # α.5 — WorldState across submodules
            f"{vision_note}"
            f"{diff_block}"
            f"{form_signal_block}"
            f"{hitl_guidance_block}"
            f"{on_track_block}"
            f"{memory_block}"
            f"{graph_block}\n"
            f"{_format_failed_approaches_block(failed_approaches)}"
            f"{_format_goal_for_prompt(goal, turn_idx=turn_idx)}\n\n"
            f"HISTORY (last few turns):\n{_format_history_for_prompt(turn_log)}\n\n"
            f"{obs_block}\n\n"
            f"This is turn {turn_idx}/{max_turns}. Pick ONE tool."
        )

        # Phase N — HITL → direct action dispatch. When the previous
        # turn(s) submitted HITL guidance, BYPASS the planner and turn
        # the human's input directly into one tool call. The user's
        # input is authoritative; we don't ask the LLM to "consider"
        # it among other options.
        hitl_direct_parsed: dict[str, Any] | None = None
        if pending_hitl_consumed_this_turn is not None:
            try:
                # Use the screenshot we already have for this turn's
                # observation — the HITL drawing is interpreted RELATIVE
                # to it.
                from app.agents.page_intel import (  # noqa: PLC0415
                    capture_screenshot_for_vision as _cap_hitl,
                )
                hitl_shot = _cap_hitl(page)
            except Exception:
                hitl_shot = None
            hitl_direct_parsed = _interpret_hitl_as_action(
                pending_hitl=pending_hitl_consumed_this_turn,
                screenshot=hitl_shot,
                provider=provider,
                cheap_provider=cheap_provider,
            )
            if hitl_direct_parsed is not None:
                _emit(emit_event, "hitl_direct_dispatch", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "turn": turn_idx,
                    "tool": hitl_direct_parsed.get("tool"),
                    "target_hint": (
                        hitl_direct_parsed.get("target_hint") or ""
                    )[:120],
                    "confidence": hitl_direct_parsed.get("confidence"),
                    "via_coord": (
                        hitl_direct_parsed.get("_hitl_coord_x") is not None
                    ),
                })
                # Account for the interpreter call's tokens.
                tin = hitl_direct_parsed.pop("_hitl_tokens_in", 0)
                tout = hitl_direct_parsed.pop("_hitl_tokens_out", 0)
                if isinstance(tin, int):
                    total_input += tin
                if isinstance(tout, int):
                    total_output += tout
                llm_calls += 1

        # Phase N — when the HITL bypass produced a tool call, use it
        # directly and skip the planner LLM call entirely. The user's
        # instruction is authoritative; we don't run the planner.
        if hitl_direct_parsed is not None:
            class _HitlResult:
                def __init__(self, parsed):
                    self.parsed = parsed
                    self.input_tokens = 0
                    self.output_tokens = 0
            llm_result = _HitlResult(hitl_direct_parsed)
            # Consume the screenshot — we already used it for the
            # HITL interpreter call above.
            pending_screenshot = None

        try:
            user_msg = ChatMessage(
                role="user",
                content=user_prompt,
                image=pending_screenshot if attach_screenshot else None,
            )
            import time as _planner_time  # noqa: PLC0415
            _planner_t0 = _planner_time.monotonic()
            if hitl_direct_parsed is not None:
                # HITL bypass already populated llm_result above.
                # Skip the planner LLM round-trip.
                pass
            else:
                llm_result = provider.chat_structured(
                    messages=[
                        ChatMessage(role="system", content=SYSTEM_PROMPT),
                        user_msg,
                    ],
                    schema=TOOL_CALL_SCHEMA,
                    schema_name="qa_tool_call",
                    temperature=0.3,
                    # TOOL_CALL_SCHEMA requires ~200-300 output tokens
                    # at the median (tool name + args + reasoning + a
                    # short page_memory_note). 512 leaves comfortable
                    # headroom for verbose reasoning while cutting the
                    # tail-latency / cost ceiling vs the prior 1024.
                    max_output_tokens=512,
                )
                # Cost: planner call goes direct to the strong provider
                # (bypasses the tier router because the planner is
                # always strong-only). Record manually with role +
                # model + duration so the drill-in view shows one
                # ``planner`` row per turn.
                try:
                    from app.llm.cost_tracker import (  # noqa: PLC0415
                        record_call,
                    )
                    record_call(
                        "strong",
                        llm_result.input_tokens,
                        llm_result.output_tokens,
                        role="planner",
                        model=getattr(provider, "model", None),
                        cached_input_tokens=getattr(
                            llm_result, "cached_input_tokens", None,
                        ),
                        duration_ms=int(
                            (_planner_time.monotonic() - _planner_t0) * 1000,
                        ),
                    )
                except Exception:
                    pass
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

        # Phase N — when the HITL bypass ran, llm_calls + token totals
        # were already credited inside the bypass block (the interpreter
        # call). Don't double-count.
        if hitl_direct_parsed is None:
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
        skip_sg_id = str(parsed.get("skip_sub_goal_id", "")).strip()
        skip_sg_reason = str(parsed.get("skip_sub_goal_reason", "")).strip()
        if goal.sub_goals:
            for sg in goal.sub_goals:
                if sg.id == current_sg_id and sg.status == "pending":
                    sg.status = "in_progress"
            if completed_sg_id:
                for sg in goal.sub_goals:
                    if sg.id == completed_sg_id and sg.status != "done":
                        # Phase A.6 Step 4 — verify-in-list gate.
                        # The auto-appended verify sub-goal has id
                        # ending in "_verify" (see
                        # ensure_create_verify_pattern). When the
                        # agent tries to close it, we deterministically
                        # check that the entity name typed earlier in
                        # this submodule actually appears in visible
                        # text on the page. If not → refuse the close
                        # + emit an event so the live feed shows why.
                        is_verify_subgoal = sg.id.endswith("_verify")
                        if is_verify_subgoal:
                            recent_typed = _last_typed_entity_value(
                                turn_log,
                            )
                            actually_visible = False
                            if recent_typed:
                                try:
                                    actually_visible = bool(
                                        page.evaluate(
                                            "(needle) => {"
                                            "  const t = (document.body."
                                            "innerText || '');"
                                            "  return t.toLowerCase()"
                                            ".includes(needle"
                                            ".toLowerCase());"
                                            "}",
                                            recent_typed,
                                        ),
                                    )
                                except Exception:
                                    actually_visible = False
                            if recent_typed and not actually_visible:
                                _emit(emit_event, "verify_check_failed", {
                                    "run_id": submodule_run_id,
                                    "step_id": submodule_step_id,
                                    "sub_goal_id": sg.id,
                                    "looked_for": recent_typed[:80],
                                    "reason": (
                                        "entity name not found in "
                                        "visible page text — search "
                                        "or scroll to locate the row "
                                        "before marking this complete"
                                    ),
                                })
                                # Refuse the close. Inject a one-shot
                                # signal block so the next turn's
                                # prompt explains why; the planner
                                # will then use the search/filter.
                                page_memory.setdefault(
                                    "_pending_signals", [],
                                ).append({
                                    "turn": turn_idx,
                                    "kind": "validation_error",
                                    "message": (
                                        f"Verify gate refused: the "
                                        f"name '{recent_typed[:60]}' "
                                        "is NOT visible in the page "
                                        "text right now. Use the "
                                        "list's search/filter input "
                                        "(or scroll) to surface the "
                                        "new row, THEN mark "
                                        f"{sg.id} done."
                                    ),
                                    "fields": [],
                                })
                                # Keep the sub-goal in_progress.
                                if sg.status == "pending":
                                    sg.status = "in_progress"
                                continue

                        # Phase O.1 — generalized success_criterion
                        # verification gate. The agent's claim to close
                        # a sub-goal is checked against the criterion
                        # text via deterministic patterns (URL contains,
                        # quoted-text visible, drawer-state matches,
                        # toast visible). When the page disagrees, we
                        # refuse the close and inject a one-shot signal
                        # so the next turn fixes it BEFORE advancing.
                        # Without this, "I clicked Save" becomes
                        # mark-the-sub-goal-done on faith — the source
                        # of the "claim success without checking"
                        # hallucination mode.
                        rsg_match = next(
                            (
                                r for r in runtime_sub_goals
                                if r.id == sg.id
                            ),
                            None,
                        )
                        criterion_text = (
                            rsg_match.success_criterion if rsg_match
                            else ""
                        )
                        if criterion_text:
                            try:
                                ok_obs, reason = (
                                    _verify_subgoal_criterion(
                                        page,
                                        criterion=criterion_text,
                                        observation=observation,
                                    )
                                )
                            except Exception:
                                ok_obs, reason = (
                                    True,
                                    "criterion check raised — gate skipped",
                                )
                            if not ok_obs:
                                _emit(
                                    emit_event,
                                    "sub_goal_criterion_unmet",
                                    {
                                        "run_id": submodule_run_id,
                                        "step_id": submodule_step_id,
                                        "sub_goal_id": sg.id,
                                        "criterion": criterion_text[:200],
                                        "reason": reason[:200],
                                    },
                                )
                                page_memory.setdefault(
                                    "_pending_signals", [],
                                ).append({
                                    "turn": turn_idx,
                                    "kind": "subgoal_unmet",
                                    "message": (
                                        f"Sub-goal {sg.id} close refused: "
                                        f"the success_criterion "
                                        f"({criterion_text[:120]!r}) is "
                                        f"NOT observable on the current "
                                        f"page — {reason[:160]}. Take "
                                        "another action to make the "
                                        "criterion observable BEFORE "
                                        "claiming completion."
                                    ),
                                    "fields": [],
                                })
                                if sg.status == "pending":
                                    sg.status = "in_progress"
                                continue
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
            # Phase B.1 — sub-goal SKIP application. Agent marks a
            # sub-goal as not applicable; emit progress so the live
            # presenter shows the skip + reason. Reason is mandatory;
            # if the agent omitted it we still apply the skip but
            # log a warning so we can spot bad flagging in telemetry.
            if skip_sg_id:
                if not skip_sg_reason:
                    logger.warning(
                        "skip_sub_goal_id=%s without reason on turn %d",
                        skip_sg_id, turn_idx,
                    )
                for sg in goal.sub_goals:
                    if sg.id == skip_sg_id and sg.status not in (
                        "done", "skipped",
                    ):
                        sg.status = "skipped"
                        sg.completed_at_turn = turn_idx
                        _emit(emit_event, "sub_goal_progress", {
                            "run_id": submodule_run_id,
                            "step_id": submodule_step_id,
                            "sub_goal_id": sg.id,
                            "description": sg.description,
                            "status": "skipped",
                            "skip_reason": skip_sg_reason[:200],
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
                "skip_sub_goal_id", "skip_sub_goal_reason",
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
            # α.7 — signal voting. When evidence_signals were
            # authored on the goal, score them at completion time
            # against the live page. ``verified`` becomes True when
            # ≥ majority match. This subsumes the older "any
            # successful verify in turn_log" rule for goals with an
            # explicit signal list — proper evidence beats a
            # historical verify that may be stale.
            signal_match_count = 0
            signal_total = 0
            signal_traces: list[dict[str, Any]] = []
            if goal.evidence_signals:
                try:
                    (
                        signal_match_count,
                        signal_total,
                        signal_traces,
                    ) = _evaluate_evidence_signals(
                        page, list(goal.evidence_signals),
                    )
                    _emit(emit_event, "signal_voting", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "matched": signal_match_count,
                        "total": signal_total,
                        "traces": signal_traces,
                    })
                except Exception as e:
                    logger.debug("signal voting skipped: %s", e)

            # Soft-guard #1: did the agent actually verify anything?
            # Two paths qualify:
            #  - ANY ok verify/extract_text in the turn log (legacy)
            #  - signal-voting majority (new, when signals were
            #    authored on the goal)
            majority = (
                (signal_total + 1) // 2 if signal_total else 0
            )
            signal_majority_met = (
                signal_total > 0
                and signal_match_count >= max(1, majority)
            )
            verified = signal_majority_met or any(
                t.tool in ("verify", "extract_text") and t.status == "ok"
                for t in turn_log
            )

            # Phase Q.2 — create-flow Save gate. When the goal text
            # says "create / add / register / make a <X>" and the
            # turn history contains NO successful submit-class action
            # (fill_form with submit_status='ok', OR a verify of a
            # post-create signal like a created-toast or list-row),
            # the role / user / record was never actually persisted —
            # the drawer might be closed but the backend never got
            # the POST. Refuse the mark_goal_complete and inject a
            # signal so the next turn knows to fix it.
            create_save_ok = True
            goal_text_lower = (goal.description or "").lower()
            is_create_goal = any(
                v in goal_text_lower for v in (
                    "create", "add new", "+add", "register",
                    "make a", "make an", "new ", "set up",
                )
            )
            if is_create_goal:
                saved = any(
                    (
                        t.tool == "fill_form"
                        and t.status == "ok"
                        and isinstance(t.details, dict)
                        and (
                            (t.details.get("fill_form") or {}).get(
                                "submit_status",
                            ) == "ok"
                        )
                    )
                    or (
                        t.tool in ("click", "type")
                        and t.status == "ok"
                        and any(
                            kw in (
                                (t.args.get("target_hint") or "")
                                + " "
                                + (t.args.get("value") or "")
                            ).lower()
                            for kw in (
                                "save", "create", "submit",
                                "confirm", "apply",
                            )
                        )
                    )
                    for t in turn_log
                    if t.args is not None
                )
                if not saved:
                    create_save_ok = False
                    _emit(emit_event, "complete_refused_no_save", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "goal_preview": goal.description[:200],
                    })
                    page_memory.setdefault(
                        "_pending_signals", [],
                    ).append({
                        "turn": turn_idx,
                        "kind": "create_no_save",
                        "message": (
                            "mark_goal_complete REFUSED: the goal is a "
                            "create flow but no successful Save / Submit "
                            "action is recorded in this submodule's "
                            "turn history. The record has NOT been "
                            "persisted. Find the Save button (it may "
                            "be at the bottom-right of the drawer) and "
                            "click it BEFORE marking the goal complete."
                        ),
                        "fields": [],
                    })
                    # Verified stays as-is; the create_save_ok flag
                    # blocks the close below.
            # Soft-guard #2: when sub-goals exist, were enough of
            # them closed out? Allow the last sub-goal to be implicit
            # (some agents call mark_goal_complete with the final
            # sub-goal still "in_progress" because the verify itself
            # closed it). So the bar is "≥ 80% done OR ≥ all-but-1".
            #
            # Phase B.1 — count SKIPPED sub-goals as "closed". An
            # agent that correctly recognizes a sub-goal is not
            # applicable (e.g. "remove cart items" when cart is
            # already empty) shouldn't be punished. SKIP requires
            # a reason and surfaces in the live feed, so it's
            # auditable.
            sub_goal_completion_ok = True
            if goal.sub_goals:
                closed = sum(
                    1 for sg in goal.sub_goals
                    if sg.status in ("done", "skipped")
                )
                total = len(goal.sub_goals)
                pct = closed / total if total else 1.0
                sub_goal_completion_ok = (
                    pct >= 0.80 or closed >= total - 1
                )

            # A4.1a: vision-grounded verdict. Only run when the
            # deterministic guards already say "OK" — otherwise we'd
            # double-fail and waste the vision call on something we'd
            # downgrade to inconclusive anyway.
            verification_record: dict[str, Any] | None = None

            if (
                not verified
                or not sub_goal_completion_ok
                or not create_save_ok
            ):
                halt_reason = "complete"
                final_status = "inconclusive"
                reasons = []
                if not verified:
                    reasons.append("no successful verify/extract_text")
                if not create_save_ok:
                    reasons.append(
                        "create-flow goal but no Save/Submit succeeded — "
                        "the record was never persisted"
                    )
                if not sub_goal_completion_ok:
                    closed = sum(
                        1 for sg in goal.sub_goals
                        if sg.status in ("done", "skipped")
                    )
                    skipped = sum(
                        1 for sg in goal.sub_goals
                        if sg.status == "skipped"
                    )
                    total = len(goal.sub_goals)
                    reasons.append(
                        f"only {closed}/{total} sub-goals closed"
                        + (
                            f" ({skipped} skipped)"
                            if skipped else ""
                        )
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
                            cheap_provider=cheap_provider,
                            on_escalate=_emit_escalation,
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

        # Phase 11 — test-case dispute. Agent flags the test step as
        # provably wrong (selector dead, action impossible, precondition
        # not met). We halt with status=blocked, halt_reason=
        # test_case_disputed, and attach the structured dispute payload
        # to the turn record so the report can render it. Frozen-path
        # capture is suppressed for this run (the disputed flow is
        # not a deterministic replay candidate).
        if tool == "flag_test_case_issue":
            issue_kind = str(args.get("issue_kind") or "").strip()
            issue_evidence = str(args.get("issue_evidence") or "").strip()
            issue_fix = str(args.get("issue_suggested_fix") or "").strip()
            valid_kinds = (
                "wrong_selector", "missing_step", "impossible_action",
                "misleading_description", "precondition_failed",
            )
            if issue_kind not in valid_kinds:
                # Malformed dispute — log, treat as a soft failure
                # rather than halting on a meta misuse.
                logger.warning(
                    "flag_test_case_issue with invalid kind=%r — "
                    "treating as ask_human", issue_kind,
                )
                halt_reason = "ask_human"
                final_status = "blocked"
                final_narration = (
                    f"Agent tried to dispute the test case but the "
                    f"issue_kind {issue_kind!r} is not recognized. "
                    f"Reasoning: {reasoning[:200]}"
                )
            else:
                halt_reason = "test_case_disputed"
                final_status = "blocked"
                final_narration = (
                    f"TEST CASE DISPUTED ({issue_kind}): "
                    f"{issue_evidence[:160]}"
                )
                if issue_fix:
                    final_narration += f" — suggested fix: {issue_fix[:120]}"
                _emit(emit_event, "test_case_disputed", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "kind": issue_kind,
                    "evidence": issue_evidence[:400],
                    "suggested_fix": issue_fix[:400],
                    "turn": turn_idx,
                    "reasoning": reasoning[:300],
                })
            dispute_record = TurnRecord(
                turn=turn_idx, tool=tool, args=args, reasoning=reasoning,
                confidence=confidence, status="blocked",
                narration=final_narration[:500],
                page_url=observation.get("url", ""),
            )
            # Reuse search_log for the structured dispute payload —
            # report_service lifts it into the per-step surface.
            dispute_record.search_log = {
                "kind": "test_case_dispute",
                "issue_kind": issue_kind,
                "evidence": issue_evidence,
                "suggested_fix": issue_fix,
                "turn": turn_idx,
            }
            turn_log.append(dispute_record)
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

        # ── Phase 14: smart candidate selection (PRE-action) ───────
        # Before a click on a target_hint that resolves to multiple
        # candidates, ask the vision LLM to pick the BEST one given
        # the goal's criteria (skip sponsored ads, items without
        # prices, items violating a goal constraint, etc.).
        #
        # Triggers when ALL of:
        #   - tool is click (extending to type/select later)
        #   - target_hint is set
        #   - the resolver finds >= 3 visible matches (ambiguous)
        #   - provider supports vision
        # On success the LLM returns either:
        #   - a precise selector → we patch args[target_hint] with it
        #     and let the normal click dispatch use it
        #   - pixel coords → we click at coords directly via mouse
        #   - "scroll" → scroll the requested direction and skip the
        #     click this turn (next turn's observation reflects the
        #     scrolled state)
        #   - "none" → leave args alone and let the regular dispatch
        #     fail; the next turn can dispute or improvise
        #
        # When provider has no vision OR the resolver finds 0-2
        # matches, this whole block is skipped — zero cost on the
        # happy path.
        smart_pick_record: dict[str, Any] | None = None
        # Phase J.5 — smart-pick + vision_only ensemble.
        # Pre-J: smart-pick + _vision_only_dispatch both fired
        #   unconditionally on every click → wasted ~50% of vision
        #   tokens with no upside on unambiguous targets.
        # Post-J.1 (too aggressive): smart-pick muted entirely in
        #   vision_only → removed the DOM-aware tie-breaker for cases
        #   where the DOM HAS ambiguity smart-pick could resolve.
        # J.5: smart-pick runs in vision_only mode ONLY when the DOM
        #   resolver returns ≥2 candidates for the target_hint —
        #   i.e. there IS DOM ambiguity worth breaking. When 0 (truly
        #   custom widget) or 1 (unambiguous) candidates exist, skip
        #   smart-pick; vision_only's coord proposal is enough.
        # Helper signals ambiguity threshold via ``min_matches``.
        if (
            tool == "click"
            and provider_supports_vision
            and isinstance(args.get("target_hint"), str)
            and args["target_hint"].strip()
        ):
            smart_pick_record = _maybe_run_smart_pick(
                min_matches=(
                    2 if agent_strategy == "vision_only" else 3
                ),
                page=page,
                provider=provider,
                cheap_provider=cheap_provider,
                goal=goal,
                tool=tool,
                args=args,
                emit_event=emit_event,
                on_escalate=_emit_escalation,
                submodule_run_id=submodule_run_id,
                submodule_step_id=submodule_step_id,
            )
            if smart_pick_record is not None:
                vision_calls += 1
                llm_calls += 1
                if isinstance(smart_pick_record.get("input_tokens"), int):
                    total_input += smart_pick_record["input_tokens"]
                if isinstance(smart_pick_record.get("output_tokens"), int):
                    total_output += smart_pick_record["output_tokens"]

        # ── Act ────────────────────────────────────────────────────
        # Phase 6 — vision-only mode. When agent_strategy == "vision_
        # only" AND tool is click/type AND we have a target_hint, route
        # via VL+coords directly (DOM resolution bypassed entirely).
        # This is the "computer use" path — slower / more vision tokens
        # but works on apps the DOM resolver can't reach (heavy canvas,
        # sealed shadow DOM, hostile rotating classes, SAP GUI for HTML
        # in legacy frames). Smart-pick still runs first because it
        # ALSO returns coords for ambiguous cases — when smart-pick
        # already preempted with a coord click, we use that and skip
        # this branch (no point re-clicking).
        vision_only_outcome: dict[str, Any] | None = None
        if (
            agent_strategy == "vision_only"
            and tool in ("click", "type")
            and provider_supports_vision
            and isinstance(args.get("target_hint"), str)
            and args["target_hint"].strip()
            and (
                smart_pick_record is None
                or smart_pick_record.get("preempt_outcome") is None
            )
        ):
            vision_only_outcome = _vision_only_dispatch(
                page=page,
                tool=tool,
                args=args,
                provider=provider,
                cheap_provider=cheap_provider,
                emit_event=emit_event,
                on_escalate=_emit_escalation,
                submodule_run_id=submodule_run_id,
                submodule_step_id=submodule_step_id,
            )
            if vision_only_outcome is not None:
                outcome = vision_only_outcome["outcome"]
                if isinstance(vision_only_outcome.get("input_tokens"), int):
                    total_input += vision_only_outcome["input_tokens"]
                if isinstance(vision_only_outcome.get("output_tokens"), int):
                    total_output += vision_only_outcome["output_tokens"]
                vision_calls += 1
                llm_calls += 1

        # Phase O.2 — pre-action existence check. Before dispatching
        # click/type, verify the proposed target_hint actually maps to
        # a visible element on the current page. If not, refuse the
        # dispatch, inject a "did you mean" signal listing visible
        # alternatives, and let the next turn re-pick. This prevents
        # the "hallucinated target" failure mode where the planner
        # invents a button name that doesn't exist and burns turns +
        # tokens trying to resolve it.
        #
        # Skipped when:
        #   - HITL bypass produced this turn (user's input is
        #     authoritative; trust them)
        #   - vision_only / smart-pick already picked coords (the
        #     visual proposer already verified visibility)
        #   - tool isn't click/type
        #   - target_hint is a selector (CSS/role-based — handled by
        #     the DOM resolver, not by label match)
        target_skipped_for_existence = False
        if (
            tool in ("click", "type")
            and hitl_direct_parsed is None
            and vision_only_outcome is None
            and (
                smart_pick_record is None
                or smart_pick_record.get("preempt_outcome") is None
            )
            and isinstance(args.get("target_hint"), str)
            and args["target_hint"].strip()
        ):
            try:
                exists, similar = _check_target_exists(
                    page, args["target_hint"],
                )
            except Exception:
                exists, similar = True, []
            # Diag.3 — recent-history allowlist. Transient dropdown
            # items (e.g. "Roles" inside the Administration menu)
            # disappear between turns when the menu auto-closes. If
            # the target_hint was seen in the AX tree / fields scan
            # in ANY of the last 3 turns, allow the click — the DOM
            # resolver downstream can take its chance. Only block
            # when the target hasn't been seen recently AND no close
            # alternatives exist (no plausible "did-you-mean").
            recently_seen = False
            hint_lower = (args["target_hint"] or "").lower().strip()
            if hint_lower and turn_log:
                for prev_rec in turn_log[-3:]:
                    prev_search = prev_rec.search_log or {}
                    if isinstance(prev_search, dict):
                        labels_seen = prev_search.get(
                            "labels_seen",
                        ) or []
                        if any(
                            hint_lower in str(lb).lower()
                            or str(lb).lower() in hint_lower
                            for lb in labels_seen
                        ):
                            recently_seen = True
                            break
                    # Also consider: a prior CLICK on this same
                    # target succeeded recently → the target exists
                    # in this app, just not THIS exact moment.
                    if (
                        prev_rec.tool == "click"
                        and prev_rec.status == "ok"
                        and hint_lower in str(
                            (prev_rec.args or {}).get(
                                "target_hint", "",
                            ),
                        ).lower()
                    ):
                        recently_seen = True
                        break
            if not exists and recently_seen:
                # Don't block; emit a soft warning event so the
                # operator sees the agent is trying a transient
                # target.
                _emit(emit_event, "target_transient_allowed", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "turn": turn_idx,
                    "target_hint": (args["target_hint"] or "")[:120],
                })
                exists = True
            if not exists:
                _emit(emit_event, "target_not_visible_refused", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "turn": turn_idx,
                    "target_hint": (args["target_hint"] or "")[:120],
                    "similar": similar[:6],
                })
                # Inject a one-shot signal so the next planner turn
                # sees the visible alternatives. The signal block is
                # already consumed by the prompt builder via
                # ``_pending_signals``.
                similar_str = (
                    ", ".join(f"'{s}'" for s in similar[:6])
                    or "(no similar labels detected)"
                )
                page_memory.setdefault("_pending_signals", []).append({
                    "turn": turn_idx,
                    "kind": "target_not_visible",
                    "message": (
                        f"Pre-action gate refused: target_hint "
                        f"'{args['target_hint'][:80]}' is NOT visible "
                        "on the page. Visible interactive labels "
                        f"include: {similar_str}. Pick one that "
                        "exists, or scroll/navigate first."
                    ),
                    "fields": [],
                })
                # Synthesize a soft-skip outcome so the rest of the
                # turn-loop accounting runs (turn_log, action counters)
                # but no DOM-side action fires. The next turn will
                # consume the signal block.
                outcome = {
                    "status": "failed",
                    "narration": (
                        f"target '{args['target_hint'][:60]}' not "
                        "visible — refused by pre-action gate"
                    ),
                    "error_message": "target_not_visible",
                    "extracted_text": "",
                    "details": {
                        "pre_action_gate": "target_not_visible",
                        "similar": similar[:6],
                    },
                }
                target_skipped_for_existence = True

        # Smart-pick may have written a coord-click outcome already;
        # in that case skip the normal dispatcher and use that result
        # directly (see ``_maybe_run_smart_pick`` for the contract).
        if target_skipped_for_existence:
            pass  # outcome already set above
        elif vision_only_outcome is not None:
            pass  # already set
        elif (
            smart_pick_record is not None
            and smart_pick_record.get("preempt_outcome") is not None
        ):
            outcome = smart_pick_record["preempt_outcome"]
        else:
            outcome = _execute_tool_call(
                page, tool, args,
                plan_target_url=plan_target_url,
                speed_config=speed_config,
                emit_event=emit_event,
                submodule_run_id=submodule_run_id,
                submodule_step_id=submodule_step_id,
                turn_idx=turn_idx,
            )

        # Phase J.4 — entity-creation tracking. When fill_form returns
        # status="ok" AND the goal text says "create / add / register
        # a <kind>", record the first textbox value as the entity's
        # identity in WorldState. The decomposer in subsequent
        # submodules reads ``entities_created`` and emits
        # "search for the existing <kind>" instead of "create a new
        # <kind>" — which prevents the second-run "already exists"
        # conflict you saw on Solar.
        if (
            tool == "fill_form"
            and isinstance(outcome, dict)
            and outcome.get("status") == "ok"
        ):
            try:
                _record_entity_from_fill_form(
                    world_state=world_state,
                    goal_text=goal.description,
                    args=args,
                    page_url=page.url if page else "",
                )
            except Exception as e:
                logger.debug(
                    "entity record post fill_form skipped: %s", e,
                )

        # ── Phase 10: popup classifier on intercepted clicks ──────
        # When a click failed because something is overlaying the
        # target (Playwright reports "intercepts pointer events"),
        # ask the vision LLM what the overlay IS — required step
        # (engage), dismissable blocker (close it), non-blocking
        # banner (ignore), or ad (close aggressively). On low
        # confidence we ENGAGE (your locked policy from plan Q3:
        # the cost of skipping a required step is much worse than
        # one extra modal click).
        popup_record: dict[str, Any] | None = None
        outcome_details_after = outcome.get("details") or {}
        if (
            outcome["status"] == "failed"
            and outcome_details_after.get("failure_kind") == "click_intercepted"
            and provider_supports_vision
        ):
            try:
                from app.agents.page_intel import (  # noqa: PLC0415
                    classify_popup, capture_screenshot_for_vision,
                )
                popup_screenshot = capture_screenshot_for_vision(page)
                pc = classify_popup(
                    provider, page,
                    goal_context=(
                        f"{goal.description} (current target: "
                        f"{args.get('target_hint', '')!r})"
                    )[:400],
                    screenshot_bytes=popup_screenshot,
                    cheap_provider=cheap_provider,
                    on_escalate=_emit_escalation,
                )
                if isinstance(pc.input_tokens, int):
                    total_input += pc.input_tokens
                if isinstance(pc.output_tokens, int):
                    total_output += pc.output_tokens
                vision_calls += 1
                llm_calls += 1
                _emit(emit_event, "popup_classified", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "kind": pc.kind,
                    "confidence": pc.confidence,
                    "reasoning": pc.reasoning[:200],
                })
                popup_record = {
                    "kind": pc.kind,
                    "dismiss_hint": pc.dismiss_hint,
                    "confidence": pc.confidence,
                    "reasoning": pc.reasoning,
                    "input_tokens": pc.input_tokens,
                    "output_tokens": pc.output_tokens,
                }

                # Resolve based on kind. Low confidence (< 0.7) →
                # default to engage (treat as required_step).
                effective_kind = (
                    pc.kind if pc.confidence >= 0.7 else "required_step"
                )

                if effective_kind in ("dismissable_blocker", "ad"):
                    # Try the LLM-supplied hint first; fall through
                    # to the heuristic dismiss_modal otherwise.
                    dismissed = False
                    if pc.dismiss_hint:
                        try:
                            page.locator(pc.dismiss_hint).first.click(
                                timeout=3_000,
                            )
                            dismissed = True
                        except Exception as e:
                            logger.debug(
                                "popup dismiss_hint %r failed: %s",
                                pc.dismiss_hint, e,
                            )
                    if not dismissed:
                        ds_status, _, _ = _do_dismiss_modal(page)
                        dismissed = ds_status == "ok"
                    if dismissed:
                        # Retry the original click ONCE post-dismiss.
                        retry_outcome = _execute_tool_call(
                            page, tool, args,
                            plan_target_url=plan_target_url,
                            speed_config=speed_config,
                        )
                        retry_outcome["narration"] = (
                            f"{retry_outcome.get('narration') or tool}"
                            f" (after popup dismiss: {pc.kind})"
                        )
                        outcome = retry_outcome
                # required_step / non_blocking_overlay / none →
                # leave the failed outcome alone; the agent's next
                # turn will see the popup in its observation and
                # engage with it via normal click.
            except Exception as e:
                logger.warning("popup classifier call failed: %s", e)

        # ── Phase 9 + 13: semantic verify escalation ──────────────
        # When a literal `verify` step fails, ask the vision LLM
        # whether the SCREENSHOT shows the expected outcome anyway.
        # Wins the spurious-fail case where the page wraps text
        # differently than the test case anticipated ("Cart" vs
        # "Your Amazon Cart") but the goal is semantically met.
        # Strict prompt — biased toward inconclusive/fail on doubt
        # so we never mask a real bug behind a generous read.
        semantic_verify_record: dict[str, Any] | None = None
        if (
            tool == "verify"
            and outcome["status"] == "failed"
            and provider_supports_vision
        ):
            expected_text = (
                str(args.get("expected") or "").strip()
                or str(args.get("target_hint") or "").strip()
            )
            if expected_text:
                _emit(emit_event, "semantic_verify_started", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "expected": expected_text[:200],
                })
                try:
                    from app.agents.page_intel import (  # noqa: PLC0415
                        verify_semantic,
                    )
                    # Phase J.1 — in vision_only mode, bypass the
                    # cheap-tier verifier. The cheap→strong escalation
                    # adds a second LLM call on every borderline
                    # confidence read (~0.6-0.7) for no real upside:
                    # vision_only already committed to spending VL
                    # tokens, and the verifier's job (was the action
                    # observable?) is too important to gamble on a
                    # weaker model. Going strong-first here is the
                    # honest cost.
                    _sv_cheap = (
                        None if agent_strategy == "vision_only"
                        else cheap_provider
                    )
                    sv = verify_semantic(
                        provider, page,
                        expected=expected_text,
                        target_hint=str(args.get("target_hint") or "") or None,
                        full_page=True,  # revalidation per user spec
                        cheap_provider=_sv_cheap,
                        on_escalate=_emit_escalation,
                    )
                    if isinstance(sv.input_tokens, int):
                        total_input += sv.input_tokens
                    if isinstance(sv.output_tokens, int):
                        total_output += sv.output_tokens
                    vision_calls += 1
                    llm_calls += 1
                    semantic_verify_record = {
                        "verdict": sv.verdict,
                        "reasoning": sv.reasoning,
                        "confidence": sv.confidence,
                        "visible_evidence": sv.visible_evidence,
                        "input_tokens": sv.input_tokens,
                        "output_tokens": sv.output_tokens,
                    }
                    _emit(emit_event, "semantic_verify_completed", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "verdict": sv.verdict,
                        "confidence": sv.confidence,
                        "reasoning": sv.reasoning[:200],
                    })
                    # Upgrade ONLY on an unambiguous "pass". Strict
                    # rubric: confidence >= 0.85 to flip a failed
                    # literal verify into ok. Anything weaker keeps
                    # the failure (we'd rather false-flag a passing
                    # test than mask a real bug).
                    if sv.verdict == "pass" and sv.confidence >= 0.85:
                        outcome["status"] = "ok"
                        outcome["narration"] = (
                            f"{outcome.get('narration') or 'verify'}"
                            f" — semantically passed via vision: "
                            f"{sv.visible_evidence[:120]}"
                        )
                        outcome["error_message"] = None
                except Exception as e:
                    logger.warning(
                        "semantic verify escalation failed: %s", e,
                    )
                    _emit(emit_event, "semantic_verify_completed", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "verdict": "error",
                        "error": str(e)[:200],
                    })

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
        # Prefer the typed ``failure_kind`` flag set by actions.py /
        # _do_extract_text. Fall back to the legacy string match so
        # any not-yet-typed call site still triggers the rescue path
        # (defence in depth — the typed flag is authoritative when
        # present).
        outcome_details = outcome.get("details") or {}
        miss_due_to_selector = (
            outcome["status"] == "failed"
            and (
                outcome_details.get("failure_kind") == "selector_not_found"
                or "target not visible" in (outcome.get("narration") or "").lower()
                or "no visible element" in (outcome.get("error_message") or "").lower()
            )
        )
        sub_goals_done = (
            bool(goal.sub_goals)
            and all(
                sg.status in ("done", "skipped") for sg in goal.sub_goals
            )
        )
        # Phase C.4 — for ``type`` actions whose DOM resolver missed,
        # fast-fail straight to coord-typing instead of going through
        # the fuzzy → AI → vision-search ladder. Empirically, hybrid's
        # type ladder loses to vision-only on complex form widgets
        # (MUI selects, React-controlled inputs, custom dropdowns)
        # because DOM resolution picks a wrapper and the typed value
        # never reaches the actual input. Coord-typing clicks the
        # visible pixel and types into whatever has focus — which is
        # almost always the right thing for a form field.
        if (
            tool == "type"
            and miss_due_to_selector
            and not sub_goals_done
            and provider_supports_vision
            and args.get("target_hint")
            and args.get("value")
        ):
            try:
                from app.agents.page_intel import (  # noqa: PLC0415
                    capture_screenshot_for_vision,
                    propose_click_coordinates,
                )
                from app.executor.actions import (  # noqa: PLC0415
                    clear_focused_field,
                )
                fast_shot = capture_screenshot_for_vision(
                    page, downscale=False,
                )
                coord_pick = propose_click_coordinates(
                    provider, page,
                    target_hint=str(args.get("target_hint", "")),
                    screenshot_bytes=fast_shot,
                )
                if (
                    coord_pick is not None
                    and coord_pick.confidence >= 0.55
                    and coord_pick.x > 0 and coord_pick.y > 0
                ):
                    page.mouse.click(coord_pick.x, coord_pick.y)
                    try:
                        page.wait_for_timeout(80)
                    except Exception:
                        pass
                    clear_focused_field(page)
                    typed_value = str(args.get("value") or "")
                    delay = (
                        20 if getattr(
                            speed_config, "typing_delay_ms", 0,
                        ) else 0
                    )
                    try:
                        if delay > 0:
                            page.keyboard.type(typed_value, delay=delay)
                        else:
                            page.keyboard.type(typed_value)
                    except Exception:
                        page.keyboard.type(typed_value)
                    # type-and-submit: respect the agent's submit flag.
                    if args.get("submit"):
                        try:
                            page.wait_for_timeout(80)
                            page.keyboard.press("Enter")
                        except Exception:
                            pass
                    coord_record = {
                        "kind": "coord_type_fast_fail",
                        "x": coord_pick.x,
                        "y": coord_pick.y,
                        "confidence": coord_pick.confidence,
                        "label_visible": coord_pick.label_visible[:120],
                        "applied": True,
                    }
                    if isinstance(coord_pick.input_tokens, int):
                        total_input += coord_pick.input_tokens
                    if isinstance(coord_pick.output_tokens, int):
                        total_output += coord_pick.output_tokens
                    vision_calls += 1
                    llm_calls += 1
                    outcome = {
                        "status": "ok",
                        "narration": (
                            f"COORD TYPE (fast-fail) at "
                            f"({coord_pick.x}, {coord_pick.y}) on "
                            f"{coord_pick.label_visible[:80]!r} — DOM "
                            f"resolution missed, vision pointed at "
                            f"pixels (confidence "
                            f"{coord_pick.confidence:.2f})."
                        ),
                        "error_message": None,
                        "extracted_text": "",
                        "details": {
                            **(outcome.get("details") or {}),
                            "coord_type_fast_fail": coord_record,
                        },
                        "search_log": {
                            "kind": "coord_type_fast_fail",
                            **coord_record,
                        },
                    }
                    _emit(emit_event, "coord_type_fast_fail", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "turn": turn_idx,
                        "target_hint": args.get("target_hint"),
                        "x": coord_pick.x,
                        "y": coord_pick.y,
                        "confidence": coord_pick.confidence,
                        "label_visible": coord_pick.label_visible[:160],
                    })
                    # Skip the heavier rescue ladder below — the
                    # fast-fail handled it.
                    miss_due_to_selector = False
            except Exception as e:
                logger.debug(
                    "coord-type fast-fail skipped (%s); falling "
                    "through to standard rescue", e,
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
                cheap_provider=cheap_provider,
                on_escalate=_emit_escalation,
            )
            # Strip the in-memory ONLY screenshot bytes out of
            # search_result before anything reads it as `search_log`:
            # search_log is folded into TurnRecord.search_log and
            # serialized to details_json, which goes through JSON.
            # Bytes aren't JSON-serializable. Coord-click can't
            # reuse this anyway — vision-search captures downscaled
            # bytes; coord-click needs original viewport pixel
            # space — so the bytes are simply discarded here.
            search_result.pop("last_screenshot", None)
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
                        # Reuse the AX-tree near-misses AND the
                        # screenshot bytes the vision search just
                        # computed for this exact page state. Saves
                        # the coord-click LLM from a cold start AND
                        # avoids recomputing the AX tree / re-
                        # screenshotting (the page hasn't moved
                        # since search exhausted).
                        coord_near_misses = (
                            search_result.get("last_near_misses") or None
                        )
                        # NOTE: do NOT reuse the vision-search
                        # screenshot here — that one is downscaled.
                        # Coord-click MUST see the page at its true
                        # pixel dimensions because its output goes
                        # straight into ``page.mouse.click(x, y)``.
                        # Letting propose_click_coordinates capture
                        # internally (with downscale=False) keeps
                        # the click coordinates in viewport space.
                        coords = propose_click_coordinates(
                            provider, page,
                            target_hint=str(args.get("target_hint", "")),
                            near_misses=coord_near_misses,
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
                                # newly-focused element. Clear first
                                # so a retry replaces instead of
                                # stacking onto a prior value.
                                if tool == "type" and args.get("value"):
                                    typed_value = str(args.get("value", ""))
                                    delay = (
                                        speed_config.type_delay_ms
                                        if hasattr(speed_config, "type_delay_ms")
                                        else 0
                                    )
                                    from app.executor.actions import (  # noqa: PLC0415
                                        clear_focused_field,
                                    )
                                    clear_focused_field(page)
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

        # Phase 14 — fold the smart-pick record into search_log so
        # the report timeline shows what was rejected and why. Done
        # AFTER any vision-search rescue so we don't clobber that
        # record (smart_pick lives alongside, not instead of).
        if smart_pick_record is not None:
            existing_log = outcome.get("search_log")
            if isinstance(existing_log, dict):
                existing_log["smart_pick"] = {
                    k: v for k, v in smart_pick_record.items()
                    if k != "preempt_outcome"
                }
            else:
                outcome["search_log"] = {
                    "kind": "smart_pick",
                    **{
                        k: v for k, v in smart_pick_record.items()
                        if k != "preempt_outcome"
                    },
                }

        # Phase 9 — fold semantic verify result into search_log so
        # the report shows BOTH the literal failure AND the LLM's
        # escalation verdict (and the visible evidence it cited).
        if semantic_verify_record is not None:
            existing_log = outcome.get("search_log")
            if isinstance(existing_log, dict):
                existing_log["semantic_verify"] = semantic_verify_record
            else:
                outcome["search_log"] = {
                    "kind": "semantic_verify",
                    "semantic_verify": semantic_verify_record,
                }

        # Phase 10 — fold popup classification + dismissal trace into
        # search_log so the report timeline shows what overlay was
        # detected and how the agent handled it.
        if popup_record is not None:
            existing_log = outcome.get("search_log")
            if isinstance(existing_log, dict):
                existing_log["popup"] = popup_record
            else:
                outcome["search_log"] = {
                    "kind": "popup",
                    "popup": popup_record,
                }

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

        # Phase O.3 — confidence + mistake memory.
        # 1. Track the failed approach when this turn's action failed
        #    so the next planner turn sees a "don't retry" list.
        # 2. Track consecutive low-confidence turns WITHOUT progress;
        #    halt to HITL escalation when the streak exceeds the cap.
        # HITL-bypassed turns + post-HITL grace window are excluded
        # (the human's input gets a fresh start).
        if (
            outcome.get("status") == "failed"
            and tool in ("click", "type", "select", "fill_form")
            and hitl_direct_parsed is None
        ):
            failed_approaches.append({
                "turn": str(turn_idx),
                "tool": tool,
                "target": (args.get("target_hint") or "")[:80],
                "value": (args.get("value") or "")[:80],
                "reason": (
                    outcome.get("error_message")
                    or outcome.get("narration") or ""
                )[:160],
            })
            # Cap memory size so the prompt block stays bounded.
            if len(failed_approaches) > MAX_FAILED_APPROACHES_REMEMBERED:
                failed_approaches = failed_approaches[
                    -MAX_FAILED_APPROACHES_REMEMBERED:
                ]

        # Confidence streak (skip during post-HITL grace and on HITL-
        # bypassed turns — we don't want HITL to count against the
        # agent's own confidence).
        if (
            hitl_direct_parsed is None
            and turns_since_hitl_consumed is None
        ):
            sub_goals_closed_now = sum(
                1 for sg in goal.sub_goals
                if sg.status in ("done", "skipped")
            ) if goal.sub_goals else 0
            if confidence < LOW_CONFIDENCE:
                if low_confidence_streak == 0:
                    sub_goals_closed_at_streak_start = sub_goals_closed_now
                low_confidence_streak += 1
            else:
                low_confidence_streak = 0
            progress_made = (
                sub_goals_closed_now > sub_goals_closed_at_streak_start
            )
            if (
                low_confidence_streak >= 2
                and not progress_made
            ):
                _emit(emit_event, "low_confidence_escalation", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "turn": turn_idx,
                    "streak": low_confidence_streak,
                    "last_confidence": confidence,
                    "sub_goals_closed": sub_goals_closed_now,
                })
                # Trigger the HITL path on the NEXT iteration via the
                # stall route — set final_error so the planner's stall
                # detector escalates instead of continuing to flail.
                if final_error is None:
                    final_error = "low_confidence_no_progress"
                # Reset so we don't keep emitting on every subsequent
                # turn until HITL fires.
                low_confidence_streak = 0

        # Phase F.1 — post-HITL no-op detector. Within the
        # POST_HITL_NOOP_WINDOW turns after a HITL submission, if
        # consecutive turns pick no-op tools AND the page hasn't
        # moved (same obs hash as before HITL), halt explicitly so
        # the operator sees the planner gave up instead of a silent
        # freeze.
        if turns_since_hitl_consumed is not None:
            turns_since_hitl_consumed += 1
            if tool in _POST_HITL_NOOP_TOOLS:
                post_hitl_noop_streak += 1
            else:
                post_hitl_noop_streak = 0
            if post_hitl_noop_streak >= POST_HITL_NOOP_WINDOW:
                _emit(emit_event, "planner_no_op_after_hitl", {
                    "run_id": submodule_run_id,
                    "step_id": submodule_step_id,
                    "turn": turn_idx,
                    "noop_streak": post_hitl_noop_streak,
                    "last_tool": tool,
                })
                halt_reason = "agent_failed"
                final_status = "blocked"
                final_narration = (
                    f"After HITL guidance, planner picked "
                    f"{post_hitl_noop_streak} consecutive no-op "
                    "tools — halting to avoid silent freeze."
                )
                final_error = "planner_no_op_after_hitl"
                break
            if turns_since_hitl_consumed >= POST_HITL_NOOP_WINDOW + 2:
                # Window expired without triggering — reset the
                # tracker so future stalls aren't misattributed.
                turns_since_hitl_consumed = None
                post_hitl_noop_streak = 0

        # Vision-on-demand: when an action tool fails AND the provider
        # supports vision, capture a screenshot now so the NEXT turn's
        # observation includes visual context. We don't take screenshots
        # on success — keeps token cost zero on the happy path.
        if outcome["status"] == "failed" and provider_supports_vision:
            try:
                from app.agents.page_intel import (  # noqa: PLC0415
                    capture_screenshot_for_vision,
                )
                pending_screenshot = capture_screenshot_for_vision(page)
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

        # Phase A.6 Step 1 — post-action form-signal observer. Runs
        # ONLY on submit-like turns (click/press_key on Save/Create/
        # Submit-like targets). When a toast or inline error appears,
        # stash the message in page_memory so the NEXT turn's prompt
        # builder folds it in as a "FORM SIGNAL" block. Also emits a
        # user-visible live-feed event so the operator sees the
        # validation message in real time.
        try:
            from app.agents.form_signals import (  # noqa: PLC0415
                is_submit_like, observe_form_signal,
            )
            target_for_signal = args.get("target_hint") or args.get("value") or ""
            if is_submit_like(tool, target_for_signal):
                sig = observe_form_signal(page)
                if sig.kind != "none" and sig.message:
                    _emit(emit_event, "form_signal_detected", {
                        "run_id": submodule_run_id,
                        "step_id": submodule_step_id,
                        "turn": turn_idx,
                        "kind": sig.kind,
                        "message": sig.message[:300],
                        "fields": sig.fields or [],
                    })
                    # Stash on page_memory so the next turn picks it up.
                    page_memory.setdefault("_pending_signals", []).append({
                        "turn": turn_idx,
                        "kind": sig.kind,
                        "message": sig.message,
                        "fields": sig.fields or [],
                    })
        except Exception as e:
            logger.debug("form-signal observer skipped: %s", e)

        # Phase 12 — page graph edge. When the URL changed during
        # this turn, record (from, to, tool) so future turns can
        # back-navigate or recognise "I've been here before via
        # this path". Edge list capped at 60 to bound memory.
        from_url = (
            prev_observation.get("url", "")
            if prev_observation else ""
        )
        to_url = observation.get("url", "")
        if from_url and to_url and from_url != to_url:
            edges = page_graph.setdefault("edges", [])
            edges.append({
                "from": from_url,
                "to": to_url,
                "tool": rec.tool,
                "turn": turn_idx,
            })
            if len(edges) > 60:
                page_graph["edges"] = edges[-60:]
        if to_url:
            visited = page_graph.setdefault("visited_urls", [])
            if not visited or visited[-1] != to_url:
                visited.append(to_url)
                if len(visited) > 60:
                    page_graph["visited_urls"] = visited[-60:]

        # Snapshot for next turn's diff. Only update at end of a
        # successful loop iteration so the diff compares against the
        # last "settled" page state, not a transient mid-action one.
        prev_observation = observation

        # Phase A — mirror sub-goal status from the planner-managed
        # static list onto the runtime list (carries reason + replan
        # iteration + audit metadata). Emits sub_goal_started /
        # sub_goal_done / sub_goal_failed / sub_goal_skipped events
        # for any transition this turn produced.
        _mirror_runtime_status()

    else:
        # for-else: loop ran to max_turns without a break
        halt_reason = "max_turns"
        final_status = "inconclusive"
        final_narration = f"Hit max_turns={max_turns} without resolution"

    # Final sub-goal status mirror (catches transitions on the
    # halt-causing turn).
    _mirror_runtime_status()

    # Mark any still-pending sub-goal as skipped with the halt reason
    # so the report timeline doesn't show ambiguous "pending" rows.
    for rsg in runtime_sub_goals:
        if rsg.status in ("pending", "in_progress"):
            rsg.status = "skipped"
            rsg.reason = (
                f"submodule halted before this sub-goal "
                f"({halt_reason})"
            )

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
        manual_intervention_used=auth_used_manual_intervention,
        sub_goals=[rsg.to_dict() for rsg in runtime_sub_goals],
        replans_used=replans_used,
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
    # Phase 4-α — HITL channel for the auth-flow orchestrator.
    # ``wait_for_intervention`` blocks until the user submits a
    # response; ``open_typed_prompt`` records the prompt's shape
    # (credentials / OTP / manual-solve) for the live presenter to
    # render. When wired, ``run_agent_for_goal`` invokes
    # ``auth_flow.run_auth_loop`` on login-looking pages before its
    # main turn loop runs.
    wait_for_intervention: Callable[[int], dict | None] | None = None,
    open_typed_prompt: Callable[..., None] | None = None,
    max_turns_per_goal: int = 30,
    max_wallclock_s_per_goal: int = 300,
    # Phase 1 — provider tiering. When the caller hasn't supplied
    # both, we look them up via build_tier_pair() from app_settings.
    # Strong = ``provider`` (existing arg, kept for back-compat).
    # Cheap = optional cheaper model that handles VL helpers with
    # escalation. None disables tiering — every call goes to strong.
    cheap_provider: LLMProvider | None = None,
    # Phase 6 — action strategy. ``hybrid`` (DOM-first) or
    # ``vision_only`` (VL+coords for click/type). Threaded into
    # each submodule's ``run_agent_for_goal``.
    agent_strategy: str = "hybrid",
    # Phase H — pre-execution Scout → Refine → Activate orchestrator.
    # "auto" runs preflight when the plan isn't yet pinned to an
    # app_map_refined version. "force" always re-scouts + re-refines.
    # "skip" disables it (legacy / debugging path).
    preflight: str = "auto",
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

    # Phase A — read SoM toggle from app_settings. NULL/missing row
    # (fresh DB, no settings yet) → default True so the SoM benefit
    # is on out-of-the-box without forcing the user into Settings first.
    som_enabled_default = True
    try:
        from app.models.app_settings import AppSettings  # noqa: PLC0415
        _settings_row = db.query(AppSettings).filter(
            AppSettings.id == 1,
        ).first()
        if _settings_row is not None:
            som_enabled_default = bool(
                getattr(_settings_row, "som_enabled_default", True),
            )
    except Exception:
        pass

    plan = db.get(TestPlan, plan_id)
    if not plan:
        raise ValueError(f"Plan {plan_id} not found")
    if not (plan.target_url and plan.target_url.strip()):
        raise ValueError(
            f"Plan {plan_id} has no target_url — cannot navigate",
        )

    # Phase H — preflight (Scout → Refine → Activate) BEFORE the
    # per-submodule agent loop. Without this, the agent walks
    # BRD-derived sub-goals that don't match the actual UI; with
    # it, the live TcNode tree the agent reads below is the refined
    # plan. Short-circuits when a fresh app_map_refined version is
    # already active.
    if preflight != "skip" and provider is not None:
        from app.services.preflight import run_preflight  # noqa: PLC0415

        _emit(emit_event, "phase", {
            "phase": "preflight",
            "message": (
                "Validating test cases against the actual UI before "
                "execution (scout + refine)"
            ),
        })
        try:
            pf = run_preflight(
                db,
                plan_id=plan_id,
                provider=provider,
                cheap_provider=cheap_provider,
                force=(preflight == "force"),
                headless=True,
                emit_event=emit_event,
                is_cancelled=is_cancelled,
            )
            if pf.status == "failed":
                logger.warning(
                    "preflight failed for plan %s: %s — proceeding "
                    "with current TC tree as-is",
                    plan_id, pf.error_message,
                )
        except Exception as e:
            logger.exception(
                "preflight raised in run_qa_agent_for_plan; "
                "continuing with live TC tree",
            )
            _emit(emit_event, "preflight_failed", {
                "plan_id": plan_id,
                "stage": "outer",
                "error": str(e)[:200],
            })

    project_id = plan.project_id

    # Phase C.3 — TC version label. Activation OVERWRITES the live
    # TcNode tree (audit-trail semantics chosen by the user), so the
    # agent always reads the live tree. ``current_tc_version_id`` is
    # kept as a UI label so the operator can see which version is
    # active + roll back via the plan editor.
    using_tc_version_id = getattr(plan, "current_tc_version_id", None)
    _emit(emit_event, "phase", {
        "phase": "loading_steps",
        "message": (
            f"Loading TC tree (active version v{using_tc_version_id}) "
            f"for plan '{plan.name}'"
            if using_tc_version_id
            else f"Loading TC tree for plan '{plan.name}' (agentic mode)"
        ),
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

            # Plan-scoped page memory — passed into every submodule's
            # agent loop. URLs catalogued during submodule N stay
            # readable to submodule N+1, so e.g. a login screen
            # already mapped during auth doesn't get re-scraped on a
            # follow-up checkout flow that bounces through it.
            plan_page_memory: dict[str, dict[str, Any]] = {}

            # γ.2 — cross-submodule frozen-flow bundle. We collect
            # the (submodule, frozen) pairs as submodules pass + bundle
            # them onto the parent MODULE node at run end. Future
            # replays can walk the parent's bundle in one shot rather
            # than re-orchestrating across submodules. Only filled
            # when the run keeps stringing together passes; any
            # blocked / failed / disputed submodule resets the streak.
            bundle_pairs: list[tuple[Any, dict[str, Any]]] = []

            # α.5 — Plan-scoped WorldState. Carried across submodules
            # so submodule N can assert preconditions set up by
            # submodule N-1 (cart_count, logged_in_as, etc.). Loaded
            # from the run row (None for legacy / fresh runs); saved
            # back at every submodule boundary.
            from app.models.agent_run import AgentRun  # noqa: PLC0415
            from app.services.world_state import (  # noqa: PLC0415
                load_world_state, save_world_state,
            )
            run_row = db.get(AgentRun, run_id)
            world_state: dict[str, Any] = (
                load_world_state(run_row) if run_row else {}
            )

            # ── Cost tracking — open the run context + snapshot ───
            # model names so the cost service can resolve pricing
            # against the model used AT RUN TIME (changing the LLM
            # config later doesn't silently re-cost historical runs
            # with the new model's name). The cost service still
            # applies the LATEST pricing — re-costing a run at
            # today's $/M is the more useful question for tracking
            # model price changes over time.
            #
            # ``run_id`` is passed in so end_run can flush the per-
            # call buffer into ``llm_call_logs`` with the right FK.
            from app.llm.cost_tracker import (  # noqa: PLC0415
                begin_run as _begin_cost,
            )
            _begin_cost(run_id=run_id)
            if run_row is not None:
                try:
                    run_row.strong_model_snapshot = getattr(
                        provider, "model", None,
                    )
                    run_row.cheap_model_snapshot = (
                        getattr(cheap_provider, "model", None)
                        if cheap_provider is not None
                        else None
                    )
                    db.commit()
                except Exception:
                    try:
                        db.rollback()
                    except Exception:
                        pass

            # Phase A.6 Step 6 — build a plan-wide submodule summary
            # once so the reconciliation pass (which fires inside the
            # first submodule, after scout) can compare the AppMap
            # against EVERY submodule in one VL call. Cheap to build,
            # tiny payload (~N short objects).
            plan_submodules_summary: list[dict[str, Any]] = []
            for sm, _steps in groups:
                plan_submodules_summary.append({
                    "submodule_id": sm.id,
                    "title": (sm.title or "")[:160],
                    "description": (
                        getattr(sm, "description_md", "") or ""
                    )[:600],
                })

            # Phase D — pre-submodule validation gate. Before each
            # submodule runs, look up the active TcVersion's
            # validation rollup for THIS submodule and emit a
            # cautionary event when confidence is low or any step is
            # marked unreachable. This mirrors what a human QA does:
            # glance at the test case, decide "this one's risky", note
            # it before pressing play. Best-effort — missing version /
            # missing validation → no warning, agent proceeds normally.
            validation_by_submodule_id: dict[int, dict[str, Any]] = {}
            try:
                if plan.current_tc_version_id:
                    from app.models.tc_version import (  # noqa: PLC0415
                        TcNodeSnapshot,
                    )
                    snaps = list(db.execute(
                        select(TcNodeSnapshot).where(
                            TcNodeSnapshot.tc_version_id ==
                            plan.current_tc_version_id,
                        ),
                    ).scalars())
                    for s in snaps:
                        if (
                            s.kind == "submodule"
                            and s.original_tc_node_id is not None
                            and s.validation_status
                            and s.validation_status != "pending"
                        ):
                            validation_by_submodule_id[
                                int(s.original_tc_node_id)
                            ] = {
                                "status": s.validation_status,
                                "confidence": s.validation_confidence,
                                "reason": s.validation_reason,
                            }
            except Exception as e:
                logger.debug(
                    "pre-submodule validation lookup skipped: %s", e,
                )

            for idx, ((submodule, steps), row) in enumerate(zip(groups, rows)):
                if is_cancelled and is_cancelled():
                    cancelled = True
                    break

                # Phase D — emit a pre-run health signal so the
                # operator sees in the live feed which submodules the
                # validator marked low-confidence. Doesn't block
                # execution; just surfaces the risk.
                pre_val = validation_by_submodule_id.get(submodule.id)
                if pre_val is not None:
                    _emit(emit_event, "submodule_pre_run_health", {
                        "run_id": run_id,
                        "step_id": row.id,
                        "submodule_id": submodule.id,
                        "title": submodule.title,
                        "validation_status": pre_val.get("status"),
                        "validation_confidence": pre_val.get("confidence"),
                        "validation_reason": pre_val.get("reason"),
                        "is_risky": (
                            pre_val.get("status") in (
                                "unresolved", "unreachable",
                            )
                            or (
                                isinstance(
                                    pre_val.get("confidence"), (int, float),
                                )
                                and float(pre_val["confidence"]) < 0.5
                            )
                        ),
                    })

                # Diag.4 — submodule state reset. Between submodules,
                # dismiss any leftover drawer / modal AND scroll back
                # to the top so the next decomposer call sees a clean
                # base state. Without this, submodule N+1 starts with
                # submodule N's drawer still open — the decomposer
                # then plans "close this drawer first" or worse picks
                # actions against the stale UI.
                # Skipped for idx==0 (no previous state to clean up).
                if idx > 0:
                    try:
                        _reset_inter_submodule_state(page)
                        _emit(emit_event, "submodule_state_reset", {
                            "step_id": row.id,
                            "submodule_id": submodule.id,
                            "ordinal": idx + 1,
                        })
                    except Exception as e:
                        logger.debug(
                            "submodule state reset failed (non-fatal): %s",
                            e,
                        )

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

                # Cost tracking: tell the per-call buffer which step
                # the upcoming calls belong to, so the drill-in view
                # can group "all calls on submodule N" cleanly.
                try:
                    from app.llm.cost_tracker import (  # noqa: PLC0415
                        set_current_step,
                    )
                    set_current_step(row.id)
                except Exception:
                    pass

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
                    page_memory=plan_page_memory,
                    cheap_provider=cheap_provider,
                    agent_strategy=agent_strategy,
                    world_state=world_state,
                    db=db,
                    plan=plan,
                    open_typed_prompt=open_typed_prompt,
                    request_intervention=wait_for_intervention,
                    som_enabled=som_enabled_default,
                    plan_submodules_summary=(
                        plan_submodules_summary if idx == 0 else None
                    ),
                )

                # α.5 — persist WorldState after every submodule so
                # the next one sees the latest cart/login/url state.
                if run_row is not None:
                    save_world_state(db, run_row, world_state)

                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens
                total_llm_calls += result.llm_calls
                total_vision_calls += result.vision_calls

                # Phase L — settle the page BEFORE capturing evidence
                # so screenshots reflect the post-action state, not a
                # mid-navigation frame. The narration overlay is hidden
                # only AFTER the capture to keep the existing "what the
                # agent said" caption on the screenshot — but the
                # underlying page is given a full settle window first.
                screenshot = _take_screenshot(
                    page, run_id, row.id,
                    speed_config=speed_config,
                    post_settle_ms=1200,
                )
                screenshot_meta = _capture_screenshot_meta(page)
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
                row.details_json = _json_safe({
                    "mode": "agentic",
                    "goal": goal.to_dict(),
                    "halt_reason": result.halt_reason,
                    "divergence": divergence,
                    # Phase L — screenshot metadata so the report can
                    # detect stale-screenshot binding (URL at capture
                    # time vs the goal's expected destination).
                    "screenshot_meta": screenshot_meta,
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
                    # Phase A — VL-derived sub-goal timeline.
                    "sub_goals": list(getattr(result, "sub_goals", []) or []),
                    "replans_used": int(
                        getattr(result, "replans_used", 0) or 0
                    ),
                })
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
                #
                # Also skip when any turn was rescued by a human
                # typing into the HITL popup (creds / OTP / captcha
                # solve). Those values are one-time and the page
                # state on a future replay won't match — freezing
                # such a path would hard-code a stale secret as a
                # selector arg or canonicalize a non-deterministic
                # step. The human still gets prompted next time.
                manual_intervention_in_run = any(
                    getattr(t, "manual_intervention_used", False)
                    for t in result.turn_log
                ) or bool(
                    getattr(result, "manual_intervention_used", False)
                )
                # Phase O.4 — gate freeze on observed-criteria verification.
                # Walk every sub-goal that was marked done AND has a
                # non-empty success_criterion; re-verify against the
                # CURRENT page (the post-submodule state). If any
                # such sub-goal fails its observable signal NOW, the
                # "passed" status was claimed but not proven — refuse
                # to freeze the recipe so the next run doesn't replay
                # a hallucinated path deterministically.
                criteria_all_verified = True
                criteria_unverified: list[dict[str, str]] = []
                try:
                    end_observation = _capture_observation(page)
                except Exception:
                    end_observation = {"url": "", "title": ""}
                runtime_sgs_for_audit = list(
                    getattr(result, "sub_goals", []) or []
                )
                for sg_dict in runtime_sgs_for_audit:
                    if not isinstance(sg_dict, dict):
                        continue
                    if sg_dict.get("status") != "done":
                        continue
                    crit = (sg_dict.get("success_criterion") or "").strip()
                    if not crit:
                        continue
                    try:
                        ok_obs, reason = _verify_subgoal_criterion(
                            page,
                            criterion=crit,
                            observation=end_observation,
                        )
                    except Exception:
                        ok_obs, reason = True, "verify raised — skipped"
                    if not ok_obs:
                        criteria_all_verified = False
                        criteria_unverified.append({
                            "sub_goal_id": str(sg_dict.get("id", "?")),
                            "criterion": crit[:160],
                            "reason": reason[:160],
                        })
                if criteria_unverified:
                    _emit(emit_event, "freeze_refused_unverified_criteria", {
                        "run_id": run_id,
                        "step_id": row.id,
                        "details": criteria_unverified[:10],
                    })

                should_freeze = (
                    result.status == "passed"
                    and vision_verdict in (None, "pass")
                    and not manual_intervention_in_run
                    and criteria_all_verified
                )

                # α.5 — apply postconditions to WorldState on success
                # so the NEXT submodule's preconditions can match. Done
                # before freeze + before save_world_state above (which
                # runs in the per-submodule loop tail).
                if result.status == "passed":
                    try:
                        from app.services.world_state import (  # noqa: PLC0415
                            apply_postconditions,
                        )
                        last_url = ""
                        if result.turn_log:
                            last_url = (
                                result.turn_log[-1].page_url or ""
                            )
                        apply_postconditions(
                            world_state,
                            list(goal.postconditions or []),
                            current_url=last_url,
                        )
                    except Exception as e:
                        logger.debug(
                            "WorldState postcondition apply skipped: %s", e,
                        )

                # γ.2 — frozen-path summary into AKB so future runs
                # querying "how do I do X on this app" see the proven
                # working flow (text only — no selectors).
                if should_freeze:
                    # Phase B Step 2 — prefer the v2 (per-sub-goal
                    # segmented) frozen path when the run had VL-
                    # derived runtime sub-goals. v2 lets replay
                    # walk sub-goal-by-sub-goal with handoff to the
                    # agent for any sub-goal that doesn't have a
                    # frozen segment. Falls back to v1 when there
                    # are no sub-goals (legacy / fallback runs).
                    runtime_sgs_for_freeze = list(
                        getattr(result, "sub_goals", []) or []
                    )
                    # Convert dict shape back into objects with .id
                    # / .description / .status / etc. for the
                    # segment builder. RuntimeSubGoal stores
                    # to_dict() into result.sub_goals; we mirror
                    # back into a light wrapper here.
                    runtime_sg_objs: list[Any] = []
                    for d in runtime_sgs_for_freeze:
                        if not isinstance(d, dict):
                            continue
                        from app.agents.sub_goals import (  # noqa: PLC0415
                            RuntimeSubGoal,
                        )
                        runtime_sg_objs.append(RuntimeSubGoal(
                            id=str(d.get("id", "")),
                            description=str(d.get("description", "")),
                            success_criterion=str(
                                d.get("success_criterion", ""),
                            ),
                            max_turns=int(d.get("max_turns", 6)),
                            status=str(d.get("status", "pending")),  # type: ignore[arg-type]
                            replan_iteration=int(
                                d.get("replan_iteration", 0),
                            ),
                        ))
                    frozen_v2: dict[str, Any] | None = None
                    if runtime_sg_objs:
                        frozen_v2 = _build_frozen_path_segments(
                            run_id=run_id,
                            goal=goal,
                            turn_log=result.turn_log,
                            runtime_sub_goals=runtime_sg_objs,
                            agent_model=getattr(provider, "model", None),
                        )
                    frozen = frozen_v2 or _build_frozen_path(
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
                            step_count = (
                                sum(
                                    len(seg.get("steps", []))
                                    for seg in frozen.get("segments", [])
                                )
                                if frozen.get("version") == 2
                                else len(frozen.get("steps", []))
                            )
                            _emit(emit_event, "frozen_path_captured", {
                                "step_id": row.id,
                                "tc_node_id": submodule.id,
                                "step_count": step_count,
                                "agent_model": frozen.get("agent_model"),
                                "version": frozen.get("version", 1),
                                "segments": (
                                    len(frozen.get("segments", []))
                                    if frozen.get("version") == 2
                                    else None
                                ),
                            })
                            logger.info(
                                "froze path for submodule %s "
                                "(%d steps, v%s) from run %s",
                                submodule.id, step_count,
                                frozen.get("version", 1), run_id,
                            )
                            # γ.2 — append to the cross-submodule
                            # bundle. The parent module gets a
                            # ``frozen_path`` whose ``steps`` is the
                            # CONCATENATION of every consecutive
                            # passing submodule's steps. Replay can
                            # walk the parent in one shot, OR fall
                            # through to per-submodule frozen paths
                            # when only some submodules under the
                            # module have passed cleanly.
                            bundle_pairs.append((submodule, frozen))
                            # γ.2 — write a text summary of the frozen
                            # flow to AKB so future RUNS asking the
                            # AKB "how do I X on this app" find it.
                            try:
                                from app.services.akb_ingest import (  # noqa: PLC0415
                                    ingest_frozen_path_summary,
                                )
                                ingest_frozen_path_summary(
                                    db,
                                    target_url=plan.target_url or "",
                                    submodule_title=(
                                        submodule.title or ""
                                    ),
                                    frozen_path=frozen,
                                    source_run_id=run_id,
                                )
                            except Exception as e:
                                logger.debug(
                                    "AKB frozen-summary write skipped: %s",
                                    e,
                                )
                else:
                    # Non-passing OR disputed submodule — reset the
                    # bundle streak. We only freeze CONTIGUOUS passing
                    # sequences as a bundle.
                    if bundle_pairs:
                        _persist_module_bundle(
                            db, bundle_pairs,
                            plan_target_url=plan.target_url or "",
                            run_id=run_id,
                            agent_model=getattr(provider, "model", None),
                        )
                        bundle_pairs = []

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
        # γ.2 — flush any in-flight cross-submodule bundle. The streak
        # might have ended on the last submodule (no "else" branch
        # fires after the loop exits cleanly). Persist whatever
        # contiguous passing streak survived to here.
        try:
            if bundle_pairs:
                _persist_module_bundle(
                    db, bundle_pairs,
                    plan_target_url=plan.target_url or "",
                    run_id=run_id,
                    agent_model=getattr(provider, "model", None),
                )
        except Exception as e:
            logger.debug(
                "module bundle final flush skipped: %s", e,
            )

        # ── Cost tracking — close the context + persist counters ──
        # to the AgentRun row + flush the per-call buffer to
        # ``llm_call_logs`` (end_run handles the call-log insert
        # internally when db is supplied). Best-effort; failure
        # here leaves the aggregate tokens (on output_summary_json)
        # still visible in the report.
        try:
            from app.llm.cost_tracker import (  # noqa: PLC0415
                end_run as _end_cost,
            )
            counters = _end_cost(db=db)
            if counters is not None and run_row is not None:
                run_row.strong_input_tokens = counters.strong_input
                run_row.strong_output_tokens = counters.strong_output
                run_row.cheap_input_tokens = counters.cheap_input
                run_row.cheap_output_tokens = counters.cheap_output
                run_row.strong_cached_input_tokens = (
                    counters.strong_cached_input
                )
                run_row.cheap_cached_input_tokens = (
                    counters.cheap_cached_input
                )
                db.commit()
        except Exception as e:
            logger.debug("cost counters persist skipped: %s", e)
            try:
                db.rollback()
            except Exception:
                pass
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
