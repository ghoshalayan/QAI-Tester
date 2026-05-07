"""AI page intelligence — LLM looks at the live page and proposes actions.

The single primitive that backs:
- :func:`propose_recovery` — when an executor step has exhausted auto-retries,
  ask the LLM to look at the live page (accessibility tree) and propose
  what to do: retry as-is, replace with a corrected step, or give up.
- (future) NL command → step generation, live state narration,
  conversational override during HITL.

Why accessibility tree over DOM HTML
------------------------------------
The AX tree is what assistive tech (and humans) see: roles, names, values,
nested structure. It's ~10x smaller than rendered HTML and free of
class-name churn, so prompts stay bounded and the LLM's pattern matching
is more reliable across page versions.

Vision escalation
-----------------
The first call is text-only (AX tree). Callers can re-call with
``include_screenshot=True`` when the first suggestion also failed —
useful when the failure is visual (overlay covering target, button
visually disabled but DOM-enabled, etc.). Vision support depends on the
configured provider/model; see ``app.llm.factory``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from playwright.sync_api import Page

from app.llm.base import ChatMessage, LLMProvider

logger = logging.getLogger(__name__)


RecoveryAction = Literal["retry", "replace", "give_up"]


@dataclass
class ImprovisationSuggestion:
    """A concrete value the LLM picked from the live page when the test
    case left it ambiguous (e.g. "search any product").

    ``value`` is empty when the LLM couldn't decide — caller should treat
    that as "no improvisation available" and let the action fail / HITL.
    """

    value: str
    reasoning: str
    confidence: float
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class RecoverySuggestion:
    """Structured proposal from the LLM after looking at a failed step."""

    action: RecoveryAction
    new_target_hint: str = ""
    new_action_type: str = ""
    new_expected: str = ""
    new_narrative: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    # Telemetry — useful for the run-detail diff view.
    input_tokens: int | None = None
    output_tokens: int | None = None
    used_vision: bool = False


# OpenAI-strict + Gemini-friendly: every property in ``required``,
# ``additionalProperties: false``. Empty strings are the contract for
# "no change to this field" since strict-JSON forbids omitted fields.
RECOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["retry", "replace", "give_up"],
        },
        "new_target_hint": {"type": "string"},
        "new_action_type": {"type": "string"},
        "new_expected": {"type": "string"},
        "new_narrative": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "action",
        "new_target_hint",
        "new_action_type",
        "new_expected",
        "new_narrative",
        "reasoning",
        "confidence",
    ],
    "additionalProperties": False,
}


IMPROVISATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "value": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["value", "reasoning", "confidence"],
    "additionalProperties": False,
}


# ── A4.1b: vision-guided target search ────────────────────────────
#
# Schema for the search-action LLM call. The model sees a screenshot
# of the live page + the target_hint that just failed to resolve, and
# returns ONE concrete next step the agent should take to find the
# target (scroll, navigate, click a card to drill in, dismiss a
# blocking modal, or give up). The orchestrator dispatches the action,
# refreshes the page state, and retries the original target.

SearchAction = Literal[
    "scroll",
    "navigate",
    "click_to_drill",
    "dismiss_modal",
    "give_up",
]


@dataclass
class SearchSuggestion:
    """Vision LLM's recommendation for how to find a missed target."""

    action: SearchAction
    # Tool-specific args. Empty strings when not applicable.
    scroll_direction: str = ""        # "down" | "up" | "left" | "right"
    scroll_amount_px: int = 0          # 0 → use a default page-height
    navigate_url: str = ""             # absolute URL to go to
    click_target_hint: str = ""        # selector / role+name / text
    reasoning: str = ""
    confidence: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


SEARCH_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "scroll",
                "navigate",
                "click_to_drill",
                "dismiss_modal",
                "give_up",
            ],
        },
        "scroll_direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right", ""],
        },
        "scroll_amount_px": {"type": "integer"},
        "navigate_url": {"type": "string"},
        "click_target_hint": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "action", "scroll_direction", "scroll_amount_px",
        "navigate_url", "click_target_hint",
        "reasoning", "confidence",
    ],
    "additionalProperties": False,
}


# ── A4.1a: vision-grounded goal verification ──────────────────────


GoalVerdict = Literal["pass", "fail", "partial"]


@dataclass
class GoalVerification:
    """LLM's screenshot-based judgment on a completed goal."""

    verdict: GoalVerdict
    reasoning: str
    confidence: float
    # Which success criteria the LLM thinks were met / missed.
    criteria_met: list[str] = field(default_factory=list)
    criteria_missed: list[str] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None


GOAL_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "fail", "partial"],
        },
        "criteria_met": {
            "type": "array",
            "items": {"type": "string"},
        },
        "criteria_missed": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "verdict", "criteria_met", "criteria_missed",
        "reasoning", "confidence",
    ],
    "additionalProperties": False,
}


GOAL_VERIFICATION_SYSTEM_PROMPT = """You verify whether a QA test goal was actually achieved by looking at a screenshot.

The agent claimed it completed a test case. You'll see:
- GOAL: one sentence describing what should be true after the test.
- SUCCESS CRITERIA: 2-4 concrete observable signals.
- SCREENSHOT: a PNG of the live page taken right after the agent
  declared completion.

Decide:
1. Is the GOAL achieved on the visible page? Look at the actual
   pixels — not what an AX tree might say, the real visual state.
2. For each success criterion, is it visibly met? Quote the visible
   evidence (toast text, URL bar contents, count badge, etc.).

Verdicts:
- "pass": ALL criteria are visibly met. The goal is achieved.
- "partial": SOME criteria are met but at least one isn't, OR the
  goal is conceptually achieved but not all criteria can be
  verified from this screenshot alone.
- "fail": At least one criterion is visibly NOT met, OR the page
  shows a clear contradiction (error banner, empty cart when the
  test was about adding to cart, login screen when the agent
  thought it was done with checkout, etc.).

Be strict — the agent has a bias toward declaring success. Your job
is to be the objective ground-truth check. If you're not sure, lean
toward "partial" rather than "pass". If the page contradicts the
claim, say "fail".

Always output:
- verdict: pass / partial / fail
- criteria_met: array of criterion strings (verbatim) that you can
  verify from the screenshot.
- criteria_missed: array of criterion strings (verbatim) that the
  screenshot does NOT support or contradicts.
- reasoning: 1-2 sentences citing what you SEE.
- confidence: 0.0 (no idea) to 1.0 (definitive).

Output JSON only.
"""


# ── A4.1c: mid-flow on-track check ────────────────────────────────


@dataclass
class OnTrackCheck:
    """Periodic mid-flow gut-check from a vision LLM."""

    on_track: bool
    suggestion: str  # short hint the agent gets next turn (or "")
    reasoning: str
    confidence: float
    input_tokens: int | None = None
    output_tokens: int | None = None


ON_TRACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "on_track": {"type": "boolean"},
        "suggestion": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["on_track", "suggestion", "reasoning", "confidence"],
    "additionalProperties": False,
}


ON_TRACK_SYSTEM_PROMPT = """You are watching a QA test agent work and assessing if it's making progress.

You see:
- GOAL: what the agent is trying to achieve.
- SUB-GOALS so far: which milestones the agent has closed and which
  are still open.
- RECENT ACTIONS: the agent's last 3-5 turns, what it tried, what
  happened.
- SCREENSHOT: the live page right now.

Decide ONE thing: is the agent on track, or is it wandering /
stalling / pursuing the wrong path?

- on_track=true: the agent's recent actions are advancing the goal,
  the screenshot shows the kind of page it should be on, sub-goal
  progress matches the actions taken. Default to true unless you
  see a clear problem.
- on_track=false: agent is on the wrong page, repeating the same
  failing action, or working on a sub-goal it has no reasonable
  path to from here. Suggest ONE concrete next action (1 sentence)
  that would get it back on track.

Examples of "off track":
- Goal is checkout, screenshot shows the help center page.
- Agent has clicked the same broken element 3 turns running.
- All sub-goals say "in progress" but no observable change happened
  on any of them in the last 5 turns.

Rules:
- Don't second-guess minor decisions; only flag genuinely-stuck
  patterns.
- ``suggestion`` should be empty string when on_track=true.
- ``confidence`` 0.0-1.0: how sure you are about your verdict.

Output JSON only.
"""


SEARCH_ACTION_SYSTEM_PROMPT = """You help a browser-testing agent find a target element it just failed to locate on the page.

You see:
- TARGET: the human-readable description of what the agent was trying
  to interact with (e.g. "Add to Cart button", "search field").
- NEAR_MISSES: the closest elements on the visible page (with their
  similarity scores) that the literal+fuzzy resolver already
  considered and rejected as too dissimilar. These are NOT the target.
- SCREENSHOT: a PNG of the page right now.

Decide ONE concrete next step the agent should take to find the target:

1. ``scroll`` — the target is plausibly OFFSCREEN in this direction.
   Use this when you can see a list / grid / form continuing past
   the visible area and the target is the kind of thing that lives
   in the offscreen continuation. Set ``scroll_direction``
   ("down"/"up"/"left"/"right") and ``scroll_amount_px`` (0 = one
   viewport's worth, the default; otherwise specific pixels).

2. ``click_to_drill`` — the target is on a DIFFERENT VIEW you can
   reach by clicking something visible (e.g. on a search results
   page, click a product card to reach its detail page where
   "Add to Cart" lives). Set ``click_target_hint`` to a Playwright-
   resolvable selector — prefer text "exact label" or
   ``role=img[name='exact alt text']`` over CSS classes.

3. ``navigate`` — the target lives at a known specific URL the user
   can deduce from the page (e.g. /cart for cart contents). Set
   ``navigate_url``. Use sparingly; prefer click_to_drill when in
   doubt.

4. ``dismiss_modal`` — a cookie banner / signup popup / age gate is
   obviously covering the page. Closing it should reveal the target.

5. ``give_up`` — the target is genuinely not on this page or
   reachable from here (page is wrong, feature missing, login
   required and not shown). Don't speculate; if you don't see a
   path, say give_up and explain why.

Rules:
- Pick the SINGLE most-likely-to-work action.
- Be specific in ``click_target_hint`` and ``navigate_url`` —
  agent will pass them straight to Playwright.
- ``confidence`` 0.0-1.0: how sure you are this finds the target.
  Below 0.4 = essentially a guess, treat give_up as the safer call.
- ``reasoning``: 1 sentence citing what you SEE in the screenshot.

Set every field; use "" / 0 for fields not applicable to your action.
Output JSON only.
"""


IMPROVISATION_SYSTEM_PROMPT = """You are testing a web application like a human tester.

You see a step that needs a CONCRETE VALUE to type or select, but the test
case doesn't specify what (e.g. "search any product", "select any option").
Look at the page's accessibility tree and pick a sensible concrete value
FROM the page itself.

Rules:
- Pick something visible and stable that's actually on the page.
- Don't fabricate — only use values that appear in the AX tree (product
  names, menu options, labels, etc.).
- For "search" / "type any X" steps: pick a real X visible on the page.
- For dropdowns: pick an option from the listed options.
- A human tester would naturally pick the FIRST sensible match — do that.
- Keep the value short and exact; copy the visible text verbatim.
- If nothing reasonable is visible, return an empty string for "value".

Always set:
- value: the concrete string to type/select, or "" if you can't decide.
- reasoning: 1 sentence. Cite the element you saw (e.g. "first product
  visible in the catalog grid: 'Wireless mouse'").
- confidence: 0.0 (wild guess) to 1.0 (clearly visible match).

Output JSON only.
"""


SYSTEM_PROMPT = """You are debugging a failed Playwright test step against a live web page.

You see the original step (title, action_type, target_hint, narrative, expected),
the recent attempt errors, and the page's accessibility tree (clean DOM-equivalent
with roles, names, values).

Decide the SMALLEST fix that makes the step succeed:

- action="retry": the page is racing or animating. Try again as-is.
  Set every "new_*" field to "" (empty string).

- action="replace": one or more fields were authored wrong. Provide
  corrections only for the fields that actually need to change; leave
  others as "". Pick selectors visible in the AX tree.

- action="give_up": the page genuinely doesn't support this step
  (page navigated away, element doesn't exist, login expired, etc.).
  Set every "new_*" field to "".

Field rules (when action="replace"):
- new_target_hint: a Playwright-resolvable hint. Either:
    * a CSS selector ("button[data-testid='signin']")
    * a text marker ("text 'Sign In'")
    * a role+name query ("role=button[name='Sign In']")
    * empty "" if the original target_hint was correct.
  Pick a STABLE selector. Avoid index-based selectors ("nth-child")
  unless nothing else works. Prefer data-testid, then text, then role.

- new_action_type: one of navigate / click / type / select / verify /
  wait / submit / screenshot. Empty "" if unchanged.

- new_expected: corrected assertion text, OR "" if unchanged.
- new_narrative: corrected human description, OR "" if unchanged.

Always set:
- reasoning: 1-2 sentences. Cite a specific element you saw in the AX tree
  (e.g. "Found a button with role=button name='Login' instead of 'Sign In'").
- confidence: 0.0 (wild guess) to 1.0 (very confident the fix will work).

Output JSON only.
"""


# Custom DOM walker — Playwright Python sync API doesn't expose
# ``page.accessibility``, so we evaluate this script to build an
# AX-tree-equivalent summary the LLM can reason about. Captures
# interactive elements with role, accessible name, stable identifiers
# (id, data-testid), and viewport-visibility flag.
_PAGE_SUMMARY_JS = r"""
() => {
  const SELECTOR = [
    'a', 'button', 'input', 'select', 'textarea',
    '[role]', '[aria-label]', '[data-testid]',
    'h1', 'h2', 'h3', 'label', 'summary',
  ].join(',');

  function visibleName(el) {
    return (
      el.getAttribute('aria-label') ||
      (el.tagName === 'INPUT' ? (el.placeholder || el.value || '') : '') ||
      (el.tagName === 'IMG' ? (el.alt || '') : '') ||
      (el.textContent || '').trim()
    ).replace(/\s+/g, ' ').slice(0, 100);
  }

  function classList(el) {
    if (typeof el.className !== 'string') return '';
    const parts = el.className.trim().split(/\s+/).filter(Boolean);
    return parts.slice(0, 3).join('.');
  }

  function selectorHints(el) {
    const out = {};
    if (el.id) out.id = '#' + el.id;
    const testid = el.getAttribute('data-testid');
    if (testid) out.testid = "[data-testid='" + testid + "']";
    const cls = classList(el);
    if (cls) out.classes = '.' + cls;
    return out;
  }

  const items = [];
  const all = document.querySelectorAll(SELECTOR);
  for (const el of all) {
    const rect = el.getBoundingClientRect();
    const visible = rect.width > 0 && rect.height > 0
      && getComputedStyle(el).visibility !== 'hidden'
      && getComputedStyle(el).display !== 'none';
    if (!visible) continue;
    items.push({
      tag: el.tagName.toLowerCase(),
      role: el.getAttribute('role') || el.tagName.toLowerCase(),
      name: visibleName(el),
      type: el.getAttribute('type') || undefined,
      ...selectorHints(el),
      // Skip the cursor/banner overlay nodes (z-index 2147483647)
    });
    if (items.length >= 120) break;
  }

  return {
    url: location.href,
    title: document.title || '',
    items: items.filter(x => !(x.id && x.id.startsWith('#__qai-'))),
  };
}
"""


def _capture_page_summary(page: Page, *, max_chars: int = 8_000) -> str:
    """Build an AX-tree-equivalent summary via a DOM walker in the page.

    Playwright Python's sync ``Page`` doesn't expose ``accessibility``, so
    we run a JS script that gathers interactive elements with role +
    accessible name + stable selector hints (id, data-testid, classes).
    Result is JSON-serialized + truncated to ``max_chars`` so prompts
    stay bounded.

    Errors return a placeholder string rather than raising — a missing
    summary shouldn't break the whole recovery flow.
    """
    try:
        summary = page.evaluate(_PAGE_SUMMARY_JS)
    except Exception as e:
        logger.warning("page summary capture failed: %s", e)
        return "(failed to capture page summary)"
    if not summary:
        return "(empty page summary)"
    s = json.dumps(summary, indent=2, ensure_ascii=False)
    if len(s) > max_chars:
        s = s[: max_chars - 24] + "\n... [truncated]"
    return s


def _safe_url(page: Page) -> str:
    try:
        return page.url
    except Exception:
        return "(unknown)"


def propose_recovery(
    provider: LLMProvider,
    page: Page,
    *,
    title: str,
    target_hint: str | None,
    action_type: str | None,
    narrative: str | None,
    expected: str | None,
    error_message: str,
    prior_attempts: list[dict[str, Any]] | None = None,
    include_screenshot: bool = False,
) -> RecoverySuggestion:
    """Ask the LLM what to do about a failed step.

    Args:
        provider: Configured LLM provider (built by the runtime via
            :func:`app.llm.factory.get_provider_from_db`).
        page: The live Playwright page the step ran against.
        title: Step title (snapshot from the TC node).
        target_hint: Original target_hint (may be None).
        action_type: Original action_type.
        narrative: Step narrative (free text from FRD→TC agent).
        expected: Step's expected text (verify steps).
        error_message: The selector-not-found / assertion error from the
            most recent failed attempt.
        prior_attempts: Optional list of previous attempt dicts (the
            ``attempts`` log produced by :func:`_execute_with_retry`).
        include_screenshot: If True, attach a screenshot of the current
            page (vision escalation). Provider must support vision.

    Returns:
        Parsed :class:`RecoverySuggestion`.

    Raises:
        RuntimeError: LLM call failed, response was malformed, or the
            ``action`` field wasn't one of the three enum values. Callers
            should fall back to HITL on this.
    """
    page_summary = _capture_page_summary(page)
    current_url = _safe_url(page)

    # Vision escalation: attach a PNG of the live page so the model can
    # reason about visual state (overlays, layout, disabled-looking-but-
    # enabled buttons, etc.). Best-effort — on capture failure or a
    # vision-incapable provider we silently fall back to text-only.
    screenshot_bytes: bytes | None = None
    if include_screenshot and getattr(provider, "supports_vision", False):
        try:
            screenshot_bytes = page.screenshot(full_page=False)
        except Exception as e:
            logger.warning(
                "propose_recovery: screenshot capture failed: %s", e,
            )
            screenshot_bytes = None
    used_vision = screenshot_bytes is not None

    user_lines = [
        f"PAGE URL: {current_url}",
        "",
        "STEP THAT FAILED:",
        f"  title:        {title or '(none)'}",
        f"  action_type:  {action_type or '(none)'}",
        f"  target_hint:  {target_hint or '(none)'}",
        f"  narrative:    {(narrative or '')[:300]}",
        f"  expected:     {(expected or '')[:300]}",
        "",
        f"ERROR: {error_message[:600]}",
    ]

    if prior_attempts:
        user_lines.extend(["", f"PRIOR ATTEMPTS ({len(prior_attempts)}):"])
        for a in prior_attempts[-3:]:
            user_lines.append(
                f"  attempt {a.get('attempt', '?')}: "
                f"{a.get('status', '?')} — "
                f"{(a.get('narration') or '')[:200]}",
            )

    user_lines.extend(
        [
            "",
            "PAGE SUMMARY (interactive elements, truncated):",
            page_summary,
        ],
    )

    if used_vision:
        user_lines.append(
            "\nA screenshot of the page is attached. Use the visual layout "
            "to disambiguate selectors when the AX tree is ambiguous "
            "(e.g. overlays, two elements with the same accessible name, "
            "visually-disabled-but-DOM-enabled controls).",
        )

    user_prompt = "\n".join(user_lines)

    try:
        result = provider.chat_structured(
            messages=[
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=user_prompt,
                    image=screenshot_bytes,
                ),
            ],
            schema=RECOVERY_SCHEMA,
            schema_name="recovery_suggestion",
            temperature=0.2,
            max_output_tokens=1024,
        )
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed for recovery suggestion: "
            f"{type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"LLM returned unexpected shape for recovery — expected dict, "
            f"got {type(parsed).__name__}",
        )

    action = parsed.get("action")
    if action not in ("retry", "replace", "give_up"):
        raise RuntimeError(
            f"LLM returned invalid action: {action!r}",
        )

    return RecoverySuggestion(
        action=action,  # type: ignore[arg-type]
        new_target_hint=str(parsed.get("new_target_hint", "")),
        new_action_type=str(parsed.get("new_action_type", "")),
        new_expected=str(parsed.get("new_expected", "")),
        new_narrative=str(parsed.get("new_narrative", "")),
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        used_vision=used_vision,
    )


def propose_improvisation(
    provider: LLMProvider,
    page: Page,
    *,
    title: str,
    action_type: str | None,
    target_hint: str | None,
    narrative: str | None,
    expected: str | None,
) -> ImprovisationSuggestion:
    """Pick a concrete value for an ambiguous type/select step.

    Triggered before action dispatch when the test case says e.g.
    "search any product" but doesn't name one. The LLM looks at the
    page's accessibility tree and picks something a human tester
    would naturally try (first visible product, first dropdown option,
    etc.).

    Returns an :class:`ImprovisationSuggestion`. ``value == ""`` means
    "no improvisation available" — caller should let the action proceed
    with whatever the test case had (likely fail → recovery / HITL).
    """
    page_summary = _capture_page_summary(page)
    user_lines = [
        "STEP THAT NEEDS A CONCRETE VALUE:",
        f"  title:        {title}",
        f"  action_type:  {action_type or '(none)'}",
        f"  target_hint:  {target_hint or '(none)'}",
        f"  narrative:    {narrative or ''}",
        f"  expected:     {expected or ''}",
        "",
        f"PAGE URL: {_safe_url(page)}",
        "PAGE SUMMARY (interactive elements visible on the page):",
        page_summary,
        "",
        "Pick a concrete value a human tester would naturally type or "
        "select here. Copy the visible text verbatim.",
    ]
    user_prompt = "\n".join(user_lines)

    try:
        result = provider.chat_structured(
            messages=[
                ChatMessage(
                    role="system",
                    content=IMPROVISATION_SYSTEM_PROMPT,
                ),
                ChatMessage(role="user", content=user_prompt),
            ],
            schema=IMPROVISATION_SCHEMA,
            schema_name="improvisation",
            temperature=0.3,
            max_output_tokens=512,
        )
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed for improvisation: "
            f"{type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"LLM returned unexpected shape for improvisation — expected "
            f"dict, got {type(parsed).__name__}",
        )

    return ImprovisationSuggestion(
        value=str(parsed.get("value", "")).strip(),
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def propose_search_action(
    provider: LLMProvider,
    page: Page,
    *,
    target_hint: str,
    near_misses: list[dict[str, Any]] | None = None,
) -> SearchSuggestion:
    """Vision LLM call: given a missed target_hint and a page screenshot,
    return ONE concrete next step (scroll / click-to-drill / navigate /
    dismiss-modal / give-up) the agent should take to find the target.

    The orchestrator then dispatches that action and retries the
    original target. Caps to a small N retries per missed target so a
    confused LLM can't loop.

    Args:
        provider: Vision-capable LLM provider. Must have
            ``supports_vision=True``; raises if not.
        page: The Playwright Page (used to capture the screenshot).
        target_hint: The hint that the resolver couldn't find.
        near_misses: Optional list of ``{role, name, score}`` dicts
            from the fuzzy resolver — the elements close enough to be
            interesting but not enough to be the target. Helps the
            LLM distinguish "target is offscreen" from "target is
            absent".

    Raises:
        RuntimeError: provider lacks vision OR LLM call fails OR
            response shape is malformed.
    """
    if not getattr(provider, "supports_vision", False):
        raise RuntimeError(
            "propose_search_action requires a vision-capable provider; "
            f"{type(provider).__name__} reports supports_vision=False",
        )

    try:
        screenshot = page.screenshot(full_page=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to capture screenshot for search-action: "
            f"{type(e).__name__}: {e}",
        ) from e

    near_text = "  (none — fuzzy resolver returned no candidates above 0.30)"
    if near_misses:
        near_text = "\n".join(
            f"  - {nm.get('role') or '?'} : "
            f"{(nm.get('name') or '')[:80]!r} (score {nm.get('score')})"
            for nm in near_misses[:5]
        )

    user_prompt = (
        f"TARGET (the agent's literal hint that just missed):\n"
        f"  {target_hint!r}\n\n"
        f"NEAR-MISSES (visible elements the resolver scored but rejected):\n"
        f"{near_text}\n\n"
        f"PAGE URL: {_safe_url(page)}\n\n"
        "Look at the attached screenshot. Pick ONE next step from "
        "{scroll, click_to_drill, navigate, dismiss_modal, give_up} "
        "and fill in the relevant fields."
    )

    try:
        result = provider.chat_structured(
            messages=[
                ChatMessage(
                    role="system",
                    content=SEARCH_ACTION_SYSTEM_PROMPT,
                ),
                ChatMessage(
                    role="user",
                    content=user_prompt,
                    image=screenshot,
                ),
            ],
            schema=SEARCH_ACTION_SCHEMA,
            schema_name="search_action",
            temperature=0.2,
            max_output_tokens=512,
        )
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed for search-action: "
            f"{type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"search-action returned unexpected shape: "
            f"{type(parsed).__name__}",
        )

    action = parsed.get("action")
    if action not in (
        "scroll", "navigate", "click_to_drill",
        "dismiss_modal", "give_up",
    ):
        raise RuntimeError(
            f"search-action returned invalid action: {action!r}",
        )

    return SearchSuggestion(
        action=action,  # type: ignore[arg-type]
        scroll_direction=str(parsed.get("scroll_direction", "")),
        scroll_amount_px=int(parsed.get("scroll_amount_px", 0) or 0),
        navigate_url=str(parsed.get("navigate_url", "")),
        click_target_hint=str(parsed.get("click_target_hint", "")),
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def verify_goal_via_screenshot(
    provider: LLMProvider,
    page: Page,
    *,
    goal_description: str,
    success_criteria: list[str],
) -> GoalVerification:
    """A4.1a: ground-truth check on a claimed-completed goal.

    Captures the live page screenshot and asks the vision LLM to
    confirm or refute the agent's "goal complete" claim against the
    written success criteria. Catches the failure mode where the
    agent saw a positive cue in the AX tree (e.g. a stale 'success'
    toast) and declared victory while the page tells a different
    story.

    Returns ``GoalVerification`` with verdict ``pass`` / ``partial``
    / ``fail`` plus per-criterion verdicts so the report can show
    "the LLM agreed with these claims, disagreed with these."

    Raises:
        RuntimeError: provider lacks vision OR LLM call fails OR
            response shape is malformed.
    """
    if not getattr(provider, "supports_vision", False):
        raise RuntimeError(
            "verify_goal_via_screenshot requires a vision-capable "
            f"provider; {type(provider).__name__} reports "
            "supports_vision=False",
        )

    try:
        screenshot = page.screenshot(full_page=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to capture screenshot for goal verification: "
            f"{type(e).__name__}: {e}",
        ) from e

    crit_block = "\n".join(
        f"  - {c}" for c in success_criteria
    ) or "  (none specified — judge by the goal alone)"
    user_prompt = (
        f"GOAL:\n  {goal_description}\n\n"
        f"SUCCESS CRITERIA:\n{crit_block}\n\n"
        f"PAGE URL: {_safe_url(page)}\n\n"
        "Look at the attached screenshot. Apply the verdict rubric "
        "and return the structured verification."
    )

    try:
        result = provider.chat_structured(
            messages=[
                ChatMessage(
                    role="system",
                    content=GOAL_VERIFICATION_SYSTEM_PROMPT,
                ),
                ChatMessage(
                    role="user",
                    content=user_prompt,
                    image=screenshot,
                ),
            ],
            schema=GOAL_VERIFICATION_SCHEMA,
            schema_name="goal_verification",
            temperature=0.1,
            max_output_tokens=512,
        )
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed for goal verification: "
            f"{type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"goal verification returned unexpected shape: "
            f"{type(parsed).__name__}",
        )

    verdict = parsed.get("verdict")
    if verdict not in ("pass", "partial", "fail"):
        raise RuntimeError(
            f"goal verification returned invalid verdict: {verdict!r}",
        )

    raw_met = parsed.get("criteria_met") or []
    raw_missed = parsed.get("criteria_missed") or []
    return GoalVerification(
        verdict=verdict,  # type: ignore[arg-type]
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        criteria_met=[
            str(c) for c in raw_met if isinstance(c, str) and c.strip()
        ],
        criteria_missed=[
            str(c) for c in raw_missed if isinstance(c, str) and c.strip()
        ],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def check_on_track(
    provider: LLMProvider,
    page: Page,
    *,
    goal_description: str,
    sub_goal_summary: str,
    recent_turns_summary: str,
) -> OnTrackCheck:
    """A4.1c: periodic mid-flow gut-check.

    Called every K turns (default 5) when the agent has unverified
    sub-goals. Catches the "wandering / stalling but technically not
    failing" pattern that the deterministic guards (stall, oscillation)
    miss — e.g. the agent is on the wrong page entirely, or has been
    re-clicking variants of a broken element.

    Returns ``OnTrackCheck`` with ``on_track`` boolean + a one-sentence
    suggestion that the agent loop can inject as a hint into the next
    turn's prompt. Cheap (~1 vision call per N turns).

    Raises:
        RuntimeError: provider lacks vision OR LLM call fails.
    """
    if not getattr(provider, "supports_vision", False):
        raise RuntimeError(
            "check_on_track requires a vision-capable provider",
        )

    try:
        screenshot = page.screenshot(full_page=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to capture screenshot for on-track check: "
            f"{type(e).__name__}: {e}",
        ) from e

    user_prompt = (
        f"GOAL:\n  {goal_description}\n\n"
        f"SUB-GOAL STATUS:\n{sub_goal_summary}\n\n"
        f"RECENT ACTIONS:\n{recent_turns_summary}\n\n"
        f"PAGE URL: {_safe_url(page)}\n\n"
        "Look at the screenshot. Is the agent on track? If not, "
        "give one concrete next-action suggestion."
    )

    try:
        result = provider.chat_structured(
            messages=[
                ChatMessage(
                    role="system",
                    content=ON_TRACK_SYSTEM_PROMPT,
                ),
                ChatMessage(
                    role="user",
                    content=user_prompt,
                    image=screenshot,
                ),
            ],
            schema=ON_TRACK_SCHEMA,
            schema_name="on_track_check",
            temperature=0.2,
            max_output_tokens=256,
        )
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed for on-track check: "
            f"{type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"on-track check returned unexpected shape: "
            f"{type(parsed).__name__}",
        )

    return OnTrackCheck(
        on_track=bool(parsed.get("on_track", True)),
        suggestion=str(parsed.get("suggestion", "")).strip(),
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
