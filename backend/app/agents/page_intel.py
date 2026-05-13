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


def capture_screenshot_for_vision(
    page: Page,
    *,
    full_page: bool = False,
    max_dim: int = 1024,
    quality: int = 80,
    downscale: bool = True,
) -> bytes:
    """Capture + downscale + JPEG-encode a screenshot meant for the
    vision LLM.

    Why this helper exists
    ----------------------
    Raw ``page.screenshot()`` produces a viewport-sized PNG (default
    1280×720 → ~1.5K image tokens at OpenAI "high detail"). Vision
    quality on UI screenshots is unaffected by:
      - dropping to JPEG quality 80 (≈ 60-70 % byte reduction)
      - clamping the longest side to 1024 px (≈ 50 % token reduction
        on most layouts; OpenAI charges per 512×512 tile)

    Combined: ~3-5× cheaper per VL call with no measurable loss in
    field-localisation accuracy on UI screens.

    ``full_page=True`` re-engages full-page capture for retry /
    revalidation passes (per the user's spec: viewport by default,
    full page on repetitive failures or end-of-flow verification
    where above-the-fold isn't enough).

    ``downscale=False`` MUST be set by callers that consume **pixel
    coordinates** from the LLM (e.g. ``propose_click_coordinates``).
    The LLM returns coords in the image's pixel space — if we shrink
    the image, the returned coords are in shrunken space and a
    naive ``page.mouse.click(x, y)`` lands at the wrong place on
    the actual viewport. Vision QUALITY is preserved by downscale;
    coordinate FIDELITY is not. Same goes for any future field-
    coordinate output (auth-flow's screen classifier, etc.).

    Falls back to the raw screenshot bytes if Pillow isn't available
    (e.g. a partially-installed dev environment) — the call still
    succeeds, it just doesn't get the size cut.

    Phase P.1 — narration overlay is HIDDEN before the screenshot and
    RESTORED immediately after. The overlay is a fixed-position banner
    at the bottom of the viewport (~60px tall); without this gate it
    occludes the drawer's Save / Submit buttons in every vision call,
    so the model literally cannot see the button to propose coords for.
    Cheap: two ``window.__qaiHideBanner / __qaiShowBanner`` evals.
    """
    # Phase P.1 — hide the narration overlay if installed. Best-effort:
    # if the overlay isn't installed (e.g. running headless without
    # the overlay) the eval no-ops via the `&&` short-circuit.
    try:
        page.evaluate(
            "window.__qaiHideBanner && window.__qaiHideBanner()",
        )
    except Exception:
        pass
    try:
        raw = page.screenshot(full_page=full_page)
    finally:
        # Restore the overlay so the operator sees live narration
        # again. The most-recent narration text is preserved (we only
        # toggled opacity, not the text content).
        try:
            page.evaluate(
                "window.__qaiShowBanner && window.__qaiShowBanner()",
            )
        except Exception:
            pass
    if not downscale:
        return raw
    try:
        from io import BytesIO  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        logger.debug(
            "Pillow not installed — sending raw screenshot bytes",
        )
        return raw

    try:
        img = Image.open(BytesIO(raw))
        # Drop alpha → JPEG can't carry it. White background matches
        # what users see on a normal page render.
        if img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        longest = max(w, h)
        if longest > max_dim:
            scale = max_dim / longest
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, Image.LANCZOS)

        out = BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()
    except Exception as e:
        logger.warning(
            "screenshot downscale failed (%s) — sending raw bytes",
            e,
        )
        return raw


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
            screenshot_bytes = capture_screenshot_for_vision(page)
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
    screenshot_bytes: bytes | None = None,
    cheap_provider: LLMProvider | None = None,
    on_escalate: Any = None,
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

    if screenshot_bytes is not None:
        screenshot = screenshot_bytes
    else:
        try:
            screenshot = capture_screenshot_for_vision(page)
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

    messages = [
        ChatMessage(
            role="system",
            content=SEARCH_ACTION_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=user_prompt,
            image=screenshot,
        ),
    ]
    try:
        # Phase 1 — route through tiered router. Cheap tier handles
        # most "should I scroll or click-to-drill?" calls fine; we
        # escalate to ``provider`` only when the cheap model's
        # confidence is < 0.7 OR returns a malformed action. When
        # ``cheap_provider`` isn't supplied (legacy callers), the
        # router routes everything to ``provider`` — original
        # behavior.
        from app.llm.router import (  # noqa: PLC0415
            LLMRole, call_for_role,
        )

        def _validate(parsed: Any) -> bool:
            if not isinstance(parsed, dict):
                return False
            return parsed.get("action") in (
                "scroll", "navigate", "click_to_drill",
                "dismiss_modal", "give_up",
            )

        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=LLMRole.VISION_SEARCH,
            messages=messages,
            schema=SEARCH_ACTION_SCHEMA,
            schema_name="search_action",
            temperature=0.2,
            max_output_tokens=512,
            validate=_validate,
            on_escalate=on_escalate,
        )
        result = tiered.chat
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
    cheap_provider: LLMProvider | None = None,
    on_escalate: Any = None,
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
        # Goal verification — full_page=True so the LLM sees the whole
        # post-flow result (e.g. confirmation messages below the fold,
        # cart contents at the bottom of a long page). The audit's
        # spec: viewport by default, full page on revalidation.
        screenshot = capture_screenshot_for_vision(page, full_page=True)
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

    messages = [
        ChatMessage(
            role="system",
            content=GOAL_VERIFICATION_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=user_prompt,
            image=screenshot,
        ),
    ]
    try:
        from app.llm.router import (  # noqa: PLC0415
            LLMRole, call_for_role,
        )

        def _validate(parsed: Any) -> bool:
            if not isinstance(parsed, dict):
                return False
            return parsed.get("verdict") in ("pass", "partial", "fail")

        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=LLMRole.GOAL_VERIFIER,
            messages=messages,
            schema=GOAL_VERIFICATION_SCHEMA,
            schema_name="goal_verification",
            temperature=0.1,
            max_output_tokens=512,
            validate=_validate,
            on_escalate=on_escalate,
        )
        result = tiered.chat
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
    cheap_provider: LLMProvider | None = None,
    on_escalate: Any = None,
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
        screenshot = capture_screenshot_for_vision(page)
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

    messages = [
        ChatMessage(role="system", content=ON_TRACK_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt, image=screenshot),
    ]
    try:
        from app.llm.router import (  # noqa: PLC0415
            LLMRole, call_for_role,
        )

        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=LLMRole.ON_TRACK_CHECK,
            messages=messages,
            schema=ON_TRACK_SCHEMA,
            schema_name="on_track_check",
            temperature=0.2,
            max_output_tokens=256,
            on_escalate=on_escalate,
        )
        result = tiered.chat
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


# ── Last-resort: pixel-coordinate click (Operator / Computer-Use pattern) ──


@dataclass
class CoordinateClick:
    """Vision LLM's pixel-level pointing for an unresolvable target.

    When DOM-based resolution exhausts every layer (literal selectors,
    fuzzy AX-tree match, vision-guided scroll/drill/dismiss/navigate)
    and the agent still can't find the target, we ask the vision LLM
    to point at PIXEL COORDINATES. The orchestrator then dispatches
    the click via ``page.mouse.click(x, y)`` directly, bypassing the
    DOM entirely.

    This is the same pattern OpenAI Operator and Anthropic Computer
    Use are built on. It works for the elements DOM can't reach:
    canvas-rendered controls, custom widgets in shadow DOM that hides
    selectors, embedded apps in cross-origin iframes, etc.
    """

    x: int
    y: int
    label_visible: str   # what the LLM saw at that location
    reasoning: str
    confidence: float
    input_tokens: int | None = None
    output_tokens: int | None = None


COORDINATE_CLICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "label_visible": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["x", "y", "label_visible", "reasoning", "confidence"],
    "additionalProperties": False,
}


COORDINATE_CLICK_SYSTEM_PROMPT = """You point at a target element with PIXEL COORDINATES so an automation agent can click it directly without using the DOM.

The agent's DOM-based resolver couldn't find this target through any
selector strategy (literal CSS, text match, role+name, fuzzy match,
vision-guided scroll/navigate/drill). But you can SEE the element
in the screenshot. Look at the image and return the (x, y) pixel
coordinate where a click would land on the target.

Rules:
- The image is the actual screenshot of the visible viewport. Origin
  is the TOP-LEFT corner; X grows rightward, Y grows downward. Pixel
  units, integer values.
- Pick the CENTER of the target's clickable area, not its edge.
- ``label_visible``: quote what you see at that location verbatim
  (e.g. "Add to cart button - black bg, white text, top-right of
  product card"). Be specific so the orchestrator can log a useful
  audit trail.
- ``confidence`` 0.0-1.0:
  * 0.9+ — you can clearly see the exact target.
  * 0.7-0.9 — you see something that's almost certainly it.
  * 0.5-0.7 — you see a candidate but it's ambiguous.
  * < 0.5 — you're guessing. The orchestrator skips clicks below 0.6.
- If you can't see the target in the visible region: x=0, y=0,
  confidence=0.0, label_visible="(not visible)".
- If the target is offscreen but you can see a scrollbar or "more"
  affordance: still return confidence=0.0 — the agent already had a
  chance to scroll. Don't try to point at something you can't see.

Output JSON only.
"""


def propose_click_coordinates(
    provider: LLMProvider,
    page: Page,
    *,
    target_hint: str,
    near_misses: list[dict[str, Any]] | None = None,
    screenshot_bytes: bytes | None = None,
) -> CoordinateClick:
    """Vision LLM call: point at pixel coordinates for a target the DOM
    chain couldn't resolve.

    Caller dispatches the click via ``page.mouse.click(x, y)``. This
    is the LAST resort — only fires when fuzzy + vision-guided search
    have both already exhausted.

    Args:
        screenshot_bytes: Pre-captured screenshot to reuse. When the
            orchestrator runs this immediately after vision-search
            exhausts (page hasn't moved since the last search
            attempt), passing the cached bytes saves one
            ``page.screenshot()`` round-trip per failed action.

    Raises:
        RuntimeError: provider lacks vision OR LLM call fails OR
            response shape is malformed.
    """
    if not getattr(provider, "supports_vision", False):
        raise RuntimeError(
            "propose_click_coordinates requires a vision-capable provider; "
            f"{type(provider).__name__} reports supports_vision=False",
        )

    if screenshot_bytes is not None:
        screenshot = screenshot_bytes
    else:
        try:
            # MUST stay at original viewport size — the LLM returns
            # pixel coords that we dispatch directly via
            # page.mouse.click(x, y). A downscaled image would yield
            # coords in shrunken space → clicks land at the wrong
            # spot on the real viewport. Cost trade-off: ~1.5K
            # image tokens vs. coordinate accuracy. Coord-click is
            # the last-resort rescue, fires rarely; worth it.
            screenshot = capture_screenshot_for_vision(
                page, downscale=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to capture screenshot for coordinate click: "
                f"{type(e).__name__}: {e}",
            ) from e

    near_text = ""
    if near_misses:
        nm_lines = "\n".join(
            f"  - {nm.get('role') or '?'}: "
            f"{(nm.get('name') or '')[:80]!r} (score {nm.get('score')})"
            for nm in near_misses[:3]
        )
        near_text = (
            f"\nNEAR-MISSES (DOM resolver scored these but they "
            f"weren't the target):\n{nm_lines}\n"
        )

    user_prompt = (
        f"TARGET (the agent's hint that the DOM resolver could not "
        f"reach):\n  {target_hint!r}\n"
        f"{near_text}\n"
        f"PAGE URL: {_safe_url(page)}\n\n"
        "Look at the screenshot. Return pixel coordinates for a "
        "direct click on the target. If you cannot see the target, "
        "set confidence to 0.0 and label_visible to '(not visible)'."
    )

    try:
        result = provider.chat_structured(
            messages=[
                ChatMessage(
                    role="system",
                    content=COORDINATE_CLICK_SYSTEM_PROMPT,
                ),
                ChatMessage(
                    role="user",
                    content=user_prompt,
                    image=screenshot,
                ),
            ],
            schema=COORDINATE_CLICK_SCHEMA,
            schema_name="coordinate_click",
            temperature=0.1,
            max_output_tokens=256,
        )
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed for coordinate click: "
            f"{type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"coordinate click returned unexpected shape: "
            f"{type(parsed).__name__}",
        )

    return CoordinateClick(
        x=int(parsed.get("x", 0) or 0),
        y=int(parsed.get("y", 0) or 0),
        label_visible=str(parsed.get("label_visible", "")),
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


# ── Phase 14: smart candidate selection ────────────────────────────
#
# When the agent's target_hint is ambiguous (resolver finds 3+
# matches) OR the goal carries explicit criteria the test case didn't
# bake into the hint (e.g. "phone under 50000 rupees"), the orchestrator
# calls this VL helper to PICK the right one among visible candidates.
#
# The LLM returns one of:
#   - selector  : a precise CSS / Playwright selector pointing at the
#                 chosen element (preferred — DOM-resolvable).
#   - coords    : pixel coords (x, y) when no clean selector exists
#                 — orchestrator dispatches via page.mouse.click().
#   - scroll    : "no visible candidate matches; scroll {direction}"
#   - none      : "no candidate matches even after a full survey"
#                 — orchestrator can flag via the test-case dispute
#                 tool (Phase 11).
#
# Bridges the DOM ↔ browser-use gap that `text 'phone'` resolution
# blew up on Amazon: DOM picks the first match (a sponsored ad);
# this helper picks the one that visibly satisfies the criteria.


SmartCandidateStrategy = Literal["selector", "coords", "scroll", "none"]


@dataclass
class SmartCandidate:
    """Vision LLM's choice of best candidate matching target + criteria."""

    strategy: SmartCandidateStrategy
    # Populated when strategy == "selector"
    selector: str = ""
    # Populated when strategy == "coords"
    x: int = 0
    y: int = 0
    # Populated when strategy == "scroll"
    scroll_direction: str = ""  # "up" | "down" | "left" | "right" | ""
    scroll_amount_px: int = 0
    # Always set
    chosen_label: str = ""    # Verbatim text of the picked candidate
    rejected_labels: list[str] = field(default_factory=list)
    # Why these candidates were rejected — one per rejected_labels entry.
    rejection_reasons: list[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


SMART_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strategy": {
            "type": "string",
            "enum": ["selector", "coords", "scroll", "none"],
        },
        "selector": {"type": "string"},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "scroll_direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right", ""],
        },
        "scroll_amount_px": {"type": "integer"},
        "chosen_label": {"type": "string"},
        "rejected_labels": {"type": "array", "items": {"type": "string"}},
        "rejection_reasons": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "strategy", "selector", "x", "y",
        "scroll_direction", "scroll_amount_px",
        "chosen_label", "rejected_labels", "rejection_reasons",
        "reasoning", "confidence",
    ],
    "additionalProperties": False,
}


SMART_CANDIDATE_SYSTEM_PROMPT = """You are a senior QA tester picking the RIGHT element from a list of visible candidates on a real browser page.

The agent's target_hint matched MULTIPLE elements (typically a list-
of-products page on e-commerce, a search-results page, a feed). Your
job is to pick the ONE that best satisfies BOTH:
1. The TARGET DESCRIPTION (what the agent is looking for)
2. The CRITERIA (constraints from the goal: price under X, "real"
   product not an ad, must have a visible price, etc.)

How to choose
-------------
- READ the labels visible on the candidates the AX tree gave you.
- LOOK at the screenshot to spot non-textual cues:
  * "Sponsored" / "Ad" / "Promoted" tags → SKIP these — they pollute
    the top of every search results page and rarely match the test's
    intent.
  * Missing price / "Currently unavailable" / "Out of stock" → SKIP
    when the test obviously needs to add to cart.
  * Visibly distinct in a way the criteria reject (price > limit,
    wrong category) → SKIP.
- Pick the FIRST candidate from the visible set that satisfies the
  criteria. Don't try to be clever — the first non-rejected match
  is usually the right one.
- When NO visible candidate satisfies (e.g. all top results are
  ads), return strategy="scroll" with a reasonable amount (~600px)
  in the natural direction (usually "down").
- When the page genuinely has no matching candidate (wrong page,
  no results, etc.), return strategy="none".

Output strategy
---------------
Prefer "selector" when you can identify a precise CSS / role+name
selector for the chosen element from the AX tree (e.g.
"role=link[name='Samsung Galaxy M14 5G ₹12,999']"). The orchestrator
will resolve via DOM, faster and more reliable than coordinates.

Use "coords" only when:
- No clean selector is identifiable (canvas-rendered content,
  highly dynamic class names, no role+name pair)
- The target is clearly visible at specific pixel coordinates

Constraints
-----------
- Coords MUST be in the FULL viewport pixel space (the screenshot's
  native dimensions). Do NOT shrink.
- chosen_label: copy the visible text VERBATIM (no paraphrasing).
- rejected_labels + rejection_reasons: same length, paired. List
  the candidates you considered AND skipped with a 1-line reason
  (e.g. "Sponsored placement", "no price visible", "₹65,999 over
  limit").
- confidence: 0.9+ when the choice is unambiguous; 0.7 when you
  picked the best of several plausible options; <0.7 when uncertain.

Output JSON only.
"""


def propose_smart_candidate(
    provider: LLMProvider,
    page: Page,
    *,
    target_description: str,
    criteria: list[str],
    visible_candidates: list[dict[str, Any]] | None = None,
    screenshot_bytes: bytes | None = None,
    cheap_provider: LLMProvider | None = None,
    on_escalate: Any = None,
) -> SmartCandidate:
    """Vision LLM call: pick the best candidate matching target + criteria.

    Phase 14 — bridges the gap that pure DOM resolution can't cross:
    when ``target_hint='text phone'`` resolves to 30 elements, the
    DOM resolver picks the first; this helper picks the BEST. Skips
    sponsored ads, products without prices, items that violate goal
    constraints (price > limit, wrong category, etc.).

    Args:
        target_description: What the agent is looking for, in natural
            language. Usually the test step's narrative or a synthesis
            of the target_hint + sub-goal context.
        criteria: List of constraint strings drawn from the goal's
            success_criteria — e.g.
            ["price under 50000 rupees", "must have a visible price"].
            Empty list is fine (will pick by general fitness).
        visible_candidates: Optional list of AX-tree items the
            resolver matched on (label, role, selector hint). When
            given, the LLM is told these are the candidates to choose
            among — narrower context, less hallucination. When None,
            the LLM picks freely from what's visible in the screenshot.
        screenshot_bytes: Pre-captured ORIGINAL-SIZE screenshot. MUST
            NOT be downscaled — the LLM may return pixel coords.
        cheap_provider: Optional cheap-tier model — see Phase 1
            tiering. Escalates to ``provider`` on low confidence /
            invalid strategy.

    Returns:
        SmartCandidate — caller dispatches based on ``.strategy``:
            "selector" → resolve(.selector) and act
            "coords"   → page.mouse.click(.x, .y) (or type, etc.)
            "scroll"   → page.mouse.wheel + retry the original step
            "none"     → flag via test-case dispute (Phase 11) +
                         halt the step

    Raises:
        RuntimeError: provider lacks vision OR LLM call fails OR
            response shape is malformed.
    """
    if not getattr(provider, "supports_vision", False):
        raise RuntimeError(
            "propose_smart_candidate requires a vision-capable provider; "
            f"{type(provider).__name__} reports supports_vision=False",
        )

    # Coord-bearing helper → MUST stay at original viewport size.
    if screenshot_bytes is not None:
        screenshot = screenshot_bytes
    else:
        try:
            screenshot = capture_screenshot_for_vision(page, downscale=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to capture screenshot for smart candidate: "
                f"{type(e).__name__}: {e}",
            ) from e

    crit_block = "\n".join(
        f"  - {c}" for c in criteria
    ) or "  (none — pick by general fitness)"

    cand_block = "  (use the screenshot — no AX-tree pre-filter passed in)"
    if visible_candidates:
        cand_lines = []
        for i, c in enumerate(visible_candidates[:25], start=1):
            role = c.get("role") or "?"
            name = (c.get("name") or "")[:120]
            sel = c.get("selector_hint") or ""
            cand_lines.append(
                f"  {i}. {role}: {name!r}"
                + (f"   [{sel}]" if sel else "")
            )
        cand_block = "\n".join(cand_lines)

    user_prompt = (
        f"TARGET DESCRIPTION:\n  {target_description}\n\n"
        f"CRITERIA (must hold for the chosen candidate):\n{crit_block}\n\n"
        f"VISIBLE CANDIDATES (AX-tree pre-filter):\n{cand_block}\n\n"
        f"PAGE URL: {_safe_url(page)}\n\n"
        "Look at the screenshot. Pick ONE candidate that matches the "
        "target AND all criteria. Skip sponsored / ad placements, "
        "out-of-stock items, and anything that visibly violates a "
        "criterion. If no visible candidate matches, return strategy="
        "'scroll' (and a reasonable direction/amount) or strategy="
        "'none' if the page genuinely has nothing."
    )

    messages = [
        ChatMessage(role="system", content=SMART_CANDIDATE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt, image=screenshot),
    ]
    try:
        from app.llm.router import (  # noqa: PLC0415
            LLMRole, call_for_role,
        )

        def _validate(parsed: Any) -> bool:
            if not isinstance(parsed, dict):
                return False
            strat = parsed.get("strategy")
            if strat not in ("selector", "coords", "scroll", "none"):
                return False
            # Strategy-specific structural checks.
            if strat == "selector" and not parsed.get("selector"):
                return False
            if strat == "coords":
                try:
                    if int(parsed.get("x", 0)) <= 0:
                        return False
                    if int(parsed.get("y", 0)) <= 0:
                        return False
                except (TypeError, ValueError):
                    return False
            if strat == "scroll" and not parsed.get("scroll_direction"):
                return False
            return True

        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=LLMRole.SMART_PICKER,
            messages=messages,
            schema=SMART_CANDIDATE_SCHEMA,
            schema_name="smart_candidate",
            temperature=0.1,
            max_output_tokens=512,
            validate=_validate,
            on_escalate=on_escalate,
        )
        result = tiered.chat
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed for smart candidate: "
            f"{type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"smart candidate returned unexpected shape: "
            f"{type(parsed).__name__}",
        )

    strat = parsed.get("strategy")
    if strat not in ("selector", "coords", "scroll", "none"):
        raise RuntimeError(
            f"smart candidate returned invalid strategy: {strat!r}",
        )

    raw_rejected = parsed.get("rejected_labels") or []
    raw_reasons = parsed.get("rejection_reasons") or []
    return SmartCandidate(
        strategy=strat,  # type: ignore[arg-type]
        selector=str(parsed.get("selector", "")),
        x=int(parsed.get("x", 0) or 0),
        y=int(parsed.get("y", 0) or 0),
        scroll_direction=str(parsed.get("scroll_direction", "")),
        scroll_amount_px=int(parsed.get("scroll_amount_px", 0) or 0),
        chosen_label=str(parsed.get("chosen_label", "")),
        rejected_labels=[
            str(s) for s in raw_rejected if isinstance(s, str) and s.strip()
        ],
        rejection_reasons=[
            str(s) for s in raw_reasons if isinstance(s, str) and s.strip()
        ],
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


# ── Phase 9 + 13: semantic verify (strict-fail mode) ───────────────
#
# When a literal verify (DOM selector resolution OR exact substring
# match) fails, the test case may still be semantically passing —
# the page just wraps "Cart" as "Your Amazon Cart", or the count text
# is "Showing 1-48 of over 100,000 results" instead of plain
# "results". A QA expert reads the page and says "yes, the goal is
# met"; literal matching says "no, the exact string isn't present."
#
# This helper is the escalation: vision LLM looks at the screenshot
# + the expected condition and rules pass/fail/inconclusive under a
# STRICT prompt — biased toward "fail" when uncertain so we never
# mask a real bug behind a generous semantic interpretation.


SemanticVerdict = Literal["pass", "fail", "inconclusive"]


@dataclass
class SemanticVerification:
    """LLM judgment on whether the screenshot satisfies an expected
    state when literal/DOM matching couldn't decide cleanly."""

    verdict: SemanticVerdict
    reasoning: str
    confidence: float
    visible_evidence: str
    input_tokens: int | None = None
    output_tokens: int | None = None


SEMANTIC_VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "fail", "inconclusive"],
        },
        "visible_evidence": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["verdict", "visible_evidence", "reasoning", "confidence"],
    "additionalProperties": False,
}


SEMANTIC_VERIFY_SYSTEM_PROMPT = """You are a strict QA verifier. The agent's literal/DOM check for a step's expected outcome did NOT match. Your job: rule on whether the SCREENSHOT shows that expected state semantically — even when the exact text differs, IF it's clearly the same outcome.

Inputs you'll see:
- EXPECTED: the test step's expected state (a phrase, sentence, or
  short claim — e.g. "Cart contains the added phone", "Search results
  for 'phone' are displayed", "Order placed successfully").
- TARGET HINT: an optional Playwright-resolvable hint that the
  literal verifier just couldn't find (selector / text marker).
- PAGE URL.
- SCREENSHOT.

Verdict rubric — STRICT
-----------------------
- "pass": The screenshot UNAMBIGUOUSLY shows the expected outcome.
  Examples that count as pass:
    * EXPECTED says "Cart" and the page header reads "Your Amazon Cart"
    * EXPECTED says "results" and the page shows "Showing 1-48 of
      over 100,000 results for ..."
    * EXPECTED says "logged in" and the header shows the user's name
      with a "Sign out" link.
  The wrapping text is fine; the substantive claim must hold.

- "fail": The screenshot CONTRADICTS the expected outcome, OR shows
  evidence the test failed:
    * EXPECTED says "added to cart" but cart count is 0 or the page
      shows "your cart is empty"
    * EXPECTED says "payment page" but the page is still a cart or
      shows a "checkout failed" banner
    * EXPECTED says "results" but the page shows "no results found"

- "inconclusive": You CANNOT tell from the screenshot alone:
    * Page is mid-load / blank / partially rendered
    * The expected state lives below the fold (you can't see it)
    * Genuine ambiguity ("login OR signup" — page shows form but
      labels are too generic to decide)

Bias toward "fail" or "inconclusive" when uncertain — DO NOT swing
toward "pass" to be helpful. The whole point of this escalation is
to catch real failures the agent's literal check missed; a loose
"pass" verdict masks bugs.

Output
------
- verdict: pass / fail / inconclusive
- visible_evidence: 1-2 phrases from the screenshot you used to
  decide (verbatim, in quotes when possible — "Your Amazon Cart",
  "Showing 1-48 of over 100,000 results", etc.).
- reasoning: 1 sentence explaining the call.
- confidence: 0.0-1.0. Only >= 0.85 counts as "definite"; <= 0.7
  ought to come with a "fail" or "inconclusive" verdict, not "pass".

Output JSON only.
"""


def verify_semantic(
    provider: LLMProvider,
    page: Page,
    *,
    expected: str,
    target_hint: str | None = None,
    full_page: bool = False,
    screenshot_bytes: bytes | None = None,
    cheap_provider: LLMProvider | None = None,
    on_escalate: Any = None,
) -> SemanticVerification:
    """Phase 9 — semantic verify escalation.

    Called when the literal ``verify`` action just failed (DOM
    resolution missed OR substring check returned False) but the
    test step's intent might still be satisfied semantically.

    Returns one of:
    - ``pass`` — vision LLM unambiguously sees the expected state;
      the literal check failed because of copy/wording mismatch.
      Caller upgrades the verify outcome to passed.
    - ``fail`` — vision LLM sees a clear contradiction; the original
      failed verdict stands.
    - ``inconclusive`` — vision LLM cannot tell from the screenshot;
      caller leaves the failed verdict as-is. We do NOT swing to
      pass on inconclusive — that would mask real bugs.

    ``full_page=True`` triggers full-page capture for revalidation
    passes (per the user's spec: viewport by default, full page when
    the expected outcome lives below the fold — confirmation banners,
    cart contents, etc.).
    """
    if not getattr(provider, "supports_vision", False):
        raise RuntimeError(
            "verify_semantic requires a vision-capable provider; "
            f"{type(provider).__name__} reports supports_vision=False",
        )

    if screenshot_bytes is not None:
        screenshot = screenshot_bytes
    else:
        try:
            screenshot = capture_screenshot_for_vision(
                page, full_page=full_page,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to capture screenshot for semantic verify: "
                f"{type(e).__name__}: {e}",
            ) from e

    target_block = (
        f"TARGET HINT (literal verifier couldn't find): "
        f"{target_hint!r}\n"
        if target_hint else ""
    )
    user_prompt = (
        f"EXPECTED:\n  {expected}\n\n"
        f"{target_block}"
        f"PAGE URL: {_safe_url(page)}\n\n"
        "Look at the screenshot. Apply the strict rubric and rule "
        "pass / fail / inconclusive. Bias toward inconclusive or "
        "fail when uncertain — do NOT pass to be helpful."
    )

    messages = [
        ChatMessage(role="system", content=SEMANTIC_VERIFY_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt, image=screenshot),
    ]
    try:
        from app.llm.router import (  # noqa: PLC0415
            LLMRole, call_for_role,
        )

        def _validate(parsed: Any) -> bool:
            if not isinstance(parsed, dict):
                return False
            return parsed.get("verdict") in ("pass", "fail", "inconclusive")

        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=LLMRole.SEMANTIC_VERIFIER,
            messages=messages,
            schema=SEMANTIC_VERIFY_SCHEMA,
            schema_name="semantic_verify",
            temperature=0.1,
            max_output_tokens=384,
            validate=_validate,
            on_escalate=on_escalate,
        )
        result = tiered.chat
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed for semantic verify: "
            f"{type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"semantic verify returned unexpected shape: "
            f"{type(parsed).__name__}",
        )

    verdict = parsed.get("verdict")
    if verdict not in ("pass", "fail", "inconclusive"):
        raise RuntimeError(
            f"semantic verify returned invalid verdict: {verdict!r}",
        )

    return SemanticVerification(
        verdict=verdict,  # type: ignore[arg-type]
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        visible_evidence=str(parsed.get("visible_evidence", "")),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


# ── Phase 10: popup / overlay classifier ───────────────────────────
#
# When the agent hits an intercepted click OR observes a modal-shaped
# DOM region after a navigation, this helper classifies the overlay
# so the orchestrator picks the right action:
#
#   required_step    : the modal IS part of the test flow (variant
#                      selector, address picker, "are you sure?" on
#                      a destructive action). Engage with it.
#   dismissable_blocker : modal blocks the page but isn't part of
#                      the test (sign-in nag, cookie consent, app
#                      install banner). Dismiss via close button or
#                      Escape.
#   non_blocking_overlay : a banner / toast at the edge of the
#                      viewport that doesn't block clicks (cookie
#                      banner at the bottom, "your order placed"
#                      toast). Ignore.
#   ad               : promotional content (interstitial signup
#                      offers, full-screen ads). Dismiss aggressively
#                      — these are the most disruptive on retail
#                      sites.
#
# Confidence: when < 0.7, the orchestrator defaults to ENGAGE
# (your answer to Q3) because the cost of skipping a required step
# is much worse than the cost of clicking through one extra modal.


PopupKind = Literal[
    "none", "required_step", "dismissable_blocker",
    "non_blocking_overlay", "ad",
]


@dataclass
class PopupClassification:
    """LLM's verdict on what the visible overlay (if any) is."""

    kind: PopupKind
    # Optional concrete dismissal hint when kind allows dismissal.
    # The LLM picks a verbatim selector / role+name from the AX
    # tree the orchestrator can resolve.
    dismiss_hint: str = ""
    # Reasoning the user sees in the live feed.
    reasoning: str = ""
    confidence: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


POPUP_CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": [
                "none", "required_step", "dismissable_blocker",
                "non_blocking_overlay", "ad",
            ],
        },
        "dismiss_hint": {"type": "string"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["kind", "dismiss_hint", "reasoning", "confidence"],
    "additionalProperties": False,
}


POPUP_CLASSIFY_SYSTEM_PROMPT = """You classify visible overlays / modals / popups on a real browser page so a QA agent decides what to do with them.

Inputs:
- GOAL CONTEXT: short description of what the agent is trying to do
  on this page (e.g. "search for phones and add to cart").
- PAGE URL.
- SCREENSHOT.

Categories
----------
- "none": No popup or overlay is visibly blocking the page. The
  agent can act normally.

- "required_step": The overlay IS part of the user's flow and must
  be engaged. Examples: a variant-selector dialog before adding to
  cart; a quantity / address picker; a "review your order" panel
  before checkout; an OTP / 2FA prompt during login; a system
  message asking "Save changes? Yes / No / Cancel" on navigation.
  CLUE: the overlay's content directly references the goal (cart,
  checkout, save, confirm an action you just took).

- "dismissable_blocker": The overlay blocks interaction but is NOT
  part of the test flow. Examples: a sign-in / sign-up nag; cookie
  consent dialog; "install our app" banner with an X button; a
  newsletter capture popup. CLUE: the overlay is generic (no link
  to the user's specific action), has a visible close affordance,
  and dismissing it returns the agent to where they were.

- "non_blocking_overlay": A banner / toast / floating element at the
  edge of the viewport that doesn't intercept clicks. Examples:
  cookie consent at the bottom (with no full-screen scrim); an
  "added to cart" toast on the right; a region-picker chip
  pinned to the top. CLUE: the rest of the page IS clickable;
  ignoring it is fine.

- "ad": Promotional content unrelated to the goal. Examples: a
  full-screen offer ("save 30% with our credit card"); a holiday
  sale interstitial; a sponsored signup form blocking the
  product list. CLUE: aggressive marketing copy, large hero
  image, irrelevant to what the user is doing. Dismiss
  aggressively.

dismiss_hint
------------
When kind is dismissable_blocker, ad, or sometimes
non_blocking_overlay, return a Playwright-resolvable hint for the
close button. Examples:
- "[aria-label='Close']"
- "role=button[name='No thanks']"
- "text 'Maybe later'"
- "" (empty) when no clean dismissal exists — orchestrator falls
  back to Escape key / close icon heuristic.

When kind is required_step or none, leave dismiss_hint empty.

confidence
----------
0.9+ when the classification is unambiguous. < 0.7 when uncertain
between two categories — the orchestrator defaults to ENGAGE
(treats as required_step) on low confidence, because skipping a
required step is worse than clicking through one extra modal.

Output JSON only.
"""


def classify_popup(
    provider: LLMProvider,
    page: Page,
    *,
    goal_context: str,
    screenshot_bytes: bytes | None = None,
    cheap_provider: LLMProvider | None = None,
    on_escalate: Any = None,
) -> PopupClassification:
    """Phase 10 — classify a visible overlay so the orchestrator picks
    the right action (engage / dismiss / ignore).

    Called when:
    - A click failed with ``failure_kind == "click_intercepted"``
      (something is overlaying the target).
    - A fresh navigation completed and the agent observes
      modal-shaped DOM (role=dialog, fixed-position large element,
      etc.). Detection lives in qa_agent.py before the next turn.

    Returns ``PopupClassification`` — caller dispatches based on
    ``.kind`` (see categories above). On low confidence (< 0.7) the
    orchestrator engages with the popup (treats as required_step)
    rather than dismissing — your locked policy from plan Q3.

    Token cost: one cheap-tier vision call per intercepted click.
    Cached per (URL, screenshot hash) at the orchestrator level so
    repeat hits don't re-pay.
    """
    if not getattr(provider, "supports_vision", False):
        raise RuntimeError(
            "classify_popup requires a vision-capable provider; "
            f"{type(provider).__name__} reports supports_vision=False",
        )

    if screenshot_bytes is not None:
        screenshot = screenshot_bytes
    else:
        try:
            screenshot = capture_screenshot_for_vision(page)
        except Exception as e:
            raise RuntimeError(
                f"Failed to capture screenshot for popup classify: "
                f"{type(e).__name__}: {e}",
            ) from e

    user_prompt = (
        f"GOAL CONTEXT:\n  {goal_context}\n\n"
        f"PAGE URL: {_safe_url(page)}\n\n"
        "Look at the screenshot. Classify any overlay you see using "
        "the four categories. If no overlay is visibly blocking, "
        "return kind='none'."
    )

    messages = [
        ChatMessage(role="system", content=POPUP_CLASSIFY_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt, image=screenshot),
    ]
    try:
        from app.llm.router import (  # noqa: PLC0415
            LLMRole, call_for_role,
        )

        def _validate(parsed: Any) -> bool:
            if not isinstance(parsed, dict):
                return False
            return parsed.get("kind") in (
                "none", "required_step", "dismissable_blocker",
                "non_blocking_overlay", "ad",
            )

        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=LLMRole.FAST_CLASSIFY,
            messages=messages,
            schema=POPUP_CLASSIFY_SCHEMA,
            schema_name="popup_classify",
            temperature=0.1,
            max_output_tokens=320,
            validate=_validate,
            on_escalate=on_escalate,
        )
        result = tiered.chat
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed for popup classify: "
            f"{type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"popup classify returned unexpected shape: "
            f"{type(parsed).__name__}",
        )

    kind = parsed.get("kind")
    if kind not in (
        "none", "required_step", "dismissable_blocker",
        "non_blocking_overlay", "ad",
    ):
        raise RuntimeError(
            f"popup classify returned invalid kind: {kind!r}",
        )

    return PopupClassification(
        kind=kind,  # type: ignore[arg-type]
        dismiss_hint=str(parsed.get("dismiss_hint", "")),
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


# ── Auth field detection (login / OTP / captcha screen intent) ─────
#
# The auth flow needs to know WHERE to type each credential without
# relying on DOM resolution. This helper does ONE vision call against
# the live screenshot and returns per-field pixel coordinates plus
# error-message detection — so the orchestrator can:
#   1. click-type the email at (x1, y1)
#   2. click-type the password at (x2, y2)
#   3. click submit at (x3, y3)
#   4. on retry: see the "Please enter a valid email" error and
#      know which field needs re-typing
#
# Critical: ``screenshot_bytes`` MUST be captured at full viewport
# resolution (downscale=False). Returned coords are in that space
# and are dispatched directly to ``page.mouse.click(x, y)``.


@dataclass
class AuthFieldsDetection:
    """Per-field coordinates + error state on an auth screen."""

    # Screen kind — drives the auth flow's branching.
    kind: Literal[
        "login", "otp", "captcha", "passkey",
        "success", "unknown",
    ]
    # Field coords (pixel center). 0/0 when not visible.
    username_x: int = 0
    username_y: int = 0
    password_x: int = 0
    password_y: int = 0
    otp_x: int = 0
    otp_y: int = 0
    submit_x: int = 0
    submit_y: int = 0
    # Per-field visibility hints — caller checks these before
    # dispatching a click (avoids clicking 0,0 when a field is absent).
    username_visible: bool = False
    password_visible: bool = False
    otp_visible: bool = False
    submit_visible: bool = False
    # Error message currently displayed under the form, if any
    # (e.g. "Please enter a valid email address"). Empty when none.
    error_text: str = ""
    # Which field the error applies to ("username" | "password" |
    # "otp" | "" when ambiguous / not field-specific).
    error_field: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


AUTH_FIELDS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": [
                "login", "otp", "captcha", "passkey",
                "success", "unknown",
            ],
        },
        "username_x": {"type": "integer"},
        "username_y": {"type": "integer"},
        "password_x": {"type": "integer"},
        "password_y": {"type": "integer"},
        "otp_x": {"type": "integer"},
        "otp_y": {"type": "integer"},
        "submit_x": {"type": "integer"},
        "submit_y": {"type": "integer"},
        "username_visible": {"type": "boolean"},
        "password_visible": {"type": "boolean"},
        "otp_visible": {"type": "boolean"},
        "submit_visible": {"type": "boolean"},
        "error_text": {"type": "string"},
        "error_field": {
            "type": "string",
            "enum": ["", "username", "password", "otp"],
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "kind",
        "username_x", "username_y",
        "password_x", "password_y",
        "otp_x", "otp_y",
        "submit_x", "submit_y",
        "username_visible", "password_visible",
        "otp_visible", "submit_visible",
        "error_text", "error_field",
        "reasoning", "confidence",
    ],
    "additionalProperties": False,
}


AUTH_FIELDS_SYSTEM_PROMPT = """You analyze a login / OTP / captcha screen and return pixel coordinates for each visible field so a QA agent can fill the form via mouse + keyboard.

Classify the screen kind FIRST:
- "login"   : email/username + password fields visible (with or
              without other fields like "remember me")
- "otp"     : a one-time-code field (6-digit, "OTP", "verification
              code", "authenticator"). Email/password may or may
              not be present too.
- "captcha" : a CAPTCHA challenge blocks the form (reCAPTCHA, hCaptcha,
              Cloudflare Turnstile). Requires human attention.
- "passkey" : a passkey / FIDO2 / WebAuthn prompt
- "success" : the post-login page (dashboard / account home —
              login already happened)
- "unknown" : screen doesn't fit any of the above

Then return PIXEL COORDINATES (full viewport space) for the center
of each VISIBLE interactive element:
- ``username_x``, ``username_y``  — the email / username INPUT field
- ``password_x``, ``password_y``  — the password INPUT field
- ``otp_x``, ``otp_y``            — the OTP / verification code field
- ``submit_x``, ``submit_y``      — the primary submit button (Login /
                                    Sign In / Continue / Verify)

Set the matching ``*_visible`` boolean to ``true`` when the field
is rendered AND ready for input (not greyed out, not hidden behind
a tab). Set it to ``false`` AND coords to 0,0 when the field isn't
present on the current screen.

Click TARGETS:
- Input fields: aim for the input's CENTER (NOT the label above
  it; clicking the label sometimes only focuses, sometimes does
  nothing depending on the framework).
- Submit button: aim for the button's center.
- Coords MUST be POSITIVE integers in the screenshot's pixel space.

Error detection:
- If you see an inline error message (red text under a field, an
  alert banner, etc.), copy it VERBATIM into ``error_text`` (trim
  to ~200 chars).
- Set ``error_field`` to which input it's about:
    "username" — "Please enter a valid email", "Email is required",
    "password" — "Password must be 8 characters", "Wrong password",
    "otp"      — "Invalid code", "Code expired",
    ""         — error is generic / cross-field / you can't tell
- Empty ``error_text`` when no error is visible.

Confidence: 0.9+ when fields are unambiguously located. <0.7 when
you're guessing on at least one coord — the caller will fall back
to HITL on low confidence.

Output JSON only.
"""


def detect_auth_fields(
    provider: LLMProvider,
    page: Page,
    *,
    screenshot_bytes: bytes | None = None,
    cheap_provider: LLMProvider | None = None,
    on_escalate: Any = None,
) -> AuthFieldsDetection:
    """Vision LLM call: locate auth fields + read errors.

    ``screenshot_bytes`` MUST be captured with ``downscale=False``;
    returned coords are dispatched directly to ``page.mouse.click``
    so they have to be in real viewport pixel space.

    Routed through ``LLMRole.COORD_PROPOSER`` (strong-only, no
    cheap-tier escalation) because pixel-coord accuracy matters
    more than the cache discount on cheap-tier calls.
    """
    if not getattr(provider, "supports_vision", False):
        raise RuntimeError(
            "detect_auth_fields requires a vision-capable provider; "
            f"{type(provider).__name__} reports supports_vision=False",
        )

    if screenshot_bytes is not None:
        screenshot = screenshot_bytes
    else:
        try:
            # Coord-bearing helper → MUST stay at original viewport size.
            screenshot = capture_screenshot_for_vision(
                page, downscale=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to capture screenshot for auth fields: "
                f"{type(e).__name__}: {e}",
            ) from e

    user_prompt = (
        f"PAGE URL: {_safe_url(page)}\n\n"
        "Look at the screenshot. Classify the auth screen and return "
        "coordinates for every visible field + any error message."
    )

    messages = [
        ChatMessage(
            role="system", content=AUTH_FIELDS_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user", content=user_prompt, image=screenshot,
        ),
    ]
    try:
        from app.llm.router import (  # noqa: PLC0415
            LLMRole, call_for_role,
        )

        def _validate(parsed: Any) -> bool:
            if not isinstance(parsed, dict):
                return False
            return parsed.get("kind") in (
                "login", "otp", "captcha", "passkey",
                "success", "unknown",
            )

        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=LLMRole.COORD_PROPOSER,
            messages=messages,
            schema=AUTH_FIELDS_SCHEMA,
            schema_name="auth_fields",
            temperature=0.1,
            max_output_tokens=512,
            validate=_validate,
            on_escalate=on_escalate,
        )
        result = tiered.chat
    except Exception as e:
        raise RuntimeError(
            f"LLM call failed for auth field detection: "
            f"{type(e).__name__}: {str(e)[:300]}",
        ) from e

    parsed = result.parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"auth fields returned unexpected shape: "
            f"{type(parsed).__name__}",
        )

    kind = parsed.get("kind") or "unknown"

    def _int(key: str) -> int:
        try:
            return max(0, int(parsed.get(key, 0) or 0))
        except (TypeError, ValueError):
            return 0

    return AuthFieldsDetection(
        kind=kind,  # type: ignore[arg-type]
        username_x=_int("username_x"),
        username_y=_int("username_y"),
        password_x=_int("password_x"),
        password_y=_int("password_y"),
        otp_x=_int("otp_x"),
        otp_y=_int("otp_y"),
        submit_x=_int("submit_x"),
        submit_y=_int("submit_y"),
        username_visible=bool(parsed.get("username_visible", False)),
        password_visible=bool(parsed.get("password_visible", False)),
        otp_visible=bool(parsed.get("otp_visible", False)),
        submit_visible=bool(parsed.get("submit_visible", False)),
        error_text=str(parsed.get("error_text", ""))[:200],
        error_field=str(parsed.get("error_field", "")),
        reasoning=str(parsed.get("reasoning", "")),
        confidence=float(parsed.get("confidence", 0.0)),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
