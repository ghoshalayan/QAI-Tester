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
from dataclasses import dataclass
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
