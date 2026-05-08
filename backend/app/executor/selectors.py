"""Selector waterfall — turn freeform ``target_hint`` strings into a visible
Playwright Locator.

The week-4 TC agent's system prompt allows several flavors:

    "button[data-testid='signin']"   # CSS
    "text 'Sign In'"                  # text marker
    "Sign In"                         # plain visible text
    "role=button[name='Sign In']"     # explicit Playwright engine
    ""                                # caller falls back to action-type only

This module probes strategies in order and returns the first that resolves
to a visible element. If nothing matches, raises :class:`SelectorNotFound`
with the strategy-by-strategy attempt log so the executor can record what
was tried.

Per-strategy probe is `count() == 0` → skip cheaply, otherwise `wait_for
(visible, timeout=timeout_ms)` on the first match. Total wait is bounded
by ``timeout_ms × strategies-tried``.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from playwright.sync_api import Locator, Page
from playwright.sync_api import TimeoutError as PWTimeoutError

logger = logging.getLogger(__name__)

# Per-strategy timeout — short so missed strategies don't tank wall-clock.
DEFAULT_PROBE_TIMEOUT_MS = 3_000

# Fuzzy stage thresholds (Phase A4.2 — selector resilience).
# A literal-miss target gets re-scored against every visible AX-tree
# element by name. Above ``ACCEPT`` we substitute and continue; between
# ``NEAR_MISS`` and ``ACCEPT`` we surface candidates in the error so
# the next agent turn can adapt. Below ``NEAR_MISS`` we fail silently
# (the page genuinely has nothing close).
FUZZY_ACCEPT_THRESHOLD = 0.60
FUZZY_NEAR_MISS_THRESHOLD = 0.30

# Pulled from a regex match on common forms; used to extract the role +
# name parts of an explicit Playwright role hint so we can score the
# NAME against AX-tree element names (not the whole role expression).
_ROLE_NAME_RE = re.compile(
    r"""^role=(\w+)\[name=['"]([^'"]+)['"]\]\s*$""",
)

# Word tokenizer used for Jaccard scoring.
_TOKEN_RE = re.compile(r"\w+")

# Native Playwright engine prefixes; if the hint starts with one we pass
# it straight through and don't fall back to other strategies.
_ENGINE_PREFIXES = (
    "css=",
    "text=",
    "role=",
    "xpath=",
    "id=",
    "data-testid=",
    ":nth=",
)

# LLMs frequently emit "text 'Sign In'" / 'text "X"' — strip and treat as text
_TEXT_MARKER_RE = re.compile(r"""^\s*text\s+["']([^"']+)["']\s*$""")

# Roles probed when the hint is plain text and nothing else worked.
_ROLE_PROBE = (
    "button",
    "link",
    "textbox",
    "menuitem",
    "checkbox",
    "tab",
    "option",
    "combobox",
)


class SelectorNotFound(RuntimeError):
    """Every strategy in the waterfall missed."""


@dataclass
class ResolvedTarget:
    """Successful match — the strategy that worked + the Locator."""

    locator: Locator
    strategy: str
    raw_hint: str
    attempts: list[str] = field(default_factory=list)
    # Populated only when the literal-strategy waterfall missed and the
    # fuzzy AX-tree fallback found a high-scoring substitute (Phase
    # A4.2). Action handlers surface this in narration + details so
    # users can see "we matched 'Add to Cart' against your hint
    # 'Add Cart' (score 0.91)" instead of silently using the wrong
    # element.
    fuzzy_match: dict[str, Any] | None = None


def _normalize(text: str) -> str:
    """Whitespace-collapsed lowercase — used for similarity scoring."""
    return " ".join((text or "").lower().split())


def _similarity(hint: str, candidate: str) -> float:
    """Score how alike ``hint`` and ``candidate`` are as element labels.

    Weighted layers, each tuned to catch a different real-world failure:

    1. **Exact match** → 1.0 (would have hit the literal strategies but
       sometimes the literal probe times out on slow pages; harmless to
       check here too).
    2. **Hint tokens are a subset of candidate tokens** → 0.7-1.0. Catches
       "Add Cart" ⊂ "Add to Cart" — the test case dropped a stop-word
       but every meaningful token matches.
    3. **Substring containment** (either direction) → 0.4-1.0 weighted by
       length ratio. Catches "search" ⊂ "search-field", "Sign" ⊂ "Sign In".
    4. **Jaccard token overlap** → 0..1 raw. Catches partial overlap like
       "submit form" vs "submit order".
    5. **SequenceMatcher** ratio × 0.85. Catches typos / reorderings.

    Returns the highest of the above. Threshold for "accept this match
    as a substitute" is ``FUZZY_ACCEPT_THRESHOLD`` (0.60); above
    ``FUZZY_NEAR_MISS_THRESHOLD`` (0.30) we surface as candidates in
    the error message so the next agent turn knows what's nearby.
    """
    h = _normalize(hint)
    c = _normalize(candidate)
    if not h or not c:
        return 0.0
    if h == c:
        return 1.0

    h_tokens = set(_TOKEN_RE.findall(h))
    c_tokens = set(_TOKEN_RE.findall(c))

    # Strong: every meaningful hint token appears in the candidate.
    if h_tokens and h_tokens.issubset(c_tokens):
        return 0.7 + 0.3 * (len(h_tokens) / max(1, len(c_tokens)))

    # Substring containment with length-weighted score.
    if h in c or c in h:
        short = min(len(h), len(c))
        long_ = max(len(h), len(c))
        return (0.4 + 0.6 * (short / long_)) if long_ else 0.0

    jaccard = (
        len(h_tokens & c_tokens) / len(h_tokens | c_tokens)
        if h_tokens and c_tokens else 0.0
    )
    seq = difflib.SequenceMatcher(None, h, c).ratio()

    return max(jaccard, seq * 0.85)


def _capture_ax_tree(page: Page) -> list[dict[str, Any]]:
    """Read the page's interactive elements via the same JS the agent's
    observation uses. Returns ``[]`` on any failure — fuzzy fallback
    is best-effort, never blocking.
    """
    # Imported here to avoid a top-level circular dep (page_intel ↔
    # selectors via execute_action).
    from app.agents.page_intel import _PAGE_SUMMARY_JS  # noqa: PLC0415
    try:
        summary = page.evaluate(_PAGE_SUMMARY_JS)
    except Exception as e:
        logger.debug("fuzzy: AX-tree capture failed: %s", e)
        return []
    items = (summary or {}).get("items") or []
    return [i for i in items if isinstance(i, dict)]


def _split_role_hint(hint: str) -> tuple[str | None, str]:
    """If the hint is ``role=button[name='Sign In']``, return
    ``("button", "Sign In")``. Otherwise ``(None, hint)``.

    Lets the fuzzy matcher score against the NAME of the requested
    role, with a small bonus for elements that share the role.
    """
    m = _ROLE_NAME_RE.match(hint.strip())
    if m:
        return m.group(1), m.group(2)
    return None, hint


def _fuzzy_resolve(
    page: Page, hint: str, *, timeout_ms: int,
) -> tuple[ResolvedTarget | None, list[dict[str, Any]]]:
    """Stage 6 of the resolver waterfall — only called when literal
    strategies all missed.

    Returns ``(ResolvedTarget | None, near_misses)``:
    - ``ResolvedTarget`` when a candidate scored ≥ 0.60 and resolved.
    - ``near_misses`` is the top 5 candidates above the 0.30 threshold,
      so the caller can include them in the error message when the
      fuzzy stage also misses.
    """
    items = _capture_ax_tree(page)
    if not items:
        return None, []

    hint_role, hint_name = _split_role_hint(hint)

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        name = item.get("name") or ""
        if not name:
            continue
        score = _similarity(hint_name, name)
        # Small bonus when the role matches what the hint requested.
        if hint_role and item.get("role") == hint_role:
            score = min(1.0, score + 0.10)
        scored.append((score, item))

    scored.sort(key=lambda t: t[0], reverse=True)

    near_misses: list[dict[str, Any]] = [
        {
            "role": item.get("role"),
            "name": item.get("name"),
            "score": round(score, 2),
        }
        for score, item in scored[:5]
        if score >= FUZZY_NEAR_MISS_THRESHOLD
    ]

    if not scored or scored[0][0] < FUZZY_ACCEPT_THRESHOLD:
        return None, near_misses

    best_score, best = scored[0]
    matched_name = best.get("name") or ""
    matched_role = best.get("role") or ""
    testid = best.get("testid")  # already in CSS form, e.g. "[data-testid='foo']"
    el_id = best.get("id")       # already in CSS form, e.g. "#foo"

    # Build a Playwright locator targeting the matched element.
    # Order favors the most stable selectors.
    builders: list[tuple[str, Callable[[], Locator]]] = []
    if testid:
        builders.append(("testid", lambda: page.locator(testid)))
    if el_id:
        builders.append(("id", lambda: page.locator(el_id)))
    if matched_role and matched_name:
        builders.append((
            "role+name",
            lambda: page.get_by_role(
                matched_role,  # type: ignore[arg-type]
                name=matched_name,
                exact=True,
            ),
        ))
    if matched_name:
        builders.append((
            "text",
            lambda: page.get_by_text(matched_name, exact=True),
        ))

    for label, builder in builders:
        try:
            loc = builder()
            if loc.count() == 0:
                continue
            loc.first.wait_for(state="visible", timeout=timeout_ms)
        except PWTimeoutError:
            continue
        except Exception as e:
            logger.debug(
                "fuzzy: builder %s raised: %s: %s",
                label, type(e).__name__, e,
            )
            continue
        return ResolvedTarget(
            locator=loc.first,
            strategy=f"fuzzy:{label}",
            raw_hint=hint,
            attempts=[f"fuzzy:{label}"],
            fuzzy_match={
                "matched_name": matched_name,
                "matched_role": matched_role,
                "score": round(best_score, 2),
                "near_misses": near_misses,
            },
        ), near_misses

    return None, near_misses


def resolve(
    page: Page,
    hint: str | None,
    *,
    timeout_ms: int = DEFAULT_PROBE_TIMEOUT_MS,
) -> ResolvedTarget:
    """Resolve ``hint`` to a visible Locator on ``page``.

    Args:
        page: Active Playwright page.
        hint: Freeform ``target_hint`` from the TC node. Empty/None raises
            immediately — the executor uses ``action_type`` to decide
            whether that's OK (e.g. ``navigate`` doesn't need a target).
        timeout_ms: Per-strategy ``wait_for(visible)`` timeout.

    Returns:
        :class:`ResolvedTarget` carrying the matched Locator + which
        strategy worked + an audit trail of attempts.

    Raises:
        SelectorNotFound: Hint was empty, or every strategy missed.
    """
    raw = (hint or "").strip()
    attempts: list[str] = []
    if not raw:
        raise SelectorNotFound("target_hint is empty — nothing to resolve")

    def probe(
        builder: Callable[[], Locator],
        label: str,
    ) -> ResolvedTarget | None:
        """One waterfall stage. Returns the resolved target or None."""
        attempts.append(label)
        try:
            loc = builder()
        except Exception as e:
            logger.debug(
                "Strategy %s: locator builder failed: %s: %s",
                label, type(e).__name__, e,
            )
            return None
        try:
            if loc.count() == 0:
                return None
            loc.first.wait_for(state="visible", timeout=timeout_ms)
        except PWTimeoutError:
            return None
        except Exception as e:
            logger.debug(
                "Strategy %s: probe failed: %s: %s",
                label, type(e).__name__, e,
            )
            return None
        return ResolvedTarget(
            locator=loc.first,
            strategy=label,
            raw_hint=raw,
            attempts=list(attempts),
        )

    # 1. Explicit Playwright engine prefix → pass-through, no fallback
    lower = raw.lower()
    for prefix in _ENGINE_PREFIXES:
        if lower.startswith(prefix):
            r = probe(lambda: page.locator(raw), f"engine:{prefix.rstrip('=')}")
            if r:
                return r
            raise SelectorNotFound(
                f"Explicit engine prefix did not match: {raw!r} "
                f"(tried: {', '.join(attempts)})",
            )

    # 2. text marker — "text 'X'" / 'text "X"'
    m = _TEXT_MARKER_RE.match(raw)
    if m:
        text = m.group(1)
        r = probe(
            lambda: page.get_by_text(text, exact=False),
            "text-marker",
        )
        if r:
            return r

    # 3. CSS — unconditional probe. Bare tag names ("h1", "button"), structural
    # selectors ("button[data-testid='x']"), and class/id forms all match here.
    # Plain English ("Sign In") gets parsed as `<Sign>` containing `<In>`,
    # matches zero elements, and falls through cheaply via the count() guard.
    r = probe(lambda: page.locator(raw), "css")
    if r:
        return r

    # 4. Plain visible text
    r = probe(lambda: page.get_by_text(raw, exact=False), "text")
    if r:
        return r

    # 5. Role probe — try common interactive roles with name=hint
    for role in _ROLE_PROBE:
        r = probe(
            lambda role=role: page.get_by_role(role, name=raw),  # type: ignore[arg-type]
            f"role:{role}",
        )
        if r:
            return r

    # 6. Fuzzy AX-tree fallback (Phase A4.2). Only runs after every
    # literal strategy missed. Catches "Add to Cart" ↔ "Add Cart",
    # "search" ↔ "search-field", "Sign In" ↔ "Sign in", etc.
    fuzzy_target, near_misses = _fuzzy_resolve(
        page, raw, timeout_ms=timeout_ms,
    )
    if fuzzy_target is not None:
        # Stitch the audit trail so callers see the literal strategies
        # tried before fuzzy got a hit.
        attempts.append(fuzzy_target.strategy)
        fuzzy_target.attempts = list(attempts)
        return fuzzy_target

    # SAP.1 — child-frame penetration. Many enterprise apps (SAP GUI
    # for HTML, ServiceNow, Salesforce embedded scenarios) render the
    # actual UI inside an iframe. The top-level locator never sees
    # those elements. Walk every same-origin child frame and re-run
    # the literal + fuzzy stack against each. Cross-origin frames
    # are silently skipped (Playwright can't access their DOM).
    frame_match = _resolve_in_frames(
        page, raw, timeout_ms=timeout_ms,
    )
    if frame_match is not None:
        attempts.append(frame_match.strategy)
        frame_match.attempts = list(attempts)
        return frame_match

    # Final failure — include near-miss candidates so the agent's next
    # turn can target them directly. Distinguishes "page has no
    # similar element" from "page has something close but not close
    # enough" (a real signal for the orchestrator's halt categorization
    # in A4.3).
    msg = (
        f"No visible element matched target_hint={raw!r}. "
        f"Tried: {', '.join(attempts)}"
    )
    if near_misses:
        candidates_str = ", ".join(
            f"{nm['role'] or '?'}:{nm['name']!r} ({nm['score']})"
            for nm in near_misses[:3]
        )
        msg += f". Closest candidates on page: {candidates_str}"
    raise SelectorNotFound(msg)


def _resolve_in_frames(
    page: Page,
    hint: str,
    *,
    timeout_ms: int,
) -> ResolvedTarget | None:
    """SAP.1 — walk same-origin child frames looking for the hint.

    SAP Fiori / GUI for HTML, plus a long tail of legacy enterprise
    apps, render their real UI inside iframes. Top-level Playwright
    selectors never see those elements; we have to search frames
    explicitly.

    Strategy: for each frame (excluding the main frame, already
    searched), try the same waterfall — text marker, CSS, get_by_text,
    role probes. Skips cross-origin frames (Playwright reports them
    but DOM access raises). First match wins; later frames aren't
    searched. Returns ``None`` when no frame yielded a hit.
    """
    raw = (hint or "").strip()
    if not raw:
        return None
    try:
        frames = list(page.frames)
    except Exception:
        return None
    main = page.main_frame
    for frame in frames:
        if frame is main:
            continue
        # Cross-origin / detached frames raise on access — skip.
        # Touching ``.url`` is the probe; the value is discarded.
        try:
            frame.url  # noqa: B018
        except Exception:
            continue

        def _frame_probe(
            builder: Callable[[], Locator],
            label: str,
        ) -> ResolvedTarget | None:
            try:
                loc = builder()
            except Exception:
                return None
            try:
                if loc.count() == 0:
                    return None
                loc.first.wait_for(state="visible", timeout=timeout_ms)
            except (PWTimeoutError, Exception):
                return None
            return ResolvedTarget(
                locator=loc.first,
                strategy=f"frame:{label}",
                raw_hint=raw,
            )

        # Same waterfall as resolve(), bound to this frame.
        m = _TEXT_MARKER_RE.match(raw)
        if m:
            text = m.group(1)
            r = _frame_probe(
                lambda: frame.get_by_text(text, exact=False),
                "text-marker",
            )
            if r:
                return r

        r = _frame_probe(lambda: frame.locator(raw), "css")
        if r:
            return r

        r = _frame_probe(
            lambda: frame.get_by_text(raw, exact=False), "text",
        )
        if r:
            return r

        for role in _ROLE_PROBE:
            r = _frame_probe(
                lambda role=role: frame.get_by_role(role, name=raw),  # type: ignore[arg-type]
                f"role:{role}",
            )
            if r:
                return r
    return None
