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

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from playwright.sync_api import Locator, Page
from playwright.sync_api import TimeoutError as PWTimeoutError

logger = logging.getLogger(__name__)

# Per-strategy timeout — short so missed strategies don't tank wall-clock.
DEFAULT_PROBE_TIMEOUT_MS = 3_000

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

    raise SelectorNotFound(
        f"No visible element matched target_hint={raw!r}. "
        f"Tried: {', '.join(attempts)}",
    )
