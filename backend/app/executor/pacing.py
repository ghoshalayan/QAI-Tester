"""Pacing — speed presets + the "is the page actually settled?" gate.

The week-5 executor went straight to selector resolution after a navigate
or a click. Real-world SPAs render skeleton DOM first, then populate from
XHR/fetch — so an element that's *visible* may still be a stale loading
shell. This module's two responsibilities:

1. :class:`SpeedConfig` — knobs that change between **slow** (default,
   for human observers + heavy-data sites), **normal**, and **fast**.
   Maps to Playwright launch ``slow_mo``, mouse ``steps``, type ``delay``,
   and the network-idle timeout.

2. :func:`wait_for_settled` — call before every action that needs the
   DOM to be ready. Waits for ``domcontentloaded`` then ``networkidle``
   within the speed-budgeted timeout. Best-effort: a missed deadline
   logs and proceeds rather than raising, because chatty SPAs (polling,
   long-poll, websocket noise) genuinely never reach networkidle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PWTimeoutError

logger = logging.getLogger(__name__)

Speed = Literal["slow", "normal", "fast"]


@dataclass(frozen=True)
class SpeedConfig:
    """Tunables that change between speed modes.

    Fields
    ------
    slow_mo_ms
        Playwright's per-action artificial delay (passed to ``launch``).
        Slows clicks, types, and navigations uniformly so a viewer can
        follow what the agent is doing.
    mouse_steps
        Number of interpolation steps in ``page.mouse.move(x, y, steps=N)``.
        Higher = smoother glide. The on-page overlay cursor tracks mousemove,
        so this is what makes the cursor visibly travel rather than teleport.
    type_delay_ms
        Per-character delay in ``locator.type(text, delay=N)``. 0 falls back
        to instant ``locator.fill`` (faster but invisible).
    network_idle_timeout_ms
        Ceiling for :func:`wait_for_settled`. Heavy SPAs may never reach
        true networkidle (polling, long-poll, ws); on timeout we proceed
        anyway with a debug log. Slow mode budgets generously.
    visibility_timeout_ms
        Per-stage timeout the selector waterfall uses for ``wait_for(visible)``.
    retry_count
        Auto-retries before escalating to the AI assist (week 7 step 6).
        Set to 0 to disable auto-retries entirely.
    retry_backoff_ms
        Initial sleep between retries; doubled each subsequent attempt.
    """

    slow_mo_ms: int
    mouse_steps: int
    type_delay_ms: int
    network_idle_timeout_ms: int
    visibility_timeout_ms: int
    retry_count: int
    retry_backoff_ms: int


SLOW = SpeedConfig(
    slow_mo_ms=500,
    mouse_steps=24,
    type_delay_ms=80,
    network_idle_timeout_ms=8_000,
    visibility_timeout_ms=8_000,
    retry_count=2,
    retry_backoff_ms=800,
)

NORMAL = SpeedConfig(
    slow_mo_ms=200,
    mouse_steps=12,
    type_delay_ms=30,
    network_idle_timeout_ms=5_000,
    visibility_timeout_ms=5_000,
    retry_count=2,
    retry_backoff_ms=500,
)

FAST = SpeedConfig(
    slow_mo_ms=0,
    mouse_steps=4,
    type_delay_ms=0,
    network_idle_timeout_ms=2_000,
    visibility_timeout_ms=3_000,
    retry_count=1,
    retry_backoff_ms=300,
)

_PRESETS: dict[Speed, SpeedConfig] = {
    "slow": SLOW,
    "normal": NORMAL,
    "fast": FAST,
}


def get_speed_config(speed: Speed | str | None) -> SpeedConfig:
    """Resolve a speed name (case-insensitive) to its preset; default SLOW.

    Defaults to slow because that's the safest setting for heavy-data sites
    where async loads bite faster runs. Callers that don't pass a speed get
    the same conservative behavior an inattentive user would.
    """
    if not speed:
        return SLOW
    key = str(speed).strip().lower()
    return _PRESETS.get(key, SLOW)  # type: ignore[arg-type]


def wait_for_settled(page: Page, config: SpeedConfig) -> bool:
    """Wait until the page is "ready to act on" — DOM parsed AND network quiet.

    Runs two sequential waits:
    - ``domcontentloaded``: HTML parsed, sync scripts done. Cheap; usually
      already true unless we just navigated.
    - ``networkidle``: no network connections for ~500ms. The expensive one;
      heavy-data sites need this to avoid clicking a skeleton row.

    Both budgeted to ``config.network_idle_timeout_ms``. Timeout is logged
    but **not raised** — a chatty SPA shouldn't kill the run; the caller's
    selector wait_for(visible) is the secondary safety net.

    Returns True when both stages settled within budget, False on timeout.
    """
    timeout = config.network_idle_timeout_ms
    settled = True
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout)
    except PWTimeoutError:
        logger.debug("wait_for_settled: domcontentloaded timeout (%dms)", timeout)
        settled = False
    try:
        page.wait_for_load_state("networkidle", timeout=timeout)
    except PWTimeoutError:
        logger.debug("wait_for_settled: networkidle timeout (%dms)", timeout)
        settled = False
    return settled
