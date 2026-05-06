"""Executor — Playwright browser automation primitives.

Modules
-------
- ``browser`` — context-manager that yields a ready-to-use Playwright Page
- ``selectors`` (week 5 step 4) — ``target_hint`` → CSS → text → role waterfall
- ``actions`` (week 5 step 5) — one handler per ``action_type``

The week-5 orchestrator (``app.agents.execute``) composes these into a
tree-walking executor that records per-step results.
"""

from app.executor.actions import (
    ActionContext,
    ActionResult,
    ActionStatus,
    execute_action,
)
from app.executor.browser import (
    BrowserNotInstalledError,
    browser_session,
    chromium_installed,
)
from app.executor.overlay import (
    hide_narration,
    highlight_target,
    install_overlay,
    update_narration,
)
from app.executor.pacing import (
    FAST,
    NORMAL,
    SLOW,
    Speed,
    SpeedConfig,
    get_speed_config,
    wait_for_settled,
)
from app.executor.selectors import (
    ResolvedTarget,
    SelectorNotFound,
    resolve,
)

__all__ = [
    "ActionContext",
    "ActionResult",
    "ActionStatus",
    "BrowserNotInstalledError",
    "FAST",
    "NORMAL",
    "ResolvedTarget",
    "SLOW",
    "SelectorNotFound",
    "Speed",
    "SpeedConfig",
    "browser_session",
    "chromium_installed",
    "execute_action",
    "get_speed_config",
    "hide_narration",
    "highlight_target",
    "install_overlay",
    "resolve",
    "update_narration",
    "wait_for_settled",
]
