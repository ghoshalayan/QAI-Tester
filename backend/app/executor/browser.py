"""Playwright browser-context factory.

One context-manager — :func:`browser_session` — yields a single ready Page
and tears down Page → Context → Browser → Playwright on exit, even on
exception. Sync API (``playwright.sync_api``) so it composes with the
sync orchestrator + sync DB session pattern from weeks 3-4.

The executor agent (``app.agents.execute``, step 6) uses this directly. The
runner (``app.services.agent_run_service.execute_run``, step 7) calls the
agent and forwards its events to the SSE bus.

First-run note
--------------
After ``uv add playwright`` you must download the Chromium binary once:

    cd v2/backend
    uv run playwright install chromium

If the binary is missing, :func:`browser_session` raises
:class:`BrowserNotInstalledError` with that exact remedy. The pre-flight
helper :func:`chromium_installed` lets the router surface this as a 4xx
before spawning a background task.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    sync_playwright,
)

from app.executor.pacing import Speed, get_speed_config

logger = logging.getLogger(__name__)


# Default browser config — the executor agent uses these directly. The
# values are tuned for "looks like a real desktop user" rather than for
# CI/headless throughput; flip ``headless`` per-run if you want speed.
DEFAULT_VIEWPORT_WIDTH = 1280
DEFAULT_VIEWPORT_HEIGHT = 800
DEFAULT_LOCALE = "en-US"
DEFAULT_TIMEOUT_MS = 30_000

# Tile the visible Chromium window into the left portion of the screen so
# the live presenter popup can sit on the right. ``--window-position`` and
# ``--window-size`` are Chromium-only Chromium flags Playwright forwards
# verbatim. Headless launches ignore these.
DEFAULT_WINDOW_POSITION = (0, 0)
DEFAULT_WINDOW_SIZE = (1300, 920)


class BrowserNotInstalledError(RuntimeError):
    """Raised when the Chromium binary isn't downloaded yet.

    Fix: ``cd v2/backend && uv run playwright install chromium``.
    """


def chromium_installed() -> bool:
    """Cheap probe — does the Chromium binary exist on disk?

    Used by the runner / router for pre-flight validation, so a missing
    binary surfaces as a clear 4xx instead of crashing a background task.
    """
    try:
        with sync_playwright() as pw:
            exe = pw.chromium.executable_path
            return bool(exe) and Path(exe).exists()
    except Exception:
        # If sync_playwright itself blows up (e.g. import error), treat as
        # not installed — the helpful remedy is the same.
        return False


@contextmanager
def browser_session(
    *,
    headless: bool = False,
    speed: Speed | str | None = None,
    viewport_width: int = DEFAULT_VIEWPORT_WIDTH,
    viewport_height: int = DEFAULT_VIEWPORT_HEIGHT,
    locale: str = DEFAULT_LOCALE,
    default_timeout_ms: int = DEFAULT_TIMEOUT_MS,
    window_position: tuple[int, int] | None = DEFAULT_WINDOW_POSITION,
    window_size: tuple[int, int] | None = DEFAULT_WINDOW_SIZE,
) -> Iterator[Page]:
    """Yield a ready-to-drive Playwright Page; clean up on exit.

    Args:
        headless: Run without a visible window. Default False — week 5 ships
            with the window visible so the user can see what's happening.
        speed: Speed preset name (``"slow"``/``"normal"``/``"fast"``) or
            None for the default (slow). Maps to Playwright's ``slow_mo``
            launch argument so every click/type/navigation is paced.
        viewport_width / viewport_height: Browser viewport in CSS pixels.
        locale: BCP-47 locale, used for ``Accept-Language`` and JS ``navigator.language``.
        default_timeout_ms: Default timeout for selectors / actions on the
            yielded Page. Per-action overrides are still possible.
        window_position: Top-left ``(x, y)`` for the headed Chromium window,
            in screen pixels. Default ``(0, 0)`` so the visible run tiles
            to the left edge, leaving the right side free for the live
            presenter popup. ``None`` disables the flag.
        window_size: ``(width, height)`` of the headed Chromium window in
            screen pixels (chrome included; differs from viewport which is
            content-only). Default leaves ~600px on a 1920-wide screen for
            the presenter popup. ``None`` disables the flag.

    Raises:
        BrowserNotInstalledError: If the Chromium binary hasn't been downloaded.
        RuntimeError: For any other Playwright launch failure (e.g. permission
            issues, port conflicts).
    """
    config = get_speed_config(speed)
    pw: Playwright | None = None
    browser: Browser | None = None
    context: BrowserContext | None = None
    try:
        try:
            pw = sync_playwright().start()
        except Exception as e:
            raise RuntimeError(f"Failed to start Playwright: {e}") from e

        # Tile the headed window so the live presenter popup can sit on
        # the right. Headless launches ignore these flags entirely, so it's
        # safe to always pass them.
        launch_args: list[str] = []
        if window_position is not None:
            launch_args.append(
                f"--window-position={window_position[0]},{window_position[1]}",
            )
        if window_size is not None:
            launch_args.append(
                f"--window-size={window_size[0]},{window_size[1]}",
            )

        try:
            browser = pw.chromium.launch(
                headless=headless,
                slow_mo=config.slow_mo_ms,
                args=launch_args or None,
            )
        except Exception as e:
            msg = str(e).lower()
            if (
                "executable doesn't exist" in msg
                or "executable_path" in msg
                or "browsers are not installed" in msg
            ):
                raise BrowserNotInstalledError(
                    "Chromium binary not installed. From v2/backend run:\n"
                    "    uv run playwright install chromium",
                ) from e
            raise

        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            locale=locale,
        )
        context.set_default_timeout(default_timeout_ms)

        page = context.new_page()
        yield page
    finally:
        # Reverse-order cleanup; swallow errors during teardown so we don't
        # mask the original exception on the way out.
        if context is not None:
            try:
                context.close()
            except Exception as e:
                logger.warning("Error closing browser context: %s", e)
        if browser is not None:
            try:
                browser.close()
            except Exception as e:
                logger.warning("Error closing browser: %s", e)
        if pw is not None:
            try:
                pw.stop()
            except Exception as e:
                logger.warning("Error stopping Playwright: %s", e)
