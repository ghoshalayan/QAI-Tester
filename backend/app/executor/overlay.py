"""Visible cursor + narration overlay — the "watch the agent work" UX.

Two pieces are injected into every page the executor visits:

1. **Cursor ring** — a circular indicator that tracks ``mousemove`` and
   pulses on click. Lets a viewer see where the agent is looking, even
   between actions, and shows up in the per-step screenshot so the
   run-detail timeline thumbnails inherit it.

2. **Narration banner** — a translucent dark pill anchored to the bottom
   of the viewport, with an action-type chip, the step title, and an
   N/M step counter on the right.

Wiring
------
- :func:`install_overlay` is called once per browser session, after the
  page is created. It registers the JS as an init-script so it re-runs on
  every navigation automatically.
- :func:`update_narration` is called at each ``step_started`` boundary by
  the orchestrator. Errors are swallowed — a page closing or navigating
  mid-call must not derail the run.
- :func:`hide_narration` clears the banner at run end so the final
  screenshot doesn't carry stale text.

Always-on
---------
The overlay installs in both headed and headless modes. Headed gives the
visible UX during the run; headless still benefits because the per-step
PNGs in ``data/screenshots/<run_id>/`` capture the cursor position and
narration that was active when the screenshot fired. The runtime cost of
the JS is negligible.
"""

from __future__ import annotations

import json
import logging

from playwright.sync_api import Page

logger = logging.getLogger(__name__)


# Top of the int32 range — beats almost every site's z-index war.
_OVERLAY_Z_INDEX = 2147483647

# Single-self-executing function: idempotent (re-checks an installed flag),
# styles inlined to bypass any host CSS reset, and works whether the script
# fires before or after DOMContentLoaded.
OVERLAY_INIT_SCRIPT = r"""
(() => {
  if (window.__qaiOverlayInstalled) return;
  window.__qaiOverlayInstalled = true;

  const Z_TOP = """ + str(_OVERLAY_Z_INDEX) + r""";

  // ── Cursor ring ───────────────────────────────────────────────
  const cursor = document.createElement('div');
  cursor.id = '__qai-cursor';
  cursor.setAttribute('aria-hidden', 'true');
  cursor.style.cssText = `
    position: fixed;
    left: -100px;
    top: -100px;
    width: 22px;
    height: 22px;
    margin: 0;
    padding: 0;
    border: 2px solid rgba(56, 132, 255, 0.85);
    border-radius: 50%;
    background: rgba(56, 132, 255, 0.18);
    pointer-events: none;
    z-index: ${Z_TOP};
    box-shadow: 0 0 0 4px rgba(56, 132, 255, 0.10),
                0 1px 4px rgba(0, 0, 0, 0.25);
    transition: transform 80ms ease-out, background 140ms ease,
                box-shadow 140ms ease;
    transform: translate(-50%, -50%) scale(1);
    will-change: left, top, transform;
  `;

  // ── Narration banner ──────────────────────────────────────────
  const banner = document.createElement('div');
  banner.id = '__qai-banner';
  banner.setAttribute('aria-hidden', 'true');
  banner.style.cssText = `
    position: fixed;
    left: 16px;
    right: 16px;
    bottom: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0;
    padding: 10px 14px;
    background: rgba(15, 18, 25, 0.92);
    -webkit-backdrop-filter: blur(12px);
    backdrop-filter: blur(12px);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    font: 500 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI",
          system-ui, sans-serif;
    pointer-events: none;
    z-index: ${Z_TOP - 1};
    opacity: 0;
    transition: opacity 220ms ease;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.30);
    max-width: calc(100vw - 32px);
  `;

  const actionChip = document.createElement('span');
  actionChip.id = '__qai-action';
  actionChip.style.cssText = `
    display: inline-flex;
    align-items: center;
    flex-shrink: 0;
    padding: 3px 8px;
    border-radius: 6px;
    background: rgba(56, 132, 255, 0.22);
    color: #7fb6ff;
    font: 600 10px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  `;
  actionChip.textContent = 'idle';

  const textSpan = document.createElement('span');
  textSpan.id = '__qai-text';
  textSpan.style.cssText = `
    flex: 1 1 auto;
    min-width: 0;
    opacity: 0.95;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  `;
  textSpan.textContent = '';

  const counter = document.createElement('span');
  counter.id = '__qai-counter';
  counter.style.cssText = `
    flex-shrink: 0;
    font: 500 11px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    opacity: 0.6;
  `;
  counter.textContent = '';

  banner.appendChild(actionChip);
  banner.appendChild(textSpan);
  banner.appendChild(counter);

  // ── Mount when the body exists ────────────────────────────────
  function mount() {
    if (!document.body) return false;
    if (!document.body.contains(cursor)) document.body.appendChild(cursor);
    if (!document.body.contains(banner)) document.body.appendChild(banner);
    return true;
  }
  if (!mount()) {
    document.addEventListener('DOMContentLoaded', mount, { once: true });
  }

  // Re-mount if the page nukes our nodes (rare but cheap to defend against)
  const remountOnVisibility = () => {
    if (document.visibilityState === 'visible') mount();
  };
  document.addEventListener('visibilitychange', remountOnVisibility);

  // ── Cursor tracking ───────────────────────────────────────────
  // Use capture-phase listeners so the page can't stopPropagation us out.
  document.addEventListener('mousemove', (e) => {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top = e.clientY + 'px';
  }, { capture: true, passive: true });

  const onPress = () => {
    cursor.style.transform = 'translate(-50%, -50%) scale(1.6)';
    cursor.style.background = 'rgba(56, 132, 255, 0.42)';
    cursor.style.boxShadow = '0 0 0 8px rgba(56, 132, 255, 0.14),' +
                             '0 1px 6px rgba(0, 0, 0, 0.30)';
  };
  const onRelease = () => {
    cursor.style.transform = 'translate(-50%, -50%) scale(1)';
    cursor.style.background = 'rgba(56, 132, 255, 0.18)';
    cursor.style.boxShadow = '0 0 0 4px rgba(56, 132, 255, 0.10),' +
                             '0 1px 4px rgba(0, 0, 0, 0.25)';
  };
  document.addEventListener('mousedown', onPress,
    { capture: true, passive: true });
  document.addEventListener('mouseup', onRelease,
    { capture: true, passive: true });

  // ── Public API for the orchestrator ───────────────────────────
  window.__qaiNarrate = function(payload) {
    if (!payload) return;
    if (typeof payload.action === 'string' && payload.action.length > 0) {
      actionChip.textContent = payload.action.slice(0, 24);
    }
    if (typeof payload.title === 'string') {
      textSpan.textContent = payload.title;
    }
    if (typeof payload.ordinal === 'number' &&
        typeof payload.total === 'number' && payload.total > 0) {
      counter.textContent = payload.ordinal + ' / ' + payload.total;
    } else {
      counter.textContent = '';
    }
    banner.style.opacity = '1';
  };

  window.__qaiHideBanner = function() {
    banner.style.opacity = '0';
  };
})();
"""


def install_overlay(page: Page) -> None:
    """Register the overlay init-script on the page's context.

    ``add_init_script`` runs in every new document, including navigations
    away from ``about:blank``, so we only need to call this once per
    browser session.
    """
    try:
        page.context.add_init_script(OVERLAY_INIT_SCRIPT)
        # Also evaluate immediately so the very first page (already loaded
        # at this point — usually about:blank) gets the overlay too.
        page.evaluate(OVERLAY_INIT_SCRIPT)
    except Exception as e:
        logger.warning("Failed to install overlay: %s", e)


def update_narration(
    page: Page,
    *,
    ordinal: int,
    total: int,
    title: str,
    action_type: str | None,
) -> None:
    """Push step metadata into the on-page banner.

    Errors are swallowed — pages can close, navigate, or block JS while
    we're trying to talk to them, and a missed narration update is never
    worth crashing the run.
    """
    payload = {
        "action": (action_type or "step").lower(),
        "title": title or "",
        "ordinal": ordinal,
        "total": total,
    }
    try:
        # `evaluate` evaluates the JS expression in the page's context.
        # Use json.dumps so quotes/backslashes in the title are safe.
        page.evaluate(
            f"window.__qaiNarrate && window.__qaiNarrate({json.dumps(payload)})",
        )
    except Exception as e:
        logger.debug("update_narration suppressed error: %s", e)


def hide_narration(page: Page) -> None:
    """Fade the banner out — used at end-of-run to leave a clean screenshot."""
    try:
        page.evaluate("window.__qaiHideBanner && window.__qaiHideBanner()")
    except Exception as e:
        logger.debug("hide_narration suppressed error: %s", e)
