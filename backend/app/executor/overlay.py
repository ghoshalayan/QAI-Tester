"""Visible cursor + narration overlay — the "watch the agent work" UX.

Three pieces are injected into every page the executor visits:

1. **Cursor ring** — visible from page load (centered in the viewport),
   tracks ``mousemove`` between actions, pulses on click, and runs a
   continuous "breathing" animation so it always looks alive even when
   nothing is moving. Lets a viewer see where the agent is looking.

2. **Narration banner** — translucent dark pill at the bottom with an
   action-type chip, title, and N/M counter. **Phase-aware**: shows
   ``About to click X`` before the action and ``Clicked X ✓`` /
   ``Click failed`` after, so each step has clear before/after states.

3. **Target highlight** — a green outline ring drawn around the element
   the agent just verified or about to act on. Auto-fades after 1.5s.
   Verify steps stop being silent.

Wiring
------
- :func:`install_overlay` is called once per browser session, after the
  page is created. It registers the JS as an init-script so it re-runs on
  every navigation automatically.
- :func:`update_narration` is called by the orchestrator at each step
  boundary with a ``phase`` of ``"about_to"`` / ``"did"`` / ``"failed"``.
- :func:`highlight_target` is called by the verify action handler to
  draw the green ring around the verified element.
- :func:`hide_narration` clears the banner at run end.

Always-on
---------
The overlay installs in both headed and headless modes so per-step PNGs
in ``data/screenshots/<run_id>/`` capture the cursor + banner + highlights.
"""

from __future__ import annotations

import json
import logging

from playwright.sync_api import Locator, Page

logger = logging.getLogger(__name__)


# Top of the int32 range — beats almost every site's z-index war.
_OVERLAY_Z_INDEX = 2147483647

# Single-self-executing function: idempotent (re-checks an installed flag),
# styles inlined to bypass any host CSS reset, and works whether the script
# fires before or after DOMContentLoaded.
OVERLAY_INIT_SCRIPT = r"""
(() => {
  // Idempotence guard: gate on the cursor element being present in THIS
  // document, not on a window flag. set_content (and other same-window
  // content swaps) replace the document body but reuse the window object,
  // so a window flag would survive while the elements get garbage-collected.
  // Checking the element catches that case correctly.
  if (document.getElementById('__qai-cursor')) return;
  window.__qaiOverlayInstalled = true;  // kept for back-compat introspection

  const Z_TOP = """ + str(_OVERLAY_Z_INDEX) + r""";

  // Build the style element in memory; it gets appended later by install()
  // alongside the cursor + banner. At init-script time on a fresh navigation,
  // BOTH document.head AND document.documentElement can be null, which would
  // throw and abort the IIFE before any later code (incl __qaiNarrate)
  // runs. Deferring the append solves that.
  const style = document.createElement('style');
  style.id = '__qai-overlay-style';
  style.textContent = `
    @keyframes __qai-breathe {
      0%, 100% { box-shadow: 0 0 0 4px rgba(56, 132, 255, 0.10),
                              0 1px 4px rgba(0, 0, 0, 0.25); }
      50%      { box-shadow: 0 0 0 9px rgba(56, 132, 255, 0.06),
                              0 1px 4px rgba(0, 0, 0, 0.25); }
    }
    @keyframes __qai-banner-pulse {
      0%   { transform: scale(0.99); }
      40%  { transform: scale(1.005); }
      100% { transform: scale(1.0); }
    }
    @keyframes __qai-highlight-fade {
      0%   { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.85),
                          0 0 0 6px rgba(34, 197, 94, 0.45); }
      100% { box-shadow: 0 0 0 8px rgba(34, 197, 94, 0.0),
                          0 0 0 14px rgba(34, 197, 94, 0.0); }
    }
  `;

  // ── Cursor ring ───────────────────────────────────────────────
  // Default to viewport center so it's visible from page load (rather
  // than waiting for the first mousemove). The on-page agent ALWAYS has
  // a visible position, even on navigate-only / verify-only steps.
  const cursor = document.createElement('div');
  cursor.id = '__qai-cursor';
  cursor.setAttribute('aria-hidden', 'true');
  cursor.style.cssText = `
    position: fixed;
    left: 50vw;
    top: 50vh;
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
                box-shadow 140ms ease, left 50ms linear, top 50ms linear;
    transform: translate(-50%, -50%) scale(1);
    will-change: left, top, transform;
    animation: __qai-breathe 2.4s ease-in-out infinite;
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
    transition: opacity 220ms ease, border-color 220ms ease;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.30);
    max-width: calc(100vw - 32px);
    transform-origin: center bottom;
  `;

  const phaseDot = document.createElement('span');
  phaseDot.id = '__qai-phase';
  phaseDot.style.cssText = `
    display: inline-block;
    flex-shrink: 0;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgba(56, 132, 255, 0.85);
    box-shadow: 0 0 6px rgba(56, 132, 255, 0.6);
    transition: background 220ms ease, box-shadow 220ms ease;
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
    transition: background 220ms ease, color 220ms ease;
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

  banner.appendChild(phaseDot);
  banner.appendChild(actionChip);
  banner.appendChild(textSpan);
  banner.appendChild(counter);

  // ── Install (style + cursor + banner) ─────────────────────────
  // Single deferred function: needs document.body to exist before any
  // DOM insertion is safe on a fresh navigation. Falls back to
  // DOMContentLoaded if called too early.
  function install() {
    if (!document.body) return false;
    // Once body exists, head/documentElement also exist — pick whichever
    // takes the style element first.
    const styleParent = document.head || document.documentElement
                        || document.body;
    if (styleParent && !document.getElementById('__qai-overlay-style')) {
      styleParent.appendChild(style);
    }
    if (!document.body.contains(cursor)) document.body.appendChild(cursor);
    if (!document.body.contains(banner)) document.body.appendChild(banner);
    return true;
  }
  if (!install()) {
    document.addEventListener('DOMContentLoaded', install, { once: true });
  }
  // Re-install if the page nukes our nodes (rare but cheap to defend against)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') install();
  });

  // ── Cursor tracking ───────────────────────────────────────────
  document.addEventListener('mousemove', (e) => {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top = e.clientY + 'px';
  }, { capture: true, passive: true });

  document.addEventListener('mousedown', () => {
    cursor.style.transform = 'translate(-50%, -50%) scale(1.6)';
    cursor.style.background = 'rgba(56, 132, 255, 0.42)';
    cursor.style.boxShadow = '0 0 0 8px rgba(56, 132, 255, 0.14),' +
                             '0 1px 6px rgba(0, 0, 0, 0.30)';
  }, { capture: true, passive: true });

  document.addEventListener('mouseup', () => {
    cursor.style.transform = 'translate(-50%, -50%) scale(1)';
    cursor.style.background = 'rgba(56, 132, 255, 0.18)';
    cursor.style.boxShadow = '0 0 0 4px rgba(56, 132, 255, 0.10),' +
                             '0 1px 4px rgba(0, 0, 0, 0.25)';
  }, { capture: true, passive: true });

  // ── Phase-aware narration ─────────────────────────────────────
  // Phases:
  //   "about_to": neutral blue, chip says lower-case action_type
  //   "did":      green tint, chip prefix " ✓ ", banner border green
  //   "failed":   red tint, chip prefix " ✗ ", banner border red
  //   "blocked":  amber tint
  // Old callers omit phase → behaves like "about_to" (the default).
  const PHASE_STYLES = {
    about_to: {
      dot: 'rgba(56, 132, 255, 0.85)',
      dotShadow: 'rgba(56, 132, 255, 0.6)',
      chipBg: 'rgba(56, 132, 255, 0.22)',
      chipColor: '#7fb6ff',
      border: 'rgba(255, 255, 255, 0.08)',
    },
    did: {
      dot: 'rgba(34, 197, 94, 0.95)',
      dotShadow: 'rgba(34, 197, 94, 0.65)',
      chipBg: 'rgba(34, 197, 94, 0.22)',
      chipColor: '#7eed9d',
      border: 'rgba(34, 197, 94, 0.45)',
    },
    failed: {
      dot: 'rgba(239, 68, 68, 0.95)',
      dotShadow: 'rgba(239, 68, 68, 0.65)',
      chipBg: 'rgba(239, 68, 68, 0.22)',
      chipColor: '#ff8b8b',
      border: 'rgba(239, 68, 68, 0.45)',
    },
    blocked: {
      dot: 'rgba(234, 179, 8, 0.95)',
      dotShadow: 'rgba(234, 179, 8, 0.65)',
      chipBg: 'rgba(234, 179, 8, 0.22)',
      chipColor: '#facc15',
      border: 'rgba(234, 179, 8, 0.45)',
    },
  };

  window.__qaiNarrate = function(payload) {
    if (!payload) return;
    const phase = (typeof payload.phase === 'string'
      ? payload.phase : 'about_to');
    const styles = PHASE_STYLES[phase] || PHASE_STYLES.about_to;

    // Apply phase styles
    phaseDot.style.background = styles.dot;
    phaseDot.style.boxShadow = '0 0 6px ' + styles.dotShadow;
    actionChip.style.background = styles.chipBg;
    actionChip.style.color = styles.chipColor;
    banner.style.borderColor = styles.border;

    if (typeof payload.action === 'string' && payload.action.length > 0) {
      const a = payload.action.slice(0, 24);
      // Decorate chip with a phase prefix so even a colorblind user gets it.
      const prefix =
        phase === 'did' ? '✓ ' :
        phase === 'failed' ? '✗ ' :
        phase === 'blocked' ? '⚠ ' : '';
      actionChip.textContent = prefix + a;
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

    // Brief pulse to draw the eye on each transition.
    banner.style.animation = 'none';
    void banner.offsetWidth; // re-trigger
    banner.style.animation = '__qai-banner-pulse 280ms ease-out';
  };

  window.__qaiHideBanner = function() {
    // Phase P.1 — fully remove from layout (visibility:hidden +
    // display:none) so the banner can't be picked up by
    // page.screenshot() even when alpha-compositing is applied at
    // the OS level (some Windows builds render opacity:0 elements
    // into the screenshot bitmap). The text content is preserved so
    // __qaiShowBanner can restore it instantly.
    banner.style.opacity = '0';
    banner.style.visibility = 'hidden';
    banner.style.pointerEvents = 'none';
  };

  window.__qaiShowBanner = function() {
    // Restore the banner after a screenshot capture. We only restore
    // visibility; the text + phase styles persist from the last
    // updateNarration call.
    banner.style.opacity = '1';
    banner.style.visibility = 'visible';
    banner.style.pointerEvents = 'none';
  };

  // ── Target highlight ──────────────────────────────────────────
  // The verify handler calls this after resolving target_hint, so a
  // verify step visibly does something. Pass an element directly (set
  // by the Python side via JSHandle) or a CSS selector; we draw a
  // green outline ring that auto-fades over 1.5s.
  window.__qaiHighlight = function(target, opts) {
    let el = null;
    if (typeof target === 'string') {
      try { el = document.querySelector(target); } catch (e) { return; }
    } else if (target && target.nodeType === 1) {
      el = target;
    }
    if (!el) return;

    const ttl = (opts && opts.duration_ms) || 1500;
    const color =
      (opts && opts.color) ||
      'rgba(34, 197, 94, 0.8)';   // green default
    const ringColor =
      (opts && opts.ringColor) ||
      'rgba(34, 197, 94, 0.30)';

    // Save the element's existing outline so we can restore it.
    const prevOutline = el.style.outline;
    const prevOutlineOffset = el.style.outlineOffset;
    const prevTransition = el.style.transition;

    el.style.transition = 'outline-color 200ms ease, outline-offset 200ms ease';
    el.style.outline = '3px solid ' + color;
    el.style.outlineOffset = '2px';
    el.style.boxShadow = (el.style.boxShadow || '') + ', 0 0 0 6px ' + ringColor;
    el.style.animation = '__qai-highlight-fade ' + ttl + 'ms ease-out forwards';

    setTimeout(() => {
      el.style.outline = prevOutline;
      el.style.outlineOffset = prevOutlineOffset;
      el.style.transition = prevTransition;
      el.style.animation = '';
      // Restore boxShadow by removing only our suffix
      el.style.boxShadow = (el.style.boxShadow || '')
        .replace(', 0 0 0 6px ' + ringColor, '');
    }, ttl);
  };

  // ── Phase W — transient rect highlight for replay ────────────
  // The recording-replay walker calls this BEFORE clicking the
  // recorded element. Draws a green ring at (x, y, w, h) in
  // viewport pixels and fades it out over ``duration`` ms. Used
  // when we don't have a direct element reference (the replay
  // resolves the element via the recorded selector, then asks
  // the overlay to ring it at the resolved bbox).
  window.__qaiHighlightRect = function(x, y, w, h, duration) {
    const ttl = Math.max(200, Math.min(5000, duration || 1800));
    const ring = document.createElement('div');
    ring.style.position = 'fixed';
    ring.style.left = (x - 4) + 'px';
    ring.style.top = (y - 4) + 'px';
    ring.style.width = (w + 8) + 'px';
    ring.style.height = (h + 8) + 'px';
    ring.style.border = '3px solid rgba(34, 197, 94, 0.95)';
    ring.style.borderRadius = '6px';
    ring.style.boxShadow =
      '0 0 0 4px rgba(34, 197, 94, 0.25), ' +
      '0 0 18px rgba(34, 197, 94, 0.5)';
    ring.style.pointerEvents = 'none';
    ring.style.zIndex = '2147483646';
    ring.style.transition =
      'opacity ' + ttl + 'ms ease-out, transform 250ms ease-out';
    ring.style.opacity = '1';
    ring.style.transform = 'scale(1)';
    // Mark the ring so the universal click listener can ignore
    // clicks that land on the ring itself (defensive — pointer-
    // events:none should prevent the click from reaching it, but
    // the dataset attr is a belt-and-braces guard).
    ring.dataset.qaiRing = '1';
    document.body.appendChild(ring);
    // Brief pulse, then fade.
    requestAnimationFrame(() => {
      ring.style.transform = 'scale(1.04)';
      requestAnimationFrame(() => {
        ring.style.transform = 'scale(1)';
        ring.style.opacity = '0';
      });
    });
    setTimeout(() => {
      try { ring.remove(); } catch (e) {}
    }, ttl + 200);
  };

  // ── Phase AD — universal click ring ──────────────────────────
  //
  // Draws the green highlight ring on ANY click that happens on
  // the page, regardless of source:
  //   - Agent's executor (page.mouse.click / locator.click)
  //   - Recording playback walker (already calls __qaiHighlightRect
  //     pre-click; this listener will add a second ring on the
  //     actual click event — visually they overlap into one)
  //   - Operator's manual clicks during a recording session
  //   - Operator's manual clicks during live-watch (post-test review)
  //
  // Capture-phase listener so it fires before the page's own
  // handlers — works on pages that stop event propagation in
  // their handlers (e.g. Angular's host stopProp wrappers).
  //
  // Idempotent install: if a prior overlay injection on this page
  // already wired the listener, skip — otherwise navigation +
  // re-injection would stack listeners and duplicate rings.
  if (!window.__qaiClickListenerInstalled) {
    window.__qaiClickListenerInstalled = true;
    document.addEventListener('click', function (e) {
      try {
        const t = e.target;
        if (!t || !t.getBoundingClientRect) return;
        // Skip our own UI: narration banner (id="__qai-banner"),
        // the ring divs themselves (dataset.qaiRing="1"), and
        // anything inside an element whose id starts with "__qai".
        if (t.dataset && t.dataset.qaiRing) return;
        const idStartsWithQai = function (el) {
          return el && el.id && typeof el.id === 'string'
            && el.id.indexOf('__qai') === 0;
        };
        if (idStartsWithQai(t)) return;
        if (t.closest) {
          let cur = t;
          while (cur && cur !== document.documentElement) {
            if (idStartsWithQai(cur)) return;
            cur = cur.parentElement;
          }
        }
        const r = t.getBoundingClientRect();
        if (r.width > 0 && r.height > 0 && window.__qaiHighlightRect) {
          window.__qaiHighlightRect(r.left, r.top, r.width, r.height, 1200);
          return;
        }
        // Zero-size target (svg use-element, span around an icon
        // that takes no layout, etc.) — fall back to a small
        // ring centred on the click coordinate.
        if (window.__qaiHighlightRect) {
          window.__qaiHighlightRect(
            e.clientX - 8, e.clientY - 8, 16, 16, 1200,
          );
        }
      } catch (err) {
        // Never break the page over a highlight failure.
      }
    }, true);
  }
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


# Phase AF — hide the OS cursor on the page.
#
# When the operator is WATCHING an agentic run (not interacting),
# their idle cursor on top of the browser clutters the view —
# distracts from the green click-rings the overlay draws and the
# screenshots the report captures. Inject a CSS rule that suppresses
# the cursor whenever the mouse is over any element on the page.
#
# Effect is scoped to the page's content area. The Chromium chrome
# (tabs / address bar / menu) keeps its normal cursor — only the
# rendered viewport hides it. The operator can still position the
# mouse there; we just don't render a cursor glyph.
#
# Recording mode deliberately does NOT call this — the operator needs
# the cursor visible to click recorded elements accurately.

_CURSOR_HIDE_INIT_SCRIPT = r"""
(() => {
  if (window.__qaiCursorHideInstalled) return;
  window.__qaiCursorHideInstalled = true;
  const inject = () => {
    if (!document.documentElement) return false;
    if (document.getElementById('__qai-cursor-hide-style')) return true;
    const style = document.createElement('style');
    style.id = '__qai-cursor-hide-style';
    // ``cursor: none`` cascades through every descendant unless they
    // override. The ``!important`` is needed because most apps set
    // explicit cursors on buttons / inputs / links and would defeat
    // the rule without it.
    style.textContent =
      '*, *::before, *::after { cursor: none !important; }';
    document.documentElement.appendChild(style);
    return true;
  };
  if (!inject()) {
    document.addEventListener('DOMContentLoaded', inject, { once: true });
  }
})();
"""


def hide_cursor_on_page(page: Page) -> None:
    """Inject a CSS rule that suppresses the cursor over the page.

    Idempotent + safe to call after :func:`install_overlay`. Failures
    are logged but never raise — visual polish, not load-bearing.

    Use from agentic / replay runs where the operator is observing.
    Do NOT use from recording runs (operator needs the cursor to
    interact with the page).
    """
    try:
        page.context.add_init_script(_CURSOR_HIDE_INIT_SCRIPT)
        page.evaluate(_CURSOR_HIDE_INIT_SCRIPT)
    except Exception as e:
        logger.debug("Failed to install cursor hide: %s", e)


def update_narration(
    page: Page,
    *,
    ordinal: int,
    total: int,
    title: str,
    action_type: str | None,
    phase: str = "about_to",
) -> None:
    """Push step metadata into the on-page banner.

    ``phase`` is one of:
    - ``"about_to"`` (default) — neutral blue, "Clicking X" tone
    - ``"did"`` — green tint, ✓ prefix, "Clicked X" tone
    - ``"failed"`` — red tint, ✗ prefix
    - ``"blocked"`` — amber, ⚠ prefix

    The phase styles the banner border + the action-chip color so the
    transition is visible from across the room.

    Errors are swallowed — pages can close, navigate, or block JS while
    we're trying to talk to them, and a missed narration update is never
    worth crashing the run.
    """
    payload = {
        "action": (action_type or "step").lower(),
        "title": title or "",
        "ordinal": ordinal,
        "total": total,
        "phase": phase,
    }
    try:
        page.evaluate(
            f"window.__qaiNarrate && window.__qaiNarrate({json.dumps(payload)})",
        )
    except Exception as e:
        logger.debug("update_narration suppressed error: %s", e)


def highlight_target(
    page: Page,
    locator: Locator,
    *,
    duration_ms: int = 1500,
) -> None:
    """Draw a green ring around the element backing ``locator`` for
    ``duration_ms`` milliseconds.

    Called by the verify handler so verify steps visibly do something
    (and any other action that wants to draw attention to a target).
    Works regardless of which strategy the selector waterfall used —
    we marshal the live ElementHandle into the JS evaluate call rather
    than re-querying by selector string.

    Errors (locator no longer attached, page closed, JS blocked) are
    swallowed silently — a missed highlight isn't worth crashing the run.
    """
    try:
        handle = locator.element_handle()
        if handle is None:
            return
        opts = {"duration_ms": duration_ms}
        page.evaluate(
            "([el, opts]) => "
            "window.__qaiHighlight && window.__qaiHighlight(el, opts)",
            [handle, opts],
        )
    except Exception as e:
        logger.debug("highlight_target suppressed error: %s", e)


def hide_narration(page: Page) -> None:
    """Fade the banner out — used at end-of-run to leave a clean screenshot."""
    try:
        page.evaluate("window.__qaiHideBanner && window.__qaiHideBanner()")
    except Exception as e:
        logger.debug("hide_narration suppressed error: %s", e)
