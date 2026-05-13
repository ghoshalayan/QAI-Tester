"""Phase A — HITL overlay rendered IN the Playwright test browser.

Per the locked design (Q1.b), when the agent gets stuck and needs
human guidance, we DON'T pop a separate OS window — we inject a
modal overlay onto the test page itself. The test browser is
already in the foreground during a run (it's the active test
surface), so a panel painted on top of it is naturally visible.

What the overlay shows
----------------------
- A screenshot of the page, pre-annotated by the system with
  colored bounding boxes (Layer 1 — red=tried-failed,
  green=tried-success, blue=VL-recommended).
- The agent's current sub-goal description + a short "why I'm
  asking" line built from the recent turn log.
- A drawing canvas overlaid on the screenshot (Layer 2 —
  rectangle / freehand pen / text label / undo).
- A free-form text input for typed guidance.
- A 15-second idle countdown timer. Resets on mousemove /
  keydown / click anywhere inside the overlay; expiry submits
  ``{status: "idle_timeout"}`` and closes the overlay.
- Submit / Skip buttons.

Round-trip
----------
1. Python calls ``open_hitl_overlay(page, ...)`` — this JS-
   injects the modal HTML and returns immediately (non-blocking).
2. Python then blocks on a ``threading.Event`` keyed by the
   overlay's ``request_id``.
3. JS in the page builds the UI, draws Layer 1, accepts user
   drawing on Layer 2, runs the idle countdown.
4. On submit / skip / timeout, JS calls a Playwright-bridged
   function ``window.qaiSubmitHitl(payload)`` which resolves
   Python's Event with the payload.
5. Python reads ``shapes_b64`` (the flattened Layer 2 canvas)
   and ``text``, sends both to the next VL call.

Why ``z-index: 2147483647`` and ``position: fixed``: those put
the overlay above ANYTHING in the page, including SAP Fiori
shells and Salesforce Lightning iframes. It can still be moved
below the OS-level windows if the user alt-tabs away, but that's
unavoidable in a non-Electron context (and the test browser is
typically already the focused window).

Stale-response guard
--------------------
Every overlay request has a unique ``request_id`` (uuid4). If
the user submits an idle-timed-out overlay (rare race), Python
discards the response because the matching Event already fired.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.sync_api import Page

logger = logging.getLogger(__name__)


# Per-request response payloads. Keyed by request_id. JS posts the
# response via ``window.qaiSubmitHitl`` which the Playwright bridge
# routes to ``_receive_response`` below.
_responses: dict[str, dict[str, Any]] = {}
_response_events: dict[str, threading.Event] = {}
_lock = threading.Lock()


def _receive_response(payload: Any) -> None:
    """Called from JS via page.expose_function. Routes the payload
    into the per-request response dict and signals the matching
    threading.Event.

    ``payload`` is whatever the JS posted — we only trust
    ``request_id`` after sanity-checking it's a known one.
    """
    if not isinstance(payload, dict):
        logger.warning("HITL overlay: bad response shape %r", type(payload))
        return
    req_id = payload.get("request_id")
    if not isinstance(req_id, str):
        return
    with _lock:
        if req_id not in _response_events:
            # Stale (already timed out / submitted) — discard.
            return
        _responses[req_id] = dict(payload)
        ev = _response_events.pop(req_id)
    ev.set()


def _ensure_bridge(page: "Page") -> None:
    """Expose ``window.qaiSubmitHitl`` on the page once per browser
    session. Playwright errors on duplicate expose, so we swallow
    the second-call exception."""
    try:
        page.expose_function("qaiSubmitHitl", _receive_response)
    except Exception:
        # Already exposed in this context — fine.
        pass


def open_and_wait(
    page: "Page",
    *,
    sub_goal_description: str,
    tried_summary: str,
    screenshot_png: bytes,
    timeout_seconds: int = 15,
    idle_skip_seconds: int = 15,
) -> dict[str, Any]:
    """Open the HITL overlay and BLOCK until user submits, skips,
    or the idle timer expires.

    Returns ``{status, text, drawing_b64, request_id}``:
    - ``status`` ∈ {"submitted", "skipped", "idle_timeout", "error"}
    - ``text`` — the user's typed guidance (may be empty)
    - ``drawing_b64`` — base64 PNG of the flattened Layer 2 canvas
      (transparent except where user drew); empty on skip/timeout

    The agent caller is responsible for handling each status:
    - submitted → fold ``text`` + ``drawing_b64`` into the next
      VL call as a ``user_guidance`` observation
    - skipped / idle_timeout → mark current sub-goal as skipped,
      advance to next sub-goal
    - error → log + fall through (the overlay never appeared;
      the agent's main loop continues)
    """
    request_id = uuid.uuid4().hex
    ev = threading.Event()
    with _lock:
        _response_events[request_id] = ev

    try:
        _ensure_bridge(page)
    except Exception as e:
        logger.warning("HITL overlay: bridge setup failed: %s", e)
        with _lock:
            _response_events.pop(request_id, None)
        return {
            "status": "error",
            "text": "",
            "drawing_b64": "",
            "request_id": request_id,
            "error": str(e)[:200],
        }

    screenshot_b64 = base64.b64encode(screenshot_png).decode("ascii")

    # Pack everything as a single JSON blob and pass to the JS
    # initializer. Avoids quote-escaping nightmares vs. inline
    # interpolation.
    init_payload = {
        "request_id": request_id,
        "screenshot_b64": screenshot_b64,
        "sub_goal_description": sub_goal_description,
        "tried_summary": tried_summary,
        "idle_skip_seconds": idle_skip_seconds,
    }

    js = _OVERLAY_JS + (
        "\n;window.__qaiHitlInit("
        + json.dumps(init_payload)
        + ");"
    )
    try:
        page.evaluate(js)
    except Exception as e:
        logger.warning("HITL overlay: inject failed: %s", e)
        with _lock:
            _response_events.pop(request_id, None)
        return {
            "status": "error",
            "text": "",
            "drawing_b64": "",
            "request_id": request_id,
            "error": str(e)[:200],
        }

    # Wait for the user. Indefinite per the user's locked policy
    # (idle timer fires inside the overlay; we don't need a Python-
    # side ceiling). We DO poll periodically so a cancelled run can
    # wake this thread up — cancel handler should call
    # ``close_overlay(page, request_id)`` to release the wait.
    deadline = None  # indefinite
    polled = ev.wait(timeout=None) if deadline is None else ev.wait(deadline)

    with _lock:
        payload = _responses.pop(request_id, None)
        _response_events.pop(request_id, None)

    if payload is None:
        # Cancelled externally or JS never reported. Best-effort
        # cleanup of the overlay.
        try:
            close_overlay(page, request_id)
        except Exception:
            pass
        return {
            "status": "error",
            "text": "",
            "drawing_b64": "",
            "request_id": request_id,
            "error": "no response received",
        }

    status = str(payload.get("status", "error"))
    return {
        "status": status if status in (
            "submitted", "skipped", "idle_timeout",
        ) else "error",
        "text": str(payload.get("text", ""))[:2000],
        "drawing_b64": str(payload.get("drawing_b64", "")),
        "request_id": request_id,
    }


def close_overlay(page: "Page", request_id: str | None = None) -> None:
    """Hide / remove the overlay if it's still open. Safe to call
    twice."""
    sel = (
        f"#qai-hitl-overlay-{request_id}" if request_id
        else "[id^='qai-hitl-overlay-']"
    )
    try:
        page.evaluate(
            f"(() => {{ const els = document.querySelectorAll('{sel}'); "
            f"els.forEach(e => e.remove()); }})()",
        )
    except Exception:
        pass


# ── JS payload — drawing canvas + idle timer + submit bridge ─────


_OVERLAY_JS = r"""
(() => {
  if (window.__qaiHitlInit) return;  // re-injected; no-op

  // Builds and mounts the overlay. Called once per request via
  // window.__qaiHitlInit(payload).
  window.__qaiHitlInit = (init) => {
    const ID = 'qai-hitl-overlay-' + init.request_id;
    if (document.getElementById(ID)) return;

    // ── Root container (modal backdrop + card) ────────────────
    const root = document.createElement('div');
    root.id = ID;
    Object.assign(root.style, {
      position: 'fixed', inset: '0',
      background: 'rgba(0,0,0,0.55)',
      zIndex: '2147483647',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
      color: '#1a1a1a',
    });

    const card = document.createElement('div');
    Object.assign(card.style, {
      background: '#ffffff', borderRadius: '12px',
      width: 'min(780px, 92vw)', maxHeight: '92vh',
      boxShadow: '0 24px 64px rgba(0,0,0,0.45)',
      display: 'flex', flexDirection: 'column',
      // Card-level vertical scroll. When the content (screenshot +
      // tools + textarea + footer) exceeds the card's max height,
      // the user scrolls inside the modal to reach the bottom
      // buttons. Horizontal overflow stays clipped (rounded corners
      // remain crisp).
      overflowY: 'auto', overflowX: 'hidden',
    });
    root.appendChild(card);

    // ── Header ────────────────────────────────────────────────
    const header = document.createElement('div');
    Object.assign(header.style, {
      padding: '14px 20px',
      borderBottom: '1px solid #e5e7eb',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    });
    const title = document.createElement('div');
    title.textContent = 'Agent is stuck — your guidance';
    Object.assign(title.style, {
      fontWeight: '600', fontSize: '16px',
    });
    const headerRight = document.createElement('div');
    Object.assign(headerRight.style, {
      display: 'flex', gap: '8px', alignItems: 'center',
    });
    const timerEl = document.createElement('div');
    Object.assign(timerEl.style, {
      fontFamily: 'monospace', fontSize: '13px',
      padding: '4px 8px', borderRadius: '6px',
      background: '#fef3c7', color: '#92400e',
    });
    timerEl.textContent = 'auto-skip in ' + init.idle_skip_seconds + 's';
    const pinBtn = document.createElement('button');
    pinBtn.textContent = 'Minimize';
    Object.assign(pinBtn.style, {
      padding: '4px 10px', fontSize: '12px', cursor: 'pointer',
      background: '#f3f4f6', border: '1px solid #d1d5db', borderRadius: '6px',
    });
    headerRight.appendChild(timerEl);
    headerRight.appendChild(pinBtn);
    header.appendChild(title);
    header.appendChild(headerRight);
    card.appendChild(header);

    // ── Sub-goal + tried summary ──────────────────────────────
    const meta = document.createElement('div');
    Object.assign(meta.style, {
      padding: '12px 20px', borderBottom: '1px solid #e5e7eb',
      fontSize: '13px', lineHeight: '1.5',
    });
    meta.innerHTML =
      '<div><strong>Sub-goal:</strong> ' +
      escapeHtml(init.sub_goal_description || '(none)') + '</div>' +
      '<div style="margin-top:6px;color:#4b5563;white-space:pre-wrap">' +
      escapeHtml(init.tried_summary || '') + '</div>';
    card.appendChild(meta);

    // ── Canvas area (Layer 1 background image + Layer 2 drawing) ──
    // Natural height (no internal scroll). When the whole card
    // exceeds 92vh the user scrolls the card itself to reach the
    // footer — see overflowY:auto on the card root above.
    const canvasWrap = document.createElement('div');
    Object.assign(canvasWrap.style, {
      position: 'relative', background: '#f9fafb',
      padding: '12px 20px',
    });
    const stage = document.createElement('div');
    Object.assign(stage.style, { position: 'relative', display: 'inline-block' });
    const img = document.createElement('img');
    img.src = 'data:image/png;base64,' + init.screenshot_b64;
    Object.assign(img.style, {
      display: 'block', maxWidth: '100%', height: 'auto',
      border: '1px solid #d1d5db', borderRadius: '6px',
    });
    stage.appendChild(img);
    const canvas = document.createElement('canvas');
    Object.assign(canvas.style, {
      position: 'absolute', inset: '0',
      cursor: 'crosshair', touchAction: 'none',
    });
    stage.appendChild(canvas);
    canvasWrap.appendChild(stage);
    card.appendChild(canvasWrap);

    // ── Drawing toolbar ───────────────────────────────────────
    const tools = document.createElement('div');
    Object.assign(tools.style, {
      padding: '10px 20px', borderTop: '1px solid #e5e7eb',
      borderBottom: '1px solid #e5e7eb',
      display: 'flex', gap: '8px', alignItems: 'center',
      fontSize: '13px',
    });
    const mkToolBtn = (label, val) => {
      const b = document.createElement('button');
      b.textContent = label;
      b.dataset.tool = val;
      Object.assign(b.style, {
        padding: '6px 12px', cursor: 'pointer',
        background: '#f3f4f6', border: '1px solid #d1d5db',
        borderRadius: '6px',
      });
      return b;
    };
    const btnRect = mkToolBtn('Rectangle', 'rect');
    const btnPen  = mkToolBtn('Freehand',  'pen');
    const btnText = mkToolBtn('Text',      'text');
    const btnUndo = mkToolBtn('Undo',      'undo');
    tools.appendChild(btnRect);
    tools.appendChild(btnPen);
    tools.appendChild(btnText);
    tools.appendChild(btnUndo);
    const hint = document.createElement('span');
    hint.textContent = 'Draw on the screenshot to point the agent at the right spot.';
    Object.assign(hint.style, { color: '#6b7280', marginLeft: '8px' });
    tools.appendChild(hint);
    card.appendChild(tools);

    // ── Free-form text input ──────────────────────────────────
    const textWrap = document.createElement('div');
    Object.assign(textWrap.style, { padding: '12px 20px' });
    const textLabel = document.createElement('label');
    textLabel.textContent = 'Guidance to the agent (optional):';
    Object.assign(textLabel.style, {
      display: 'block', fontSize: '13px', marginBottom: '6px', color: '#374151',
    });
    const textArea = document.createElement('textarea');
    textArea.rows = 2;
    textArea.placeholder = 'e.g. "Click the +Add New Role button at the top right"';
    Object.assign(textArea.style, {
      width: '100%', padding: '8px 10px', fontSize: '13px',
      border: '1px solid #d1d5db', borderRadius: '6px',
      resize: 'vertical', boxSizing: 'border-box', fontFamily: 'inherit',
    });
    textWrap.appendChild(textLabel);
    textWrap.appendChild(textArea);
    card.appendChild(textWrap);

    // ── Footer (Submit / Skip) ────────────────────────────────
    const footer = document.createElement('div');
    Object.assign(footer.style, {
      padding: '12px 20px', borderTop: '1px solid #e5e7eb',
      display: 'flex', justifyContent: 'flex-end', gap: '8px',
    });
    const skipBtn = document.createElement('button');
    skipBtn.textContent = 'Skip this sub-goal';
    Object.assign(skipBtn.style, {
      padding: '8px 14px', cursor: 'pointer',
      background: '#f3f4f6', border: '1px solid #d1d5db',
      borderRadius: '6px', fontSize: '13px',
    });
    const submitBtn = document.createElement('button');
    submitBtn.textContent = 'Submit guidance';
    Object.assign(submitBtn.style, {
      padding: '8px 14px', cursor: 'pointer',
      background: '#2563eb', color: 'white', border: 'none',
      borderRadius: '6px', fontSize: '13px', fontWeight: '600',
    });
    footer.appendChild(skipBtn);
    footer.appendChild(submitBtn);
    card.appendChild(footer);

    document.body.appendChild(root);

    // ── Drawing state ─────────────────────────────────────────
    let currentTool = 'rect';
    const shapes = [];   // { kind, points } | { kind, x, y, w, h } | { kind, x, y, text }
    let drawing = false;
    let startX = 0, startY = 0;
    let penPoints = null;
    const COLOR = '#06b6d4';  // bright cyan — user marks color

    const sizeCanvas = () => {
      const r = img.getBoundingClientRect();
      canvas.width = r.width;
      canvas.height = r.height;
      canvas.style.width = r.width + 'px';
      canvas.style.height = r.height + 'px';
    };
    const setActiveTool = (t) => {
      currentTool = t;
      [btnRect, btnPen, btnText].forEach(b => {
        b.style.background = (b.dataset.tool === t) ? '#dbeafe' : '#f3f4f6';
      });
    };
    setActiveTool('rect');

    img.addEventListener('load', () => { sizeCanvas(); redraw(); });
    if (img.complete) { sizeCanvas(); redraw(); }
    window.addEventListener('resize', () => { sizeCanvas(); redraw(); });

    const redraw = () => {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = COLOR;
      ctx.fillStyle = COLOR;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.font = 'bold 14px system-ui';
      for (const s of shapes) {
        if (s.kind === 'rect') {
          ctx.strokeRect(s.x, s.y, s.w, s.h);
        } else if (s.kind === 'pen' && s.points && s.points.length > 1) {
          ctx.beginPath();
          ctx.moveTo(s.points[0].x, s.points[0].y);
          for (let i = 1; i < s.points.length; i++) {
            ctx.lineTo(s.points[i].x, s.points[i].y);
          }
          ctx.stroke();
        } else if (s.kind === 'text') {
          const padding = 4;
          const w = ctx.measureText(s.text).width + 2 * padding;
          const h = 20;
          ctx.fillRect(s.x, s.y, w, h);
          ctx.fillStyle = 'white';
          ctx.fillText(s.text, s.x + padding, s.y + 14);
          ctx.fillStyle = COLOR;
        }
      }
    };

    btnRect.onclick = () => { resetIdle(); setActiveTool('rect'); };
    btnPen.onclick  = () => { resetIdle(); setActiveTool('pen');  };
    btnText.onclick = () => { resetIdle(); setActiveTool('text'); };
    btnUndo.onclick = () => {
      resetIdle();
      shapes.pop();
      redraw();
    };

    const canvasPos = (e) => {
      const r = canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    };

    canvas.addEventListener('mousedown', (e) => {
      resetIdle();
      const p = canvasPos(e);
      if (currentTool === 'text') {
        const t = prompt('Label text:');
        if (t && t.trim()) {
          shapes.push({ kind: 'text', x: p.x, y: p.y, text: t.trim().slice(0, 60) });
          redraw();
        }
        return;
      }
      drawing = true;
      startX = p.x; startY = p.y;
      if (currentTool === 'pen') {
        penPoints = [{ x: p.x, y: p.y }];
        shapes.push({ kind: 'pen', points: penPoints });
      }
    });
    canvas.addEventListener('mousemove', (e) => {
      if (!drawing) return;
      resetIdle();
      const p = canvasPos(e);
      if (currentTool === 'rect') {
        // Live preview: redraw + overlay a "current" rect.
        redraw();
        const ctx = canvas.getContext('2d');
        ctx.strokeStyle = COLOR;
        ctx.lineWidth = 3;
        ctx.strokeRect(startX, startY, p.x - startX, p.y - startY);
      } else if (currentTool === 'pen' && penPoints) {
        penPoints.push({ x: p.x, y: p.y });
        redraw();
      }
    });
    canvas.addEventListener('mouseup', (e) => {
      if (!drawing) return;
      drawing = false;
      const p = canvasPos(e);
      if (currentTool === 'rect') {
        shapes.push({
          kind: 'rect', x: startX, y: startY,
          w: p.x - startX, h: p.y - startY,
        });
        redraw();
      } else if (currentTool === 'pen') {
        penPoints = null;
      }
    });

    // ── Idle countdown ────────────────────────────────────────
    let idleRemaining = init.idle_skip_seconds;
    let idleTick = null;
    const resetIdle = () => {
      idleRemaining = init.idle_skip_seconds;
      timerEl.textContent = 'auto-skip in ' + idleRemaining + 's';
      timerEl.style.background = '#fef3c7';
      timerEl.style.color = '#92400e';
    };
    const startIdle = () => {
      if (idleTick) clearInterval(idleTick);
      idleTick = setInterval(() => {
        idleRemaining -= 1;
        timerEl.textContent = 'auto-skip in ' + idleRemaining + 's';
        if (idleRemaining <= 5) {
          timerEl.style.background = '#fee2e2';
          timerEl.style.color = '#991b1b';
        }
        if (idleRemaining <= 0) {
          clearInterval(idleTick); idleTick = null;
          finalize('idle_timeout');
        }
      }, 1000);
    };
    ['mousemove','keydown','click','wheel','pointerdown'].forEach(t =>
      root.addEventListener(t, resetIdle, true)
    );

    // ── Pin / minimize toggle ─────────────────────────────────
    let minimized = false;
    pinBtn.onclick = (e) => {
      e.stopPropagation();
      resetIdle();
      minimized = !minimized;
      if (minimized) {
        root.style.background = 'transparent';
        root.style.pointerEvents = 'none';
        Object.assign(card.style, {
          position: 'fixed', bottom: '16px', right: '16px',
          width: '360px', maxHeight: '320px', pointerEvents: 'auto',
        });
        pinBtn.textContent = 'Expand';
      } else {
        root.style.background = 'rgba(0,0,0,0.55)';
        root.style.pointerEvents = 'auto';
        Object.assign(card.style, {
          position: 'static', width: 'min(780px, 92vw)',
          maxHeight: '92vh',
        });
        pinBtn.textContent = 'Minimize';
      }
    };

    // ── Submit / Skip / Timeout finalize ──────────────────────
    const finalize = (status) => {
      if (idleTick) clearInterval(idleTick);
      let drawing_b64 = '';
      try {
        // Always flatten the drawing layer (even on skip /
        // timeout — the agent may still find the partial draw
        // useful for context).
        drawing_b64 = canvas.toDataURL('image/png').split(',')[1] || '';
      } catch (e) { /* tainted? unlikely — we drew our own image */ }
      const payload = {
        request_id: init.request_id,
        status: status,
        text: textArea.value || '',
        drawing_b64: drawing_b64,
      };
      try {
        if (window.qaiSubmitHitl) window.qaiSubmitHitl(payload);
      } catch (e) { /* bridge gone — agent will time out */ }
      try { root.remove(); } catch (e) {}
    };
    submitBtn.onclick = (e) => { e.stopPropagation(); finalize('submitted'); };
    skipBtn.onclick   = (e) => { e.stopPropagation(); finalize('skipped'); };

    // Kick off idle countdown after a beat (give the user time to
    // notice the overlay opened).
    setTimeout(startIdle, 500);
  };

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
})();
"""
