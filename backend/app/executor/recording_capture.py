"""Phase W — JS injection that captures user actions for recording.

Installed via ``page.add_init_script`` so it runs on EVERY page
(post-navigation, post-redirect, post-iframe-load). The script:

- Listens for ``click`` events at the document level (capture phase,
  before the page's own handlers swallow them).
- Buffers ``input`` events into a per-field "typed value" snapshot,
  flushing the final string on ``blur``.
- Captures ``keydown`` for Enter / Escape / Tab (control keys that
  don't show up as input events).
- Detects navigation via ``visibilitychange`` + the unload event.

Batches events in-memory and posts them to ``/api/recordings/{run_id}
/events`` every 1000ms or when the buffer hits 20 events. Network
failure → re-queue (operator's manual recording can't be lost just
because the backend hiccupped).

Per-target metadata captured
----------------------------
For each clicked element we capture:
- tag (BUTTON / INPUT / A / DIV / etc.)
- ARIA role
- visible text (trimmed)
- id, name, type, placeholder, aria-label, title
- a best-effort stable selector: id → data-testid → unique class
  combo → role+nth → coords as fallback
- bounding rect (in viewport pixels) for the highlight overlay
  during replay

This metadata is enough for the replay walker to either re-locate
the element via DOM resolution OR fall back to clicking at the
recorded coords.
"""

from __future__ import annotations


def build_capture_init_script(run_id: int, backend_origin: str) -> str:
    """Return the JS source to inject. ``run_id`` is the agent_run
    row id; ``backend_origin`` is the http://host:port the JS posts
    events to (e.g. http://localhost:8000).
    """
    return _CAPTURE_TEMPLATE.replace(
        "__RUN_ID__", str(int(run_id)),
    ).replace(
        "__BACKEND_ORIGIN__", backend_origin.rstrip("/"),
    )


_CAPTURE_TEMPLATE = r"""
(() => {
  // ── Phase W — user-action capture ──────────────────────────
  // Idempotent: if a prior injection on this page already ran,
  // skip — we only want one capture installed per page lifetime.
  if (window.__qaiCaptureInstalled) return;
  window.__qaiCaptureInstalled = true;

  const RUN_ID = __RUN_ID__;
  const BACKEND = "__BACKEND_ORIGIN__";
  const POST_URL = BACKEND + "/api/recordings/" + RUN_ID + "/events";

  const buffer = [];
  let flushTimer = null;

  function flushNow() {
    if (flushTimer) {
      clearTimeout(flushTimer);
      flushTimer = null;
    }
    if (buffer.length === 0) return;
    const batch = buffer.splice(0, buffer.length);
    try {
      fetch(POST_URL, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({events: batch}),
        keepalive: true,
        // Avoid being blocked by CORS preflight on slow apps —
        // send with no credentials so the backend can use *.
        credentials: "omit",
      }).catch((e) => {
        // Re-queue on failure; next tick will retry.
        buffer.unshift(...batch);
      });
    } catch (e) {
      buffer.unshift(...batch);
    }
  }

  function scheduleFlush() {
    if (flushTimer) return;
    flushTimer = setTimeout(flushNow, 1000);
  }

  function pushEvent(ev) {
    ev.t = Date.now();
    buffer.push(ev);
    if (buffer.length >= 20) {
      flushNow();
    } else {
      scheduleFlush();
    }
  }

  // ── Element metadata helpers ───────────────────────────────
  function shortText(el) {
    if (!el) return "";
    const t = (el.innerText || el.textContent || el.value || "").trim();
    return t.slice(0, 120);
  }
  function attr(el, name) {
    const v = el && el.getAttribute && el.getAttribute(name);
    return v ? String(v) : "";
  }
  function bestSelector(el) {
    if (!el) return "";
    if (el.id && /^[A-Za-z_][\w-]*$/.test(el.id)) return "#" + el.id;
    const testId = attr(el, "data-testid");
    if (testId) return '[data-testid="' + testId + '"]';
    const ariaLabel = attr(el, "aria-label");
    if (ariaLabel) return '[aria-label="' + ariaLabel.replace(/"/g, '\\"') + '"]';
    // Tag + class combination (only stable-looking classes).
    const tag = (el.tagName || "").toLowerCase();
    const classes = (el.className || "").toString()
      .split(/\s+/)
      .filter(c => c && !/^css-|^Mui[A-Z]/.test(c))  // skip hashed
      .slice(0, 2);
    if (classes.length > 0) {
      return tag + "." + classes.join(".");
    }
    // Role + text fallback.
    const role = attr(el, "role") || tag;
    const t = shortText(el).replace(/"/g, '\\"');
    if (t && t.length < 40) {
      return role + "[text=" + JSON.stringify(t) + "]";
    }
    return tag;
  }
  function rectOf(el) {
    if (!el || !el.getBoundingClientRect) return null;
    const r = el.getBoundingClientRect();
    return {
      x: Math.round(r.left),
      y: Math.round(r.top),
      w: Math.round(r.width),
      h: Math.round(r.height),
    };
  }
  function describeTarget(el) {
    if (!el) return null;
    return {
      tag: (el.tagName || "").toLowerCase(),
      role: attr(el, "role"),
      text: shortText(el),
      id: el.id || "",
      name: attr(el, "name"),
      type: attr(el, "type"),
      placeholder: attr(el, "placeholder"),
      aria_label: attr(el, "aria-label"),
      title: attr(el, "title"),
      selector: bestSelector(el),
      rect: rectOf(el),
    };
  }

  // ── Click capture (capture phase) ──────────────────────────
  document.addEventListener("click", (e) => {
    const target = e.target;
    pushEvent({
      kind: "click",
      x: e.clientX,
      y: e.clientY,
      button: e.button,
      target: describeTarget(target),
      url: location.href,
    });
  }, true);

  // ── Input / typing capture ─────────────────────────────────
  // We DON'T capture every keystroke; we snapshot the field's
  // value on blur or after a 600ms quiet period. That gives us
  // one ``type`` event per filled field rather than 30+ per word.
  const fieldQuietTimers = new WeakMap();
  function snapshotInputValue(el) {
    if (!el) return;
    const value = (el.value !== undefined)
      ? String(el.value)
      : (el.innerText || el.textContent || "").trim();
    pushEvent({
      kind: "type",
      value: value,
      target: describeTarget(el),
      url: location.href,
    });
  }
  document.addEventListener("input", (e) => {
    const target = e.target;
    if (!target) return;
    // Reset / start the quiet timer; on idle, push the snapshot.
    const prev = fieldQuietTimers.get(target);
    if (prev) clearTimeout(prev);
    const tid = setTimeout(() => {
      snapshotInputValue(target);
      fieldQuietTimers.delete(target);
    }, 600);
    fieldQuietTimers.set(target, tid);
  }, true);
  document.addEventListener("blur", (e) => {
    const target = e.target;
    if (!target) return;
    const t = fieldQuietTimers.get(target);
    if (t) {
      clearTimeout(t);
      fieldQuietTimers.delete(target);
      snapshotInputValue(target);
    }
  }, true);

  // ── Key capture (Enter / Escape / Tab / arrow keys) ──────
  // These often drive form submission / dropdown navigation
  // without a click — must capture them to replay correctly.
  const CAPTURED_KEYS = new Set([
    "Enter", "Escape", "Tab",
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  ]);
  document.addEventListener("keydown", (e) => {
    if (!CAPTURED_KEYS.has(e.key)) return;
    pushEvent({
      kind: "key",
      key: e.key,
      ctrl: e.ctrlKey,
      shift: e.shiftKey,
      alt: e.altKey,
      url: location.href,
    });
  }, true);

  // ── Navigation capture ────────────────────────────────────
  // Pushed at page-load time so the replay knows when a URL
  // change happened. Combined with the next event's ``url``
  // field, the walker can detect mid-flow navigations.
  pushEvent({kind: "navigate", url: location.href});

  // Flush on unload — best-effort, browser may kill the request.
  window.addEventListener("beforeunload", () => {
    flushNow();
  });
})();
"""
