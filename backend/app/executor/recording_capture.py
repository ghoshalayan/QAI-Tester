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
  // ── Phase Z.1 — Tricentis-grade fingerprint capture ──────────
  //
  // The basic ``selector + rect`` capture was too fragile on Angular
  // and PrimeNG apps: the recorded selectors (``button.p-ripple``,
  // ``span.layout-menuitem-text.ng-star-inserted``) match dozens of
  // elements at replay, and the recorder picks the first one — wrong.
  //
  // Tricentis Tosca solves this with a richer per-element fingerprint
  // that's framework-aware and semantically anchored. We mirror it:
  //
  //   - **accessibleName**     The element's computed accessible
  //                            name (ARIA spec) — the single string
  //                            screen readers would announce. Stable
  //                            across class-hash churn.
  //   - **componentAnchor**    Walks up to the nearest ancestor with
  //                            a stable identifier (id / data-testid
  //                            / formControlName / aria-label) and
  //                            captures THAT. Replay can locate the
  //                            anchor first, then narrow within.
  //   - **container**          Nearest semantic container (form,
  //                            dialog, section, [role=…]) — gives
  //                            replay a scope to constrain text-only
  //                            lookups to.
  //   - **labelText**          Visible label associated with an
  //                            input via <label for=>, aria-
  //                            labelledby, or wrapping <label>.
  //   - **siblingIndex**       Index among same-tag siblings of the
  //                            parent. Lets replay say "the 3rd
  //                            button in this form" when text-based
  //                            matching is ambiguous.
  //   - **ngReflect**          Compact dump of ng-reflect-* attrs
  //                            (Angular's data-binding hints) —
  //                            framework-specific anchor when CSS
  //                            classes are useless.
  //   - **formControl**        formControlName / [ng-reflect-name]
  //                            value. The Angular FORMS API name is
  //                            usually unique within a form scope.
  //
  // Every field is BEST-EFFORT and may be empty. Replay falls back
  // through them in priority order.

  function isHashedClass(c) {
    // Match common build-hash patterns: css-XXX, Mui[Word], _ngcontent-*,
    // *-c123, _xyzABC (Tailwind JIT), bem__hash, etc.
    if (!c) return true;
    if (/^css-/.test(c)) return true;
    if (/^Mui[A-Z]/.test(c)) return true;
    if (/^_ngcontent/.test(c)) return true;
    if (/^_nghost/.test(c)) return true;
    if (/^ng-star-inserted$/.test(c)) return true;
    if (/^ng-tns-/.test(c)) return true;
    if (/^p-element$/.test(c)) return true;
    return false;
  }

  function computeAccessibleName(el) {
    // Subset of the ARIA accessible-name algorithm. Good enough to
    // anchor replay; not 100% spec-compliant.
    if (!el) return "";
    // 1) aria-labelledby — concatenate referenced elements' text.
    const labelledBy = attr(el, "aria-labelledby");
    if (labelledBy) {
      const parts = labelledBy.split(/\s+/)
        .map(id => document.getElementById(id))
        .filter(Boolean)
        .map(e => (e.innerText || e.textContent || "").trim())
        .filter(Boolean);
      if (parts.length) return parts.join(" ").slice(0, 120);
    }
    // 2) aria-label
    const ariaLabel = attr(el, "aria-label");
    if (ariaLabel) return ariaLabel.slice(0, 120);
    // 3) <label for="id"> association
    if (el.id) {
      try {
        const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
        if (lab) {
          const t = (lab.innerText || lab.textContent || "").trim();
          if (t) return t.slice(0, 120);
        }
      } catch (e) {}
    }
    // 4) Wrapping <label>
    let p = el.parentElement;
    let depth = 0;
    while (p && depth < 3) {
      if ((p.tagName || "").toLowerCase() === "label") {
        const t = (p.innerText || p.textContent || "").trim();
        if (t) return t.slice(0, 120);
      }
      p = p.parentElement;
      depth++;
    }
    // 5) title attribute
    const title = attr(el, "title");
    if (title) return title.slice(0, 120);
    // 6) alt for images
    const alt = attr(el, "alt");
    if (alt) return alt.slice(0, 120);
    // 7) placeholder for inputs
    const placeholder = attr(el, "placeholder");
    if (placeholder) return placeholder.slice(0, 120);
    // 8) Text content (last resort)
    const text = (el.innerText || el.textContent || "").trim();
    return text.slice(0, 120);
  }

  function stableAnchorSelector(el) {
    // Return a stable CSS selector if THIS element has one; empty
    // string otherwise. Doesn't walk up — caller does that.
    if (!el) return "";
    if (el.id && /^[A-Za-z_][\w-]*$/.test(el.id)) {
      // Bail on common framework-generated ids like ``pn_id_3``,
      // ``mat-input-7``, ``cdk-overlay-12`` — those rotate.
      if (!/^(pn_id_|mat-|cdk-|mui-|radix-)/.test(el.id)) {
        return "#" + el.id;
      }
    }
    const testId = attr(el, "data-testid")
      || attr(el, "data-test")
      || attr(el, "data-cy");
    if (testId) return '[data-testid="' + testId.replace(/"/g, '\\"') + '"]';
    const formControl = attr(el, "formcontrolname")
      || attr(el, "ng-reflect-name");
    if (formControl) {
      return '[formcontrolname="' + formControl.replace(/"/g, '\\"') + '"]';
    }
    const ariaLabel = attr(el, "aria-label");
    if (ariaLabel) {
      return '[aria-label="' + ariaLabel.replace(/"/g, '\\"') + '"]';
    }
    return "";
  }

  function findComponentAnchor(el) {
    // Walk up until we find an element with a stable selector.
    // Returns null if no anchor within 8 levels (whole page is the
    // anchor — useless for replay narrowing).
    let cur = el;
    let depth = 0;
    while (cur && cur.nodeType === 1 && depth < 8) {
      const sel = stableAnchorSelector(cur);
      if (sel) {
        return {
          selector: sel,
          tag: (cur.tagName || "").toLowerCase(),
          role: attr(cur, "role"),
          accessible_name: computeAccessibleName(cur),
          depth_from_target: depth,
        };
      }
      cur = cur.parentElement;
      depth++;
    }
    return null;
  }

  function findContainer(el) {
    // Nearest semantic container: form / dialog / section / nav /
    // main / aside / [role="dialog"|"form"|"region"|"main"|"navigation"].
    const CONTAINER_TAGS = new Set([
      "form", "dialog", "section", "nav", "main", "aside", "fieldset",
    ]);
    const CONTAINER_ROLES = new Set([
      "dialog", "form", "region", "main", "navigation", "tabpanel",
      "alertdialog", "search",
    ]);
    let cur = el && el.parentElement;
    let depth = 0;
    while (cur && cur.nodeType === 1 && depth < 12) {
      const tag = (cur.tagName || "").toLowerCase();
      const role = (attr(cur, "role") || "").toLowerCase();
      if (CONTAINER_TAGS.has(tag) || CONTAINER_ROLES.has(role)) {
        return {
          tag: tag,
          role: role,
          accessible_name: computeAccessibleName(cur),
          selector: stableAnchorSelector(cur) || "",
        };
      }
      cur = cur.parentElement;
      depth++;
    }
    return null;
  }

  function findLabelText(el) {
    // For inputs / textareas / selects, the visible label text. We
    // already roll some of this into accessibleName; expose it
    // separately for replay strategies that need the label literally.
    if (!el) return "";
    const tag = (el.tagName || "").toLowerCase();
    if (!["input", "textarea", "select"].includes(tag)) return "";
    if (el.id) {
      try {
        const lab = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
        if (lab) return (lab.innerText || lab.textContent || "").trim().slice(0, 80);
      } catch (e) {}
    }
    // Wrapping <label>
    let p = el.parentElement;
    let depth = 0;
    while (p && depth < 3) {
      if ((p.tagName || "").toLowerCase() === "label") {
        return (p.innerText || p.textContent || "").trim().slice(0, 80);
      }
      p = p.parentElement;
      depth++;
    }
    return "";
  }

  function siblingIndexOf(el) {
    if (!el || !el.parentElement) return -1;
    const tag = el.tagName;
    let idx = 0;
    for (const sib of el.parentElement.children) {
      if (sib === el) return idx;
      if (sib.tagName === tag) idx++;
    }
    return -1;
  }

  function ngReflectAttrs(el) {
    // Dump ng-reflect-* attributes — Angular's data-binding hints.
    // Stable across class-hash changes; often carry semantic info
    // ("ng-reflect-form-control-name=email", "ng-reflect-router-
    // link=/admin/roles"). Returns object, empty when none present.
    if (!el || !el.attributes) return {};
    const out = {};
    for (const a of el.attributes) {
      const n = a.name || "";
      if (n.startsWith("ng-reflect-")) {
        const key = n.slice("ng-reflect-".length);
        out[key] = String(a.value || "").slice(0, 80);
      }
    }
    return out;
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
      // Phase Z.1 — Tricentis-grade fingerprint additions.
      accessible_name: computeAccessibleName(el),
      component_anchor: findComponentAnchor(el),
      container: findContainer(el),
      label_text: findLabelText(el),
      sibling_index: siblingIndexOf(el),
      ng_reflect: ngReflectAttrs(el),
      form_control: attr(el, "formcontrolname")
        || attr(el, "ng-reflect-name") || "",
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
