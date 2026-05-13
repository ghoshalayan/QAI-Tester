"""Phase A.6 Step 1 — post-action form-signal observer.

Watches for toasts, inline validation errors, ARIA-live updates, and
``aria-invalid`` field markers that appear after a submit-style action.
Without this, a form rejected with "Display Name is required" looks
identical to a successful submit from the agent's POV (Save was
clicked; drawer didn't close; agent moves on or stalls).

Called from :func:`qa_agent.run_agent_for_goal` after every action
turn whose tool is submit-like (submit / click on a button labeled
Save / Create / Submit / Confirm / Update). Returns a :class:`FormSignal`
record describing what — if anything — appeared. The qa_agent folds
the message into the NEXT turn's prompt block, plus emits a live-feed
event so the user sees the toast as it happens.

Detection strategy (all in one JS pass — single page.evaluate)
--------------------------------------------------------------
1. ARIA-live regions (``[aria-live]``, ``[role=alert]``,
   ``[role=status]``) — most modern apps put toasts here.
2. Toast-style elements by class hint (``.toast``, ``.MuiAlert-root``,
   ``.snackbar``, ``.notification``, ``[class*="toast"]``,
   ``[class*="alert"]``).
3. Inline form errors — elements with ``aria-invalid="true"`` plus
   their associated error message (``aria-describedby`` → element).
4. Generic error containers (``.error``, ``.field-error``,
   ``.form-error``, ``.invalid-feedback``).

The JS returns the FIRST plausible signal it finds (toasts take
priority over inline errors — the user usually shows the toast and
the agent should react to the explicit message first).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from playwright.sync_api import Page

logger = logging.getLogger(__name__)


SignalKind = Literal[
    "toast_success",
    "toast_error",
    "toast_warning",
    "toast_info",
    "inline_error",
    "validation_error",
    "none",
]


@dataclass
class FormSignal:
    """One signal observed on the page after a submit action.

    Empty ``message`` + kind="none" means "no signal — proceed as
    normal". The qa_agent only injects a prompt block when the kind
    is NOT "none".
    """
    kind: SignalKind
    message: str = ""
    # Field labels mentioned in the message, when the signal is a
    # validation error (e.g. ["Display Name", "Email"]). Used by
    # the runtime safety net (Step 3) to know which fields to fill.
    fields: list[str] | None = None


_FORM_SIGNAL_JS = r"""
() => {
  const out = { kind: 'none', message: '', fields: [] };
  const VISIBLE = (el) => {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' ||
        parseFloat(cs.opacity) < 0.1) return false;
    return true;
  };
  const TEXT = (el) => (el.innerText || el.textContent || '').trim().slice(0, 400);

  // 1) ARIA-live + role=alert / role=status — strongest signal.
  const aria = [
    ...document.querySelectorAll(
      '[aria-live]:not([aria-live="off"]), [role=alert], [role=status]'
    ),
  ].filter(VISIBLE).filter(e => TEXT(e).length > 0);
  if (aria.length > 0) {
    const el = aria[0];
    const text = TEXT(el);
    const lower = text.toLowerCase();
    let kind = 'toast_info';
    if (/error|fail|invalid|required|missing|reject/i.test(lower)) {
      kind = 'toast_error';
    } else if (/success|created|saved|updated|added/i.test(lower)) {
      kind = 'toast_success';
    } else if (/warning|caution/i.test(lower)) {
      kind = 'toast_warning';
    }
    out.kind = kind; out.message = text;
    return out;
  }

  // 2) Toast-class elements (MUI Alert, Sonner, Toastify, AntD, etc.)
  const toastSels = [
    '.toast', '.toast-error', '.toast-success', '.snackbar',
    '.MuiAlert-root', '.ant-message', '.ant-notification-notice',
    '.notification', '[class*="Toast"]', '[class*="toast"]',
    '[class*="Snackbar"]', '[data-sonner-toast]',
    '[class*="Notification"]', '[class*="Alert"]'
  ];
  for (const sel of toastSels) {
    const nodes = [...document.querySelectorAll(sel)].filter(VISIBLE);
    for (const el of nodes) {
      const text = TEXT(el);
      if (!text) continue;
      const lower = text.toLowerCase();
      let kind = 'toast_info';
      if (/error|fail|invalid|required|missing|reject/i.test(lower)) {
        kind = 'toast_error';
      } else if (/success|created|saved|updated|added/i.test(lower)) {
        kind = 'toast_success';
      } else if (/warning|caution/i.test(lower)) {
        kind = 'toast_warning';
      }
      // Skip very-long toast text — it's probably not a toast.
      if (text.length > 280) continue;
      out.kind = kind; out.message = text;
      return out;
    }
  }

  // 3) aria-invalid fields + their described-by error labels.
  const invalids = [...document.querySelectorAll(
    '[aria-invalid="true"], [aria-invalid="grammar"], [aria-invalid="spelling"]'
  )].filter(VISIBLE);
  if (invalids.length > 0) {
    const fieldLabels = [];
    let firstError = '';
    for (const inv of invalids) {
      // Find this field's accessible name.
      const label = inv.getAttribute('aria-label')
        || inv.getAttribute('name')
        || inv.getAttribute('placeholder')
        || '';
      if (label && !fieldLabels.includes(label)) {
        fieldLabels.push(label.trim().slice(0, 60));
      }
      // Try to read the associated error message.
      const desc = inv.getAttribute('aria-describedby');
      if (desc) {
        for (const id of desc.split(/\s+/)) {
          const errEl = document.getElementById(id);
          if (errEl && VISIBLE(errEl)) {
            const t = TEXT(errEl);
            if (t && !firstError) firstError = t;
          }
        }
      }
    }
    out.kind = 'inline_error';
    out.message = firstError ||
      ('Validation failed on: ' + fieldLabels.join(', '));
    out.fields = fieldLabels;
    return out;
  }

  // 4) Generic error containers — last resort.
  const errSels = [
    '.error', '.field-error', '.form-error', '.invalid-feedback',
    '.help-block.error', '[class*="ErrorMessage"]', '[class*="errorMessage"]',
  ];
  for (const sel of errSels) {
    const nodes = [...document.querySelectorAll(sel)].filter(VISIBLE);
    for (const el of nodes) {
      const text = TEXT(el);
      if (!text || text.length > 280) continue;
      out.kind = 'validation_error';
      out.message = text;
      return out;
    }
  }
  return out;
};
"""


def observe_form_signal(
    page: "Page",
    *,
    settle_ms: int = 400,
) -> FormSignal:
    """Run the post-action form-signal scan.

    Waits ``settle_ms`` for any toast / inline error animations to
    arrive (most apps animate them in over ~200-300 ms), then runs
    the single-pass detection JS.

    Returns ``FormSignal(kind="none")`` on any error — observer is
    strictly best-effort and never raises.
    """
    try:
        page.wait_for_timeout(settle_ms)
    except Exception:
        pass
    try:
        raw = page.evaluate(_FORM_SIGNAL_JS)
    except Exception as e:
        logger.debug("form_signal: evaluate failed: %s", e)
        return FormSignal(kind="none")
    if not isinstance(raw, dict):
        return FormSignal(kind="none")
    kind_raw = str(raw.get("kind") or "none")
    if kind_raw not in (
        "toast_success", "toast_error", "toast_warning", "toast_info",
        "inline_error", "validation_error", "none",
    ):
        kind_raw = "none"
    message = str(raw.get("message") or "")[:600]
    fields_raw = raw.get("fields") or []
    fields = [
        str(f)[:80] for f in fields_raw if isinstance(f, str) and f
    ] if isinstance(fields_raw, list) else None
    return FormSignal(
        kind=kind_raw,  # type: ignore[arg-type]
        message=message,
        fields=fields,
    )


# Tool names that should trigger the form-signal observer.
_SUBMIT_LIKE_TOOLS: frozenset[str] = frozenset({"click", "press_key"})
# Lowercased substrings that mark a submit-like target.
_SUBMIT_LIKE_HINTS: tuple[str, ...] = (
    "save", "create", "submit", "confirm", "update", "add",
    "register", "send", "publish", "apply",
)


# ── Phase A.6 Step 3 — required-field completeness check ─────────


@dataclass
class EmptyRequiredField:
    """One required field that's still empty when the agent is about
    to click submit. Pulled into the next turn's prompt block so
    the planner re-fills before resubmitting.
    """
    label: str
    role: str  # "textbox" | "combobox" | "checkbox" | "textarea"


_EMPTY_REQUIRED_JS = r"""
() => {
  // Scan the visible form for required fields with empty values.
  // Required is detected via: [required], [aria-required=true], or
  // a visible asterisk near the field's label (handled below).
  const out = [];
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' ||
        parseFloat(cs.opacity) < 0.05) return false;
    return true;
  };
  const labelFor = (el) => {
    let lab = el.getAttribute('aria-label') || el.getAttribute('placeholder')
      || el.getAttribute('name') || '';
    if (lab) return lab.trim().slice(0, 80);
    // Find a <label for="..."> referencing this element.
    if (el.id) {
      const lbl = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (lbl) return (lbl.innerText || lbl.textContent || '').trim().slice(0, 80);
    }
    // Wrapping <label>.
    let p = el.parentElement;
    for (let i = 0; i < 3 && p; i++) {
      if (p.tagName === 'LABEL') {
        return (p.innerText || p.textContent || '').trim().slice(0, 80);
      }
      p = p.parentElement;
    }
    return '';
  };
  const roleOf = (el) => {
    const t = (el.tagName || '').toLowerCase();
    if (t === 'select') return 'combobox';
    if (t === 'textarea') return 'textarea';
    if (t === 'input') {
      const it = (el.getAttribute('type') || 'text').toLowerCase();
      if (it === 'checkbox' || it === 'radio') return 'checkbox';
      return 'textbox';
    }
    if (el.getAttribute('role') === 'combobox') return 'combobox';
    if (el.getAttribute('role') === 'checkbox') return 'checkbox';
    return 'textbox';
  };
  const inputs = document.querySelectorAll(
    'input[required], input[aria-required="true"], ' +
    'select[required], select[aria-required="true"], ' +
    'textarea[required], textarea[aria-required="true"], ' +
    '[role=combobox][aria-required="true"], ' +
    '[role=checkbox][aria-required="true"]'
  );
  const seen = new Set();
  for (const el of inputs) {
    if (seen.has(el) || !VISIBLE(el)) continue;
    seen.add(el);
    const role = roleOf(el);
    let empty = false;
    if (role === 'checkbox') {
      empty = !el.checked && el.getAttribute('aria-checked') !== 'true';
    } else if (role === 'combobox') {
      const v = (el.value || '').trim();
      empty = !v || v === '0' || v === '-1';
    } else {
      empty = !(el.value && el.value.trim());
    }
    if (!empty) continue;
    const lab = labelFor(el);
    if (!lab) continue;
    out.push({ label: lab, role: role });
    if (out.length >= 10) break;
  }
  return out;
};
"""


def find_empty_required_fields(
    page: "Page",
) -> list[EmptyRequiredField]:
    """Return required fields visible on the page that are still empty.

    Best-effort: silently returns ``[]`` on any failure (page closed,
    JS exception). Used by the qa_agent's pre-submit safety net to
    catch "agent forgot to fill Display Name and clicked Save"
    BEFORE Save fires + the form rejects.
    """
    try:
        raw = page.evaluate(_EMPTY_REQUIRED_JS)
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[EmptyRequiredField] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "").strip()
        role = str(entry.get("role") or "")
        if not label or role not in (
            "textbox", "combobox", "checkbox", "textarea",
        ):
            continue
        out.append(EmptyRequiredField(label=label, role=role))
    return out


def is_submit_like(tool: str, target_hint: str | None = None) -> bool:
    """Heuristic: should we run the form-signal observer after this tool?

    Cheap pre-filter so we don't pay 400ms per turn on read-only
    actions (extract_text, scroll, navigate). True when the tool
    semantically commits a form: a click or press_key on a button
    whose label contains a submit-like word.
    """
    if tool not in _SUBMIT_LIKE_TOOLS:
        return False
    if not target_hint:
        # Press-key on Enter is often a form submit when in a focused
        # text input; default True to be safe.
        return tool == "press_key"
    h = target_hint.lower()
    return any(s in h for s in _SUBMIT_LIKE_HINTS)
