"""Phase F.2/F.3/F.4 — FormFillRoutine.

A meta-tool the agent calls with "fill this form with these values".
The routine orchestrates the brittle parts (enumerate the form's
fields, classify each, fill with the appropriate per-widget
strategy, observe inline errors, re-fill on validation feedback,
scroll, submit, watch for the post-submit signal) as ONE unit
instead of leaving the agent to compose them turn-by-turn.

Why this exists
---------------
The agent's general loop can fill a textbox + click Save just fine
when the form is simple. It struggles on:

- Custom dropdowns (not a native ``<select>``) — needs click +
  filter + click pattern, not Playwright's ``select_option``
- React-controlled inputs — DOM resolution finds a wrapper; type
  goes to the wrong node; value doesn't stick
- MUI-style checkboxes — clicking the visible icon misses; the real
  hidden input must be clicked via the label
- Submit buttons below the drawer scroll-fold
- Forms that report inline ``aria-invalid="true"`` after submit —
  the agent doesn't know which field to fix without a routine
  that maps the error back to a fill action

FormFillRoutine reads each field's role first (textbox / textarea /
native_select / custom_combobox / checkbox / radio / date / file),
runs a strategy that verifies the value stuck, retries on miss
with coord-typing fallback, then submits with a per-field error
retry loop.

Caller contract
---------------
The agent calls ``run_form_fill(page, fields, ...)`` with a list of
``FormField`` records. Each field has:
- ``label``: the field's visible label (used for fuzzy matching)
- ``value``: what to type / select / check (None for "leave default")
- ``role_hint``: optional hint to skip auto-classification
- ``required``: True if this field MUST succeed (False = best-effort)

Returns a ``FormFillResult`` with per-field outcomes plus the
submit outcome.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

logger = logging.getLogger(__name__)


# ── Public types ──────────────────────────────────────────────────


FieldRole = Literal[
    "textbox", "textarea", "native_select", "custom_combobox",
    "checkbox", "radio", "date", "file",
    # Phase G.3/G.4 — compound widgets a single FormField targets as
    # a whole. The ``value`` string carries a small DSL the handler
    # parses (see strategy docstrings).
    "permission_tree", "paginated_resource_table",
    "unknown",
]

FillStatus = Literal[
    "filled", "verified", "miss", "skipped", "error",
]


@dataclass
class FormField:
    """One value to set on the form.

    ``label`` matches against the field's accessible name (placeholder /
    aria-label / wrapping <label> text / data-testid). Fuzzy match —
    we accept partial substrings (case-insensitive) so the agent can
    pass "first name" and we'll find the "First Name *" field.

    ``value``:
      - textbox / textarea / date / native_select / custom_combobox:
        the string value to set
      - checkbox: "true"/"false"/"on"/"off"/"1"/"0" - the desired
        state
      - radio: the value of the radio option to pick
      - file: the absolute path to the file (or comma-separated paths)
    """
    label: str
    value: str
    role_hint: FieldRole | None = None
    required: bool = False


@dataclass
class FieldOutcome:
    label: str
    role: FieldRole
    status: FillStatus
    final_value: str = ""
    error: str = ""
    attempts: int = 1


@dataclass
class FormFillResult:
    fields: list[FieldOutcome] = field(default_factory=list)
    submit_status: Literal["ok", "validation_error", "no_submit", "error"] = "no_submit"
    submit_message: str = ""
    validation_fields: list[str] = field(default_factory=list)
    total_seconds: float = 0.0
    # Convenience rollups for the live feed.
    @property
    def filled_count(self) -> int:
        return sum(
            1 for f in self.fields if f.status in ("filled", "verified")
        )
    @property
    def miss_count(self) -> int:
        return sum(1 for f in self.fields if f.status == "miss")


# ── JS helpers ───────────────────────────────────────────────────


# Enumerate visible interactive fields in the current viewport (or
# the largest visible drawer / dialog). Returns role + label + a
# stable selector that the runtime can pass to ``page.locator()``.
_ENUMERATE_FIELDS_JS = r"""
() => {
  // Drawer / dialog detection — extended to catch custom-styled
  // drawers that don't carry MuiDrawer / Dialog class names. We
  // pick the largest visible "high-z-index right-side panel" as
  // the drawer when no explicit dialog matches; otherwise fall
  // through to body so simple inline forms still work.
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden' &&
           parseFloat(cs.opacity) > 0.05;
  };
  const findDrawer = () => {
    // Pass 1 — explicit ARIA / framework selectors.
    const explicit = [
      '[role=dialog]', '[role=alertdialog]',
      '[aria-modal="true"]',
      '.MuiDialog-paper', '.MuiDrawer-paper', '.MuiModal-root',
      '[class*="Drawer"]', '[class*="drawer"]',
      '[class*="Modal"]', '[class*="modal"]',
      '[data-state="open"][role="dialog"]',
    ].join(',');
    let cands = [...document.querySelectorAll(explicit)]
      .filter(VISIBLE)
      .filter(el => {
        const r = el.getBoundingClientRect();
        return r.width >= 240 && r.height >= 240;
      });
    if (cands.length === 0) {
      // Pass 2 — heuristic: position:fixed, high z-index, covers a
      // large viewport area, has form inputs inside. Catches Solar's
      // custom-styled drawer.
      const vw = window.innerWidth || 1280;
      const vh = window.innerHeight || 720;
      cands = [...document.querySelectorAll('div, section, aside')]
        .filter(el => {
          const cs = getComputedStyle(el);
          const pos = cs.position;
          if (pos !== 'fixed' && pos !== 'absolute') return false;
          const z = parseInt(cs.zIndex || '0', 10);
          if (z < 100) return false;
          if (!VISIBLE(el)) return false;
          const r = el.getBoundingClientRect();
          // Must cover a sizable chunk of the viewport (≥ 30% width or
          // height of the viewport).
          if (r.width < vw * 0.3 && r.height < vh * 0.4) return false;
          // Must contain at least one form input — drawers with forms.
          return !!el.querySelector(
            'input, textarea, select, [role=combobox], [role=textbox]',
          );
        });
    }
    if (cands.length === 0) return document.body;
    cands.sort((a, b) => {
      const ar = a.getBoundingClientRect();
      const br = b.getBoundingClientRect();
      return (br.width * br.height) - (ar.width * ar.height);
    });
    return cands[0];
  };
  const root = findDrawer();

  // Label resolution — extended to handle:
  //   1. MUI floating labels (label as sibling of the input wrapper)
  //   2. Stacked layouts (label as a preceding <div>/<span> above
  //      the input)
  //   3. aria-describedby chains
  //   4. Closest text-only element within parent 3 levels up
  const labelOf = (el) => {
    // 1. aria-label
    const al = el.getAttribute('aria-label');
    if (al && al.trim()) return al.trim().slice(0, 200);
    // 2. aria-labelledby → resolve IDs to text
    const alb = el.getAttribute('aria-labelledby');
    if (alb && alb.trim()) {
      const t = alb.split(/\s+/)
        .map(id => document.getElementById(id))
        .filter(Boolean)
        .map(n => (n.innerText || n.textContent || '').trim())
        .filter(Boolean)
        .join(' ');
      if (t) return t.slice(0, 200);
    }
    // 3. <label for=ID> elsewhere in the document
    if (el.id) {
      const lbl = document.querySelector(
        'label[for="' + CSS.escape(el.id) + '"]',
      );
      if (lbl) {
        const t = (lbl.innerText || lbl.textContent || '').trim();
        if (t) return t.slice(0, 200);
      }
    }
    // 4. Wrapping <label>
    let p = el.parentElement;
    for (let i = 0; i < 5 && p; i++) {
      if (p.tagName === 'LABEL') {
        const t = (p.innerText || p.textContent || '').trim();
        if (t) return t.slice(0, 200);
      }
      p = p.parentElement;
    }
    // 5. MUI / stacked-layout: floating label is a sibling text node
    // above the input. Walk up 3 levels and look at preceding
    // siblings for short text.
    p = el.parentElement;
    for (let i = 0; i < 3 && p; i++) {
      let sib = p.previousElementSibling;
      let hops = 0;
      while (sib && hops < 2) {
        const t = (sib.innerText || sib.textContent || '').trim();
        // Short, label-like text only — skip paragraphs.
        if (t && t.length <= 60 && !t.includes('\n')) {
          return t.slice(0, 200);
        }
        sib = sib.previousElementSibling;
        hops += 1;
      }
      // Also check the FIRST text-only descendant of `p` that's NOT
      // the input itself — handles "<div><span>Name</span><input
      // .../></div>" layouts.
      const labelChild = [...p.children].find(c =>
        c !== el && !c.contains(el)
        && c.tagName !== 'INPUT' && c.tagName !== 'TEXTAREA'
        && c.tagName !== 'SELECT'
        && !c.querySelector('input, textarea, select')
        && (c.innerText || c.textContent || '').trim().length > 0
        && (c.innerText || c.textContent || '').trim().length <= 60
      );
      if (labelChild) {
        const t = (labelChild.innerText || labelChild.textContent || '').trim();
        if (t) return t.slice(0, 200);
      }
      p = p.parentElement;
    }
    // 6. placeholder (only when nothing else matched — placeholders
    // are unreliable and often disappear when the field is focused).
    const ph = el.getAttribute('placeholder');
    if (ph && ph.trim()) return ph.trim().slice(0, 200);
    // 7. name attr — last resort.
    const nm = el.getAttribute('name');
    if (nm && nm.trim()) return nm.trim().slice(0, 200);
    return '';
  };
  const roleOf = (el) => {
    const t = (el.tagName || '').toLowerCase();
    const ariaRole = (el.getAttribute('role') || '').toLowerCase();
    if (t === 'select') return 'native_select';
    if (t === 'textarea') return 'textarea';
    if (t === 'input') {
      const it = (el.getAttribute('type') || 'text').toLowerCase();
      if (it === 'checkbox') return 'checkbox';
      if (it === 'radio') return 'radio';
      if (it === 'file') return 'file';
      if (it === 'date' || it === 'datetime-local' || it === 'time') return 'date';
      return 'textbox';
    }
    if (ariaRole === 'combobox' || ariaRole === 'listbox') {
      return 'custom_combobox';
    }
    if (ariaRole === 'checkbox') return 'checkbox';
    if (ariaRole === 'radio') return 'radio';
    if (ariaRole === 'textbox') return 'textbox';
    return 'unknown';
  };
  const isRequired = (el) => {
    if (el.hasAttribute('required')) return true;
    const a = el.getAttribute('aria-required');
    return a === 'true' || a === '1';
  };
  const currentValue = (el, role) => {
    if (role === 'checkbox' || role === 'radio') {
      if (el.checked !== undefined) return el.checked ? '1' : '0';
      const ac = el.getAttribute('aria-checked');
      return (ac === 'true') ? '1' : '0';
    }
    return el.value || el.getAttribute('value') || '';
  };

  const inputs = root.querySelectorAll(
    'input, textarea, select, ' +
    '[role=combobox], [role=listbox], [role=checkbox], [role=radio], [role=textbox]'
  );
  const out = [];
  const seen = new Set();
  let idx = 0;
  for (const el of inputs) {
    if (seen.has(el) || !VISIBLE(el)) continue;
    seen.add(el);
    const role = roleOf(el);
    const lab = labelOf(el);
    const r = el.getBoundingClientRect();
    // Stash an addressable id on the element if it doesn't already
    // have one. Callers locate via the data-attr we mint.
    let key = el.getAttribute('data-qai-form-key');
    if (!key) {
      key = 'qai-ff-' + (++idx) + '-' + Math.random().toString(36).slice(2, 8);
      el.setAttribute('data-qai-form-key', key);
    }
    out.push({
      key,
      role,
      label: lab,
      required: isRequired(el),
      current_value: currentValue(el, role),
      rect: [Math.round(r.left), Math.round(r.top),
             Math.round(r.width), Math.round(r.height)],
      type_attr: el.getAttribute('type') || '',
      name_attr: el.getAttribute('name') || '',
    });
    if (out.length >= 60) break;
  }
  return { fields: out };
};
"""


# Detect inline aria-invalid fields + their error messages. Used
# AFTER submit attempt to know WHICH field to retry.
_INVALID_FIELDS_JS = r"""
() => {
  // Phase V.2 — multi-strategy validation-error detection.
  // Detects:
  //   (a) aria-invalid="true" + aria-describedby chain (the
  //       textbook accessible form pattern)
  //   (b) Visible error text near each input via CSS class
  //       heuristics (Mui-error, .error, .invalid, [class*="helper"
  //       text-error"]) — covers Solar's UI which renders errors
  //       as <p class="MuiFormHelperText-root Mui-error"> beneath
  //       the input WITHOUT setting aria-invalid on the input
  //   (c) Inline red text under a field (siblings within 80px below
  //       the input with text-color #f44 / #d32 / class names
  //       containing "error" / "danger" / "required")
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  const isHiddenStyle = (el) => {
    const cs = getComputedStyle(el);
    return cs.display === 'none' || cs.visibility === 'hidden' ||
      parseFloat(cs.opacity) < 0.05;
  };
  const looksLikeError = (text) => {
    if (!text) return false;
    const t = text.toLowerCase();
    return (
      t.includes('required') ||
      t.includes('invalid') ||
      t.includes('must') ||
      t.includes('cannot') ||
      t.includes('only') && t.length < 120 ||
      t.includes('please') && t.length < 120 ||
      t.includes('error') ||
      t.includes('exists') ||
      t.includes('already') ||
      t.includes('duplicate') ||
      t.includes('format')
    );
  };
  const labelOf = (el) => {
    let lab = el.getAttribute('aria-label')
      || el.getAttribute('placeholder')
      || el.getAttribute('name') || '';
    if (lab) return String(lab).trim().slice(0, 200);
    if (el.id) {
      const lbl = document.querySelector(
        'label[for="' + CSS.escape(el.id) + '"]',
      );
      if (lbl) return (lbl.innerText || lbl.textContent || '').trim().slice(0, 200);
    }
    let p = el.parentElement;
    for (let i = 0; i < 5 && p; i++) {
      if (p.tagName === 'LABEL') {
        return (p.innerText || p.textContent || '').trim().slice(0, 200);
      }
      // Stacked layout — label is a sibling of the input wrapper.
      const sib = p.previousElementSibling;
      if (sib && sib.tagName !== 'INPUT' && sib.tagName !== 'TEXTAREA') {
        const t = (sib.innerText || sib.textContent || '').trim();
        if (t && t.length <= 60) return t.slice(0, 200);
      }
      p = p.parentElement;
    }
    return '';
  };

  const out = [];
  const seen = new Set();

  // Strategy (a) — aria-invalid + aria-describedby
  const ariaNodes = document.querySelectorAll(
    '[aria-invalid="true"], [aria-invalid="grammar"], [aria-invalid="spelling"]',
  );
  for (const el of ariaNodes) {
    if (!VISIBLE(el) || isHiddenStyle(el)) continue;
    const label = labelOf(el);
    let errText = '';
    const desc = el.getAttribute('aria-describedby');
    if (desc) {
      for (const id of desc.split(/\s+/)) {
        const en = document.getElementById(id);
        if (en && VISIBLE(en) && !isHiddenStyle(en)) {
          const t = (en.innerText || en.textContent || '').trim();
          if (t) { errText = t; break; }
        }
      }
    }
    const key = el.getAttribute('data-qai-form-key') || '';
    const dedup_key = key || label || '';
    if (seen.has(dedup_key)) continue;
    seen.add(dedup_key);
    out.push({ label, error: errText, key });
  }

  // Strategy (b) — Mui-error / .error / similar class names on a
  // FormHelperText-style sibling. Walk every visible input and look
  // up its surrounding form control for an error-styled helper text.
  const inputs = document.querySelectorAll(
    'input, textarea, [role=combobox], [role=textbox]',
  );
  for (const inp of inputs) {
    if (!VISIBLE(inp) || isHiddenStyle(inp)) continue;
    // Find the nearest form-control / wrapper.
    let wrapper = inp.closest(
      '.MuiFormControl-root, .ant-form-item, .form-group, ' +
      '[class*="form-control"], [class*="field-wrapper"]',
    );
    if (!wrapper) wrapper = inp.parentElement;
    if (!wrapper) continue;
    // Look for an error-flagged helper text inside the wrapper.
    const helper = wrapper.querySelector(
      '.Mui-error, .ant-form-item-explain-error, ' +
      '[class*="helper-text"][class*="error" i], ' +
      '[class*="error-message" i], [class*="error-text" i], ' +
      '[role=alert]',
    );
    let errText = '';
    if (helper && VISIBLE(helper) && !isHiddenStyle(helper)) {
      errText = (helper.innerText || helper.textContent || '').trim();
    }
    // If no explicit error class, look for sibling text within 80px
    // below that LOOKS like an error message.
    if (!errText) {
      const r = inp.getBoundingClientRect();
      const sibs = [...wrapper.querySelectorAll('p, span, div')]
        .filter(s => s !== inp && VISIBLE(s) && !isHiddenStyle(s));
      for (const s of sibs) {
        const sr = s.getBoundingClientRect();
        if (sr.top < r.bottom - 4 || sr.top > r.bottom + 80) continue;
        const t = (s.innerText || s.textContent || '').trim();
        if (t && t.length < 200 && looksLikeError(t)) {
          errText = t;
          break;
        }
      }
    }
    if (!errText) continue;
    const label = labelOf(inp);
    const key = inp.getAttribute('data-qai-form-key') || '';
    const dedup_key = key || label || '';
    if (seen.has(dedup_key)) continue;
    seen.add(dedup_key);
    out.push({ label, error: errText.slice(0, 240), key });
  }

  return out;
};
"""


# Scroll the active drawer / dialog to make a key visible. Returns
# whether the element was found.
_SCROLL_INTO_VIEW_JS = r"""
(key) => {
  const el = document.querySelector('[data-qai-form-key="' + key + '"]');
  if (!el) return false;
  el.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'instant' });
  return true;
};
"""


# Click a custom-combobox option by visible text after the listbox
# popup opens. Walks the document for the FIRST visible role=option
# whose text matches (case-insensitive substring).
_PICK_COMBOBOX_OPTION_JS = r"""
(needle) => {
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  };
  const want = (needle || '').trim().toLowerCase();
  if (!want) return false;
  // role=option, then MUI-MenuItem, then AntD select item.
  const sels = [
    '[role=option]',
    '.MuiMenuItem-root', '.MuiAutocomplete-option',
    '.ant-select-item-option',
    '[class*="option"]',
  ];
  for (const sel of sels) {
    const nodes = document.querySelectorAll(sel);
    for (const el of nodes) {
      if (!VISIBLE(el)) continue;
      const text = ((el.innerText || el.textContent) || '').trim().toLowerCase();
      if (!text) continue;
      if (text === want || text.startsWith(want) || text.includes(want)) {
        el.click();
        return true;
      }
    }
  }
  return false;
};
"""


# Phase G.3 — click Expand All inside the active drawer if present.
# Returns true when something was clicked. Idempotent — safe to call
# repeatedly (subsequent calls find nothing to click).
_EXPAND_ALL_JS = r"""
() => {
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  };
  const drawerSel = [
    '[role=dialog]', '[role=alertdialog]',
    '.MuiDialog-paper', '.MuiDrawer-paper',
    '[class*="Drawer"]', '[class*="drawer"]',
    '[class*="Modal"]', '[class*="modal"]',
  ].join(',');
  const drawers = [...document.querySelectorAll(drawerSel)].filter(VISIBLE);
  drawers.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return (br.width * br.height) - (ar.width * ar.height);
  });
  const root = drawers[0] || document.body;
  const btns = [...root.querySelectorAll('button, [role=button], a')]
    .filter(VISIBLE);
  for (const b of btns) {
    const t = ((b.innerText || b.textContent) || '').trim().toLowerCase();
    if (t === 'expand all' || t === 'expand-all' || t === 'expand' ||
        t.startsWith('expand all')) {
      b.click();
      return true;
    }
  }
  return false;
};
"""


# Phase G.3 — enumerate permission-tree parent rows + (after expand)
# leaf checkboxes. Returns a flat list of all checkboxes inside the
# tree along with their text label + a parent-text bread-crumb so the
# strategy can target "only:Module.Action" precisely.
_ENUM_TREE_CHECKBOXES_JS = r"""
() => {
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  };
  const drawerSel = [
    '[role=dialog]', '[role=alertdialog]',
    '.MuiDialog-paper', '.MuiDrawer-paper',
    '[class*="Drawer"]', '[class*="drawer"]',
    '[class*="Modal"]', '[class*="modal"]',
  ].join(',');
  const drawers = [...document.querySelectorAll(drawerSel)].filter(VISIBLE);
  drawers.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return (br.width * br.height) - (ar.width * ar.height);
  });
  const root = drawers[0] || document.body;

  // Find tree container — same heuristic as the Scout (role=tree
  // OR aria-expanded stack).
  let tree = root.querySelector('[role=tree], .MuiTreeView-root, .ant-tree');
  if (!tree) {
    // Fall back to the first ancestor of the first aria-expanded row
    // whose children look tree-like.
    const exp = [...root.querySelectorAll('[aria-expanded]')]
      .find(el => VISIBLE(el) && el.querySelector(
        'input[type=checkbox], [role=checkbox]'
      ));
    if (exp) tree = exp.parentElement;
  }
  if (!tree) return { items: [] };

  const items = [];
  const cbs = [...tree.querySelectorAll(
    'input[type=checkbox], [role=checkbox]'
  )].filter(VISIBLE);
  for (let i = 0; i < cbs.length; i++) {
    const cb = cbs[i];
    // Best label = closest LABEL ancestor / sibling, else nearest text.
    let lab = '';
    if (cb.id) {
      const l = document.querySelector(
        'label[for="' + CSS.escape(cb.id) + '"]'
      );
      if (l) lab = (l.innerText || l.textContent || '').trim();
    }
    if (!lab) {
      let p = cb.parentElement;
      for (let j = 0; j < 4 && p && !lab; j++) {
        if (p.tagName === 'LABEL') {
          lab = (p.innerText || p.textContent || '').trim();
        }
        p = p.parentElement;
      }
    }
    if (!lab && cb.parentElement) {
      const sib = cb.parentElement;
      lab = (sib.innerText || sib.textContent || '').trim();
    }
    lab = (lab || '').slice(0, 120);
    // Tag for click targeting.
    let key = cb.getAttribute('data-qai-tree-key');
    if (!key) {
      key = 'qai-tree-' + (i + 1);
      cb.setAttribute('data-qai-tree-key', key);
    }
    // Find ancestor parent label — first ancestor treeitem / aria-expanded.
    let parentLabel = '';
    let anc = cb.parentElement;
    while (anc && anc !== tree) {
      if (anc.getAttribute('role') === 'treeitem' ||
          anc.hasAttribute('aria-expanded')) {
        // Get this ancestor's OWN label (not including descendant labels).
        const own = [...anc.childNodes]
          .filter(n => n.nodeType === Node.TEXT_NODE)
          .map(n => n.textContent.trim())
          .join(' ').trim();
        if (own) { parentLabel = own.slice(0, 120); break; }
      }
      anc = anc.parentElement;
    }
    const r = cb.getBoundingClientRect();
    items.push({
      key,
      label: lab,
      parent_label: parentLabel,
      checked: !!cb.checked || cb.getAttribute('aria-checked') === 'true',
      rect: [Math.round(r.left), Math.round(r.top),
             Math.round(r.width), Math.round(r.height)],
    });
  }
  return { items };
};
"""


# Phase G.4 — enumerate the paginated resource table inside the
# active drawer. Returns header columns + per-row metadata including
# the row label + each checkbox keyed by column header.
_ENUM_RESOURCE_TABLE_JS = r"""
() => {
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  };
  const textOf = (el) =>
    el ? ((el.innerText || el.textContent) || '').trim().slice(0, 80) : '';

  const drawerSel = [
    '[role=dialog]', '[role=alertdialog]',
    '.MuiDialog-paper', '.MuiDrawer-paper',
    '[class*="Drawer"]', '[class*="drawer"]',
    '[class*="Modal"]', '[class*="modal"]',
  ].join(',');
  const drawers = [...document.querySelectorAll(drawerSel)].filter(VISIBLE);
  drawers.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return (br.width * br.height) - (ar.width * ar.height);
  });
  const root = drawers[0] || document.body;
  const tables = [...root.querySelectorAll(
    'table, [role=table], [role=grid], .ant-table-content table, .MuiTable-root'
  )].filter(VISIBLE);
  if (tables.length === 0) return null;
  // Pick the largest table.
  tables.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return (br.width * br.height) - (ar.width * ar.height);
  });
  const t = tables[0];
  const headerCells = [...t.querySelectorAll('thead th, [role=columnheader]')];
  const headers = headerCells.map(textOf);
  // Column master checkboxes: same indices as columns; null when absent.
  const columnMasters = [];
  for (let i = 0; i < headerCells.length; i++) {
    const hc = headerCells[i];
    const cb = hc.querySelector('input[type=checkbox], [role=checkbox]');
    if (cb) {
      let key = cb.getAttribute('data-qai-tab-col');
      if (!key) {
        key = 'qai-tab-col-' + (i + 1);
        cb.setAttribute('data-qai-tab-col', key);
      }
      columnMasters.push({ index: i, key, label: headers[i] || '' });
    }
  }
  // Rows.
  const rows = [...t.querySelectorAll('tbody tr, [role=row]')]
    .filter(VISIBLE);
  const rowDump = [];
  for (let ri = 0; ri < rows.length; ri++) {
    const r = rows[ri];
    const cells = [...r.querySelectorAll('td, [role=cell], th')];
    if (cells.length === 0) continue;
    const rowLabel = textOf(cells[0]);
    const cbs = [];
    for (let ci = 0; ci < cells.length; ci++) {
      const cb = cells[ci].querySelector(
        'input[type=checkbox], [role=checkbox]'
      );
      if (!cb) continue;
      let key = cb.getAttribute('data-qai-tab-cell');
      if (!key) {
        key = 'qai-tab-cell-' + (ri + 1) + '-' + (ci + 1);
        cb.setAttribute('data-qai-tab-cell', key);
      }
      cbs.push({
        col_index: ci,
        col_label: headers[ci] || '',
        key,
        checked: !!cb.checked ||
          cb.getAttribute('aria-checked') === 'true',
      });
    }
    rowDump.push({
      row_index: ri,
      row_label: rowLabel,
      cells: cbs,
    });
  }
  // Pagination — find "Next" button + total page count if visible.
  let nextSelector = '';
  let isNextEnabled = false;
  const par = t.closest('div, section, form') || t.parentElement || root;
  if (par) {
    const cands = [...par.querySelectorAll(
      'button, [role=button], a, .MuiPaginationItem-root, .ant-pagination-next'
    )].filter(VISIBLE);
    for (const c of cands) {
      const tx = textOf(c).toLowerCase();
      if (tx === 'next' || tx === '›' || tx === '>' ||
          c.classList.contains('ant-pagination-next') ||
          c.getAttribute('aria-label') === 'Go to next page') {
        let key = c.getAttribute('data-qai-tab-next');
        if (!key) {
          key = 'qai-tab-next';
          c.setAttribute('data-qai-tab-next', key);
        }
        nextSelector = '[data-qai-tab-next="' + key + '"]';
        isNextEnabled = !(c.disabled ||
          c.getAttribute('aria-disabled') === 'true' ||
          c.classList.contains('ant-pagination-disabled'));
        break;
      }
    }
  }
  return {
    headers,
    column_masters: columnMasters,
    rows: rowDump,
    next_selector: nextSelector,
    next_enabled: isNextEnabled,
  };
};
"""


# ── Public API ────────────────────────────────────────────────────


def run_form_fill(
    page: "Page",
    *,
    fields: list[FormField],
    submit_label: str = "Save",
    settle_ms: int = 600,
    max_validation_retries: int = 2,
    emit_event: Callable[[str, dict], None] | None = None,
    vision_provider: Any = None,
) -> FormFillResult:
    """Orchestrated form fill — see module docstring.

    ``submit_label`` is fuzzy-matched against visible buttons in the
    form region; pass ``""`` to skip the submit phase (caller dispatches
    submit separately).

    ``vision_provider`` (Phase U) — when supplied AND the DOM scanner
    fails to match any of the requested fields (Solar's custom drawer
    has no MUI / dialog class names, so the scanner returns 0 fields),
    the routine falls back to ONE vision call that returns pixel
    coordinates for each requested field. Fields are then filled by
    clicking at the returned (x, y) and typing. Same fallback applies
    to the Save button finder. Optional; when not supplied, the
    routine fails with the same outcomes as before.

    The routine never raises — failures are recorded in the per-field
    outcomes + the submit_status field of the result.
    """
    t0 = time.monotonic()
    result = FormFillResult()

    def _emit(t: str, d: dict) -> None:
        if emit_event:
            try:
                emit_event(t, d)
            except Exception:
                pass

    _emit("form_fill_started", {
        "fields_requested": [
            {"label": f.label, "role_hint": f.role_hint or "",
             "required": f.required}
            for f in fields
        ],
        "submit_label": submit_label,
    })

    # 1) Enumerate visible form fields + tag with data-qai-form-key.
    # Diag.1 — scroll the drawer to the top first, scan, then scroll
    # to the bottom and scan again. Merge unique fields. Solar's
    # drawers push the Resource Access Control table below the fold;
    # without this, fill_form would only see the top half.
    _drawer_scroll_top_js = (
        "(() => {const sel=['[role=dialog]','[role=alertdialog]',"
        "'.MuiDialog-paper','.MuiDrawer-paper','[class*=\"Drawer\"]',"
        "'[class*=\"drawer\"]','[class*=\"Modal\"]','[class*=\"modal\"]']"
        ".join(',');const ds=[...document.querySelectorAll(sel)]"
        ".filter(el=>{const r=el.getBoundingClientRect();"
        "return r.width>=100&&r.height>=100;});if(ds.length===0)return false;"
        "ds.sort((a,b)=>{const ar=a.getBoundingClientRect();"
        "const br=b.getBoundingClientRect();return (br.width*br.height)-(ar.width*ar.height);});"
        "let s=ds[0];if(s.scrollHeight<=s.clientHeight+4){"
        "const inner=[...s.querySelectorAll('*')].find(el=>{"
        "const cs=getComputedStyle(el);"
        "return (cs.overflowY==='auto'||cs.overflowY==='scroll')&&el.scrollHeight>el.clientHeight+4;});"
        "if(inner)s=inner;}s.scrollTop=0;return true;})()"
    )
    try:
        page.evaluate(_drawer_scroll_top_js)
        page.wait_for_timeout(200)
    except Exception:
        pass
    try:
        scan_top = page.evaluate(_ENUMERATE_FIELDS_JS)
    except Exception as e:
        result.submit_status = "error"
        result.submit_message = f"scan failed: {e}"
        return result
    detected_top = (
        (scan_top.get("fields") or []) if isinstance(scan_top, dict) else []
    )
    # Now scroll to the bottom and re-scan for below-the-fold fields.
    try:
        from app.executor.actions import scroll_drawer_to_bottom  # noqa: PLC0415
        scroll_drawer_to_bottom(page)
        page.wait_for_timeout(200)
        scan_bottom = page.evaluate(_ENUMERATE_FIELDS_JS)
    except Exception:
        scan_bottom = None
    detected_bottom = (
        (scan_bottom.get("fields") or [])
        if isinstance(scan_bottom, dict) else []
    )
    # Merge — dedupe by data-qai-form-key (set by the JS scanner).
    seen_keys: set[str] = set()
    detected: list[dict[str, Any]] = []
    for d in list(detected_top) + list(detected_bottom):
        k = d.get("key") or ""
        if not k or k in seen_keys:
            continue
        seen_keys.add(k)
        detected.append(d)
    # Scroll back to top so fill_one's per-field scroll-into-view
    # starts from a clean state.
    try:
        page.evaluate(_drawer_scroll_top_js)
        page.wait_for_timeout(120)
    except Exception:
        pass
    _emit("form_fill_scanned", {
        "detected_count": len(detected),
        "detected": [
            {"key": d.get("key"), "role": d.get("role"),
             "label": d.get("label"), "required": d.get("required")}
            for d in detected[:30]
        ],
    })

    # 2) For each requested field, find a matching detected field
    # by fuzzy label match.
    matched: list[tuple[FormField, dict[str, Any]]] = []
    unmatched: list[FormField] = []
    used_keys: set[str] = set()
    for ff in fields:
        m = _match_field(ff, detected, used_keys)
        if m is None:
            unmatched.append(ff)
        else:
            used_keys.add(m["key"])
            matched.append((ff, m))

    # Phase U — vision-coord fallback. When the DOM scanner failed to
    # match ANY of the requested fields (Solar's custom drawer, an
    # iframe we can't reach, shadow DOM) — fire ONE vision call that
    # returns pixel coords for every requested label. Fill those via
    # click+type. Only fires when:
    #   - vision_provider is configured AND vision-capable
    #   - matched is empty OR < 50% of requested fields matched
    #   - the requested field is a simple text input
    # Compound widgets (permission_tree, paginated_resource_table)
    # are skipped — they need DOM key-tagging.
    vision_coord_fields: dict[str, tuple[int, int]] = {}
    vision_submit_coord: tuple[int, int] | None = None
    miss_ratio = (
        len(unmatched) / len(fields) if fields else 0.0
    )
    should_try_vl_fallback = (
        vision_provider is not None
        and unmatched
        and miss_ratio >= 0.5
    )
    if should_try_vl_fallback:
        try:
            # Phase X.1 — exclude dropdowns from the VL-coord path.
            # The path types directly at the returned (x,y) via
            # keyboard.type, which works for textbox / textarea / date
            # / file but NOT for custom_combobox or native_select
            # (those need click → wait for popup → click option /
            # press Enter — see ``_fill_custom_combobox``). Send them
            # back as ``miss`` so the agent's next turn dispatches a
            # proper combobox interaction via the DOM path.
            vl_labels = [
                ff.label for ff in unmatched
                if ff.role_hint not in (
                    "permission_tree", "paginated_resource_table",
                    "custom_combobox", "native_select",
                )
            ]
            vision_coord_fields, vision_submit_coord = (
                _locate_form_fields_via_vision(
                    page,
                    labels=vl_labels,
                    submit_label=submit_label or "Save",
                    vision_provider=vision_provider,
                )
            )
            _emit("form_fill_vl_fallback", {
                "labels_requested": vl_labels[:10],
                "labels_located": list(vision_coord_fields.keys())[:10],
                "submit_located": vision_submit_coord is not None,
            })
        except Exception as e:
            logger.debug("VL fallback raised: %s", e)
            vision_coord_fields = {}

    # 3) Fill each matched field with the right strategy. We handle
    # required-field MISSES with an immediate retry via coord-typing
    # fallback.
    for ff, info in matched:
        outcome = _fill_one(
            page, ff, info,
            settle_ms=settle_ms,
            emit_event=_emit,
        )
        result.fields.append(outcome)
        _emit("form_fill_field", {
            "label": ff.label,
            "role": outcome.role,
            "status": outcome.status,
            "final_value": outcome.final_value[:120],
            "error": outcome.error[:200] if outcome.error else "",
            "attempts": outcome.attempts,
        })

    # Phase U — coord-fill unmatched fields the VL locator found.
    still_unmatched: list[FormField] = []
    for ff in unmatched:
        coord = vision_coord_fields.get(ff.label)
        if coord is None:
            still_unmatched.append(ff)
            continue
        try:
            page.mouse.click(coord[0], coord[1])
            try:
                page.wait_for_timeout(100)
            except Exception:
                pass
            # Clear any pre-existing value before typing.
            page.keyboard.press("Control+A")
            page.keyboard.press("Delete")
            page.keyboard.type(ff.value, delay=20)
            # Phase V.1 — REACT-COMPATIBLE INPUT EVENT DISPATCH.
            # ``page.keyboard.type`` fires native key events, but
            # React's controlled inputs track value via the synthetic
            # event system and the native HTMLInputElement value
            # setter. If we don't trigger the React setter manually,
            # the form's internal state stays empty even though the
            # input visually shows the typed text. Save stays
            # disabled / validation rejects on submit.
            #
            # The fix: locate the focused element at (x, y), call the
            # NATIVE value setter, then dispatch input + change events
            # with bubbles=true. This nudges React (and MUI/AntD/
            # Formik/RHF on top of it) to update controlled state.
            try:
                page.evaluate(
                    """({x, y, value}) => {
                        const el = document.elementFromPoint(x, y);
                        if (!el) return false;
                        // Locate the actual input — elementFromPoint
                        // may land on a wrapper.
                        const input =
                          el.tagName === 'INPUT' || el.tagName === 'TEXTAREA'
                            ? el
                            : el.querySelector('input, textarea')
                              || el.closest('label, .MuiFormControl-root')?.querySelector('input, textarea');
                        if (!input) return false;
                        const proto = input.tagName === 'TEXTAREA'
                          ? HTMLTextAreaElement.prototype
                          : HTMLInputElement.prototype;
                        const setter = Object.getOwnPropertyDescriptor(
                          proto, 'value',
                        )?.set;
                        if (setter) setter.call(input, value);
                        else input.value = value;
                        input.dispatchEvent(new Event('input', {bubbles: true}));
                        input.dispatchEvent(new Event('change', {bubbles: true}));
                        // Blur to commit the value — many forms only
                        // run validation on blur.
                        input.dispatchEvent(new Event('blur', {bubbles: true}));
                        return true;
                    }""",
                    {"x": coord[0], "y": coord[1], "value": ff.value},
                )
            except Exception as e:
                logger.debug(
                    "VL-coord React event dispatch failed: %s", e,
                )
            outcome = FieldOutcome(
                label=ff.label,
                role=ff.role_hint or "textbox",  # type: ignore[arg-type]
                status="filled",
                final_value=ff.value,
                attempts=1,
            )
            outcome.error = (
                f"VL-coord fallback @ ({coord[0]},{coord[1]})"
            )
        except Exception as e:
            outcome = FieldOutcome(
                label=ff.label,
                role=ff.role_hint or "textbox",  # type: ignore[arg-type]
                status="error",
                error=f"VL-coord click/type failed: {e}",
            )
        result.fields.append(outcome)
        _emit("form_fill_field", {
            "label": ff.label,
            "role": outcome.role,
            "status": outcome.status,
            "final_value": outcome.final_value[:120],
            "error": outcome.error[:200] if outcome.error else "",
            "attempts": outcome.attempts,
            "via_vl_coord": True,
        })

    # Record still-unmatched as misses (VL also couldn't find them).
    for ff in still_unmatched:
        result.fields.append(FieldOutcome(
            label=ff.label,
            role=ff.role_hint or "unknown",
            status="miss",
            error="no matching field on the visible form",
        ))
        _emit("form_fill_field", {
            "label": ff.label,
            "role": ff.role_hint or "unknown",
            "status": "miss",
            "error": "no matching field on the visible form",
            "attempts": 0,
        })

    # 4) Submit + per-field validation-error retry loop.
    if submit_label:
        for attempt in range(max_validation_retries + 1):
            ok, msg, invalids = _submit_and_observe(
                page,
                submit_label=submit_label,
                settle_ms=settle_ms,
                vision_submit_coord=vision_submit_coord,
                # Phase X.3b — pass the VL provider so a fresh
                # ``propose_click_coordinates`` call can fire if
                # JS scorer + cached coord both miss.
                vision_provider=vision_provider,
            )
            _emit("form_fill_submit_attempt", {
                "attempt": attempt + 1,
                "max": max_validation_retries + 1,
                "ok": ok,
                "message": msg[:200],
                "invalid_fields": [
                    iv.get("label", "") for iv in invalids[:10]
                ],
            })
            if ok:
                result.submit_status = "ok"
                result.submit_message = msg
                break
            if not invalids:
                result.submit_status = "error"
                result.submit_message = msg
                break
            result.submit_status = "validation_error"
            result.submit_message = msg
            result.validation_fields = [
                iv.get("label", "") for iv in invalids if iv.get("label")
            ]
            if attempt >= max_validation_retries:
                break
            # Retry: read each error message + regenerate a compliant
            # value from the constraint (Phase I.3). When the regen
            # produces a NEW value, re-fill with it; otherwise the
            # caller falls back to the same-value retry (which clears
            # the field first — useful for stale-state errors).
            for iv in invalids:
                err_label = (iv.get("label") or "").lower().strip()
                err_msg = iv.get("error") or ""
                target = next(
                    (m for (ff, m) in matched
                     if err_label
                     and err_label in (m.get("label") or "").lower()),
                    None,
                )
                target_ff = next(
                    (ff for (ff, m) in matched
                     if target is m), None,
                ) if target else None
                if target is None or target_ff is None:
                    continue

                # Phase I.3 — constraint-aware value regeneration.
                new_value, regen_reason = _regenerate_value_from_error(
                    field_label=target_ff.label,
                    original_value=target_ff.value,
                    error_message=err_msg,
                )
                if new_value != target_ff.value:
                    # Mutate the FormField in-place so subsequent
                    # retries (still inside the same retry-loop) keep
                    # using the corrected value.
                    target_ff.value = new_value

                retry_outcome = _fill_one(
                    page, target_ff, target,
                    settle_ms=settle_ms,
                    emit_event=_emit,
                )
                retry_outcome.attempts += 1
                retry_outcome.error = (
                    f"retry after validation: {err_msg[:140]} "
                    f"[regen: {regen_reason[:60]}]"
                )
                # Update the recorded outcome in place so the
                # final result reflects the latest state.
                for i, prev in enumerate(result.fields):
                    if prev.label == target_ff.label:
                        result.fields[i] = retry_outcome
                        break
                _emit("form_fill_field_retry", {
                    "label": target_ff.label,
                    "role": retry_outcome.role,
                    "status": retry_outcome.status,
                    "validation_error": err_msg[:200],
                    "regen_reason": regen_reason,
                    "regenerated_value_preview": new_value[:60],
                    "attempts": retry_outcome.attempts,
                })

    result.total_seconds = round(time.monotonic() - t0, 2)
    _emit("form_fill_completed", {
        "filled": result.filled_count,
        "miss": result.miss_count,
        "submit_status": result.submit_status,
        "validation_fields": result.validation_fields,
        "seconds": result.total_seconds,
    })
    return result


# ── Matching ────────────────────────────────────────────────────


def _match_field(
    requested: FormField,
    detected: list[dict[str, Any]],
    used_keys: set[str],
) -> dict[str, Any] | None:
    """Fuzzy-match a requested field to a detected one.

    Order of preference:
      1. Exact label match (case-insensitive)
      2. requested.label is substring of detected.label
      3. detected.label is substring of requested.label
      4. role_hint matches AND requested label words ⊆ detected label
    """
    needle = (requested.label or "").lower().strip()
    if not needle:
        return None
    # Pass 1: exact.
    for d in detected:
        if d["key"] in used_keys:
            continue
        if (d.get("label") or "").lower().strip() == needle:
            return d
    # Pass 2: needle in label.
    for d in detected:
        if d["key"] in used_keys:
            continue
        if needle in (d.get("label") or "").lower():
            return d
    # Pass 3: label in needle.
    for d in detected:
        if d["key"] in used_keys:
            continue
        det = (d.get("label") or "").lower().strip()
        if det and det in needle:
            return d
    # Pass 4: word overlap + role hint.
    if requested.role_hint:
        words = set(needle.split())
        best: tuple[int, dict[str, Any]] | None = None
        for d in detected:
            if d["key"] in used_keys:
                continue
            if d.get("role") != requested.role_hint:
                continue
            dw = set((d.get("label") or "").lower().split())
            overlap = len(words & dw)
            if overlap == 0:
                continue
            if best is None or overlap > best[0]:
                best = (overlap, d)
        if best is not None:
            return best[1]
    return None


# ── Per-widget strategies ────────────────────────────────────────


def _fill_one(
    page: "Page",
    ff: FormField,
    info: dict[str, Any],
    *,
    settle_ms: int,
    emit_event: Callable[[str, dict], None] | None = None,
) -> FieldOutcome:
    """Dispatch to the per-role fill strategy + verify."""
    role: FieldRole = (
        ff.role_hint or info.get("role") or "unknown"
    )  # type: ignore[assignment]
    key = info.get("key", "")
    locator: "Locator" = page.locator(f'[data-qai-form-key="{key}"]')

    # Scroll the field into view so coord-typing fallback has a hope.
    try:
        page.evaluate(_SCROLL_INTO_VIEW_JS, key)
        page.wait_for_timeout(120)
    except Exception:
        pass

    try:
        if role in ("textbox", "textarea"):
            return _fill_textbox(
                page, locator, ff, info, settle_ms=settle_ms,
            )
        if role == "native_select":
            return _fill_native_select(
                page, locator, ff, info,
            )
        if role == "custom_combobox":
            return _fill_custom_combobox(
                page, locator, ff, info, settle_ms=settle_ms,
            )
        if role == "checkbox":
            return _fill_checkbox(
                page, locator, ff, info,
            )
        if role == "radio":
            return _fill_radio(
                page, locator, ff, info,
            )
        if role == "date":
            return _fill_date(
                page, locator, ff, info, settle_ms=settle_ms,
            )
        if role == "file":
            return _fill_file(
                page, locator, ff, info,
            )
        if role == "permission_tree":
            return _fill_permission_tree(
                page, ff, info,
                settle_ms=settle_ms,
                emit_event=emit_event,
            )
        if role == "paginated_resource_table":
            return _fill_paginated_resource_table(
                page, ff, info, settle_ms=settle_ms,
            )
        return FieldOutcome(
            label=ff.label,
            role=role,
            status="skipped",
            error=f"unsupported role: {role}",
        )
    except Exception as e:
        return FieldOutcome(
            label=ff.label,
            role=role,
            status="error",
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )


def _fill_textbox(
    page: "Page",
    locator: "Locator",
    ff: FormField,
    info: dict[str, Any],
    *,
    settle_ms: int,
) -> FieldOutcome:
    role: FieldRole = info.get("role") or "textbox"  # type: ignore[assignment]
    value = ff.value
    try:
        locator.scroll_into_view_if_needed(timeout=2000)
    except Exception:
        pass
    # Strategy A — Playwright's high-level fill.
    try:
        locator.fill("", timeout=1500)
    except Exception:
        pass
    try:
        locator.fill(value, timeout=3000)
        cur = locator.input_value(timeout=1000)
        if (cur or "").strip() == value.strip():
            return FieldOutcome(
                label=ff.label, role=role, status="verified",
                final_value=cur,
            )
    except Exception:
        pass
    # Strategy B — focus + clear + type (key-by-key). Helps with
    # React-controlled inputs that drop fill() values.
    try:
        locator.click(timeout=2000)
        page.wait_for_timeout(80)
        page.keyboard.press("Control+A")
        page.keyboard.press("Delete")
        page.keyboard.type(value, delay=20)
        cur = locator.input_value(timeout=1000)
        if (cur or "").strip() == value.strip():
            return FieldOutcome(
                label=ff.label, role=role, status="verified",
                final_value=cur, attempts=2,
            )
    except Exception:
        pass
    # Strategy C — coord-based fallback (last resort).
    try:
        rect = info.get("rect") or [0, 0, 0, 0]
        cx = int(rect[0]) + int(rect[2]) // 2
        cy = int(rect[1]) + int(rect[3]) // 2
        page.mouse.click(cx, cy)
        page.wait_for_timeout(80)
        page.keyboard.press("Control+A")
        page.keyboard.press("Delete")
        page.keyboard.type(value, delay=20)
        try:
            cur = locator.input_value(timeout=800)
        except Exception:
            cur = value  # best-effort verify
        return FieldOutcome(
            label=ff.label, role=role,
            status="filled" if cur == value else "miss",
            final_value=cur, attempts=3,
        )
    except Exception as e:
        return FieldOutcome(
            label=ff.label, role=role, status="error",
            error=f"{type(e).__name__}: {str(e)[:200]}",
            attempts=3,
        )


def _fill_native_select(
    page: "Page",
    locator: "Locator",
    ff: FormField,
    info: dict[str, Any],
) -> FieldOutcome:
    try:
        # Try by value first, then label.
        try:
            locator.select_option(value=ff.value, timeout=2000)
        except Exception:
            locator.select_option(label=ff.value, timeout=2000)
        cur = locator.input_value(timeout=1000)
        return FieldOutcome(
            label=ff.label, role="native_select",
            status="verified", final_value=cur,
        )
    except Exception as e:
        return FieldOutcome(
            label=ff.label, role="native_select", status="error",
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )


def _fill_custom_combobox(
    page: "Page",
    locator: "Locator",
    ff: FormField,
    info: dict[str, Any],
    *,
    settle_ms: int,
) -> FieldOutcome:
    """Open the combobox, type to filter, click the matching option.

    Used for MUI Autocomplete, AntD Select, custom listboxes, etc.
    The strategy:
      1. Click the trigger.
      2. Type the value to filter the popup (most custom widgets
         filter as you type).
      3. JS-click the first visible role=option whose text matches.
      4. If no option found by JS, press Enter to commit (works for
         creatable selects).
    """
    try:
        locator.click(timeout=2500)
    except Exception:
        # Fallback: coord click.
        rect = info.get("rect") or [0, 0, 0, 0]
        try:
            page.mouse.click(
                int(rect[0]) + int(rect[2]) // 2,
                int(rect[1]) + int(rect[3]) // 2,
            )
        except Exception:
            return FieldOutcome(
                label=ff.label, role="custom_combobox", status="error",
                error="could not click trigger",
            )

    try:
        page.wait_for_timeout(settle_ms // 2)
        # Type the value to filter — works on most autocomplete-style
        # widgets. Harmless on widgets that don't filter.
        page.keyboard.type(ff.value, delay=20)
        page.wait_for_timeout(settle_ms // 2)
        # JS-pick the matching option.
        picked = page.evaluate(_PICK_COMBOBOX_OPTION_JS, ff.value)
        if picked:
            page.wait_for_timeout(150)
            return FieldOutcome(
                label=ff.label, role="custom_combobox",
                status="filled", final_value=ff.value,
            )
        # Last resort: press Enter — creatable selects + some
        # uncontrolled comboboxes accept this.
        page.keyboard.press("Enter")
        return FieldOutcome(
            label=ff.label, role="custom_combobox",
            status="filled", final_value=ff.value, attempts=2,
        )
    except Exception as e:
        return FieldOutcome(
            label=ff.label, role="custom_combobox", status="error",
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )


def _fill_checkbox(
    page: "Page",
    locator: "Locator",
    ff: FormField,
    info: dict[str, Any],
) -> FieldOutcome:
    """State-aware checkbox fill. Only clicks if the desired state
    differs from the current state — so calling twice with the same
    value is idempotent."""
    desired = _parse_bool(ff.value)
    try:
        current = info.get("current_value") in ("1", "true", "True")
        if current == desired:
            return FieldOutcome(
                label=ff.label, role="checkbox", status="verified",
                final_value="1" if current else "0",
            )
        # Playwright's check/uncheck handles MUI-style hidden inputs
        # via the label automatically.
        if desired:
            locator.check(timeout=2000)
        else:
            locator.uncheck(timeout=2000)
        # Verify by re-reading from the DOM.
        try:
            ok = bool(locator.is_checked(timeout=800))
        except Exception:
            ok = desired  # best-effort
        if ok == desired:
            return FieldOutcome(
                label=ff.label, role="checkbox", status="verified",
                final_value="1" if ok else "0",
            )
        # Try clicking the bbox (handles fully custom checkboxes).
        rect = info.get("rect") or [0, 0, 0, 0]
        page.mouse.click(
            int(rect[0]) + int(rect[2]) // 2,
            int(rect[1]) + int(rect[3]) // 2,
        )
        try:
            ok = bool(locator.is_checked(timeout=800))
        except Exception:
            ok = desired
        return FieldOutcome(
            label=ff.label, role="checkbox",
            status="filled" if ok == desired else "miss",
            final_value="1" if ok else "0", attempts=2,
        )
    except Exception as e:
        return FieldOutcome(
            label=ff.label, role="checkbox", status="error",
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )


def _fill_radio(
    page: "Page",
    locator: "Locator",
    ff: FormField,
    info: dict[str, Any],
) -> FieldOutcome:
    """Radio is simpler — just click the matching radio. The
    enumerate JS produces one entry PER radio option, so the agent
    is expected to pass the specific radio it wants (label = "Male"
    not "Gender")."""
    try:
        locator.check(timeout=2000)
        try:
            ok = bool(locator.is_checked(timeout=800))
        except Exception:
            ok = True
        return FieldOutcome(
            label=ff.label, role="radio",
            status="verified" if ok else "miss",
            final_value="1" if ok else "0",
        )
    except Exception as e:
        return FieldOutcome(
            label=ff.label, role="radio", status="error",
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )


def _fill_date(
    page: "Page",
    locator: "Locator",
    ff: FormField,
    info: dict[str, Any],
    *,
    settle_ms: int,
) -> FieldOutcome:
    """Date inputs: try fill() first (works for native `input
    type=date` / `datetime-local`). Custom date pickers fall back to
    the textbox strategy — agents authoring tests against custom
    pickers should provide the value in the format the field expects
    (e.g. "12/25/2026")."""
    try:
        locator.fill(ff.value, timeout=2000)
        try:
            cur = locator.input_value(timeout=800)
        except Exception:
            cur = ff.value
        return FieldOutcome(
            label=ff.label, role="date", status="verified",
            final_value=cur,
        )
    except Exception:
        # Fall through to textbox-style.
        return _fill_textbox(
            page, locator, ff, info, settle_ms=settle_ms,
        )


def _fill_file(
    page: "Page",
    locator: "Locator",
    ff: FormField,
    info: dict[str, Any],
) -> FieldOutcome:
    """File upload via Playwright's set_input_files. Value is a
    file path (or comma-separated multiple paths)."""
    paths = [
        p.strip() for p in (ff.value or "").split(",")
        if p.strip()
    ]
    if not paths:
        return FieldOutcome(
            label=ff.label, role="file", status="skipped",
            error="no file paths supplied",
        )
    try:
        locator.set_input_files(paths)
        return FieldOutcome(
            label=ff.label, role="file", status="filled",
            final_value=",".join(paths),
        )
    except Exception as e:
        return FieldOutcome(
            label=ff.label, role="file", status="error",
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )


# ── Phase G.3 — permission tree strategy ─────────────────────────


def _parse_permission_tree_value(raw: str) -> dict[str, Any]:
    """Parse the small DSL ``ff.value`` carries for a permission_tree.

    Forms:
      - "all"                 → tick every leaf checkbox
      - "none"                → untick every leaf (defaults already
                                 unticked, so this becomes idempotent)
      - "only:A,B,C"          → tick only leaves whose label OR parent
                                 contains A or B or C (case-insensitive)
      - "all_except:X,Y"      → tick every leaf except those matching
                                 X or Y

    Anything else falls back to "all". The match is on label substring
    (so "Roles" picks "Role Management → Add Role" and "Remove Role").
    """
    text = (raw or "").strip().lower()
    if not text or text == "all":
        return {"mode": "all", "items": []}
    if text == "none":
        return {"mode": "none", "items": []}
    if text.startswith("only:"):
        items = [
            s.strip()
            for s in text[len("only:"):].split(",")
            if s.strip()
        ]
        return {"mode": "only", "items": items}
    if text.startswith("all_except:"):
        items = [
            s.strip()
            for s in text[len("all_except:"):].split(",")
            if s.strip()
        ]
        return {"mode": "all_except", "items": items}
    return {"mode": "all", "items": []}


def _fill_permission_tree(
    page: "Page",
    ff: FormField,
    info: dict[str, Any],
    *,
    settle_ms: int,
    emit_event: Callable[[str, dict], None] | None = None,
) -> FieldOutcome:
    """Permission-tree strategy.

    Steps:
      1. Click Expand All until everything is expanded (looped — some
         UIs only expand one level per click).
      2. Phase Q.3 — mode="all" parent-first optimization: try ticking
         top-level PARENT checkboxes first and verify the children
         auto-cascaded. This handles Solar's Material-UI tree where
         clicking the parent propagates to every descendant — a 3-
         click solution vs 30+ per-child clicks (which also burns
         turns and can flip-flop indeterminate state).
      3. Fall back to per-leaf clicks for any leaf whose state still
         differs after the parent pass.
      4. Verify; report mismatches as miss.

    Best-effort: failures on individual leaves are recorded in the
    error but don't crash the routine. A tree with 0 detected leaves
    returns "miss" so the agent escalates.
    """
    plan = _parse_permission_tree_value(ff.value)

    # 1) Expand all branches — loop so multi-level trees fully open.
    # Solar's Material-UI tree only expands one depth-level per
    # "Expand all" click; a second click is needed to reach
    # grandchildren.
    for _ in range(3):
        try:
            expanded_something = bool(
                page.evaluate(_EXPAND_ALL_JS),
            )
        except Exception:
            expanded_something = False
        try:
            page.wait_for_timeout(settle_ms // 2)
        except Exception:
            pass
        if not expanded_something:
            break

    # 2) Enumerate checkboxes.
    try:
        raw = page.evaluate(_ENUM_TREE_CHECKBOXES_JS)
    except Exception as e:
        return FieldOutcome(
            label=ff.label, role="permission_tree", status="error",
            error=f"tree enumeration failed: {e}",
        )
    items: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        items = [
            it for it in (raw.get("items") or [])
            if isinstance(it, dict)
        ]
    if not items:
        return FieldOutcome(
            label=ff.label, role="permission_tree", status="miss",
            error=(
                "no tree checkboxes detected — verify the AppMap has "
                "has_permission_tree=true AND the drawer is open at "
                "the moment fill_form runs"
            ),
        )

    # Phase Q.3 / Q.4 — parent-first cascade optimization. RBAC trees
    # typically cascade parent → all descendants in one click. So if
    # a top-level parent's ENTIRE subtree is desired, clicking that
    # ONE parent is faster than clicking every leaf, and avoids the
    # flip-flop where re-clicking a leaf after parent-cascade
    # accidentally unchecks the parent + all its siblings.
    #
    # Q.3 supported mode="all" (every subtree is desired).
    # Q.4 extends to mode="only:..." when the named tokens match a
    # complete top-level parent subtree — e.g. "only:Administration,
    # Management" picks every Admin perm AND every Mgmt perm but
    # NOT My Profile, so we want to click the two parents and skip
    # the My Profile subtree entirely.
    needles_for_match = [s for s in plan["items"] if s]

    def _is_desired_for_planning(label: str, parent: str) -> bool:
        """Compute desired state per-leaf for the planning pass.
        Mirrors the per-leaf logic below so the parent-cascade can
        check 'is every child in this subtree desired?' before
        clicking the parent."""
        if plan["mode"] == "all":
            return True
        if plan["mode"] == "none":
            return False
        if plan["mode"] == "only" and needles_for_match:
            hay = (label + " | " + parent).lower()
            return any(n in hay for n in needles_for_match)
        if plan["mode"] == "all_except" and needles_for_match:
            hay = (label + " | " + parent).lower()
            return not any(n in hay for n in needles_for_match)
        return True

    if plan["mode"] in ("all", "only"):
        # Group items by their parent's label so we can ask "is the
        # whole subtree desired?" before clicking the parent.
        # Top-level parents have parent_label="".
        top_parents = [
            it for it in items
            if not str(it.get("parent_label") or "").strip()
        ]
        # For mode="only", restrict to parents whose subtree is
        # fully desired. For mode="all", every subtree is desired.
        cascadable_parents = []
        for pr in top_parents:
            pr_label = str(pr.get("label") or "")
            if not _is_desired_for_planning(pr_label, ""):
                continue
            children = [
                it for it in items
                if str(it.get("parent_label") or "").strip().lower()
                == pr_label.lower()
            ]
            # If ALL children are also desired (or there are no
            # children — rare but possible), the parent cascades
            # safely. Otherwise fall through to per-leaf for this
            # subtree.
            if all(
                _is_desired_for_planning(
                    str(c.get("label") or ""),
                    pr_label,
                )
                for c in children
            ):
                cascadable_parents.append(pr)

        parent_clicks_attempted = 0

        def _emit_tree(action: str, pr_item: dict[str, Any], err: str = "") -> None:
            # Phase X.4 — telemetry so the next run reveals which
            # parents the cascade attempted + why it succeeded /
            # failed. Silent before; from now on the live feed shows
            # one ``permission_tree_parent_attempted`` per parent.
            if emit_event is None:
                return
            try:
                emit_event("permission_tree_parent_attempted", {
                    "label": str(pr_item.get("label") or "")[:120],
                    "key": str(pr_item.get("key") or "")[:80],
                    "current_checked": bool(pr_item.get("checked")),
                    "action": action,
                    "error": err[:200] if err else "",
                })
            except Exception:
                pass

        for pr in cascadable_parents:
            if bool(pr.get("checked")):
                _emit_tree("skipped_already_checked", pr)
                continue
            key = str(pr.get("key") or "")
            if not key:
                _emit_tree("skipped_no_key", pr)
                continue
            try:
                loc = page.locator(f'[data-qai-tree-key="{key}"]')
                loc.scroll_into_view_if_needed(timeout=1200)
                try:
                    loc.check(timeout=1200)
                    _emit_tree("checked_via_check", pr)
                except Exception as e_check:
                    try:
                        loc.click(timeout=1200)
                        _emit_tree("checked_via_click", pr)
                    except Exception as e_click:
                        _emit_tree(
                            "failed",
                            pr,
                            err=(
                                f"check={type(e_check).__name__}; "
                                f"click={type(e_click).__name__}: "
                                f"{str(e_click)[:120]}"
                            ),
                        )
                        continue
                parent_clicks_attempted += 1
                page.wait_for_timeout(80)
            except Exception as e:
                _emit_tree(
                    "failed",
                    pr,
                    err=f"{type(e).__name__}: {str(e)[:120]}",
                )
                continue
        # Re-enumerate to see what the cascade actually achieved.
        if parent_clicks_attempted > 0:
            try:
                raw2 = page.evaluate(_ENUM_TREE_CHECKBOXES_JS)
                items2 = (
                    raw2.get("items") or []
                ) if isinstance(raw2, dict) else []
                if items2:
                    items = [it for it in items2 if isinstance(it, dict)]
            except Exception:
                pass

    # 3) Decide desired state per leaf.
    needles = [s for s in plan["items"] if s]

    def _matches_any(label: str, parent: str) -> bool:
        hay = (label + " | " + parent).lower()
        return any(n in hay for n in needles)

    desired_state: dict[str, bool] = {}
    for it in items:
        key = str(it.get("key") or "")
        if not key:
            continue
        label = str(it.get("label") or "")
        parent = str(it.get("parent_label") or "")
        if plan["mode"] == "all":
            desired_state[key] = True
        elif plan["mode"] == "none":
            desired_state[key] = False
        elif plan["mode"] == "only":
            desired_state[key] = (
                _matches_any(label, parent) if needles else False
            )
        elif plan["mode"] == "all_except":
            desired_state[key] = not _matches_any(label, parent)
        else:
            desired_state[key] = True

    # 4) Click each leaf whose current state ≠ desired state.
    clicks = 0
    errors: list[str] = []
    for it in items:
        key = str(it.get("key") or "")
        if not key or key not in desired_state:
            continue
        cur = bool(it.get("checked"))
        want = desired_state[key]
        if cur == want:
            continue
        try:
            locator = page.locator(f'[data-qai-tree-key="{key}"]')
            locator.scroll_into_view_if_needed(timeout=1500)
            if want:
                try:
                    locator.check(timeout=1500)
                except Exception:
                    locator.click(timeout=1500)
            else:
                try:
                    locator.uncheck(timeout=1500)
                except Exception:
                    locator.click(timeout=1500)
            clicks += 1
            # Drip — let the propagation handlers (some RBAC UIs
            # cascade parent → child on click) settle between clicks.
            page.wait_for_timeout(40)
        except Exception as e:
            errors.append(
                f"{(it.get('label') or '')[:40]}: {type(e).__name__}",
            )
            if len(errors) > 5:
                break

    # 5) Verify.
    page.wait_for_timeout(settle_ms // 2)
    try:
        raw2 = page.evaluate(_ENUM_TREE_CHECKBOXES_JS)
        items2 = (raw2.get("items") or []) if isinstance(raw2, dict) else []
    except Exception:
        items2 = items
    mismatched = 0
    total = 0
    for it in items2:
        key = str(it.get("key") or "")
        if key not in desired_state:
            continue
        total += 1
        if bool(it.get("checked")) != desired_state[key]:
            mismatched += 1

    if total == 0:
        return FieldOutcome(
            label=ff.label, role="permission_tree", status="miss",
            error="no tree leaves matched after enumeration",
        )
    if mismatched == 0:
        return FieldOutcome(
            label=ff.label, role="permission_tree", status="verified",
            final_value=f"{plan['mode']} ({clicks} clicks, {total} leaves)",
        )
    return FieldOutcome(
        label=ff.label, role="permission_tree",
        status="filled" if mismatched < total // 2 else "miss",
        final_value=f"{plan['mode']} ({total - mismatched}/{total} matched)",
        error="; ".join(errors)[:200] if errors else (
            f"{mismatched}/{total} leaves still mismatched"
        ),
    )


# ── Phase G.4 — paginated resource table strategy ─────────────────


def _parse_resource_table_value(raw: str) -> dict[str, Any]:
    """Parse ``ff.value`` for paginated_resource_table.

    Forms:
      - "all:read,update"
          → tick every row's read + update column.
            If column-master checkboxes exist they're clicked instead.
      - "specific:CH-0001:read,update;CH-0002:read"
          → tick only the named rows / cols. Pagination walks until
            every named row is found.
      - "none"
          → untick everything (visible rows only — pagination NOT
            walked since the resting state IS unticked).

    Returns ``{"mode", "actions", "rows"}`` — see code for shape.
    """
    text = (raw or "").strip()
    if not text or text.lower() == "none":
        return {"mode": "none", "actions": [], "rows": {}}
    if text.lower().startswith("all:"):
        actions = [
            a.strip().lower()
            for a in text[len("all:"):].split(",")
            if a.strip()
        ]
        return {"mode": "all", "actions": actions, "rows": {}}
    if text.lower().startswith("specific:"):
        # specific:ROW:a,b;ROW:c
        body = text[len("specific:"):]
        rows: dict[str, list[str]] = {}
        for chunk in body.split(";"):
            chunk = chunk.strip()
            if not chunk or ":" not in chunk:
                continue
            row_id, acts = chunk.split(":", 1)
            rows[row_id.strip()] = [
                a.strip().lower()
                for a in acts.split(",")
                if a.strip()
            ]
        return {"mode": "specific", "actions": [], "rows": rows}
    # Unknown form — default to all:read (safest broad permission).
    return {"mode": "all", "actions": ["read"], "rows": {}}


def _column_indices_for_actions(
    headers: list[str], actions: list[str],
) -> list[int]:
    """Map requested action names to column indices via substring match."""
    out: list[int] = []
    for a in actions:
        a_low = (a or "").lower().strip()
        for i, h in enumerate(headers):
            if a_low and a_low in (h or "").lower():
                if i not in out:
                    out.append(i)
                break
    return out


def _click_table_checkbox(
    page: "Page", key_attr: str, key: str, *, desired: bool,
) -> bool:
    """Click a tagged table checkbox if its current state differs."""
    try:
        loc = page.locator(f'[{key_attr}="{key}"]')
        # Try check/uncheck first — most robust.
        try:
            if desired:
                loc.check(timeout=1200)
            else:
                loc.uncheck(timeout=1200)
            return True
        except Exception:
            loc.click(timeout=1200)
            return True
    except Exception:
        return False


def _fill_paginated_resource_table(
    page: "Page",
    ff: FormField,
    info: dict[str, Any],
    *,
    settle_ms: int,
) -> FieldOutcome:
    """Paginated resource-table strategy.

    Solar's Resource Access Control is the canonical example: a table
    of chainages × permission columns, with column-master checkboxes
    in the header that grant the action across ALL pages at once.

    Strategy:
      - mode=all:    prefer column-master checkboxes (1 click per
                     action). Fall back to per-row when masters absent.
      - mode=specific: walk pagination until every named row is found
                     and its requested cells ticked.
      - mode=none:   untick every visible checkbox once (no pagination).
    """
    plan = _parse_resource_table_value(ff.value)

    try:
        snap = page.evaluate(_ENUM_RESOURCE_TABLE_JS)
    except Exception as e:
        return FieldOutcome(
            label=ff.label, role="paginated_resource_table",
            status="error",
            error=f"table enumeration failed: {e}",
        )
    if not isinstance(snap, dict):
        return FieldOutcome(
            label=ff.label, role="paginated_resource_table",
            status="miss",
            error="no resource table detected",
        )

    headers = [str(h or "") for h in (snap.get("headers") or [])]
    column_masters = snap.get("column_masters") or []

    if plan["mode"] == "all":
        target_cols = _column_indices_for_actions(headers, plan["actions"])
        if not target_cols:
            return FieldOutcome(
                label=ff.label, role="paginated_resource_table",
                status="miss",
                error=(
                    f"no table columns matched actions: {plan['actions']}"
                ),
            )

        # Prefer column-master checkboxes when present.
        master_by_index = {
            int(cm.get("index", -1)): cm for cm in column_masters
            if isinstance(cm, dict)
        }
        used_masters = 0
        leftover_cols: list[int] = []
        for col_idx in target_cols:
            if col_idx in master_by_index:
                key = str(master_by_index[col_idx].get("key") or "")
                if key and _click_table_checkbox(
                    page, "data-qai-tab-col", key, desired=True,
                ):
                    used_masters += 1
                    page.wait_for_timeout(40)
                    continue
            leftover_cols.append(col_idx)

        # Per-row fallback for columns without masters. Pagination
        # walked here so we don't miss rows on later pages.
        per_row_clicks = 0
        if leftover_cols:
            pages_walked = 0
            max_pages = 8
            while pages_walked < max_pages:
                cur_snap = page.evaluate(_ENUM_RESOURCE_TABLE_JS)
                if not isinstance(cur_snap, dict):
                    break
                for row in cur_snap.get("rows") or []:
                    if not isinstance(row, dict):
                        continue
                    for cell in row.get("cells") or []:
                        if not isinstance(cell, dict):
                            continue
                        if int(cell.get("col_index", -1)) not in leftover_cols:
                            continue
                        if cell.get("checked"):
                            continue
                        key = str(cell.get("key") or "")
                        if not key:
                            continue
                        if _click_table_checkbox(
                            page, "data-qai-tab-cell", key, desired=True,
                        ):
                            per_row_clicks += 1
                            page.wait_for_timeout(20)
                next_sel = cur_snap.get("next_selector") or ""
                if not next_sel or not cur_snap.get("next_enabled"):
                    break
                try:
                    page.locator(next_sel).first.click(timeout=1500)
                    page.wait_for_timeout(settle_ms // 2)
                except Exception:
                    break
                pages_walked += 1

        return FieldOutcome(
            label=ff.label, role="paginated_resource_table",
            status="verified" if used_masters or per_row_clicks else "miss",
            final_value=(
                f"all: masters={used_masters} rows={per_row_clicks}"
            ),
        )

    if plan["mode"] == "specific":
        rows_wanted = plan["rows"]
        if not rows_wanted:
            return FieldOutcome(
                label=ff.label, role="paginated_resource_table",
                status="miss", error="no row IDs supplied",
            )
        remaining = {k.lower(): list(v) for k, v in rows_wanted.items()}
        total_clicks = 0
        pages_walked = 0
        max_pages = 12
        while pages_walked < max_pages and remaining:
            try:
                cur_snap = page.evaluate(_ENUM_RESOURCE_TABLE_JS)
            except Exception:
                break
            if not isinstance(cur_snap, dict):
                break
            cur_headers = [
                str(h or "") for h in (cur_snap.get("headers") or [])
            ]
            for row in cur_snap.get("rows") or []:
                if not isinstance(row, dict):
                    continue
                row_label = (row.get("row_label") or "").lower()
                if not row_label:
                    continue
                matched_key = next(
                    (rk for rk in remaining if rk in row_label),
                    None,
                )
                if matched_key is None:
                    continue
                wanted_actions = remaining.pop(matched_key)
                want_cols = _column_indices_for_actions(
                    cur_headers, wanted_actions,
                )
                for cell in row.get("cells") or []:
                    if not isinstance(cell, dict):
                        continue
                    if int(cell.get("col_index", -1)) not in want_cols:
                        continue
                    if cell.get("checked"):
                        continue
                    key = str(cell.get("key") or "")
                    if key and _click_table_checkbox(
                        page, "data-qai-tab-cell", key, desired=True,
                    ):
                        total_clicks += 1
                        page.wait_for_timeout(25)
            if not remaining:
                break
            next_sel = cur_snap.get("next_selector") or ""
            if not next_sel or not cur_snap.get("next_enabled"):
                break
            try:
                page.locator(next_sel).first.click(timeout=1500)
                page.wait_for_timeout(settle_ms // 2)
            except Exception:
                break
            pages_walked += 1
        if remaining:
            return FieldOutcome(
                label=ff.label, role="paginated_resource_table",
                status="filled" if total_clicks else "miss",
                final_value=f"specific: {total_clicks} clicks",
                error=(
                    "rows not found after pagination: "
                    + ", ".join(remaining.keys())
                )[:200],
            )
        return FieldOutcome(
            label=ff.label, role="paginated_resource_table",
            status="verified",
            final_value=f"specific: {total_clicks} clicks",
        )

    # mode=none — untick visible rows only.
    unticks = 0
    for row in snap.get("rows") or []:
        if not isinstance(row, dict):
            continue
        for cell in row.get("cells") or []:
            if not isinstance(cell, dict) or not cell.get("checked"):
                continue
            key = str(cell.get("key") or "")
            if key and _click_table_checkbox(
                page, "data-qai-tab-cell", key, desired=False,
            ):
                unticks += 1
                page.wait_for_timeout(20)
    return FieldOutcome(
        label=ff.label, role="paginated_resource_table",
        status="verified", final_value=f"none: {unticks} unticks",
    )


# ── Submit + validation observer ─────────────────────────────────


# Phase U — vision-coord fallback for form fields.
#
# Fires when the DOM scanner finds 0 fields OR fewer than the
# requested set matched. One VL call returns {label: (x, y)} for
# every requested label. The fill loop then clicks at each coord and
# types the value — bypasses DOM resolution entirely. The fallback
# does NOT support custom comboboxes / trees / paginated tables
# (those need DOM-keyed JS); textbox + textarea + simple inputs only.


_VL_LOCATE_FIELDS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "x": {"type": ["integer", "null"]},
                    "y": {"type": ["integer", "null"]},
                    "confidence": {
                        "type": "number", "minimum": 0.0, "maximum": 1.0,
                    },
                },
                "required": ["label", "x", "y", "confidence"],
                "additionalProperties": False,
            },
        },
        "submit_x": {"type": ["integer", "null"]},
        "submit_y": {"type": ["integer", "null"]},
        "submit_confidence": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
        },
    },
    "required": ["fields", "submit_x", "submit_y", "submit_confidence"],
    "additionalProperties": False,
}


_VL_LOCATE_FIELDS_SYSTEM_PROMPT = (
    "You are a UI element locator. The agent is filling a form on a "
    "screenshot. The DOM scanner failed (custom-styled drawer, shadow "
    "DOM, iframe, etc.). Your ONE job: return pixel coordinates for "
    "the CENTER of each input field matching the requested labels, "
    "plus the Save / Submit button.\n\n"
    "Rules:\n"
    "1. (x, y) are in the screenshot's pixel space — the same space "
    "page.mouse.click expects.\n"
    "2. For each requested field, find the INPUT box (textbox / "
    "textarea / dropdown / checkbox) associated with that label. "
    "Return the center of the INPUT, NOT the label text.\n"
    "3. If you can't see a field's input, set x=null, y=null, "
    "confidence=0.0 for that entry.\n"
    "4. submit_x / submit_y point at the primary Save / Submit / "
    "Create button at the bottom of the form. Null when not visible.\n"
    "5. confidence: 0.95+ when the field is unambiguous; 0.6-0.8 "
    "when uncertain; <0.5 when guessing.\n"
    "Output strict JSON only."
)


def _locate_form_fields_via_vision(
    page: "Page",
    *,
    labels: list[str],
    submit_label: str,
    vision_provider: Any,
) -> tuple[dict[str, tuple[int, int]], tuple[int, int] | None]:
    """Phase U — fallback vision call. Returns ``(label_coords,
    submit_coord_or_None)``.

    Single LLM call covers ALL requested labels + the Save button.
    Empty dict + None when the call fails or the provider can't see
    images.
    """
    if vision_provider is None or not getattr(
        vision_provider, "supports_vision", False,
    ):
        return {}, None
    if not labels:
        return {}, None
    try:
        from app.agents.page_intel import (  # noqa: PLC0415
            capture_screenshot_for_vision,
        )
        from app.llm.base import ChatMessage  # noqa: PLC0415
    except Exception:
        return {}, None
    try:
        # downscale=False — we need pixel-accurate coords; downscaling
        # would shift the returned (x, y) relative to the live viewport.
        shot = capture_screenshot_for_vision(page, downscale=False)
    except Exception:
        return {}, None

    user_text = (
        "FIELDS TO LOCATE (return coords for each in the same order):\n"
        + "\n".join(f"  - {lab!r}" for lab in labels)
        + f"\n\nSUBMIT BUTTON LABEL: {submit_label!r}\n\n"
        "Return JSON with one entry per field and the submit coords."
    )
    try:
        result = vision_provider.chat_structured(
            messages=[
                ChatMessage(
                    role="system",
                    content=_VL_LOCATE_FIELDS_SYSTEM_PROMPT,
                ),
                ChatMessage(role="user", content=user_text, image=shot),
            ],
            schema=_VL_LOCATE_FIELDS_SCHEMA,
            schema_name="form_field_coords",
            temperature=0.1,
            max_output_tokens=800,
        )
    except Exception as e:
        logger.warning(
            "vision-coord field locator failed: %s: %s",
            type(e).__name__, e,
        )
        return {}, None
    parsed = result.parsed
    if not isinstance(parsed, dict):
        return {}, None

    coords: dict[str, tuple[int, int]] = {}
    for entry in (parsed.get("fields") or []):
        if not isinstance(entry, dict):
            continue
        lab = str(entry.get("label") or "").strip()
        x = entry.get("x")
        y = entry.get("y")
        conf = entry.get("confidence", 0.0)
        if not lab or not isinstance(x, int) or not isinstance(y, int):
            continue
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = 0.0
        if conf_f < 0.5:
            continue
        coords[lab] = (x, y)

    submit_coord: tuple[int, int] | None = None
    sx = parsed.get("submit_x")
    sy = parsed.get("submit_y")
    sconf = parsed.get("submit_confidence", 0.0)
    try:
        sconf_f = float(sconf)
    except (TypeError, ValueError):
        sconf_f = 0.0
    if isinstance(sx, int) and isinstance(sy, int) and sconf_f >= 0.5:
        submit_coord = (sx, sy)
    return coords, submit_coord


_FIND_PRIMARY_SUBMIT_JS = r"""
(opts) => {
  // Phase I.2 — find the most likely SUBMIT BUTTON, distinguishing
  // it from a form-title heading or static label with the same text.
  //
  // Strategy:
  //   1. Pick the active drawer/dialog (largest visible role=dialog).
  //   2. Scan ACTIONABLE elements only: <button>, [role=button],
  //      <input type=submit|button>. Headings + plain text are
  //      excluded by construction.
  //   3. Score each candidate by:
  //      - text exact-match (highest)
  //      - text contains needle (medium)
  //      - bottom-right position bias (drawer footers)
  //      - primary visual styling (background, contrast)
  //      - NOT disabled
  //   4. Return the best candidate's bbox + tagging key so the caller
  //      can click via Playwright with a stable selector.
  const needle = (opts.needle || '').trim().toLowerCase();
  const fallback = opts.fallback_needles || [];
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return false;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    if (parseFloat(cs.opacity) < 0.1) return false;
    return true;
  };
  const labelOf = (el) => {
    const al = el.getAttribute('aria-label');
    if (al) return al.trim();
    const v = el.value || '';
    if (v) return String(v).trim();
    const txt = (el.innerText || el.textContent || '').trim();
    return txt;
  };
  const isDisabled = (el) =>
    el.disabled || el.getAttribute('aria-disabled') === 'true';
  const isPrimary = (el) => {
    const cs = getComputedStyle(el);
    const cls = (el.className || '').toString().toLowerCase();
    // Heuristic: class names that almost universally signal primary
    // action ("Save"-style) buttons. Plus a contrast check: button
    // background materially different from its parent's background.
    if (/primary|cta|action|submit/i.test(cls)) return true;
    try {
      const myBg = cs.backgroundColor;
      const parentBg = el.parentElement
        ? getComputedStyle(el.parentElement).backgroundColor : '';
      if (myBg && parentBg && myBg !== parentBg &&
          myBg !== 'rgba(0, 0, 0, 0)' && myBg !== 'transparent') {
        return true;
      }
    } catch (e) {}
    return false;
  };

  // Diag.2 — drawer detection mirrors _ENUMERATE_FIELDS_JS so the
  // SAME drawer is scanned for the SAME form. Includes the
  // heuristic fallback (position:fixed + high z-index + form inputs
  // inside) for custom-styled drawers like Solar's.
  const findDrawer = () => {
    const explicit = [
      '[role=dialog]', '[role=alertdialog]',
      '[aria-modal="true"]',
      '.MuiDialog-paper', '.MuiDrawer-paper', '.MuiModal-root',
      '[class*="Drawer"]', '[class*="drawer"]',
      '[class*="Modal"]', '[class*="modal"]',
    ].join(',');
    let cands2 = [...document.querySelectorAll(explicit)]
      .filter(VISIBLE)
      .filter(el => {
        const r = el.getBoundingClientRect();
        return r.width >= 240 && r.height >= 240;
      });
    if (cands2.length === 0) {
      const vw = window.innerWidth || 1280;
      const vh = window.innerHeight || 720;
      cands2 = [...document.querySelectorAll('div, section, aside')]
        .filter(el => {
          const cs = getComputedStyle(el);
          const pos = cs.position;
          if (pos !== 'fixed' && pos !== 'absolute') return false;
          const z = parseInt(cs.zIndex || '0', 10);
          if (z < 100) return false;
          if (!VISIBLE(el)) return false;
          const r = el.getBoundingClientRect();
          if (r.width < vw * 0.3 && r.height < vh * 0.4) return false;
          return !!el.querySelector(
            'input, textarea, select, [role=combobox], [role=textbox]',
          );
        });
    }
    if (cands2.length === 0) return document.body;
    cands2.sort((a, b) => {
      const ar = a.getBoundingClientRect();
      const br = b.getBoundingClientRect();
      return (br.width * br.height) - (ar.width * ar.height);
    });
    return cands2[0];
  };
  const root = findDrawer();
  const drawerRect = root.getBoundingClientRect();

  // Diag.2 — broaden the action selector. Some apps wrap Save in a
  // <div role="button"> / <a> / <span tabindex="0"> / any element
  // with an onclick. We collect:
  //   - canonical <button> / input[type=submit|button]
  //   - role=button / role=link with text
  //   - clickable non-button (cursor:pointer + text) within the
  //     drawer's bottom 30% (footer region) as a last-resort
  //     candidate — only when canonical search returns nothing.
  const actionSel =
    'button, [role=button], input[type=submit], input[type=button], ' +
    '[role=link], a[role=button]';
  let cands = [...root.querySelectorAll(actionSel)].filter(VISIBLE);
  if (cands.length === 0) {
    const footerCutY = drawerRect.top + drawerRect.height * 0.7;
    cands = [...root.querySelectorAll('*')]
      .filter(el => {
        if (!VISIBLE(el)) return false;
        const r = el.getBoundingClientRect();
        if (r.top < footerCutY) return false;
        const cs = getComputedStyle(el);
        if (cs.cursor !== 'pointer') return false;
        const t = (el.innerText || el.textContent || '').trim();
        return t.length > 0 && t.length < 40;
      });
  }

  function scoreCandidate(el) {
    if (isDisabled(el)) return -1;
    const lab = labelOf(el).toLowerCase();
    if (!lab) return -1;
    let textScore = 0;
    if (lab === needle) textScore = 100;
    else if (lab.startsWith(needle) && needle) textScore = 80;
    else if (needle && lab.includes(needle)) textScore = 60;
    else {
      // No needle match — check fallbacks (e.g. ["save", "create"]).
      for (const f of fallback) {
        const fl = f.toLowerCase();
        if (lab === fl) { textScore = 50; break; }
        if (lab.startsWith(fl)) { textScore = 40; break; }
        if (lab.includes(fl)) { textScore = 30; break; }
      }
    }
    if (textScore === 0) return -1;

    const r = el.getBoundingClientRect();
    // Bottom-bias: closer to the drawer's bottom edge → higher score.
    const drawerBottom = drawerRect.top + drawerRect.height;
    const yFromBottom = drawerBottom - (r.top + r.height);
    const bottomBias = Math.max(0, 40 - yFromBottom / 4);
    // Right-bias: closer to the drawer's right edge → higher.
    const drawerRight = drawerRect.left + drawerRect.width;
    const xFromRight = drawerRight - (r.left + r.width);
    const rightBias = Math.max(0, 20 - xFromRight / 10);
    const primaryBias = isPrimary(el) ? 25 : 0;
    return textScore + bottomBias + rightBias + primaryBias;
  }

  let best = null;
  let bestScore = -1;
  for (const el of cands) {
    const s = scoreCandidate(el);
    if (s > bestScore) {
      bestScore = s;
      best = el;
    }
  }
  // Phase X.3a — positional fallback. When the text-matching scorer
  // returns nothing (button text doesn't match "save / create /
  // submit / confirm / ok / apply" — e.g. Solar's "Add Role" or a
  // locale-translated string), fall back to a PRIMARY-styled button
  // in the drawer's bottom 30% (footer region). If exactly one,
  // use it. If multiple, pick the rightmost (drawer footers put
  // primary action bottom-right). Cancel buttons are usually NOT
  // primary-styled (outlined / muted), so this won't pick them.
  if (!best || bestScore < 30) {
    const footerCutY = drawerRect.top + drawerRect.height * 0.7;
    let primaryFooter = cands.filter(el => {
      if (isDisabled(el)) return false;
      const r = el.getBoundingClientRect();
      if (r.top < footerCutY) return false;
      return isPrimary(el);
    });
    if (primaryFooter.length === 0) {
      // No primary-styled candidate — relax to ANY non-disabled
      // button in the bottom 30%. Last resort before giving up.
      primaryFooter = cands.filter(el => {
        if (isDisabled(el)) return false;
        const r = el.getBoundingClientRect();
        return r.top >= footerCutY;
      });
    }
    if (primaryFooter.length === 1) {
      best = primaryFooter[0];
      bestScore = 25;
    } else if (primaryFooter.length > 1) {
      primaryFooter.sort((a, b) => {
        const ar = a.getBoundingClientRect();
        const br = b.getBoundingClientRect();
        return (br.left + br.width) - (ar.left + ar.width);
      });
      best = primaryFooter[0];
      bestScore = 25;
    }
  }
  if (!best || bestScore < 25) return null;
  let key = best.getAttribute('data-qai-submit-key');
  if (!key) {
    key = 'qai-submit-' + Math.random().toString(36).slice(2, 9);
    best.setAttribute('data-qai-submit-key', key);
  }
  const r = best.getBoundingClientRect();
  return {
    key,
    label: labelOf(best),
    score: bestScore,
    rect: [Math.round(r.left), Math.round(r.top),
           Math.round(r.width), Math.round(r.height)],
  };
};
"""


def _submit_and_observe(
    page: "Page",
    *,
    submit_label: str,
    settle_ms: int,
    vision_submit_coord: tuple[int, int] | None = None,
    vision_provider: Any = None,
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Click the submit button matching ``submit_label``, then watch
    for inline aria-invalid feedback. Returns
    ``(ok, message, invalid_fields)``.

    ``ok=True`` only when:
      - the click succeeded, AND
      - no aria-invalid fields are visible after settle_ms.

    Doesn't try to interpret success signals (drawer-closed / toast)
    here — those come from the caller's existing form_signals
    observer.
    """
    # Scroll the active drawer so the submit is in view.
    try:
        from app.executor.actions import scroll_drawer_to_bottom  # noqa: PLC0415
        scroll_drawer_to_bottom(page)
    except Exception:
        pass

    # Phase I.2 — find the actual action button via JS scoring that
    # filters out form-title headings and labels with the same text.
    # The legacy ``get_by_text`` fallback used to fire on the heading
    # "Create Role" (which appears at the top of Solar's role drawer)
    # instead of the footer Save button.
    clicked = False
    primary_pick: dict[str, Any] | None = None
    try:
        raw = page.evaluate(
            _FIND_PRIMARY_SUBMIT_JS,
            {
                "needle": submit_label,
                "fallback_needles": [
                    "save", "create", "submit", "confirm", "ok", "apply",
                ],
            },
        )
        if isinstance(raw, dict) and raw.get("key"):
            primary_pick = raw
    except Exception:
        primary_pick = None
    if primary_pick is not None:
        try:
            page.locator(
                f'[data-qai-submit-key="{primary_pick["key"]}"]',
            ).first.click(timeout=2500)
            clicked = True
        except Exception:
            pass
    if not clicked:
        try:
            page.get_by_role("button", name=submit_label).first.click(
                timeout=2500,
            )
            clicked = True
        except Exception:
            pass
    # Phase U — vision-coord fallback for Save. When neither the JS
    # scorer nor the role-based locator found Save (custom-styled
    # drawer, shadow DOM, etc.), use the coords the form-locator VL
    # call returned. The locator was called ONCE per fill_form
    # invocation; we reuse its submit_coord here instead of firing
    # another VL call.
    if not clicked and vision_submit_coord is not None:
        try:
            page.mouse.click(
                vision_submit_coord[0],
                vision_submit_coord[1],
            )
            clicked = True
        except Exception:
            pass
    # Phase X.3b — fresh VL fallback for Save. When the JS scorer
    # AND the role locator AND the cached vision coord ALL failed,
    # fire a dedicated VL call asking ONLY for the Save button.
    # Decoupled from the form-locator's initial call so we don't lose
    # Save just because the form locator's submit_confidence was low.
    if not clicked and vision_provider is not None and getattr(
        vision_provider, "supports_vision", False,
    ):
        try:
            from app.agents.page_intel import (  # noqa: PLC0415
                propose_click_coordinates,
            )
            coords = propose_click_coordinates(
                vision_provider,
                page,
                target_hint=(
                    submit_label
                    or "Save / Submit / Create button at the "
                    "bottom of the drawer (the primary action that "
                    "persists the form)"
                ),
            )
            if (
                coords is not None
                and getattr(coords, "confidence", 0.0) >= 0.5
                and isinstance(getattr(coords, "x", None), int)
                and isinstance(getattr(coords, "y", None), int)
            ):
                page.mouse.click(coords.x, coords.y)
                clicked = True
        except Exception as e:
            logger.debug(
                "fresh-VL Save fallback failed: %s: %s",
                type(e).__name__, e,
            )
    # Note: deliberately NO get_by_text fallback here — that's exactly
    # what mis-fires on form-title headings. If both methods above
    # failed, we'd rather report "submit not found" than click the wrong
    # element.
    if not clicked:
        return (
            False,
            (
                f"submit click failed: no button matching {submit_label!r}"
                + (
                    " (vision-coord fallback also unavailable)"
                    if vision_submit_coord is None
                    and vision_provider is None
                    else " (all fallbacks exhausted)"
                )
            ),
            [],
        )

    # Wait for inline errors to settle.
    try:
        page.wait_for_timeout(settle_ms)
    except Exception:
        pass

    invalids: list[dict[str, Any]] = []
    try:
        raw = page.evaluate(_INVALID_FIELDS_JS)
        if isinstance(raw, list):
            invalids = [r for r in raw if isinstance(r, dict)]
    except Exception:
        invalids = []

    # Phase X.2 — async-validation double-check. On Solar (and any
    # React app that fires validation through useEffect / async
    # form-resolver chain), inline errors render AFTER ``settle_ms``
    # — the first scan declares clean, the agent calls
    # mark_goal_complete, the operator sees a red error on screen.
    # If the first scan is empty, wait another 1500ms and re-scan.
    # Any errors found on the second pass are treated as if they had
    # fired the first time → routed through the regenerator on retry.
    if not invalids:
        try:
            page.wait_for_timeout(1500)
        except Exception:
            pass
        try:
            raw2 = page.evaluate(_INVALID_FIELDS_JS)
            if isinstance(raw2, list):
                invalids = [r for r in raw2 if isinstance(r, dict)]
        except Exception:
            invalids = []

    if invalids:
        names = ", ".join(
            r.get("label") or "(unlabeled)"
            for r in invalids[:6]
        )
        return (
            False,
            f"validation errors on: {names}",
            invalids,
        )
    return (True, "submit OK; no inline validation errors visible", [])


# ── Phase I.3 — validation-aware value regeneration ──────────────


def _regenerate_value_from_error(
    *,
    field_label: str,
    original_value: str,
    error_message: str,
) -> tuple[str, str]:
    """Read the error_message + original_value, return ``(new_value, reason)``.

    Deterministic, dependency-free, fast. Covers the constraint families
    that produce the bulk of real validation feedback:

    - "only letters" / "only alphabets" / "alphabetic only"
      → strip digits + special chars from the value
    - "no spaces" → strip whitespace
    - "letters and spaces only" → strip everything except letters + spaces
    - "must be a valid email" / "invalid email"
      → if value isn't email-shaped, attach a synthetic suffix
    - "numeric only" / "must be a number" / "digits only"
      → strip non-digits
    - "phone number" / "10 digits" → produce a 10-digit fallback
    - "minimum N characters" → pad with safe chars
    - "maximum N characters" → truncate

    When no rule matches, return ``(original_value, "no rule matched")`` —
    the caller falls back to the same-value retry.
    """
    import re  # noqa: PLC0415

    err = (error_message or "").lower()
    val = original_value or ""

    # Strip operations — order matters; check the most-specific first.
    if (
        "only letters and spaces" in err
        or "letters and spaces only" in err
        or "alphabet and spaces" in err
    ):
        cleaned = re.sub(r"[^A-Za-z\s]", "", val).strip() or "QA Test"
        return cleaned, "stripped non-letter/non-space chars"

    if (
        "only letters" in err
        or "letters only" in err
        or "only alphabet" in err
        or "alphabet only" in err
        or "alphabetic" in err
    ):
        cleaned = re.sub(r"[^A-Za-z]", "", val) or "QATest"
        return cleaned, "stripped non-letter chars"

    if "no spaces" in err or "spaces are not allowed" in err:
        return re.sub(r"\s+", "", val), "stripped spaces"

    if (
        "only digits" in err
        or "digits only" in err
        or "numeric only" in err
        or "must be a number" in err
        or "must be numeric" in err
    ):
        digits = re.sub(r"\D", "", val) or "1234567890"
        return digits, "stripped non-digit chars"

    if (
        "valid email" in err
        or "invalid email" in err
        or "email format" in err
        or "must be email" in err
    ):
        if "@" not in val:
            base = re.sub(r"[^A-Za-z0-9._-]", "", val) or "qa.tester"
            return f"{base}@example.com", "appended @example.com"
        # Has @ but invalid — probably bad domain.
        local = val.split("@", 1)[0] or "qa.tester"
        return f"{local}@example.com", "normalised email domain"

    if (
        "phone" in err
        and (
            "10 digit" in err or "ten digit" in err
            or "digits" in err or "number" in err
        )
    ):
        digits = re.sub(r"\D", "", val)
        if len(digits) < 10:
            digits = (digits + "9999999999")[:10]
        else:
            digits = digits[:10]
        return digits, "10-digit phone fallback"

    # Length constraints — "minimum N characters" / "at least N".
    m_min = re.search(
        r"(?:minimum|at least)\s+(\d+)\s*(?:characters|chars|letters)?",
        err,
    )
    if m_min:
        n = int(m_min.group(1))
        if len(val) < n:
            padded = (val + "X" * n)[:n]
            return padded, f"padded to {n} chars"
    m_max = re.search(
        r"(?:maximum|at most)\s+(\d+)\s*(?:characters|chars|letters)?",
        err,
    )
    if m_max:
        n = int(m_max.group(1))
        if len(val) > n:
            return val[:n], f"truncated to {n} chars"

    # "Required" / "cannot be empty" — supply a safe non-empty.
    if (
        "required" in err
        or "cannot be empty" in err
        or "is empty" in err
        or "please enter" in err
    ):
        if not val.strip():
            return f"QA-{field_label[:20].strip() or 'field'}", "filled in required field"

    # Phase J.2 — duplicate / uniqueness conflict. The value the test
    # case mandated already exists (e.g. role "QA Tester" was created
    # on the previous run). Append a millisecond timestamp suffix so
    # the second creation succeeds without rejecting the test intent.
    # Pattern variations seen in real apps:
    #   "Already exists" / "already exists"
    #   "must be unique" / "should be unique"
    #   "duplicate" (used in some APIs returning a friendly toast)
    #   "name is taken" / "X is taken" / "in use"
    #   "conflict" / "already in use"
    if (
        "already exists" in err
        or "already in use" in err
        or "already taken" in err
        or "must be unique" in err
        or "should be unique" in err
        or "duplicate" in err
        or "is taken" in err
        or (
            "exists" in err and (
                "name" in err or "username" in err or "email" in err
            )
        )
        or ("conflict" in err and "data" not in err)  # avoid 'data conflict'
    ):
        import time as _time  # noqa: PLC0415
        suffix = f"-{int(_time.time() * 1000) % 1_000_000}"
        # Preserve the original look; append the suffix at the end.
        # If the field has a length limit observed in another rule
        # earlier, we still defer to that — this regen runs OUTSIDE
        # those caps.
        if val.strip():
            return f"{val.rstrip()}{suffix}", f"appended {suffix} to dedupe"
        # Empty value + duplicate error is rare but happens on
        # auto-generated codes — synthesize a fresh value.
        return (
            f"QA-{field_label[:20].strip() or 'item'}{suffix}",
            "synthesised unique value",
        )

    # No deterministic rule matched.
    return val, "no rule matched"


# ── Helpers ───────────────────────────────────────────────────────


def _parse_bool(s: str) -> bool:
    return (s or "").strip().lower() in (
        "1", "true", "yes", "on", "checked",
    )
