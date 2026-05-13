"""Phase A.5 — Authenticated Scout.

The existing :mod:`app.agents.recon` walker STOPS at the auth wall.
For private admin apps (Solar, SAP Fiori, internal CMSes), that
means it learns essentially nothing — the entire app surface lives
behind login.

This module runs AFTER ``auth_flow.run_auth_loop`` succeeds for the
first time on a given ``target_url``. It walks the post-login surface,
captures the primary nav, opens each main section, opens the primary
create-form (drawer / modal) on each list page, and emits one
:class:`ScoutedPage` per landing.

The captured pages flow into :mod:`app.agents.app_map` which uses ONE
vision call to consolidate them into a structured :class:`AppMap` —
modules → sections → forms / tables / dropdowns → action patterns.
The qa_agent's sub-goal decomposer reads the map at submodule start
so its plans are anchored to the REAL UI, not the BRD's idea of it.

Why a separate module from ``recon``
------------------------------------
- Different lifecycle: triggered mid-run (after first auth), not
  pre-run by user.
- Different output shape: structured per-page records, not free-form
  AKB notes.
- Different walking strategy: nav-driven (look at the top bar →
  visit each item) rather than link-following.

Cost ceiling
------------
- "shallow" depth: top-level nav only, no drawer opens. ~6-8 screenshots
  + ~1 consolidator call. ≈ $0.03 on a strong-tier provider.
- "deep" depth (default): top nav + open the primary +Create button
  on each list page to capture drawer fields. ~12-20 screenshots.
  ≈ $0.06-0.10. Cached for all subsequent runs on the same target_url.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal
from urllib.parse import urlparse

if TYPE_CHECKING:
    from playwright.sync_api import Page

    from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


ScoutDepth = Literal["shallow", "deep"]


@dataclass
class CapturedElement:
    """One interactive element a Scout captured on a page.

    ``role`` follows ARIA / accessibility-tree terminology so the
    consolidator can reason about what a thing IS, not just what it
    looks like. ``label`` is the visible text. ``rect`` is pixel
    bbox in the viewport — used downstream by SoM annotation.
    """
    role: str          # "button", "link", "textbox", "combobox",
                       # "checkbox", "tab", "menuitem", "heading", "row"
    label: str
    rect: tuple[int, int, int, int]  # x, y, w, h
    # Lightweight selector the runtime can try first; may be None when
    # the element only has visible text + no stable attrs.
    hint_selector: str | None = None


@dataclass
class ScoutedPage:
    """One page the Scout visited.

    Carries the screenshot bytes (decimated PNG) + structured
    element summary + the URL. The consolidator (next step) reads
    a batch of these and produces the :class:`AppMap`.
    """
    url: str
    title: str
    nav_path: list[str]                  # ["Administration", "Roles"]
    elements: list[CapturedElement] = field(default_factory=list)
    screenshot_png: bytes | None = None
    # When this page has a "+Add"/"+Create"/"+New" button and the
    # Scout opened it, this is the drawer's captured screenshot +
    # form fields. None when no create surface was found / opened.
    create_surface: "CreateSurface | None" = None


@dataclass
class TreeStructure:
    """Phase G.1 — a permission tree captured inside a create-surface.

    Solar's role drawer (and many RBAC admin forms) renders permissions
    as a hierarchical tree: parent nodes (modules) with child checkboxes
    (actions). There's no "Select All" — to grant every permission you
    click each parent. The runtime needs the shape to dispatch the
    right strategy.

    ``label`` is the section title above the tree ("Permissions",
    "Access"). ``parents`` are the labels of the top-level nodes
    (each typically a module name). ``has_expand_all`` reflects
    whether an "Expand All" / "Collapse All" toggle is visible.
    """
    label: str = ""
    parents: list[str] = field(default_factory=list)
    has_expand_all: bool = False
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class ResourceTable:
    """Phase G.1 — a paginated checkbox table captured in the drawer.

    Solar's user-create flow has a Resource Access Control section with
    a paginated table: each row is a resource (chainage) and each column
    is a permission action (read / update / delete). The header has an
    "All read" / "All update" master checkbox per column. Without
    knowing this exists, the agent never figures out how to grant
    "read access on all chainages".
    """
    label: str = ""
    columns: list[str] = field(default_factory=list)  # action column headers
    row_label_sample: list[str] = field(default_factory=list)
    has_pagination: bool = False
    has_column_masters: bool = False
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class ConditionalSection:
    """Phase G.1 / G.5 — a section that only renders after a trigger.

    Solar's user drawer shows "Resource Access Control" only after a
    role is picked; another flow shows the Phone country-code field
    only after the Phone field is typed. The Scout's probe step
    surfaces these — once the trigger fires, the new section's title
    + field labels are captured here so the decomposer knows to plan
    around ordering.
    """
    label: str = ""
    trigger_field_label: str = ""    # which field's value made it appear
    trigger_value: str = ""          # what value
    new_fields: list[CapturedElement] = field(default_factory=list)


@dataclass
class CreateSurface:
    """Captured when Scout opens a list page's primary create-form.

    The decomposer uses these fields to generate accurate "fill the
    create form" sub-goals — e.g. for Solar's Create User drawer it
    knows the form has First Name + Last Name + Email + Phone + Role
    (not just "user form fields" like the BRD claimed).
    """
    label_of_trigger: str        # the button the Scout clicked to open
    drawer_title: str            # text in the drawer header
    fields: list[CapturedElement] = field(default_factory=list)
    primary_submit_label: str = ""   # "Save", "Create", "Submit"
    has_close_button: bool = True
    screenshot_png: bytes | None = None
    # Phase G.1 — captured nested structures inside the drawer.
    tree_structures: list[TreeStructure] = field(default_factory=list)
    resource_tables: list[ResourceTable] = field(default_factory=list)
    conditional_sections: list[ConditionalSection] = field(default_factory=list)
    # Phase G.2 — what kind of nav got us here. Set from the parent
    # nav walk when the surface was reached via a dropdown menu (e.g.
    # Administration → Roles where Administration itself isn't a page).
    nav_type: str = "page"            # "page" | "dropdown"
    nav_chain: list[str] = field(default_factory=list)  # ["Administration", "Roles"]


@dataclass
class ScoutResult:
    """Aggregate output of one authenticated-scout run."""
    target_url: str
    pages: list[ScoutedPage] = field(default_factory=list)
    landing_url: str = ""
    landing_title: str = ""
    notes: list[str] = field(default_factory=list)
    error_message: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    vision_calls: int = 0


# ── Page-summary JS (lightweight DOM scrape) ─────────────────────


# Pulls visible interactive elements with role + label + bbox. Cap
# at 80 elements per page (admin tables can have hundreds of rows —
# we only need the schema, not every cell).
_SCOUT_PAGE_JS = r"""
() => {
  const out = { title: document.title || '', elements: [] };
  const ROLES_SELECTORS = [
    ['button',  'button, [role=button], input[type=submit], input[type=button]'],
    ['link',    'a[href]'],
    ['textbox', 'input[type=text], input[type=email], input[type=tel], input[type=url], input[type=search], input[type=password], input:not([type]), textarea'],
    ['combobox','select, [role=combobox], [role=listbox]'],
    ['checkbox','input[type=checkbox], [role=checkbox]'],
    ['tab',     '[role=tab]'],
    ['menuitem','[role=menuitem]'],
    ['heading', 'h1, h2, h3, [role=heading]'],
    ['row',     'tr, [role=row]'],
  ];
  const seen = new Set();
  const labelOf = (el) => {
    const al = el.getAttribute('aria-label');
    if (al) return al.trim().slice(0, 200);
    const lt = el.getAttribute('title') || el.getAttribute('placeholder') || el.getAttribute('name');
    if (lt && lt.trim()) return lt.trim().slice(0, 200);
    const txt = (el.innerText || el.textContent || '').trim();
    return txt.slice(0, 200);
  };
  const isVisible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || parseFloat(cs.opacity) < 0.05) return false;
    return true;
  };
  for (const [role, sel] of ROLES_SELECTORS) {
    const nodes = document.querySelectorAll(sel);
    for (const el of nodes) {
      if (seen.has(el)) continue;
      seen.add(el);
      if (!isVisible(el)) continue;
      const r = el.getBoundingClientRect();
      const label = labelOf(el);
      if (!label && role === 'heading') continue;
      out.elements.push({
        role,
        label,
        rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
        // Best-effort selector — id wins, then data-testid, then nothing.
        hint_selector: el.id ? '#' + CSS.escape(el.id)
          : el.getAttribute('data-testid') ? '[data-testid="' + el.getAttribute('data-testid') + '"]'
          : null,
      });
      if (out.elements.length >= 80) return out;
    }
  }
  return out;
}
"""


# Heuristic phrases the Scout uses to find "this is a +Create button".
# Match-by-substring against element labels — case-insensitive.
_CREATE_TRIGGER_HINTS: tuple[str, ...] = (
    "add new", "+ add", "add ", "create new", "+ create", "create ",
    "+ new", "new role", "new user", "new project", "new ",
)
# Match-by-substring against element labels for "this is a submit
# button inside an open drawer". Case-insensitive.
_SUBMIT_HINTS: tuple[str, ...] = (
    "save", "create", "submit", "add", "confirm",
)
# Match-by-substring against element labels for nav items in the
# top-bar / side-bar — visited in order.
_NAV_ROLE_HINTS: tuple[str, ...] = (
    "link", "tab", "menuitem", "button",
)


def _norm_host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_same_host(a: str, b: str) -> bool:
    ah, bh = _norm_host(a), _norm_host(b)
    if not ah or not bh:
        return False
    if ah == bh:
        return True
    return ah.lstrip("www.") == bh.lstrip("www.")


def _capture_page(
    page: "Page",
    nav_path: list[str],
) -> ScoutedPage:
    """Pull title + interactive elements + screenshot for the current page."""
    from app.agents.page_intel import (  # noqa: PLC0415
        capture_screenshot_for_vision,
    )
    try:
        summary = page.evaluate(_SCOUT_PAGE_JS)
    except Exception as e:
        logger.debug("scout: page summary failed: %s", e)
        summary = {"title": "", "elements": []}

    raw_els = (summary.get("elements") or []) if isinstance(summary, dict) else []
    els: list[CapturedElement] = []
    for e in raw_els:
        if not isinstance(e, dict):
            continue
        rect = e.get("rect") or [0, 0, 0, 0]
        if not isinstance(rect, list) or len(rect) != 4:
            continue
        els.append(CapturedElement(
            role=str(e.get("role") or ""),
            label=str(e.get("label") or "")[:200],
            rect=(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])),
            hint_selector=(e.get("hint_selector") or None),
        ))

    try:
        shot = capture_screenshot_for_vision(page)
    except Exception as e:
        logger.debug("scout: screenshot failed: %s", e)
        shot = None

    try:
        cur_url = page.url
    except Exception:
        cur_url = ""

    title = ""
    if isinstance(summary, dict):
        title = str(summary.get("title", ""))[:200]

    return ScoutedPage(
        url=cur_url,
        title=title,
        nav_path=list(nav_path),
        elements=els,
        screenshot_png=shot,
    )


def _find_create_trigger(
    elements: list[CapturedElement],
) -> CapturedElement | None:
    """Pick the most likely "+Add"/"+Create" button on a list page.

    Heuristic order:
    1. button-role elements whose label contains a create-hint and
       appears in the top-right quadrant of the viewport (admin
       convention).
    2. button-role elements anywhere with a create-hint.
    3. None — no obvious create surface; Scout skips drawer capture.
    """
    candidates: list[tuple[int, CapturedElement]] = []
    for el in elements:
        if el.role != "button":
            continue
        label = el.label.lower()
        if not any(hint in label for hint in _CREATE_TRIGGER_HINTS):
            continue
        # Score: top-right preference. Higher x + lower y → higher score.
        score = el.rect[0] - el.rect[1] // 2
        candidates.append((score, el))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _find_submit_in_drawer(
    elements: list[CapturedElement],
    drawer_y_min: int = 0,
) -> CapturedElement | None:
    """Pick the most likely submit button in a freshly-opened drawer.

    Drawers in admin UIs put their primary submit button BOTTOM-RIGHT.
    We score by y (lower-on-page = higher score) + x (further-right =
    higher score), filtering to buttons with submit-hint labels.
    """
    candidates: list[tuple[int, CapturedElement]] = []
    for el in elements:
        if el.role != "button":
            continue
        label = el.label.lower().strip()
        if not label:
            continue
        if not any(label == h or label.startswith(h) for h in _SUBMIT_HINTS):
            continue
        if el.rect[1] < drawer_y_min:
            continue
        score = el.rect[0] + el.rect[1]  # bottom-right
        candidates.append((score, el))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


# Phase G.1 — detect permission-tree structures inside the active
# drawer / dialog. A "tree" is a vertical stack of disclosure rows
# (parent label + chevron / +/- icon) each with a checkbox and nested
# child checkboxes. We accept a few common shapes:
#
#   1. <li role="treeitem"> ... <input type=checkbox> ... </li>
#   2. role=tree / role=group containers
#   3. MUI TreeView (`.MuiTreeItem-root`)
#   4. AntD Tree (`.ant-tree-treenode`)
#   5. Generic disclosure pattern: a row with aria-expanded + a sibling
#      list of checkbox rows.
#
# Returns a list of trees with their parent labels + the rect of the
# whole tree container.
_DETECT_TREE_JS = r"""
() => {
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  };
  const labelOf = (el) => {
    const al = el.getAttribute('aria-label');
    if (al) return al.trim().slice(0, 120);
    const t = (el.innerText || el.textContent || '').trim();
    return t.split('\n')[0].slice(0, 120);
  };
  // Search inside the largest visible drawer / dialog (consistent
  // with the form-fill scanner), fall back to whole document.
  const drawerSel = [
    '[role=dialog]', '[role=alertdialog]',
    '.MuiDialog-paper', '.MuiDrawer-paper',
    '[class*="Drawer"]', '[class*="drawer"]',
    '[class*="Modal"]', '[class*="modal"]',
  ].join(',');
  const drawers = [...document.querySelectorAll(drawerSel)]
    .filter(VISIBLE);
  drawers.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return (br.width * br.height) - (ar.width * ar.height);
  });
  const root = drawers[0] || document.body;

  const out = [];

  // Shape A — explicit role=tree.
  const trees = [...root.querySelectorAll('[role=tree], .MuiTreeView-root, .ant-tree')]
    .filter(VISIBLE);
  for (const tree of trees) {
    const items = [...tree.querySelectorAll(
      '[role=treeitem], .MuiTreeItem-root, .ant-tree-treenode'
    )].filter(VISIBLE);
    // Top-level only — depth-1 nodes are the "modules" the decomposer
    // hands back to the agent.
    const tops = items.filter(it => {
      const par = it.parentElement && it.parentElement.closest(
        '[role=treeitem], .MuiTreeItem-root, .ant-tree-treenode'
      );
      return !par;
    });
    if (tops.length < 2) continue;
    const parents = tops.map(labelOf).filter(s => s.length > 0);
    const r = tree.getBoundingClientRect();
    out.push({
      label: '',
      parents: parents.slice(0, 30),
      has_expand_all: false,
      rect: [Math.round(r.left), Math.round(r.top),
             Math.round(r.width), Math.round(r.height)],
    });
  }

  // Shape B — disclosure-row pattern: rows with aria-expanded that
  // sit in a vertical stack with checkboxes. Catches Solar's roles
  // permissions UI (no role=tree attribute).
  if (out.length === 0) {
    const stacks = new Map();  // parent container → list of disclosure rows
    const rows = [...root.querySelectorAll('[aria-expanded]')].filter(VISIBLE);
    for (const r of rows) {
      // Must have at least one checkbox in its subtree (its own group
      // of permissions) — otherwise it's just an accordion.
      const hasCb = r.querySelector(
        'input[type=checkbox], [role=checkbox]'
      ) || (r.parentElement && r.parentElement.querySelector(
        'input[type=checkbox], [role=checkbox]'
      ));
      if (!hasCb) continue;
      // Group by the row's parent so we coalesce siblings into ONE
      // tree.
      const par = r.parentElement || root;
      const list = stacks.get(par) || [];
      list.push(r);
      stacks.set(par, list);
    }
    for (const [par, list] of stacks) {
      if (list.length < 2) continue;
      const parents = list.map(labelOf).filter(s => s.length > 0);
      const pr = par.getBoundingClientRect();
      out.push({
        label: '',
        parents: parents.slice(0, 30),
        has_expand_all: false,
        rect: [Math.round(pr.left), Math.round(pr.top),
               Math.round(pr.width), Math.round(pr.height)],
      });
    }
  }

  // Look for Expand All / Collapse All controls anywhere in the
  // drawer — flag the FIRST detected tree as having one.
  if (out.length > 0) {
    const buttons = [...root.querySelectorAll(
      'button, [role=button], a, [class*="expand"]'
    )].filter(VISIBLE);
    for (const b of buttons) {
      const t = ((b.innerText || b.textContent) || '').trim().toLowerCase();
      if (t.includes('expand all') || t.includes('expand-all') ||
          t === 'expand' || t.includes('collapse all')) {
        out[0].has_expand_all = true;
        break;
      }
    }
  }

  return out;
};
"""


# Phase G.1 — detect paginated resource tables inside the drawer.
# A "resource table" = <table>-like structure with:
#   1. Multiple rows of checkboxes (per-resource controls)
#   2. Optional column-header checkboxes (master toggles per column)
#   3. Optional pagination control nearby (Next / page numbers)
_DETECT_RESOURCE_TABLE_JS = r"""
() => {
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 100 || r.height < 60) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  };
  const textOf = (el) => {
    if (!el) return '';
    return ((el.innerText || el.textContent) || '').trim().slice(0, 120);
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

  const tables = [...root.querySelectorAll(
    'table, [role=table], [role=grid], .ant-table, .MuiTable-root'
  )].filter(VISIBLE);
  const out = [];
  for (const t of tables) {
    const rows = [...t.querySelectorAll(
      'tbody tr, [role=row]'
    )];
    if (rows.length < 2) continue;
    // Count checkbox-bearing rows.
    let cbRows = 0;
    for (const r of rows) {
      if (r.querySelector('input[type=checkbox], [role=checkbox]')) cbRows++;
    }
    if (cbRows < 2) continue;
    // Header columns: prefer <th> text, fall back to first row TDs
    // that LOOK like header labels (short, all-caps-ish).
    let headers = [...t.querySelectorAll('thead th, [role=columnheader]')]
      .map(textOf).filter(s => s.length > 0);
    if (headers.length === 0) {
      // Walk the first row.
      const first = rows[0];
      headers = [...first.children].map(textOf);
    }
    // Column masters: do any header cells contain a checkbox?
    const headerHasCb = !![...t.querySelectorAll(
      'thead input[type=checkbox], thead [role=checkbox], ' +
      '[role=columnheader] input[type=checkbox], [role=columnheader] [role=checkbox]'
    )].length;
    // Row label sample = first column text of the first 3 rows.
    const sampleLabels = rows.slice(0, 3).map(r => {
      const firstCell = r.querySelector('td, [role=cell]');
      return firstCell ? textOf(firstCell) : '';
    }).filter(s => s.length > 0);
    // Pagination = look for "Next"/"Previous"/page-number-button near
    // the table (within the next 2 siblings of the table OR within
    // its drawer container with .pagination class).
    let hasPag = false;
    const pagSel = [
      '[aria-label*="pagin" i]', '[class*="pagination" i]',
      '[class*="Pagination"]', '.MuiPagination-root', '.ant-pagination',
    ].join(',');
    const par = t.parentElement;
    if (par && par.querySelector(pagSel)) hasPag = true;
    if (!hasPag && par && par.parentElement &&
        par.parentElement.querySelector(pagSel)) hasPag = true;
    // Also check buttons with text "Next" / "Previous" near the table.
    if (!hasPag && par) {
      const btns = [...par.querySelectorAll('button, [role=button]')]
        .map(b => (b.innerText || '').toLowerCase());
      hasPag = btns.some(t =>
        t === 'next' || t === 'previous' || t === 'prev' ||
        /^\d+$/.test(t.trim())
      );
    }

    const r = t.getBoundingClientRect();
    out.push({
      label: '',
      columns: headers.slice(0, 12),
      row_label_sample: sampleLabels,
      has_pagination: hasPag,
      has_column_masters: headerHasCb,
      rect: [Math.round(r.left), Math.round(r.top),
             Math.round(r.width), Math.round(r.height)],
    });
  }
  return out;
};
"""


# Phase G.2 — after clicking a top-nav item, did a DROPDOWN appear?
# Signals: a popup element with role=menu / aria-orientation=vertical
# anchored under the clicked button, OR the URL didn't change but new
# menuitem/link elements just appeared just below the clicked button.
_DETECT_DROPDOWN_JS = r"""
(anchorRect) => {
  const ax = anchorRect[0], ay = anchorRect[1];
  const aw = anchorRect[2], ah = anchorRect[3];
  const VISIBLE = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  };
  // Look for an explicit menu popup.
  const menus = [...document.querySelectorAll(
    '[role=menu], [role=listbox], .MuiMenu-list, .ant-dropdown-menu'
  )].filter(VISIBLE);
  for (const m of menus) {
    const r = m.getBoundingClientRect();
    // Must be roughly under or near the anchor.
    if (r.top >= ay - 4 && r.top < ay + ah + 200) {
      // Pull menuitem labels.
      const items = [...m.querySelectorAll(
        '[role=menuitem], li, a, button'
      )].filter(VISIBLE);
      const labels = items.map(el =>
        ((el.innerText || el.textContent) || '').trim().slice(0, 80)
      ).filter(s => s.length > 0 && s.length < 80);
      // Dedupe.
      const dedup = [];
      for (const l of labels) if (!dedup.includes(l)) dedup.push(l);
      return { is_dropdown: true, items: dedup.slice(0, 12) };
    }
  }
  return { is_dropdown: false, items: [] };
};
"""


_SCROLLABLE_DRAWER_JS = r"""
(() => {
  // Find the dominant scrollable drawer / dialog and scroll it to
  // the bottom. Heuristic: pick the topmost element with role=dialog
  // OR a high z-index OR a fixed-position container whose
  // overflow-y suggests scrolling. Fall back to scrolling the page.
  const candidates = [
    ...document.querySelectorAll(
      '[role=dialog], [role=alertdialog], .MuiDialog-paper, ' +
      '.MuiDrawer-paper, [class*="Drawer"], [class*="drawer"], ' +
      '[class*="Modal"], [class*="modal"]'
    ),
  ].filter(el => {
    const r = el.getBoundingClientRect();
    if (r.width < 100 || r.height < 100) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden';
  });
  // Pick the largest visible drawer.
  candidates.sort((a, b) => {
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    return (br.width * br.height) - (ar.width * ar.height);
  });
  let target = candidates[0] || null;
  // If the candidate isn't internally scrollable, look for a
  // scrollable descendant.
  if (target) {
    let scroller = target;
    if (scroller.scrollHeight <= scroller.clientHeight + 4) {
      const inner = [...target.querySelectorAll('*')].find(el => {
        const cs = getComputedStyle(el);
        return (cs.overflowY === 'auto' || cs.overflowY === 'scroll')
          && el.scrollHeight > el.clientHeight + 4;
      });
      if (inner) scroller = inner;
    }
    scroller.scrollTop = scroller.scrollHeight;
    return true;
  }
  // No drawer found — scroll the document.
  window.scrollTo(0, document.documentElement.scrollHeight);
  return false;
})();
"""


def _try_open_create_surface(
    page: "Page",
    list_page: ScoutedPage,
    *,
    settle_ms: int = 600,
) -> CreateSurface | None:
    """Click the page's create-trigger and capture the resulting drawer.

    Best-effort — many admin pages don't have a create surface
    (read-only dashboards, log viewers). On any exception we return
    None and the Scout continues to the next nav item.
    """
    trigger = _find_create_trigger(list_page.elements)
    if trigger is None:
        return None

    # Try clicking via the hint selector first (stable), fall back to
    # text + coords. We use Playwright's get_by_text since the agent's
    # selector resolver is overkill for one click.
    clicked = False
    if trigger.hint_selector:
        try:
            page.locator(trigger.hint_selector).first.click(
                timeout=3_000,
            )
            clicked = True
        except Exception:
            pass
    if not clicked:
        try:
            page.get_by_text(trigger.label, exact=False).first.click(
                timeout=3_000,
            )
            clicked = True
        except Exception:
            pass
    if not clicked:
        # Last resort: click the bbox center.
        try:
            cx = trigger.rect[0] + trigger.rect[2] // 2
            cy = trigger.rect[1] + trigger.rect[3] // 2
            page.mouse.click(cx, cy)
            clicked = True
        except Exception:
            return None
    if not clicked:
        return None

    # Let the drawer settle.
    try:
        page.wait_for_timeout(settle_ms)
    except Exception:
        time.sleep(settle_ms / 1000.0)

    # Re-scrape — drawer should now be in the DOM.
    captured = _capture_page(page, list_page.nav_path + ["+create"])

    # Phase A.6 Step 2 — scroll the drawer to the bottom and re-scrape,
    # then merge element sets. Many admin forms push the Save button
    # below the fold on viewports with > 4 fields; without this the
    # scout's submit_label comes back empty and the runtime can't
    # find the button at all.
    captured_bottom: ScoutedPage | None = None
    try:
        page.evaluate(_SCROLLABLE_DRAWER_JS)
        try:
            page.wait_for_timeout(250)
        except Exception:
            pass
        captured_bottom = _capture_page(
            page, list_page.nav_path + ["+create", "scrolled"],
        )
    except Exception as e:
        logger.debug("scout: drawer scroll failed: %s", e)

    # Merge element lists — keep both sets but dedupe by (role, label).
    if captured_bottom is not None and captured_bottom.elements:
        seen: set[tuple[str, str]] = set(
            (e.role, e.label) for e in captured.elements
        )
        for e in captured_bottom.elements:
            key = (e.role, e.label)
            if key not in seen:
                captured.elements.append(e)
                seen.add(key)
        # Prefer the BOTTOM screenshot (shows the submit button) for
        # the create_surface record so the consolidator's prompt
        # actually sees Save.
        if captured_bottom.screenshot_png:
            captured.screenshot_png = captured_bottom.screenshot_png

    # Drawer fields: prefer textbox / combobox / checkbox in the
    # bottom 75% of the viewport (drawer slides up over the page).
    drawer_y_min = 0
    # Try to find the drawer's title (heading near the top of a
    # right-aligned drawer; or a "Create X" heading).
    drawer_title = ""
    for el in captured.elements:
        if el.role == "heading":
            label = el.label.lower()
            if any(h in label for h in _CREATE_TRIGGER_HINTS) or "create" in label:
                drawer_title = el.label
                drawer_y_min = el.rect[1]
                break

    # Form fields = textbox / combobox / checkbox / textarea in the
    # drawer's vertical band. If we couldn't find drawer_y_min we
    # use 0 (whole page; OK, the drawer dominates).
    fields = [
        e for e in captured.elements
        if e.role in ("textbox", "combobox", "checkbox")
        and e.rect[1] >= drawer_y_min
    ]

    # Phase A.6 Step 3 — probe textboxes with a clearly-fake value to
    # surface CONDITIONAL fields (a Role dropdown that only appears
    # after First Name is typed; a Phone country code that only
    # appears after Phone is filled). We type "qai-scout-probe-N"
    # into the first ~5 textboxes (skipping comboboxes which fire
    # onChange backend queries), capture after each, and merge any
    # new fields into the list.
    #
    # Safety: the probe value is unique + recognizable so accidental
    # backend hits show up in logs as "qai-scout-probe-..." values.
    # The drawer is closed via Escape afterwards so probe values
    # don't get persisted.
    probe_inputs = [
        f for f in fields if f.role == "textbox"
    ][:5]
    for probe_idx, probe_field in enumerate(probe_inputs):
        probe_value = f"qai-scout-probe-{probe_idx + 1}"
        try:
            # Focus + type via coords (cheap, no need for selector).
            cx = probe_field.rect[0] + probe_field.rect[2] // 2
            cy = probe_field.rect[1] + probe_field.rect[3] // 2
            page.mouse.click(cx, cy)
            try:
                page.wait_for_timeout(80)
            except Exception:
                pass
            page.keyboard.type(probe_value, delay=15)
            try:
                page.wait_for_timeout(200)
            except Exception:
                pass
            re_cap = _capture_page(
                page,
                list_page.nav_path + ["+create", f"probe{probe_idx + 1}"],
            )
            # Merge any NEW fields (not seen in current `fields`
            # list by (role, label) key) into the captured surface.
            seen_keys = {(f.role, f.label) for f in fields}
            new_fields_count = 0
            for el in re_cap.elements:
                if el.role not in ("textbox", "combobox", "checkbox"):
                    continue
                if el.rect[1] < drawer_y_min:
                    continue
                key = (el.role, el.label)
                if key in seen_keys:
                    continue
                fields.append(el)
                seen_keys.add(key)
                new_fields_count += 1
            if new_fields_count > 0:
                logger.debug(
                    "scout probe %s revealed %s new fields",
                    probe_idx + 1, new_fields_count,
                )
        except Exception as e:
            logger.debug("scout: probe %s failed: %s", probe_idx + 1, e)

    submit = _find_submit_in_drawer(captured.elements, drawer_y_min)

    # Phase G.1 — detect nested structures inside the drawer.
    tree_structures: list[TreeStructure] = []
    resource_tables: list[ResourceTable] = []
    try:
        raw_trees = page.evaluate(_DETECT_TREE_JS)
        if isinstance(raw_trees, list):
            for t in raw_trees:
                if not isinstance(t, dict):
                    continue
                rect_l = t.get("rect") or [0, 0, 0, 0]
                tree_structures.append(TreeStructure(
                    label=str(t.get("label") or ""),
                    parents=[
                        str(p) for p in (t.get("parents") or [])
                    ][:30],
                    has_expand_all=bool(t.get("has_expand_all")),
                    rect=(
                        int(rect_l[0]), int(rect_l[1]),
                        int(rect_l[2]), int(rect_l[3]),
                    ) if len(rect_l) == 4 else (0, 0, 0, 0),
                ))
    except Exception as e:
        logger.debug("scout: tree detection failed: %s", e)
    try:
        raw_tables = page.evaluate(_DETECT_RESOURCE_TABLE_JS)
        if isinstance(raw_tables, list):
            for rt in raw_tables:
                if not isinstance(rt, dict):
                    continue
                rect_l = rt.get("rect") or [0, 0, 0, 0]
                resource_tables.append(ResourceTable(
                    label=str(rt.get("label") or ""),
                    columns=[
                        str(c) for c in (rt.get("columns") or [])
                    ][:12],
                    row_label_sample=[
                        str(s) for s in (rt.get("row_label_sample") or [])
                    ][:3],
                    has_pagination=bool(rt.get("has_pagination")),
                    has_column_masters=bool(rt.get("has_column_masters")),
                    rect=(
                        int(rect_l[0]), int(rect_l[1]),
                        int(rect_l[2]), int(rect_l[3]),
                    ) if len(rect_l) == 4 else (0, 0, 0, 0),
                ))
    except Exception as e:
        logger.debug("scout: resource-table detection failed: %s", e)

    # Phase G.5 — conditional section probe. Phase A.6 already
    # records new INDIVIDUAL fields revealed by probing. G.5 goes
    # further: if a probe reveals an entire new SECTION (i.e. a
    # heading we didn't have before + multiple fields under it), we
    # record it as a ConditionalSection so the decomposer knows the
    # ordering constraint (e.g. "fill role first, THEN configure
    # resource access").
    conditional_sections: list[ConditionalSection] = []
    pre_headings = {
        e.label for e in captured.elements if e.role == "heading"
    }
    try:
        post_scan = page.evaluate(_SCOUT_PAGE_JS)
    except Exception:
        post_scan = None
    if isinstance(post_scan, dict):
        post_headings: list[CapturedElement] = []
        for raw in (post_scan.get("elements") or []):
            if not isinstance(raw, dict):
                continue
            if raw.get("role") != "heading":
                continue
            lab = str(raw.get("label") or "").strip()
            if not lab or lab in pre_headings:
                continue
            rect_l = raw.get("rect") or [0, 0, 0, 0]
            post_headings.append(CapturedElement(
                role="heading",
                label=lab,
                rect=(
                    int(rect_l[0]), int(rect_l[1]),
                    int(rect_l[2]), int(rect_l[3]),
                ) if len(rect_l) == 4 else (0, 0, 0, 0),
            ))
        for h in post_headings:
            nearby_fields = [
                e for e in fields
                if e.rect[1] >= h.rect[1] - 8
                and e.rect[1] <= h.rect[1] + 400
            ]
            if len(nearby_fields) >= 1:
                conditional_sections.append(ConditionalSection(
                    label=h.label,
                    trigger_field_label=(
                        probe_inputs[0].label if probe_inputs else ""
                    ),
                    trigger_value="(probe)",
                    new_fields=nearby_fields[:10],
                ))

    surface = CreateSurface(
        label_of_trigger=trigger.label,
        drawer_title=drawer_title or f"Create from '{trigger.label}'",
        fields=fields[:40],
        primary_submit_label=(submit.label if submit else ""),
        screenshot_png=captured.screenshot_png,
        tree_structures=tree_structures,
        resource_tables=resource_tables,
        conditional_sections=conditional_sections,
    )

    # Close the drawer so the Scout returns to a clean state for the
    # next nav item.
    _try_close_drawer(page)
    return surface


def _try_close_drawer(page: "Page") -> None:
    """Best-effort drawer dismissal — Escape key, then look for
    Cancel / Close button. Failures are non-fatal."""
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(150)
    except Exception:
        pass
    for label in ("Cancel", "Close", "Discard", "X"):
        try:
            page.get_by_role("button", name=label).first.click(timeout=500)
            return
        except Exception:
            continue


def _find_top_nav_items(
    elements: list[CapturedElement],
    *,
    max_items: int = 8,
) -> list[CapturedElement]:
    """Pick the top-level nav items from a captured landing page.

    Heuristic: links + buttons + tabs + menuitems in the TOP STRIP
    of the viewport (y < 120). Ranked by label length asc (short =
    "Administration", "Management"; long = breadcrumbs / random
    inline text).
    """
    candidates: list[CapturedElement] = []
    for el in elements:
        if el.role not in _NAV_ROLE_HINTS:
            continue
        if el.rect[1] > 120:
            continue
        if not el.label.strip():
            continue
        # Skip generic strings that look like icons.
        if len(el.label) < 2:
            continue
        # De-dup by label — the same nav might be wrapped in a button
        # AND a link (a11y workaround for some shells).
        if any(c.label == el.label for c in candidates):
            continue
        candidates.append(el)
    candidates.sort(key=lambda e: len(e.label))
    return candidates[:max_items]


def run_authenticated_scout(
    page: "Page",
    *,
    target_url: str,
    depth: ScoutDepth = "deep",
    max_pages: int = 12,
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    submodule_run_id: int | None = None,
) -> ScoutResult:
    """Walk the post-auth surface and collect :class:`ScoutedPage` records.

    Expects the page to ALREADY be logged in (call after
    ``auth_flow.run_auth_loop`` returns status="ok"). Walks the
    landing's top nav; on "deep" mode also opens each list page's
    primary create-trigger to capture the drawer/form structure.

    The :class:`ScoutResult` is fed into the AppMap consolidator
    which produces the structured map the decomposer queries.

    Failures here are non-fatal — partial results are returned with
    ``error_message`` set so the qa_agent can fall through to legacy
    behavior (no mindmap, BRD-only sub-goals).
    """
    out = ScoutResult(target_url=target_url)

    def _emit(t: str, d: dict) -> None:
        if emit_event:
            try:
                emit_event(t, d)
            except Exception:
                pass

    _emit("auth_scout_started", {
        "run_id": submodule_run_id,
        "target_url": target_url,
        "depth": depth,
    })

    # Capture the landing — this is the first page post-login.
    try:
        page.wait_for_load_state(
            "domcontentloaded", timeout=10_000,
        )
    except Exception:
        pass
    try:
        page.wait_for_load_state(
            "networkidle", timeout=6_000,
        )
    except Exception:
        pass

    landing = _capture_page(page, nav_path=["(landing)"])
    out.landing_url = landing.url
    out.landing_title = landing.title
    out.pages.append(landing)
    _emit("auth_scout_page", {
        "run_id": submodule_run_id,
        "url": landing.url,
        "title": landing.title,
        "elements": len(landing.elements),
        "nav_path": landing.nav_path,
    })

    # Identify top-level nav.
    nav_items = _find_top_nav_items(landing.elements)
    if not nav_items:
        out.notes.append(
            "no top-level navigation detected on landing page",
        )

    visited_labels: set[str] = set()

    for nav_el in nav_items:
        if is_cancelled and is_cancelled():
            out.error_message = "cancelled mid-scout"
            return out
        if len(out.pages) >= max_pages:
            out.notes.append(
                f"scout halted at max_pages={max_pages}",
            )
            break
        label_key = nav_el.label.strip().lower()
        if label_key in visited_labels:
            continue
        visited_labels.add(label_key)

        # Click the nav item.
        clicked = False
        for attempt_selector in (
            nav_el.hint_selector,
            None,  # text fallback
        ):
            try:
                if attempt_selector:
                    page.locator(attempt_selector).first.click(
                        timeout=3_000,
                    )
                else:
                    page.get_by_text(
                        nav_el.label, exact=False,
                    ).first.click(timeout=3_000)
                clicked = True
                break
            except Exception:
                continue
        if not clicked:
            # Click the bbox center as a last resort.
            try:
                cx = nav_el.rect[0] + nav_el.rect[2] // 2
                cy = nav_el.rect[1] + nav_el.rect[3] // 2
                page.mouse.click(cx, cy)
                clicked = True
            except Exception:
                logger.debug(
                    "scout: click failed for %r", nav_el.label,
                )
                continue

        try:
            page.wait_for_load_state("networkidle", timeout=5_000)
        except Exception:
            pass

        # Phase G.2 — detect dropdown menu. If the click exposed a
        # role=menu popup anchored under the trigger, this nav item
        # is a DROPDOWN, not a page. Tag downstream surfaces so the
        # decomposer emits a two-step "open menu → click item"
        # navigation sub-goal.
        nav_is_dropdown = False
        try:
            dd_res = page.evaluate(
                _DETECT_DROPDOWN_JS, list(nav_el.rect),
            )
            if isinstance(dd_res, dict) and dd_res.get("is_dropdown"):
                nav_is_dropdown = True
        except Exception:
            pass

        # Some nav items are MENUS — clicking exposes a sub-menu we
        # should also walk. Capture and check for fresh elements.
        section_page = _capture_page(page, nav_path=[nav_el.label])
        out.pages.append(section_page)
        _emit("auth_scout_page", {
            "run_id": submodule_run_id,
            "url": section_page.url,
            "title": section_page.title,
            "nav_path": section_page.nav_path,
            "elements": len(section_page.elements),
        })

        # If the nav item opened a SUBMENU (e.g. Administration → Roles
        # + Users), there are new menuitem-style elements visible that
        # weren't on the landing. Walk those too, one level deep.
        new_menuitems: list[CapturedElement] = []
        landing_labels = {e.label for e in landing.elements}
        for sm in section_page.elements:
            if sm.role not in ("menuitem", "link", "button"):
                continue
            if sm.label in landing_labels:
                continue
            if sm.label.strip().lower() in visited_labels:
                continue
            # Sub-menu items typically appear right under the clicked
            # nav (top-strip + just below). Filter to y < 220.
            if sm.rect[1] > 220:
                continue
            if not sm.label.strip():
                continue
            new_menuitems.append(sm)

        for sub_el in new_menuitems[:6]:
            if is_cancelled and is_cancelled():
                out.error_message = "cancelled mid-scout"
                return out
            if len(out.pages) >= max_pages:
                break
            sub_key = sub_el.label.strip().lower()
            if sub_key in visited_labels:
                continue
            visited_labels.add(sub_key)
            try:
                if sub_el.hint_selector:
                    page.locator(sub_el.hint_selector).first.click(
                        timeout=3_000,
                    )
                else:
                    page.get_by_text(
                        sub_el.label, exact=False,
                    ).first.click(timeout=3_000)
            except Exception:
                continue
            try:
                page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception:
                pass
            leaf_page = _capture_page(
                page, nav_path=[nav_el.label, sub_el.label],
            )
            out.pages.append(leaf_page)
            _emit("auth_scout_page", {
                "run_id": submodule_run_id,
                "url": leaf_page.url,
                "title": leaf_page.title,
                "nav_path": leaf_page.nav_path,
                "elements": len(leaf_page.elements),
            })

            # On "deep" mode, try opening the primary create-trigger
            # on this leaf to capture the drawer's form fields.
            if depth == "deep":
                surface = _try_open_create_surface(page, leaf_page)
                if surface is not None:
                    # Phase G.2 — tag nav type so the AppMap consolidator
                    # + decomposer know to emit a two-step "open menu →
                    # click item" navigation sub-goal.
                    if nav_is_dropdown:
                        surface.nav_type = "dropdown"
                    surface.nav_chain = [nav_el.label, sub_el.label]
                    leaf_page.create_surface = surface
                    _emit("auth_scout_create_captured", {
                        "run_id": submodule_run_id,
                        "nav_path": leaf_page.nav_path,
                        "trigger": surface.label_of_trigger,
                        "fields": len(surface.fields),
                        "submit_label": surface.primary_submit_label,
                        "trees": len(surface.tree_structures),
                        "resource_tables": len(surface.resource_tables),
                        "conditional_sections": len(
                            surface.conditional_sections,
                        ),
                        "nav_type": surface.nav_type,
                    })

    _emit("auth_scout_completed", {
        "run_id": submodule_run_id,
        "pages_captured": len(out.pages),
        "create_surfaces": sum(
            1 for p in out.pages if p.create_surface is not None
        ),
    })
    return out
