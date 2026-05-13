"""Phase A.5 — AppMap consolidator + persistence.

After the authenticated Scout collects :class:`ScoutedPage` records,
this module:

1. Consolidates them into a structured :class:`AppMap` via ONE
   strong-tier LLM call. The LLM reads the page-summary JSON for
   each scouted page (no images — text-only is enough since each
   page already has labelled elements) and emits a structured map.

2. Persists the map to the AKB as a single row with
   ``kind="app_map"`` and content = ``json.dumps(map.to_dict())``.
   Re-saves OVERWRITE the existing row (we keep ONE current map per
   target_url; previous versions are not retained).

3. Provides :func:`load_app_map` for the decomposer to read the map
   at submodule start.

Why one VL call vs. per-page calls
----------------------------------
The scout has already classified element roles + extracted labels +
bounding boxes. We don't need the LLM to do that work again — we
need it to RECOGNIZE PATTERNS across pages: "this app has an
Administration → Roles list with a +Add New Role button that opens
a drawer with Name + Display Name + permissions tree". One call
keeps cost predictable and gives the LLM full cross-page context to
spot the create→list→verify pattern.

Cost
----
Input: ~3-6 KB of structured JSON (per-page summaries). Output:
~2-3 KB structured map. ~$0.005-0.015 on a strong-tier provider.
Cached for all subsequent runs against the same target_url.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.agents.authenticated_scout import ScoutResult
    from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class FormFieldSpec:
    """One field the consolidator extracted from a create-surface."""
    label: str
    role: str  # "textbox" | "combobox" | "checkbox" | "textarea"
    required: bool = False


@dataclass
class TreeSpec:
    """Phase G.1/G.2 — a permission tree referenced from a create-flow."""
    label: str = ""
    parents: list[str] = field(default_factory=list)
    has_expand_all: bool = False


@dataclass
class ResourceTableSpec:
    """Phase G.1/G.2 — a paginated checkbox resource table."""
    label: str = ""
    columns: list[str] = field(default_factory=list)
    row_label_sample: list[str] = field(default_factory=list)
    has_pagination: bool = False
    has_column_masters: bool = False


@dataclass
class ConditionalSectionSpec:
    """Phase G.5 — a section that only appears after a trigger field
    is filled. The decomposer uses ``trigger_field_label`` to order
    sub-goals: fill trigger first, then handle the new section."""
    label: str = ""
    trigger_field_label: str = ""
    new_field_labels: list[str] = field(default_factory=list)


@dataclass
class CreateFlowSpec:
    """A repeatable "create X" pattern the app exposes.

    Built from one or more :class:`CreateSurface` captures (the same
    create-flow may show up on multiple list pages). The decomposer
    uses this directly when a sub-goal is "create a new <entity>".
    """
    entity: str                          # "Role", "User", "Project"
    section_path: list[str]              # ["Administration", "Roles"]
    trigger_label: str                   # "+ Add New Role"
    submit_label: str                    # "Save", "Create"
    fields: list[FormFieldSpec] = field(default_factory=list)
    # When True, the list page has a search/filter input the agent
    # should use to verify the new entity instead of scrolling.
    list_has_search: bool = False
    # Heuristic: when True, the form contains a tree-like permission
    # selector (Solar's role drawer). Decomposer emits "expand parent
    # → check leaves" sub-goals instead of "select all".
    has_permission_tree: bool = False
    # Phase G.2 — how to reach this flow. "page" = direct nav; "dropdown"
    # = open a dropdown menu first, then click the leaf.
    nav_type: str = "page"
    # Phase G.1 — captured nested structures (when has_permission_tree
    # or has_resource_table is true the agent reads these to dispatch
    # the right widget handler).
    trees: list[TreeSpec] = field(default_factory=list)
    resource_tables: list[ResourceTableSpec] = field(default_factory=list)
    conditional_sections: list[ConditionalSectionSpec] = field(default_factory=list)
    has_resource_table: bool = False
    has_conditional_section: bool = False


@dataclass
class ModuleSpec:
    """One top-level module in the app's nav."""
    name: str                            # "Administration"
    sections: list[str] = field(default_factory=list)
    landing_url: str = ""
    notes: str = ""
    # Phase G.2 — "page" = the module IS a destination URL; "dropdown"
    # = clicking it expands a menu and you click a section item next.
    nav_type: str = "page"


@dataclass
class AppMap:
    """The consolidated mindmap of an authenticated app.

    Stored as one AKB row per target_url with kind="app_map" + JSON
    content. Reloaded at submodule start so the decomposer's sub-goal
    plans align with the REAL UI.
    """
    target_url: str
    landing_url: str = ""
    landing_title: str = ""
    modules: list[ModuleSpec] = field(default_factory=list)
    create_flows: list[CreateFlowSpec] = field(default_factory=list)
    # Free-text notes — anything the LLM noticed that doesn't fit
    # the structured fields. Rendered into the decomposer's prompt
    # so cross-cutting UX observations (e.g. "Save buttons are
    # always bottom-right of the drawer") are available.
    cross_cutting_notes: list[str] = field(default_factory=list)
    # Metadata
    pages_scouted: int = 0
    scout_depth: str = "deep"
    scout_version: int = 1
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AppMap":
        modules: list[ModuleSpec] = []
        for m in (d.get("modules") or []):
            if not isinstance(m, dict):
                continue
            modules.append(ModuleSpec(
                name=str(m.get("name", "")),
                sections=[str(s) for s in (m.get("sections") or [])],
                landing_url=str(m.get("landing_url", "")),
                notes=str(m.get("notes", "")),
                nav_type=str(m.get("nav_type", "page")),
            ))
        flows: list[CreateFlowSpec] = []
        for f in (d.get("create_flows") or []):
            if not isinstance(f, dict):
                continue
            fields_list = [
                FormFieldSpec(**ff) for ff in (f.get("fields") or [])
                if isinstance(ff, dict)
            ]
            trees_list = [
                TreeSpec(
                    label=str(t.get("label", "")),
                    parents=[str(p) for p in (t.get("parents") or [])],
                    has_expand_all=bool(t.get("has_expand_all", False)),
                )
                for t in (f.get("trees") or [])
                if isinstance(t, dict)
            ]
            tables_list = [
                ResourceTableSpec(
                    label=str(t.get("label", "")),
                    columns=[str(c) for c in (t.get("columns") or [])],
                    row_label_sample=[
                        str(s) for s in (t.get("row_label_sample") or [])
                    ],
                    has_pagination=bool(t.get("has_pagination", False)),
                    has_column_masters=bool(
                        t.get("has_column_masters", False),
                    ),
                )
                for t in (f.get("resource_tables") or [])
                if isinstance(t, dict)
            ]
            cond_list = [
                ConditionalSectionSpec(
                    label=str(c.get("label", "")),
                    trigger_field_label=str(
                        c.get("trigger_field_label", ""),
                    ),
                    new_field_labels=[
                        str(x) for x in (c.get("new_field_labels") or [])
                    ],
                )
                for c in (f.get("conditional_sections") or [])
                if isinstance(c, dict)
            ]
            flows.append(CreateFlowSpec(
                entity=str(f.get("entity", "")),
                section_path=list(f.get("section_path") or []),
                trigger_label=str(f.get("trigger_label", "")),
                submit_label=str(f.get("submit_label", "")),
                fields=fields_list,
                list_has_search=bool(f.get("list_has_search", False)),
                # Roll-up flags: prefer the LLM's read, but also OR in
                # the structural fact (a non-empty list trivially implies
                # the flag) so the agent isn't tripped by an LLM omission.
                has_permission_tree=bool(
                    f.get("has_permission_tree", False),
                ) or bool(trees_list),
                nav_type=str(f.get("nav_type", "page")),
                trees=trees_list,
                resource_tables=tables_list,
                conditional_sections=cond_list,
                has_resource_table=bool(
                    f.get("has_resource_table", False),
                ) or bool(tables_list),
                has_conditional_section=bool(
                    f.get("has_conditional_section", False),
                ) or bool(cond_list),
            ))
        return cls(
            target_url=str(d.get("target_url", "")),
            landing_url=str(d.get("landing_url", "")),
            landing_title=str(d.get("landing_title", "")),
            modules=modules,
            create_flows=flows,
            cross_cutting_notes=[
                str(n) for n in (d.get("cross_cutting_notes") or [])
            ],
            pages_scouted=int(d.get("pages_scouted", 0)),
            scout_depth=str(d.get("scout_depth", "deep")),
            scout_version=int(d.get("scout_version", 1)),
            reasoning=str(d.get("reasoning", "")),
        )

    def format_for_prompt(self) -> str:
        """Render the map as the decomposer's prompt context block.

        Compact enough to fit alongside the goal + screenshot without
        blowing the planner's input budget. Each section is dropped
        when empty so unused fields don't waste tokens.
        """
        lines: list[str] = [
            f"APP: {self.landing_title or self.target_url}",
            f"  landing: {self.landing_url}",
        ]
        if self.modules:
            lines.append("MODULES:")
            for m in self.modules:
                section_str = (
                    " → " + " | ".join(m.sections) if m.sections else ""
                )
                nav_marker = (
                    " [dropdown]" if m.nav_type == "dropdown" else ""
                )
                lines.append(f"  - {m.name}{nav_marker}{section_str}")
                if m.notes:
                    lines.append(f"    note: {m.notes}")
        if self.create_flows:
            lines.append("CREATE FLOWS:")
            for fl in self.create_flows:
                path = " > ".join(fl.section_path) or "(unknown)"
                nav_marker = (
                    " (via dropdown)" if fl.nav_type == "dropdown" else ""
                )
                lines.append(
                    f"  - Create {fl.entity} at [{path}]{nav_marker}: "
                    f"trigger=\"{fl.trigger_label}\" "
                    f"submit=\"{fl.submit_label}\""
                )
                if fl.fields:
                    fld_str = ", ".join(
                        f"{f.label}({f.role})"
                        + ("*" if f.required else "")
                        for f in fl.fields[:10]
                    )
                    lines.append(f"    fields: {fld_str}")
                flags: list[str] = []
                if fl.list_has_search:
                    flags.append("searchable list")
                if fl.has_permission_tree:
                    flags.append("permission tree")
                if fl.has_resource_table:
                    flags.append("paginated resource table")
                if fl.has_conditional_section:
                    flags.append("conditional sections")
                if flags:
                    lines.append(f"    flags: {', '.join(flags)}")
                # Phase G.1 — surface tree parents so the decomposer
                # knows the module names it must expand + check.
                for t in fl.trees:
                    if t.parents:
                        lines.append(
                            "    tree parents: "
                            + ", ".join(t.parents[:12])
                            + (" [Expand All available]"
                               if t.has_expand_all else "")
                        )
                for rt in fl.resource_tables:
                    cols = ", ".join(rt.columns[:6]) or "(no columns)"
                    sample = (
                        " sample=" + " / ".join(rt.row_label_sample[:3])
                        if rt.row_label_sample else ""
                    )
                    pag = " [paginated]" if rt.has_pagination else ""
                    mast = " [column masters]" if rt.has_column_masters else ""
                    lines.append(
                        f"    resource table cols: {cols}{sample}"
                        f"{pag}{mast}"
                    )
                for cs in fl.conditional_sections:
                    new_fs = ", ".join(cs.new_field_labels[:6])
                    lines.append(
                        f"    conditional section \"{cs.label}\" "
                        f"appears after \"{cs.trigger_field_label}\""
                        + (f"; fields: {new_fs}" if new_fs else "")
                    )
        if self.cross_cutting_notes:
            lines.append("PATTERNS:")
            for n in self.cross_cutting_notes[:6]:
                lines.append(f"  - {n}")
        return "\n".join(lines)

    def create_flow_for_entity(
        self, entity_keyword: str,
    ) -> "CreateFlowSpec | None":
        """Find the create-flow matching an entity keyword (case-insensitive).

        Used by the decomposer when a sub-goal mentions creating
        something — e.g. ``entity_keyword="role"`` matches the
        ``Create Role`` flow even when the BRD says "make a new role".
        """
        k = entity_keyword.lower().strip()
        if not k:
            return None
        for f in self.create_flows:
            if f.entity.lower() == k:
                return f
        for f in self.create_flows:
            if k in f.entity.lower() or f.entity.lower() in k:
                return f
        return None


# ── Strict JSON schema for the consolidator ──────────────────────


_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "role": {
            "type": "string",
            "enum": ["textbox", "combobox", "checkbox", "textarea"],
        },
        "required": {"type": "boolean"},
    },
    "required": ["label", "role", "required"],
    "additionalProperties": False,
}

_TREE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "parents": {"type": "array", "items": {"type": "string"}},
        "has_expand_all": {"type": "boolean"},
    },
    "required": ["label", "parents", "has_expand_all"],
    "additionalProperties": False,
}

_RES_TABLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "columns": {"type": "array", "items": {"type": "string"}},
        "row_label_sample": {
            "type": "array", "items": {"type": "string"},
        },
        "has_pagination": {"type": "boolean"},
        "has_column_masters": {"type": "boolean"},
    },
    "required": [
        "label", "columns", "row_label_sample",
        "has_pagination", "has_column_masters",
    ],
    "additionalProperties": False,
}

_COND_SECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "trigger_field_label": {"type": "string"},
        "new_field_labels": {
            "type": "array", "items": {"type": "string"},
        },
    },
    "required": [
        "label", "trigger_field_label", "new_field_labels",
    ],
    "additionalProperties": False,
}

_CREATE_FLOW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entity": {"type": "string"},
        "section_path": {
            "type": "array",
            "items": {"type": "string"},
        },
        "trigger_label": {"type": "string"},
        "submit_label": {"type": "string"},
        "fields": {
            "type": "array",
            "items": _FIELD_SCHEMA,
        },
        "list_has_search": {"type": "boolean"},
        "has_permission_tree": {"type": "boolean"},
        "nav_type": {"type": "string", "enum": ["page", "dropdown"]},
        "trees": {"type": "array", "items": _TREE_SCHEMA},
        "resource_tables": {
            "type": "array", "items": _RES_TABLE_SCHEMA,
        },
        "conditional_sections": {
            "type": "array", "items": _COND_SECTION_SCHEMA,
        },
        "has_resource_table": {"type": "boolean"},
        "has_conditional_section": {"type": "boolean"},
    },
    "required": [
        "entity", "section_path", "trigger_label", "submit_label",
        "fields", "list_has_search", "has_permission_tree",
        "nav_type", "trees", "resource_tables", "conditional_sections",
        "has_resource_table", "has_conditional_section",
    ],
    "additionalProperties": False,
}

_MODULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {"type": "string"},
        },
        "landing_url": {"type": "string"},
        "notes": {"type": "string"},
        "nav_type": {"type": "string", "enum": ["page", "dropdown"]},
    },
    "required": [
        "name", "sections", "landing_url", "notes", "nav_type",
    ],
    "additionalProperties": False,
}

APP_MAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "modules": {"type": "array", "items": _MODULE_SCHEMA},
        "create_flows": {
            "type": "array",
            "items": _CREATE_FLOW_SCHEMA,
        },
        "cross_cutting_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
        "confidence": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
        },
    },
    "required": [
        "modules", "create_flows", "cross_cutting_notes",
        "reasoning", "confidence",
    ],
    "additionalProperties": False,
}


CONSOLIDATOR_SYSTEM_PROMPT = """You are a senior QA architect. You receive a JSON
dump of pages a Scout walked through in an authenticated web app —
each page has a URL, a title, a navigation path (e.g. ["Administration",
"Roles"]), and a list of visible interactive elements with their role
+ label + bounding box. Some leaf pages also carry a CREATE_SURFACE
recording what happened when the Scout clicked the page's primary
+Add/+Create button, including any nested permission TREES, paginated
RESOURCE_TABLES, or conditional sub-SECTIONS the Scout detected.

Your job: produce a structured AppMap that captures:

1. MODULES — top-level sections of the app's main navigation. For
   each, list the immediate child sections / pages. Set
   ``nav_type="dropdown"`` when the module ISN'T a destination page but
   a menu that expands on click to reveal sub-items (e.g. Solar's
   "Administration" → opens a dropdown with Roles / Users / etc).
   Set ``nav_type="page"`` for direct-navigate modules.
2. CREATE_FLOWS — every distinct "create an X" pattern observed.
   For each one, record:
   - entity: short noun ("Role", "User", "Project", "Chainage")
   - section_path: where to find the create-trigger ["Administration", "Roles"]
   - trigger_label: the exact text of the button that opens the form
   - submit_label: the exact text of the button that persists the form
   - fields: the form's visible inputs (textbox / combobox / checkbox /
     textarea). Mark required=true when the field had a visible
     required marker (asterisk, "Required" text, red border).
   - list_has_search: true when the list page has a search / filter
     input above the table.
   - has_permission_tree: true when the form contains a tree-shaped
     permission selector with expandable parent nodes + child checkboxes.
   - nav_type: "dropdown" when reaching this flow requires opening a
     dropdown menu first (parent module is a dropdown), otherwise "page".
   - trees: ONE entry PER tree captured. ``parents`` are the top-level
     node labels the agent will need to expand + tick. ``has_expand_all``
     is true when the drawer has an "Expand All" affordance.
   - resource_tables: ONE entry PER paginated checkbox table inside
     the drawer (e.g. Solar's Resource Access Control). ``columns`` are
     the action column headers (read / update / delete). ``has_pagination``
     and ``has_column_masters`` reflect the table's structure.
   - conditional_sections: ONE entry PER section that only appears after
     a trigger field is filled. The agent will fill the trigger first,
     THEN the new fields.
   - has_resource_table / has_conditional_section: roll-up booleans
     mirroring the lists above; set true when the corresponding list
     is non-empty.
3. CROSS_CUTTING_NOTES — short, factual observations that apply to
   the WHOLE app, not one specific page. Examples:
   - "Save buttons appear bottom-right of all drawer forms"
   - "List pages show a Total count badge above the table"
   - "Status badges are color-coded (green=Active, gray=Inactive)"
   These help the agent verify creation succeeded.

Rules
=====
- Be CONSERVATIVE: only emit a CREATE_FLOW when the Scout actually
  captured a create_surface for it (you'll see fields populated). Do
  not invent flows from list pages alone.
- Use the EXACT visible labels from the Scout's element dump. If
  the trigger says "+ Add New Role", that's what trigger_label says.
- For fields, prefer the input's visible label or placeholder over
  its internal name. "Email Address" beats "user_email".
- For tree/table/conditional structures: re-emit what the Scout
  passed you. Do not invent parents or columns; if the Scout's array
  is empty, return an empty array. The runtime's heuristics are the
  source of truth.
- ``confidence`` is your self-assessment of how complete the map is.
  < 0.6 if the scout missed sections or pages have empty element lists.

Output: strict JSON matching the schema. No prose outside the JSON.
"""


def consolidate_app_map(
    provider: "LLMProvider",
    *,
    scout_result: "ScoutResult",
    cheap_provider: "LLMProvider | None" = None,
    on_escalate: Callable[[str, str, str], None] | None = None,
) -> tuple[AppMap, int, int]:
    """Run the consolidator LLM call.

    Returns ``(AppMap, input_tokens, output_tokens)``. The AppMap is
    minimally populated when the LLM call fails — landing URL/title +
    pages_scouted are filled from the scout result so the decomposer
    at least knows the scout ran.
    """
    from app.agents.authenticated_scout import (  # noqa: PLC0415
        ScoutedPage,
    )
    from app.llm.base import ChatMessage  # noqa: PLC0415
    from app.llm.router import (  # noqa: PLC0415
        LLMRole, call_for_role,
    )

    # Build a compact JSON dump of the scouted pages. Drop the
    # screenshot bytes (we're text-only here) and trim element
    # lists to the most useful 30 per page so the input stays small.
    def _page_dump(p: ScoutedPage) -> dict[str, Any]:
        d: dict[str, Any] = {
            "url": p.url,
            "title": p.title,
            "nav_path": list(p.nav_path),
            "elements": [
                {
                    "role": e.role,
                    "label": e.label,
                    "rect": list(e.rect),
                }
                for e in p.elements[:30]
            ],
        }
        if p.create_surface is not None:
            cs = p.create_surface
            d["create_surface"] = {
                "trigger_label": cs.label_of_trigger,
                "drawer_title": cs.drawer_title,
                "submit_label": cs.primary_submit_label,
                "nav_type": getattr(cs, "nav_type", "page"),
                "nav_chain": list(getattr(cs, "nav_chain", [])),
                "fields": [
                    {
                        "role": f.role,
                        "label": f.label,
                    }
                    for f in cs.fields[:25]
                ],
                "trees": [
                    {
                        "label": t.label,
                        "parents": list(t.parents),
                        "has_expand_all": t.has_expand_all,
                    }
                    for t in getattr(cs, "tree_structures", [])
                ],
                "resource_tables": [
                    {
                        "label": rt.label,
                        "columns": list(rt.columns),
                        "row_label_sample": list(rt.row_label_sample),
                        "has_pagination": rt.has_pagination,
                        "has_column_masters": rt.has_column_masters,
                    }
                    for rt in getattr(cs, "resource_tables", [])
                ],
                "conditional_sections": [
                    {
                        "label": c.label,
                        "trigger_field_label": c.trigger_field_label,
                        "new_field_labels": [
                            f.label for f in c.new_fields[:8]
                        ],
                    }
                    for c in getattr(cs, "conditional_sections", [])
                ],
            }
        return d

    pages_payload = [_page_dump(p) for p in scout_result.pages]
    user_text = (
        f"TARGET_URL: {scout_result.target_url}\n"
        f"LANDING: {scout_result.landing_url} — {scout_result.landing_title}\n"
        f"PAGES_SCOUTED: {len(scout_result.pages)}\n\n"
        "SCOUTED_PAGES (JSON):\n"
        + json.dumps(pages_payload, ensure_ascii=False)
        + "\n\n"
        "Produce the AppMap matching the schema."
    )

    messages = [
        ChatMessage(role="system", content=CONSOLIDATOR_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_text),
    ]

    fallback = AppMap(
        target_url=scout_result.target_url,
        landing_url=scout_result.landing_url,
        landing_title=scout_result.landing_title,
        pages_scouted=len(scout_result.pages),
    )

    def _validate(parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            return False
        return (
            isinstance(parsed.get("modules"), list)
            and isinstance(parsed.get("create_flows"), list)
        )

    try:
        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=LLMRole.PLANNER,
            messages=messages,
            schema=APP_MAP_SCHEMA,
            schema_name="app_map",
            temperature=0.2,
            max_output_tokens=2400,
            validate=_validate,
            on_escalate=on_escalate,
        )
        chat = tiered.chat
    except Exception as e:
        logger.warning(
            "AppMap consolidator LLM call failed: %s: %s",
            type(e).__name__, e,
        )
        return fallback, 0, 0

    parsed = chat.parsed
    if not isinstance(parsed, dict):
        return fallback, chat.input_tokens or 0, chat.output_tokens or 0

    out = AppMap.from_dict({
        "target_url": scout_result.target_url,
        "landing_url": scout_result.landing_url,
        "landing_title": scout_result.landing_title,
        "modules": parsed.get("modules") or [],
        "create_flows": parsed.get("create_flows") or [],
        "cross_cutting_notes": parsed.get("cross_cutting_notes") or [],
        "pages_scouted": len(scout_result.pages),
        "scout_depth": "deep",
        "scout_version": 1,
        "reasoning": str(parsed.get("reasoning", "")),
    })
    return (
        out,
        chat.input_tokens or 0,
        chat.output_tokens or 0,
    )


# ── AKB persistence (one row per target_url, kind=app_map) ────────


_APP_MAP_KIND = "app_map"


# ── Phase A.6 Step 6 — plan ↔ AppMap reconciliation ───────────────


@dataclass
class SubmoduleReconciliation:
    """One row of the plan-vs-AppMap reconciliation report.

    Emitted as a ``plan_reconciled`` live-feed event after the scout
    completes (first run only). Lets the user see which submodules
    the agent will likely struggle with BEFORE execution burns turns
    on them.
    """
    submodule_id: int
    title: str
    status: str   # "ok" | "uncertain" | "mismatch" | "missing"
    reason: str
    matched_module: str = ""
    matched_create_flow: str = ""


_RECON_ROW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "submodule_id": {"type": "integer"},
        "title": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["ok", "uncertain", "mismatch", "missing"],
        },
        "reason": {"type": "string"},
        "matched_module": {"type": "string"},
        "matched_create_flow": {"type": "string"},
    },
    "required": [
        "submodule_id", "title", "status", "reason",
        "matched_module", "matched_create_flow",
    ],
    "additionalProperties": False,
}

RECONCILIATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": _RECON_ROW_SCHEMA,
        },
        "overall_notes": {"type": "string"},
        "confidence": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
        },
    },
    "required": ["rows", "overall_notes", "confidence"],
    "additionalProperties": False,
}


RECONCILIATION_SYSTEM_PROMPT = """You are a senior QA architect. You're given:
- An APP MAP — the structured navigation, create-flows, and patterns
  the system learned about a web app via the authenticated Scout pass.
- A list of SUBMODULES from a test plan, each with id + title +
  short description.

For each submodule, decide whether the app actually supports what
the submodule intends to test. Use ONE of these statuses:

- "ok" — the submodule maps cleanly to something in the AppMap.
  Provide ``matched_module`` (e.g. "Administration") and, when
  relevant, ``matched_create_flow`` (e.g. "Role").
- "uncertain" — the AppMap is silent on this submodule's surface
  (e.g. the scout didn't open the relevant section). Run may
  succeed or fail; flag for the user's attention.
- "mismatch" — the submodule references UI that conflicts with what
  the map shows. Example: BRD says "Select All permissions" but the
  map's role create-flow has a permission TREE (no Select All
  control). Provide a concrete ``reason``.
- "missing" — the submodule references UI the map can't find at all.
  Example: a "chainage" module that doesn't appear in the modules
  list. The submodule will likely be blocked / disputed.

Return strict JSON. ``overall_notes`` is a 1-2 sentence rollup
("most submodules map cleanly; chainage flow may be unreachable
without further nav").
"""


def reconcile_plan_with_map(
    provider: "LLMProvider",
    *,
    app_map: AppMap,
    submodules: list[dict[str, Any]],
    cheap_provider: "LLMProvider | None" = None,
    on_escalate: Callable[[str, str, str], None] | None = None,
) -> tuple[list[SubmoduleReconciliation], int, int]:
    """Run ONE VL call to compare the AppMap against the plan's
    submodule list.

    Returns ``(rows, input_tokens, output_tokens)``. Empty list on
    LLM failure (caller treats reconciliation as best-effort —
    a missing report doesn't block execution).
    """
    from app.llm.base import ChatMessage  # noqa: PLC0415
    from app.llm.router import (  # noqa: PLC0415
        LLMRole, call_for_role,
    )

    if not submodules:
        return [], 0, 0

    submodule_dump = json.dumps([
        {
            "submodule_id": int(sm.get("submodule_id", 0)),
            "title": str(sm.get("title", ""))[:160],
            "description": str(sm.get("description", ""))[:600],
        }
        for sm in submodules
    ], ensure_ascii=False)

    user_text = (
        f"APP MAP:\n{app_map.format_for_prompt()}\n\n"
        f"SUBMODULES (JSON):\n{submodule_dump}\n\n"
        "For each submodule, return one row in ``rows`` (preserve "
        "the input ordering)."
    )
    messages = [
        ChatMessage(role="system", content=RECONCILIATION_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_text),
    ]

    def _validate(parsed: Any) -> bool:
        return isinstance(parsed, dict) and isinstance(
            parsed.get("rows"), list,
        )

    try:
        tiered = call_for_role(
            strong=provider,
            cheap=cheap_provider,
            role=LLMRole.GOAL_VERIFIER,
            messages=messages,
            schema=RECONCILIATION_SCHEMA,
            schema_name="plan_reconciliation",
            temperature=0.2,
            max_output_tokens=1800,
            validate=_validate,
            on_escalate=on_escalate,
        )
        chat = tiered.chat
    except Exception as e:
        logger.warning(
            "plan reconciliation LLM call failed: %s: %s",
            type(e).__name__, e,
        )
        return [], 0, 0

    parsed = chat.parsed
    if not isinstance(parsed, dict):
        return [], chat.input_tokens or 0, chat.output_tokens or 0

    rows_out: list[SubmoduleReconciliation] = []
    for r in (parsed.get("rows") or []):
        if not isinstance(r, dict):
            continue
        status = str(r.get("status") or "")
        if status not in ("ok", "uncertain", "mismatch", "missing"):
            continue
        try:
            sid = int(r.get("submodule_id", 0))
        except (TypeError, ValueError):
            continue
        rows_out.append(SubmoduleReconciliation(
            submodule_id=sid,
            title=str(r.get("title", ""))[:200],
            status=status,
            reason=str(r.get("reason", ""))[:400],
            matched_module=str(r.get("matched_module", ""))[:80],
            matched_create_flow=str(r.get("matched_create_flow", ""))[:80],
        ))
    return rows_out, chat.input_tokens or 0, chat.output_tokens or 0


def save_app_map(
    db: "Session",
    *,
    target_url: str,
    app_map: AppMap,
    source_run_id: int | None = None,
) -> int | None:
    """Persist the AppMap. OVERWRITES any prior map for this target_url.

    We deliberately keep only the current map — a stale map is more
    dangerous than no map. The "refresh mindmap" button on the UI
    re-runs the scout + consolidator + saves; the new save displaces
    the old one in the same transaction.

    Returns the AKB row id, or None on failure.
    """
    from app.models.app_knowledge import AppKnowledge  # noqa: PLC0415
    from app.services.akb import (  # noqa: PLC0415
        _normalise_pattern, write_chunk,
    )
    from sqlalchemy import select as _select  # noqa: PLC0415

    pattern = _normalise_pattern(target_url)
    if not pattern:
        return None

    # Delete prior app_map rows for this pattern so the dedup probe
    # in write_chunk doesn't merge confidence onto a stale row.
    existing = db.execute(
        _select(AppKnowledge).where(
            AppKnowledge.target_url_pattern == pattern,
            AppKnowledge.kind == _APP_MAP_KIND,
        ),
    ).scalars().all()
    for row in existing:
        db.delete(row)
    if existing:
        db.flush()

    content = json.dumps(app_map.to_dict(), ensure_ascii=False)
    return write_chunk(
        db,
        target_url_pattern=pattern,
        kind=_APP_MAP_KIND,
        content=content,
        tags=["app_map", f"v{app_map.scout_version}"],
        confidence=0.9,
        source_run_id=source_run_id,
    )


def load_app_map(
    db: "Session",
    *,
    target_url: str,
) -> AppMap | None:
    """Read the current AppMap for a target_url, or None when absent."""
    from app.models.app_knowledge import AppKnowledge  # noqa: PLC0415
    from app.services.akb import _normalise_pattern  # noqa: PLC0415
    from sqlalchemy import select as _select  # noqa: PLC0415

    pattern = _normalise_pattern(target_url)
    if not pattern:
        return None
    row = db.execute(
        _select(AppKnowledge).where(
            AppKnowledge.target_url_pattern == pattern,
            AppKnowledge.kind == _APP_MAP_KIND,
        ).order_by(AppKnowledge.updated_at.desc()),
    ).scalar_one_or_none()
    if row is None:
        return None
    try:
        return AppMap.from_dict(json.loads(row.content))
    except Exception as e:
        logger.warning(
            "AppMap row %s present but unparseable (%s); ignoring",
            row.id, e,
        )
        return None
