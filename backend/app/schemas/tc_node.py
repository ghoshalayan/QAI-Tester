"""Pydantic schemas for the TC nodes router.

Read shapes
-----------
- :class:`TcNodeRead`     — flat row, no children
- :class:`TcNodeTreeRead` — recursive tree (children populated)

The list endpoint returns ``list[TcNodeTreeRead]`` (one item per root module
with nested submodules and steps). The detail endpoint returns the same shape
rooted at the requested node.

Update shapes
-------------
- :class:`TcNodeUpdate`            — partial PATCH
- :class:`TcNodeBulkUpdateRequest` — apply ``approve``/``archive``/``delete``
  to a set selected by ids and/or status/kind filters
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TcNodeKind = Literal["module", "submodule", "step"]
TcNodeStatus = Literal["draft", "approved", "archived"]
TcNodeBulkAction = Literal["approve", "archive", "delete", "select", "deselect"]


# ── Read ──────────────────────────────────────────────────────────


class TcNodeRead(BaseModel):
    """Flat read — used as base for the recursive tree shape and on its own
    for endpoints that don't need children."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    plan_id: int
    parent_id: int | None = None

    kind: TcNodeKind
    ordinal: int
    depth: int
    path_cached: str

    title: str
    description_md: str | None = None

    # Step-only fields (NULL for module/submodule)
    action_type: str | None = None
    target_hint: str | None = None
    narrative: str | None = None
    expected: str | None = None
    data_needs_json: list[dict[str, Any]] | None = None

    selectable_default: bool
    status: TcNodeStatus
    source_requirement_ids: list[int]

    # Phase E — lightweight flag so the test-cases viewer can show
    # "Save as module" only on submodules that actually have a
    # frozen path. We don't ship the full ``frozen_path`` JSON to
    # the frontend (can be large) — just a boolean + the version.
    has_frozen_path: bool = False
    frozen_path_version: int | None = None

    created_at: datetime
    updated_at: datetime
    reviewed_at: datetime | None = None


class TcNodeTreeRead(TcNodeRead):
    """Recursive tree shape. ``children`` is empty for leaves (steps)."""

    children: list["TcNodeTreeRead"] = Field(default_factory=list)


# Pydantic v2: explicitly resolve the forward reference to ``TcNodeTreeRead``
TcNodeTreeRead.model_rebuild()


# ── Update / Bulk ─────────────────────────────────────────────────


class TcNodeUpdate(BaseModel):
    """Partial update.

    Unlike requirements, TC nodes don't auto-demote on body edits — there's
    no ``edited`` status. Caller must pass ``status`` explicitly to change it.
    """

    title: str | None = Field(default=None, min_length=1, max_length=512)
    description_md: str | None = None

    # Step-only edits
    action_type: str | None = Field(default=None, max_length=64)
    target_hint: str | None = None
    narrative: str | None = None
    expected: str | None = None
    data_needs_json: list[dict[str, Any]] | None = None

    selectable_default: bool | None = None
    status: TcNodeStatus | None = None


class TcNodeBulkUpdateRequest(BaseModel):
    """Apply ``action`` to a set of nodes.

    Provide at least one filter:
    - ``node_ids``      — explicit list (intersected with the other filters)
    - ``filter_status`` — narrow to nodes currently in this status
    - ``filter_kind``   — narrow to a kind (e.g. only step leaves)

    Multiple filters are AND-combined.
    """

    node_ids: list[int] | None = None
    filter_status: TcNodeStatus | None = None
    filter_kind: TcNodeKind | None = None
    action: TcNodeBulkAction


class TcNodeBulkUpdateResponse(BaseModel):
    affected: int
    affected_ids: list[int]
    action: TcNodeBulkAction
