"""Phase E — Sub-flow modules router.

Project-scoped CRUD + promote + import. Mounted at::

    /api/projects/{project_id}/sub-flow-modules
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.project import Project

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/projects/{project_id}/sub-flow-modules",
    tags=["sub-flow-modules"],
)


def _require_project(db: Session, project_id: int) -> Project:
    proj = db.get(Project, project_id)
    if proj is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Project {project_id} not found",
        )
    return proj


# ── Schemas ──────────────────────────────────────────────────────


class PromoteRequest(BaseModel):
    submodule_tc_node_id: int
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    target_url_pattern: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_run_id: int | None = None


class ImportRequest(BaseModel):
    plan_id: int
    parent_module_tc_node_id: int | None = None


class UpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    target_url_pattern: str | None = None
    tags: list[str] | None = None


# ── Endpoints ─────────────────────────────────────────────────────


@router.get("")
def list_sub_flow_modules(
    project_id: int,
    db: Session = Depends(get_db),
):
    _require_project(db, project_id)
    from app.services.sub_flow_module import list_modules  # noqa: PLC0415
    return list_modules(db, project_id=project_id)


@router.get("/{module_id}")
def get_sub_flow_module(
    project_id: int,
    module_id: int,
    db: Session = Depends(get_db),
):
    _require_project(db, project_id)
    from app.services.sub_flow_module import get_module  # noqa: PLC0415
    m = get_module(db, project_id=project_id, module_id=module_id)
    if m is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Module {module_id} not found",
        )
    return m


@router.post("/promote", status_code=status.HTTP_201_CREATED)
def promote_to_module(
    project_id: int,
    payload: PromoteRequest,
    db: Session = Depends(get_db),
):
    """Promote a passed submodule's frozen v2 segments into a
    reusable module."""
    _require_project(db, project_id)
    from app.services.sub_flow_module import (  # noqa: PLC0415
        promote_submodule_to_module,
    )
    try:
        result = promote_submodule_to_module(
            db,
            project_id=project_id,
            submodule_tc_node_id=payload.submodule_tc_node_id,
            name=payload.name,
            description=payload.description,
            target_url_pattern=payload.target_url_pattern,
            tags=payload.tags,
            source_run_id=payload.source_run_id,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {
        "module_id": result.module_id,
        "name": result.name,
        "segments": result.segments,
        "steps": result.steps,
    }


@router.post(
    "/{module_id}/import", status_code=status.HTTP_201_CREATED,
)
def import_to_plan(
    project_id: int,
    module_id: int,
    payload: ImportRequest,
    db: Session = Depends(get_db),
):
    """Import a module into a plan's TC tree as a new submodule
    with the frozen path pre-populated."""
    _require_project(db, project_id)
    from app.services.sub_flow_module import (  # noqa: PLC0415
        import_module_into_plan,
    )
    try:
        result = import_module_into_plan(
            db,
            project_id=project_id,
            plan_id=payload.plan_id,
            module_id=module_id,
            parent_module_tc_node_id=payload.parent_module_tc_node_id,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {
        "new_submodule_id": result.new_submodule_id,
        "parent_module_id": result.parent_module_id,
        "steps_created": result.steps_created,
    }


@router.patch("/{module_id}")
def update_metadata(
    project_id: int,
    module_id: int,
    payload: UpdateRequest,
    db: Session = Depends(get_db),
):
    _require_project(db, project_id)
    from app.services.sub_flow_module import (  # noqa: PLC0415
        update_module_metadata,
    )
    ok = update_module_metadata(
        db,
        project_id=project_id,
        module_id=module_id,
        name=payload.name,
        description=payload.description,
        target_url_pattern=payload.target_url_pattern,
        tags=payload.tags,
    )
    if not ok:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Module {module_id} not found",
        )
    return {"ok": True}


@router.delete(
    "/{module_id}", status_code=status.HTTP_204_NO_CONTENT,
)
def delete(
    project_id: int,
    module_id: int,
    db: Session = Depends(get_db),
):
    _require_project(db, project_id)
    from app.services.sub_flow_module import delete_module  # noqa: PLC0415
    ok = delete_module(
        db, project_id=project_id, module_id=module_id,
    )
    if not ok:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Module {module_id} not found",
        )
