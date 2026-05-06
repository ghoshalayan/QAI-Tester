"""Projects CRUD.

Delete cascade
--------------
The DB-level FK chain ``documents.project_id → projects.id`` (ON DELETE
CASCADE) and ``document_chunks.document_id → documents.id`` (ON DELETE
CASCADE) means a single ``DELETE FROM projects WHERE id = ?`` flushes the
whole content tree atomically, provided ``PRAGMA foreign_keys=ON`` is set
(it is — see ``app/db.py``).

After the SQL delete, we also wipe two on-disk directories:

- ``data/faiss/<project_id>/``  — the project's FAISS index files
- ``data/docs/<project_id>/``   — original uploaded files + parsed.md outputs

Order: SQL first, then disk wipes. If a disk wipe fails, the project is
already gone from the DB (no orphaned UI references); only stale files
remain — easy to clean up manually.
"""

import logging
import shutil

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.faiss_store.store import get_store
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectRead, ProjectUpdate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    project = Project(
        name=payload.name.strip(),
        description=payload.description,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("", response_model=list[ProjectRead])
def list_projects(db: Session = Depends(get_db)):
    """Newest activity first."""
    stmt = select(Project).order_by(Project.updated_at.desc())
    return list(db.scalars(stmt))


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


@router.patch("/{project_id}", response_model=ProjectRead)
def update_project(
    project_id: int,
    payload: ProjectUpdate,
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    if payload.name is not None:
        project.name = payload.name.strip()
    if payload.description is not None:
        project.description = payload.description

    db.commit()
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")

    # 1. Delete the project row. SQLite FK CASCADE removes all rows in
    #    documents and document_chunks for this project atomically.
    db.delete(project)
    db.commit()

    # 2. Wipe FAISS indices for this project.
    try:
        get_store().reset(project_id)
    except Exception as e:
        logger.warning(
            "FAISS reset failed for project %s after DB delete: %s",
            project_id,
            e,
        )

    # 3. Wipe the on-disk docs dir (originals + parsed.md outputs).
    docs_dir = settings.docs_dir / str(project_id)
    if docs_dir.exists():
        shutil.rmtree(docs_dir, ignore_errors=True)
