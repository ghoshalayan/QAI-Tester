"""Test Plans router.

Mounted at ``/api/projects/{project_id}/plans``. Owns:

- Plans CRUD
- Per-plan credentials sub-CRUD (``/credentials`` and ``/credentials/{id}``)
- Heading suggestions endpoint (``/heading-suggestions``) — used by the Plan
  editor's scope dropdown to surface module names extracted from linked docs.

Doc-link management is folded into ``PATCH /plans/{id}`` via
``linked_document_ids`` for atomic save.

Route ordering note
-------------------
Literal routes (``/heading-suggestions``) are declared **before** parametric
ones (``/{plan_id}``) so the router doesn't try to parse "heading-suggestions"
as a numeric plan id.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.models.document import Document, DocumentChunk
from app.models.project import Project
from app.models.test_plan import TestPlan, TestPlanCredential, TestPlanDocument
from app.schemas.test_plan import (
    CredentialCreate,
    CredentialRead,
    CredentialUpdate,
    HeadingSuggestionsResponse,
    LinkedDocSummary,
    PlanCreate,
    PlanReadCompact,
    PlanReadDetail,
    PlanUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/projects/{project_id}/plans",
    tags=["Test Plans"],
)


# ── Helpers ───────────────────────────────────────────────────────


def _require_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


def _require_plan(
    db: Session, project_id: int, plan_id: int,
) -> TestPlan:
    plan = db.get(TestPlan, plan_id)
    if not plan or plan.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Plan not found")
    return plan


def _require_credential(
    db: Session, project_id: int, plan_id: int, cred_id: int,
) -> TestPlanCredential:
    cred = db.get(TestPlanCredential, cred_id)
    if not cred or cred.plan_id != plan_id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "Credential not found",
        )
    if cred.plan and cred.plan.project_id != project_id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "Credential not found",
        )
    return cred


def _to_credential_read(c: TestPlanCredential) -> CredentialRead:
    # Phase 3 — decrypt the username for display ONLY (password &
    # TOTP secret are never echoed). When decryption fails (key
    # drift, corrupt row), surface a placeholder so the UI doesn't
    # crash — user can re-enter to fix.
    username_display = c.username or ""
    if getattr(c, "encrypted", False) and c.username:
        try:
            from app.security.vault import decrypt_str  # noqa: PLC0415

            username_display = decrypt_str(c.username)
        except Exception:
            username_display = "(decryption failed — re-enter)"
    return CredentialRead(
        id=c.id,
        plan_id=c.plan_id,
        label=c.label,
        username=username_display,
        password_set=bool(c.password),
        totp_set=bool(getattr(c, "totp_secret", None)),
        url_pattern=c.url_pattern,
        username_selector_hint=c.username_selector_hint,
        password_selector_hint=c.password_selector_hint,
        notes=c.notes,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


def _to_plan_compact(plan: TestPlan) -> PlanReadCompact:
    return PlanReadCompact(
        id=plan.id,
        project_id=plan.project_id,
        name=plan.name,
        target_url=plan.target_url,
        scope=list(plan.scope or []),
        status=plan.status,  # type: ignore[arg-type]
        credential_count=len(plan.credentials),
        linked_document_count=len(plan.linked_docs),
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


def _to_plan_detail(
    db: Session, plan: TestPlan,
) -> PlanReadDetail:
    # Resolve linked-doc summaries. Plan.linked_docs are TestPlanDocument
    # rows; we eagerly access .document on each.
    linked: list[LinkedDocSummary] = []
    for link in plan.linked_docs:
        d = link.document
        if d is None:
            continue
        linked.append(
            LinkedDocSummary(
                document_id=d.id,
                filename=d.filename,
                kind=d.kind,  # type: ignore[arg-type]
                status=d.status,  # type: ignore[arg-type]
                chunk_count=d.chunk_count,
            ),
        )

    return PlanReadDetail(
        id=plan.id,
        project_id=plan.project_id,
        name=plan.name,
        target_url=plan.target_url,
        description=plan.description,
        scope=list(plan.scope or []),
        status=plan.status,  # type: ignore[arg-type]
        credentials=[_to_credential_read(c) for c in plan.credentials],
        linked_documents=linked,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


def _validate_doc_ids_for_project(
    db: Session, project_id: int, doc_ids: list[int],
) -> list[Document]:
    """Ensure every doc id exists AND belongs to the given project."""
    if not doc_ids:
        return []
    unique = list(dict.fromkeys(doc_ids))  # preserve order, drop dupes
    docs = list(
        db.scalars(select(Document).where(Document.id.in_(unique))),
    )
    by_id = {d.id: d for d in docs}
    missing = [i for i in unique if i not in by_id]
    if missing:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Document(s) not found: {missing}",
        )
    wrong = [d.id for d in docs if d.project_id != project_id]
    if wrong:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Document(s) belong to a different project: {wrong}",
        )
    return docs


def _replace_linked_docs(
    db: Session, plan: TestPlan, document_ids: list[int],
) -> None:
    """Replace the plan's linked-doc set with exactly ``document_ids``."""
    _validate_doc_ids_for_project(db, plan.project_id, document_ids)

    # Delete existing links
    for link in list(plan.linked_docs):
        db.delete(link)
    db.flush()

    # Insert new links
    for did in dict.fromkeys(document_ids):  # dedupe
        db.add(TestPlanDocument(plan_id=plan.id, document_id=did))


# ── Heading suggestions (literal route — declared before /{plan_id}) ──


@router.get("/heading-suggestions", response_model=HeadingSuggestionsResponse)
def heading_suggestions(
    project_id: int,
    document_ids: list[int] = Query(default_factory=list),
    db: Session = Depends(get_db),
):
    """Top-level headings (split on the first '> ') extracted from the chunks
    of the given documents. Powers the Plan editor's scope dropdown.

    The Plan editor calls this dynamically as the user (un)checks document
    links — even before the plan itself has been saved.
    """
    _require_project(db, project_id)

    if not document_ids:
        return HeadingSuggestionsResponse(
            suggestions=[], document_count=0, chunk_count=0,
        )

    _validate_doc_ids_for_project(db, project_id, document_ids)

    heading_paths = list(
        db.scalars(
            select(DocumentChunk.heading_path)
            .where(DocumentChunk.document_id.in_(document_ids))
            .where(DocumentChunk.heading_path.isnot(None))
            .where(DocumentChunk.heading_path != ""),
        ),
    )

    top_level = sorted(
        {
            (hp or "").split(" > ", 1)[0].strip()
            for hp in heading_paths
            if hp and hp.strip()
        },
    )

    return HeadingSuggestionsResponse(
        suggestions=[s for s in top_level if s],
        document_count=len(set(document_ids)),
        chunk_count=len(heading_paths),
    )


# ── Plans CRUD ────────────────────────────────────────────────────


@router.post(
    "",
    response_model=PlanReadDetail,
    status_code=status.HTTP_201_CREATED,
)
def create_plan(
    project_id: int,
    payload: PlanCreate,
    db: Session = Depends(get_db),
):
    _require_project(db, project_id)

    # Validate linked docs (if any) belong to the project
    if payload.linked_document_ids:
        _validate_doc_ids_for_project(
            db, project_id, payload.linked_document_ids,
        )

    plan = TestPlan(
        project_id=project_id,
        name=payload.name.strip(),
        target_url=payload.target_url.strip(),
        description=payload.description,
        scope=list(payload.scope or []),
        status=payload.status,
    )
    db.add(plan)
    db.flush()  # populate plan.id

    for did in dict.fromkeys(payload.linked_document_ids):
        db.add(TestPlanDocument(plan_id=plan.id, document_id=did))

    for cred in payload.credentials:
        db.add(
            TestPlanCredential(
                plan_id=plan.id,
                label=cred.label.strip(),
                username=cred.username,
                password=cred.password,
                url_pattern=cred.url_pattern,
                username_selector_hint=cred.username_selector_hint,
                password_selector_hint=cred.password_selector_hint,
                notes=cred.notes,
            ),
        )

    db.commit()
    db.refresh(plan)
    return _to_plan_detail(db, plan)


@router.get("", response_model=list[PlanReadCompact])
def list_plans(project_id: int, db: Session = Depends(get_db)):
    _require_project(db, project_id)
    stmt = (
        select(TestPlan)
        .where(TestPlan.project_id == project_id)
        .options(
            selectinload(TestPlan.credentials),
            selectinload(TestPlan.linked_docs),
        )
        .order_by(TestPlan.updated_at.desc())
    )
    return [_to_plan_compact(p) for p in db.scalars(stmt)]


@router.get("/{plan_id}", response_model=PlanReadDetail)
def get_plan(
    project_id: int, plan_id: int, db: Session = Depends(get_db),
):
    plan = _require_plan(db, project_id, plan_id)
    return _to_plan_detail(db, plan)


@router.patch("/{plan_id}", response_model=PlanReadDetail)
def update_plan(
    project_id: int,
    plan_id: int,
    payload: PlanUpdate,
    db: Session = Depends(get_db),
):
    plan = _require_plan(db, project_id, plan_id)

    if payload.name is not None:
        plan.name = payload.name.strip()
    if payload.target_url is not None:
        plan.target_url = payload.target_url.strip()
    if payload.description is not None:
        plan.description = payload.description
    if payload.scope is not None:
        plan.scope = list(payload.scope)
    if payload.status is not None:
        plan.status = payload.status

    if payload.linked_document_ids is not None:
        _replace_linked_docs(db, plan, payload.linked_document_ids)

    db.commit()
    db.refresh(plan)
    return _to_plan_detail(db, plan)


@router.delete("/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_plan(
    project_id: int, plan_id: int, db: Session = Depends(get_db),
):
    plan = _require_plan(db, project_id, plan_id)
    db.delete(plan)  # CASCADE removes credentials and doc-links
    db.commit()


# ── Credentials sub-CRUD ─────────────────────────────────────────


@router.get(
    "/{plan_id}/credentials",
    response_model=list[CredentialRead],
)
def list_credentials(
    project_id: int, plan_id: int, db: Session = Depends(get_db),
):
    plan = _require_plan(db, project_id, plan_id)
    return [_to_credential_read(c) for c in plan.credentials]


@router.post(
    "/{plan_id}/credentials",
    response_model=CredentialRead,
    status_code=status.HTTP_201_CREATED,
)
def create_credential(
    project_id: int,
    plan_id: int,
    payload: CredentialCreate,
    db: Session = Depends(get_db),
):
    plan = _require_plan(db, project_id, plan_id)
    # Phase 3 — encrypt at write. Resolve the vault key (env var,
    # then file, then auto-generate on first use) and Fernet-encrypt
    # username / password / totp_secret. ``encrypted=True`` flags the
    # row so the read path knows to decrypt.
    from app.security.vault import encrypt_for_write  # noqa: PLC0415

    enc_user, enc_pass, enc_totp = encrypt_for_write(
        payload.username,
        payload.password,
        getattr(payload, "totp_secret", None),
    )
    cred = TestPlanCredential(
        plan_id=plan.id,
        label=payload.label.strip(),
        username=enc_user,
        password=enc_pass,
        totp_secret=enc_totp,
        encrypted=True,
        url_pattern=payload.url_pattern,
        username_selector_hint=payload.username_selector_hint,
        password_selector_hint=payload.password_selector_hint,
        notes=payload.notes,
    )
    db.add(cred)
    db.commit()
    db.refresh(cred)
    return _to_credential_read(cred)


@router.patch(
    "/{plan_id}/credentials/{cred_id}",
    response_model=CredentialRead,
)
def update_credential(
    project_id: int,
    plan_id: int,
    cred_id: int,
    payload: CredentialUpdate,
    db: Session = Depends(get_db),
):
    cred = _require_credential(db, project_id, plan_id, cred_id)

    # Phase 3 — partial update with re-encrypt. Each field that
    # changes goes back through the vault. Existing fields stay
    # in their current encryption state (legacy plaintext rows
    # remain plaintext on a label-only update — they migrate to
    # encrypted form only when the user re-enters credentials).
    from app.security.vault import encrypt_str  # noqa: PLC0415

    if payload.label is not None:
        cred.label = payload.label.strip()
    if payload.username is not None:
        cred.username = encrypt_str(payload.username)
        cred.encrypted = True
    # Only replace password if a non-empty value is provided
    if payload.password is not None and payload.password.strip():
        cred.password = encrypt_str(payload.password)
        cred.encrypted = True
    # TOTP secret update — empty string clears it.
    if getattr(payload, "totp_secret", None) is not None:
        from app.security.vault import (  # noqa: PLC0415
            _normalize_totp_seed,
        )
        seed_raw = (payload.totp_secret or "").strip()
        if seed_raw:
            seed = _normalize_totp_seed(seed_raw)
            cred.totp_secret = encrypt_str(seed) if seed else None
            cred.encrypted = True
        else:
            cred.totp_secret = None
    if payload.url_pattern is not None:
        cred.url_pattern = payload.url_pattern or None
    if payload.username_selector_hint is not None:
        cred.username_selector_hint = (
            payload.username_selector_hint or None
        )
    if payload.password_selector_hint is not None:
        cred.password_selector_hint = (
            payload.password_selector_hint or None
        )
    if payload.notes is not None:
        cred.notes = payload.notes or None

    db.commit()
    db.refresh(cred)
    return _to_credential_read(cred)


@router.delete(
    "/{plan_id}/credentials/{cred_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_credential(
    project_id: int,
    plan_id: int,
    cred_id: int,
    db: Session = Depends(get_db),
):
    cred = _require_credential(db, project_id, plan_id, cred_id)
    db.delete(cred)
    db.commit()
