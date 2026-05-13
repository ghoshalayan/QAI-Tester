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
from datetime import datetime, timezone

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
        max_replans_per_submodule=int(
            getattr(plan, "max_replans_per_submodule", 2),
        ),
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
        max_replans_per_submodule=int(
            payload.max_replans_per_submodule,
        ),
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

    # α.3 — push linked BRD/FRD chunks into AKB scoped to the plan's
    # target_url so the runtime agent's RAG sees them. Best-effort
    # (don't fail the plan-create on AKB errors); first run on this
    # target_url pays the embed cost.
    try:
        from app.services.akb_ingest import (  # noqa: PLC0415
            ingest_plan_documents_to_akb,
        )
        ingest_plan_documents_to_akb(db, plan)
    except Exception as e:
        logger.warning(
            "AKB ingest on plan create failed (non-fatal): %s", e,
        )

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
    if payload.max_replans_per_submodule is not None:
        plan.max_replans_per_submodule = int(
            payload.max_replans_per_submodule,
        )

    docs_changed = payload.linked_document_ids is not None
    if docs_changed:
        _replace_linked_docs(db, plan, payload.linked_document_ids)

    db.commit()
    db.refresh(plan)

    # α.3 — re-ingest AKB when docs OR target_url changed. The
    # write_chunk helper deduplicates so re-saves on unchanged
    # documents are cheap. Best-effort (non-fatal on AKB errors).
    if docs_changed or payload.target_url is not None:
        try:
            from app.services.akb_ingest import (  # noqa: PLC0415
                ingest_plan_documents_to_akb,
            )
            ingest_plan_documents_to_akb(db, plan)
        except Exception as e:
            logger.warning(
                "AKB ingest on plan update failed (non-fatal): %s", e,
            )

    return _to_plan_detail(db, plan)


@router.post("/{plan_id}/scout")
def scout_app(
    project_id: int,
    plan_id: int,
    db: Session = Depends(get_db),
):
    """β.1 — "Scout this app". Walks the plan's target URL 2-3 levels
    deep and writes recon notes to AKB so subsequent runs have a
    mental model of the app.

    Returns a summary the UI renders. Synchronous for v1 — recon
    typically completes in 30-60s. If we move to background-task
    later, add a polling endpoint.
    """
    plan = _require_plan(db, project_id, plan_id)
    target_url = (plan.target_url or "").strip()
    if not target_url:
        raise HTTPException(
            400,
            "Plan has no target_url; can't scout an empty URL.",
        )

    from app.agents.recon import run_recon  # noqa: PLC0415
    from app.executor.browser import browser_session  # noqa: PLC0415
    from app.llm.cost_tracker import (  # noqa: PLC0415
        begin_run as _begin_cost,
        end_run as _end_cost,
    )
    from app.models.agent_run import AgentRun  # noqa: PLC0415

    try:
        from app.llm.router import build_tier_pair  # noqa: PLC0415

        provider, cheap_provider = build_tier_pair(db)
    except Exception as e:
        logger.info(
            "recon: LLM unavailable, walking text-only: %s", e,
        )
        provider = None
        cheap_provider = None

    # Create a kind=recon AgentRun row so scout activity surfaces in
    # the Runs list with its own cost breakdown, same as a regular
    # execute run. Model names are snapshotted so cost-tracking
    # stays correct after the user changes models.
    run_row = AgentRun(
        project_id=project_id,
        plan_id=plan.id,
        kind="recon",
        status="running",
        input_json={
            "target_url": target_url,
            "max_pages": 8,
        },
        output_summary_json={},
        started_at=datetime.now(timezone.utc),
        strong_model_snapshot=getattr(provider, "model", None),
        cheap_model_snapshot=(
            getattr(cheap_provider, "model", None)
            if cheap_provider is not None
            else None
        ),
    )
    db.add(run_row)
    db.commit()
    db.refresh(run_row)

    _begin_cost(run_id=run_row.id)
    try:
        # ``browser_session`` yields a Page directly; ``speed=None``
        # defaults to the configured speed preset. The previous call
        # used ``speed_config=`` + ``bs.context.new_page()`` which
        # were stale APIs left over from an earlier wrapper.
        with browser_session(headless=True, speed=None) as page:
            try:
                result = run_recon(
                    page,
                    db,
                    target_url=target_url,
                    provider=provider,
                    cheap_provider=cheap_provider,
                    max_pages=8,
                )
            finally:
                try:
                    page.close()
                except Exception:
                    pass
    finally:
        # Persist cost counters + flush per-call logs to llm_call_logs
        # (handled inside end_run when db is supplied). Run status
        # flips regardless of success / failure.
        counters = _end_cost(db=db)
        if counters is not None:
            run_row.strong_input_tokens = counters.strong_input
            run_row.strong_output_tokens = counters.strong_output
            run_row.cheap_input_tokens = counters.cheap_input
            run_row.cheap_output_tokens = counters.cheap_output
            run_row.strong_cached_input_tokens = counters.strong_cached_input
            run_row.cheap_cached_input_tokens = counters.cheap_cached_input

    run_row.status = (
        "failed" if (
            "result" not in locals()
            or getattr(result, "error_message", None)
        ) else "completed"
    )
    run_row.completed_at = datetime.now(timezone.utc)
    res_obj = locals().get("result")
    run_row.output_summary_json = {
        "target_url": target_url,
        "pages_visited": (
            res_obj.pages_visited if res_obj is not None else 0
        ),
        "auth_surface": (
            res_obj.auth_surface if res_obj is not None else None
        ),
        "vision_calls": (
            res_obj.vision_calls if res_obj is not None else 0
        ),
    }
    db.commit()

    return {
        "run_id": run_row.id,
        "target_url": result.target_url,
        "pages_visited": result.pages_visited,
        "pages": result.pages,
        "auth_surface": result.auth_surface,
        "primary_nav_items": result.primary_nav_items,
        "notes": result.notes,
        "error_message": result.error_message,
        "vision_calls": result.vision_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "strong_model": run_row.strong_model_snapshot,
        "cheap_model": run_row.cheap_model_snapshot,
        "strong_input_tokens": run_row.strong_input_tokens,
        "strong_output_tokens": run_row.strong_output_tokens,
        "cheap_input_tokens": run_row.cheap_input_tokens,
        "cheap_output_tokens": run_row.cheap_output_tokens,
    }


@router.delete("/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_plan(
    project_id: int, plan_id: int, db: Session = Depends(get_db),
):
    plan = _require_plan(db, project_id, plan_id)
    db.delete(plan)  # CASCADE removes credentials and doc-links
    db.commit()


# ── Phase A.5 — AppMap (mindmap) inspection + refresh ──────────────


@router.get("/{plan_id}/app-map")
def get_app_map(
    project_id: int,
    plan_id: int,
    db: Session = Depends(get_db),
):
    """Return the current AppMap for the plan's target_url, or 404
    when none exists yet. Lets the UI render the mindmap viewer
    + show "no map yet — first run will build one"."""
    plan = _require_plan(db, project_id, plan_id)
    target_url = (plan.target_url or "").strip()
    if not target_url:
        raise HTTPException(
            400,
            "Plan has no target_url; can't load an app map.",
        )
    from app.agents.app_map import load_app_map  # noqa: PLC0415

    m = load_app_map(db, target_url=target_url)
    if m is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No app map exists for {target_url} yet. The next "
            "agentic run will build one automatically after login.",
        )
    return m.to_dict()


# ── Phase C — TC version listing + refinement endpoint ────────────


@router.get("/{plan_id}/tc-versions")
def list_tc_versions(
    project_id: int,
    plan_id: int,
    db: Session = Depends(get_db),
):
    """Return all TcVersions for this plan, newest first. Lets the
    run-start dialog show "use v1 BRD-initial / v2 app-map-refined /
    v3 manual" choices."""
    plan = _require_plan(db, project_id, plan_id)
    from app.models.tc_version import TcVersion  # noqa: PLC0415
    from sqlalchemy import select as _select  # noqa: PLC0415

    rows = list(db.execute(
        _select(TcVersion)
        .where(TcVersion.plan_id == plan.id)
        .order_by(TcVersion.version_number.desc()),
    ).scalars())
    return {
        "current_tc_version_id": plan.current_tc_version_id,
        "versions": [
            {
                "id": r.id,
                "version_number": r.version_number,
                "source": r.source,
                "label": r.label or f"v{r.version_number} ({r.source})",
                "created_at": r.created_at.isoformat()
                if r.created_at else None,
                "notes": r.notes_json or None,
            }
            for r in rows
        ],
    }


@router.get("/{plan_id}/tc-versions/{version_id}")
def get_tc_version(
    project_id: int,
    plan_id: int,
    version_id: int,
    db: Session = Depends(get_db),
):
    """Return one version's full snapshot tree — used by the diff
    dialog after refinement so the user can review each change
    before activating the version."""
    plan = _require_plan(db, project_id, plan_id)
    from app.models.tc_version import (  # noqa: PLC0415
        TcVersion, TcNodeSnapshot,
    )
    from sqlalchemy import select as _select  # noqa: PLC0415

    version = db.get(TcVersion, version_id)
    if version is None or version.plan_id != plan.id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Version {version_id} not found for plan {plan.id}",
        )
    snaps = list(db.execute(
        _select(TcNodeSnapshot)
        .where(TcNodeSnapshot.tc_version_id == version.id)
        .order_by(
            TcNodeSnapshot.depth,
            TcNodeSnapshot.parent_snapshot_id,
            TcNodeSnapshot.ordinal,
        ),
    ).scalars())
    return {
        "id": version.id,
        "plan_id": version.plan_id,
        "version_number": version.version_number,
        "source": version.source,
        "label": version.label or f"v{version.version_number} ({version.source})",
        "created_at": version.created_at.isoformat()
        if version.created_at else None,
        "notes": version.notes_json or None,
        "snapshots": [
            {
                "id": s.id,
                "original_tc_node_id": s.original_tc_node_id,
                "parent_snapshot_id": s.parent_snapshot_id,
                "kind": s.kind,
                "ordinal": s.ordinal,
                "depth": s.depth,
                "title": s.title,
                "description_md": s.description_md,
                "action_type": s.action_type,
                "target_hint": s.target_hint,
                "narrative": s.narrative,
                "expected": s.expected,
                "change_kind": s.change_kind,
                "change_reason": s.change_reason,
                "selectable_default": s.selectable_default,
                # Phase D — validation surface.
                "validation_status": getattr(
                    s, "validation_status", "pending",
                ),
                "validation_confidence": getattr(
                    s, "validation_confidence", None,
                ),
                "validation_reason": getattr(
                    s, "validation_reason", None,
                ),
                "validation_at": (
                    s.validation_at.isoformat()
                    if getattr(s, "validation_at", None) else None
                ),
            }
            for s in snaps
        ],
    }


@router.put("/{plan_id}/tc-versions/{version_id}/activate")
def activate_tc_version(
    project_id: int,
    plan_id: int,
    version_id: int,
    db: Session = Depends(get_db),
):
    """Activate this TcVersion — OVERWRITE the live TcNode tree with
    the version's snapshot tree (per the user's audit-trail
    semantics). What you see in the test-cases viewer is what runs
    against. The version snapshots remain queryable for audit /
    rollback (activate v1 = revert to the BRD-initial baseline).

    Pass version_id=0 to clear the pointer without changing the
    tree (rare; advanced use).
    """
    plan = _require_plan(db, project_id, plan_id)
    if version_id == 0:
        plan.current_tc_version_id = None
        db.commit()
        return {"current_tc_version_id": None}
    from app.models.tc_version import TcVersion  # noqa: PLC0415
    from app.services.tc_refinement import (  # noqa: PLC0415
        apply_tc_version_to_live,
    )

    version = db.get(TcVersion, version_id)
    if version is None or version.plan_id != plan.id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Version {version_id} not found for plan {plan.id}",
        )

    try:
        counts = apply_tc_version_to_live(
            db, plan_id=plan.id, version_id=version.id,
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            500,
            f"Failed to apply version to live tree: {e}",
        ) from e

    plan.current_tc_version_id = version.id
    db.commit()
    return {
        "current_tc_version_id": version.id,
        "version_number": version.version_number,
        "applied": counts,
    }


@router.post("/{plan_id}/tc-versions/{version_id}/validate")
def validate_tc_version(
    project_id: int,
    plan_id: int,
    version_id: int,
    db: Session = Depends(get_db),
):
    """Phase D — dry-run validate a TcVersion against the live UI.

    Opens a (headless) browser, logs into the target app via
    ``auth_flow``, walks each refined step, and probes each
    target_hint against the live DOM **without dispatching the
    action**. Writes ``validation_status`` + ``validation_confidence``
    onto each snapshot row.

    Synchronous (~60-90s for Solar's 30+ steps). The dialog shows a
    spinner. Cancellable via the same cancel registry used for runs
    (planned).
    """
    plan = _require_plan(db, project_id, plan_id)
    from app.models.tc_version import TcVersion  # noqa: PLC0415

    version = db.get(TcVersion, version_id)
    if version is None or version.plan_id != plan.id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Version {version_id} not found for plan {plan.id}",
        )
    if not (plan.target_url or "").strip():
        raise HTTPException(
            400,
            "Plan has no target_url — can't dry-run validate.",
        )
    from app.llm.router import build_tier_pair  # noqa: PLC0415
    try:
        provider, cheap_provider = build_tier_pair(db)
    except Exception:
        provider, cheap_provider = None, None
    from app.services.tc_validation import (  # noqa: PLC0415
        validate_version_against_live,
    )
    result = validate_version_against_live(
        db,
        plan_id=plan.id,
        version_id=version.id,
        headless=True,
        provider=provider,
        cheap_provider=cheap_provider,
    )
    return {
        "plan_id": plan.id,
        "version_id": version.id,
        "total_probed": result.total_probed,
        "total_seconds": result.total_seconds,
        "error_message": result.error_message,
        "cancelled": result.cancelled,
        "submodules": [
            {
                "submodule_snapshot_id": sm.submodule_snapshot_id,
                "title": sm.submodule_title,
                "confirmed": sm.confirmed,
                "partial": sm.partial,
                "unresolved": sm.unresolved,
                "unreachable": sm.unreachable,
                "skipped": sm.skipped,
                "confidence": round(sm.confidence, 3),
            }
            for sm in result.submodules
        ],
    }


@router.post("/{plan_id}/refine-from-app-map")
def refine_from_app_map(
    project_id: int,
    plan_id: int,
    db: Session = Depends(get_db),
):
    """Phase C.2 — kick off the per-submodule TC refinement.

    Synchronous (the call returns when refinement is done). For
    Solar (7 submodules) this is ~10-30s wallclock. The response
    carries the new version_id so the UI can immediately fetch the
    diff via ``GET /tc-versions/{id}``.
    """
    plan = _require_plan(db, project_id, plan_id)
    target_url = (plan.target_url or "").strip()
    if not target_url:
        raise HTTPException(
            400,
            "Plan has no target_url; can't refine without an AppMap.",
        )
    from app.agents.app_map import load_app_map  # noqa: PLC0415

    if load_app_map(db, target_url=target_url) is None:
        raise HTTPException(
            400,
            "No AppMap exists for this target_url yet. Run agentic "
            "mode once OR click 'Scout this app' to build one first.",
        )

    from app.llm.router import build_tier_pair  # noqa: PLC0415

    try:
        provider, cheap_provider = build_tier_pair(db)
    except Exception as e:
        raise HTTPException(
            400,
            f"LLM provider not configured: {e}",
        ) from e

    from app.services.tc_refinement import refine_plan  # noqa: PLC0415

    result = refine_plan(
        db,
        plan_id=plan_id,
        provider=provider,
        cheap_provider=cheap_provider,
    )

    if result.error_message:
        raise HTTPException(
            400,
            f"Refinement failed: {result.error_message}",
        )
    return {
        "plan_id": plan_id,
        "version_id": result.version_id,
        "version_number": result.version_number,
        "submodule_count": len(result.submodules),
        "input_tokens": result.total_input_tokens,
        "output_tokens": result.total_output_tokens,
        "submodule_summaries": [
            {
                "submodule_id": rs.submodule_id,
                "title": rs.submodule_title,
                "step_count": len(rs.steps),
                "kept": sum(
                    1 for s in rs.steps if s.change_kind == "kept"
                ),
                "rewritten": sum(
                    1 for s in rs.steps
                    if s.change_kind == "rewritten"
                ),
                "added": sum(
                    1 for s in rs.steps if s.change_kind == "added"
                ),
                "flagged_missing": sum(
                    1 for s in rs.steps
                    if s.change_kind == "flagged_missing"
                ),
                "confidence": rs.confidence,
                "error": rs.error_message,
            }
            for rs in result.submodules
        ],
    }


@router.post("/{plan_id}/preflight")
def preflight_plan(
    project_id: int,
    plan_id: int,
    db: Session = Depends(get_db),
    force: bool = False,
    skip_scout: bool = False,
):
    """Phase H — run the full Scout → Refine → Activate preflight.

    Convenience endpoint. Identical to what auto-runs at the head of
    an agentic run, but exposed so the tester can validate + refine
    the plan WITHOUT actually starting execution. Useful when:

    - Adding new submodules to an existing plan (refine before run).
    - Switching the target_url to a new environment (re-scout +
      re-refine).
    - Reviewing what the refiner would change before committing.

    Synchronous. Scout-needed first runs are ~30-90s; cached re-runs
    skip straight to refinement (~10-30s for a 7-submodule plan).
    """
    plan = _require_plan(db, project_id, plan_id)
    target_url = (plan.target_url or "").strip()
    if not target_url:
        raise HTTPException(
            400,
            "Plan has no target_url; can't preflight without one.",
        )

    from app.llm.router import build_tier_pair  # noqa: PLC0415
    try:
        provider, cheap_provider = build_tier_pair(db)
    except Exception as e:
        raise HTTPException(
            400,
            f"LLM provider not configured: {e}",
        ) from e

    from app.services.preflight import run_preflight  # noqa: PLC0415

    result = run_preflight(
        db,
        plan_id=plan_id,
        provider=provider,
        cheap_provider=cheap_provider,
        force=force,
        skip_scout=skip_scout,
        headless=True,
    )
    if result.status == "failed":
        raise HTTPException(
            400,
            f"Preflight failed: {result.error_message}",
        )
    return {
        "plan_id": plan_id,
        "status": result.status,
        "scout_ran": result.scout_ran,
        "scout_pages": result.scout_pages,
        "scout_create_surfaces": result.scout_create_surfaces,
        "refine_ran": result.refine_ran,
        "new_version_id": result.new_version_id,
        "refined_submodules": result.refined_submodules,
        "rewritten": result.refined_rewritten,
        "added": result.refined_added,
        "flagged_missing": result.refined_flagged_missing,
        "activated_version_id": result.activated_version_id,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_seconds": result.total_seconds,
        "notes": result.notes,
    }


@router.post("/{plan_id}/submodules/{submodule_id}/convert-to-goal-mode")
def convert_submodule_to_goal_mode(
    project_id: int,
    plan_id: int,
    submodule_id: int,
    db: Session = Depends(get_db),
):
    """Phase I.4 — drop a submodule's step children so the agentic
    runner treats it as goal-mode.

    Goal-mode contract: the submodule's ``description_md`` IS the
    goal (intent); ``evidence_signals`` are the success criteria;
    ``preconditions`` / ``postconditions`` / ``alternative_paths``
    are passed as context. The agent decomposes from goal +
    screenshot + AppMap, ignoring step-level prescription entirely.

    Idempotent: calling on a submodule that's already goal-mode
    (no step children) is a no-op.
    """
    plan = _require_plan(db, project_id, plan_id)
    from app.models.tc_node import TcNode  # noqa: PLC0415
    from sqlalchemy import select as _select  # noqa: PLC0415

    submodule = db.execute(
        _select(TcNode).where(
            TcNode.id == submodule_id,
            TcNode.plan_id == plan.id,
            TcNode.kind == "submodule",
        ),
    ).scalar_one_or_none()
    if submodule is None:
        raise HTTPException(
            404,
            f"submodule {submodule_id} not found on plan {plan_id}",
        )

    children = list(db.execute(
        _select(TcNode).where(
            TcNode.parent_id == submodule.id,
            TcNode.kind == "step",
        ),
    ).scalars())
    removed = 0
    for c in children:
        db.delete(c)
        removed += 1
    db.commit()
    return {
        "submodule_id": submodule.id,
        "title": submodule.title,
        "goal_mode": True,
        "steps_removed": removed,
        "description_md": submodule.description_md,
        "has_evidence_signals": bool(submodule.evidence_signals),
        "has_postconditions": bool(submodule.postconditions),
    }


@router.delete(
    "/{plan_id}/app-map",
    status_code=status.HTTP_204_NO_CONTENT,
)
def clear_app_map(
    project_id: int,
    plan_id: int,
    db: Session = Depends(get_db),
):
    """Delete the AppMap so the next agentic run rebuilds it.

    We don't run the scout synchronously here — that needs a
    logged-in browser session, which only exists during an agent
    run. Instead the "refresh" UX deletes the stale map; the next
    run notices it's missing and inline-scouts.
    """
    plan = _require_plan(db, project_id, plan_id)
    target_url = (plan.target_url or "").strip()
    if not target_url:
        raise HTTPException(
            400,
            "Plan has no target_url; can't clear an app map.",
        )
    from app.models.app_knowledge import AppKnowledge  # noqa: PLC0415
    from app.services.akb import _normalise_pattern  # noqa: PLC0415
    from sqlalchemy import select as _select  # noqa: PLC0415

    pattern = _normalise_pattern(target_url)
    rows = db.execute(
        _select(AppKnowledge).where(
            AppKnowledge.target_url_pattern == pattern,
            AppKnowledge.kind == "app_map",
        ),
    ).scalars().all()
    for row in rows:
        db.delete(row)
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
