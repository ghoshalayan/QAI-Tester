"""FRD → TC synthesis orchestrator (pure function).

Given a TestPlan, produces a hierarchical test-case tree:
``Module → Submodule(s) → Step(s)`` (3 levels for MVP; schema permits more).

Pipeline (one LLM call per module in plan.scope)
------------------------------------------------
For each module name in plan.scope (or one synthetic ["All test cases"]):

    1. Embed query "<module_name>: <plan.description>"
    2. FAISS search project's ``frd_requirements`` namespace → top-K approved FRDs
    3. (If linked docs) FAISS search ``chunks`` namespace, filter to plan's
       linked_document_ids → top-K source chunks for additional context
    4. Build prompt: target_url + plan.description + FRD list + chunk list
    5. Call structured LLM with the per-module schema → tree payload
    6. Persist Module + Submodules + Steps under that root
    7. Emit ``module_completed`` event (current/total + nodes_added)

Cancellation between modules. Mid-LLM cancellation requires streaming
(deferred to week 5's execution agent).

Re-synthesis appends a new root Module sibling — never wipes existing trees.
The ordinal of the new root is ``MAX(ordinal) + 1`` for that plan's roots.

Plan validation (pre-flight)
----------------------------
The plan must have at least ONE of:
- approved FRDs in the project (FAISS will retrieve relevant ones)
- non-empty plan.description
- linked documents

Otherwise the agent has no signal and we return a clear 400-class error.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.brd_to_frd import AgentCancelled  # reused exception
from app.embeddings.bge import get_embedder
from app.faiss_store.store import get_store
from app.llm.base import ChatMessage, LLMProvider
from app.models.document import Document, DocumentChunk
from app.models.requirement import Requirement
from app.models.tc_node import TcNode
from app.models.test_plan import TestPlan

logger = logging.getLogger(__name__)


@dataclass
class TcSynthesisResult:
    plan_id: int
    modules_requested: list[str]
    modules_generated: int
    modules_skipped: list[str]   # modules where no relevant context was found
    nodes_total: int             # all kinds — modules + submodules + steps
    input_tokens: int
    output_tokens: int


# ── LLM contract ──────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a senior QA engineer designing a hierarchical test-case tree for a web application.

For each module produce a clean tree:
  Module
   ├─ Submodule(s)
   │   └─ Step(s)

Each step describes ONE observable user action with a clear pass/fail criterion.

Step fields:
- title       : short action label (5-12 words)
- narrative   : full sentence describing what the user (or agent) does
- action_type : one of "navigate", "click", "type", "select", "verify", "wait", "submit", "screenshot"
                (use "verify" for assertion-only steps; use the closest fit)
- target_hint : optional CSS selector or visible text snippet — e.g. "button[data-testid='signin']"
                or "text 'Sign In'". Empty string if the agent should resolve it from context.
- expected    : what should be observable after the step
- data_needs  : array of { kind, notes }, kind ∈ "credentials" | "otp" | "data".
                Use this when the step needs login creds, an OTP code, or specific test data.
                Empty array if the step needs none.
- source_frd_codes : list of FRD codes (e.g. ["FRD-3", "FRD-7"]) that motivated this step.
                     Empty array if the step comes from your own reasoning.

Quality bar:
- Each step is atomic — one action, one observation. Split compound steps.
- Submodules group related steps (e.g. "Happy path login", "Error states").
- Don't fabricate FRD codes; cite only codes from the provided list.
- Don't repeat steps verbatim; phrase them as testable user actions.
- Aim for 2-5 submodules per module, 3-8 steps per submodule. Be thorough but not exhaustive."""


PER_MODULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "module": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description_md": {"type": "string"},
                "source_frd_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "submodules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description_md": {"type": "string"},
                            "source_frd_codes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "narrative": {"type": "string"},
                                        "action_type": {"type": "string"},
                                        "target_hint": {"type": "string"},
                                        "expected": {"type": "string"},
                                        "data_needs": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "kind": {"type": "string"},
                                                    "notes": {"type": "string"},
                                                },
                                                "required": ["kind", "notes"],
                                                "additionalProperties": False,
                                            },
                                        },
                                        "source_frd_codes": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": [
                                        "title",
                                        "narrative",
                                        "action_type",
                                        "target_hint",
                                        "expected",
                                        "data_needs",
                                        "source_frd_codes",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "title",
                            "description_md",
                            "source_frd_codes",
                            "steps",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "title",
                "description_md",
                "source_frd_codes",
                "submodules",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["module"],
    "additionalProperties": False,
}


# ── Helpers ───────────────────────────────────────────────────────


def _check_cancel(
    is_cancelled: Callable[[], bool] | None, where: str,
) -> None:
    if is_cancelled and is_cancelled():
        raise AgentCancelled(f"Cancelled at: {where}")


def _emit(
    emit_event: Callable[[str, dict], None] | None,
    event_type: str,
    data: dict,
) -> None:
    if emit_event:
        try:
            emit_event(event_type, data)
        except Exception as e:
            logger.warning("emit_event raised, continuing: %s", e)


def _next_root_ordinal(db: Session, plan_id: int) -> int:
    """Next ordinal for a new root module (parent_id IS NULL)."""
    stmt = select(func.max(TcNode.ordinal)).where(
        TcNode.plan_id == plan_id,
        TcNode.parent_id.is_(None),
    )
    max_ord = db.scalar(stmt)
    return (max_ord + 1) if max_ord is not None else 0


def _project_has_approved_frds(db: Session, project_id: int) -> bool:
    return (
        db.scalar(
            select(Requirement.id)
            .where(
                Requirement.project_id == project_id,
                Requirement.kind == "FRD",
                Requirement.status == "approved",
            )
            .limit(1),
        )
        is not None
    )


def _retrieve_frds(
    db: Session,
    project_id: int,
    query_vec,
    cap: int,
) -> list[Requirement]:
    """Top-K approved FRDs for the query vector (filters out rejected/edited)."""
    store = get_store()
    hits = store.search(project_id, "frd_requirements", query_vec, k=cap * 2)
    if not hits:
        return []

    score_by_id = dict(hits)
    rows = list(
        db.scalars(
            select(Requirement).where(
                Requirement.id.in_(score_by_id.keys()),
                Requirement.status == "approved",
            ),
        ),
    )
    rows.sort(key=lambda r: -score_by_id.get(r.id, 0.0))
    return rows[:cap]


def _retrieve_chunks(
    db: Session,
    project_id: int,
    linked_doc_ids: list[int],
    query_vec,
    cap: int,
) -> tuple[list[DocumentChunk], dict[int, str]]:
    """Top-K relevant chunks restricted to the plan's linked docs."""
    if not linked_doc_ids:
        return [], {}

    store = get_store()
    hits = store.search(project_id, "chunks", query_vec, k=cap * 3)
    if not hits:
        return [], {}

    score_by_id = dict(hits)
    rows = list(
        db.execute(
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(
                DocumentChunk.id.in_(score_by_id.keys()),
                DocumentChunk.document_id.in_(linked_doc_ids),
            ),
        ).all(),
    )
    rows.sort(key=lambda pair: -score_by_id.get(pair[0].id, 0.0))
    rows = rows[:cap]

    chunks = [pair[0] for pair in rows]
    by_doc = {pair[1].id: pair[1].filename for pair in rows}
    return chunks, by_doc


def _build_user_prompt(
    module_name: str,
    plan: TestPlan,
    frds: list[Requirement],
    chunks: list[DocumentChunk],
    chunks_by_doc: dict[int, str],
) -> str:
    lines: list[str] = [
        f"TARGET URL: {plan.target_url}",
        f"PLAN: {plan.name}",
        f"MODULE TO GENERATE: {module_name}",
    ]

    if plan.description and plan.description.strip():
        lines.extend(["", "PLAN INSTRUCTIONS:", plan.description.strip()])

    if frds:
        lines.extend(["", "RELEVANT APPROVED FRDs (cite by code):"])
        for frd in frds:
            body_excerpt = (frd.body_md or "")[:400].replace("\n", " ")
            lines.append(f"  [{frd.code}] {frd.title}")
            if body_excerpt:
                lines.append(f"    {body_excerpt}")

    if chunks:
        lines.extend(["", "RELEVANT BRD/FRD DOC CHUNKS (context only):"])
        for ch in chunks:
            filename = chunks_by_doc.get(ch.document_id, f"doc-{ch.document_id}")
            heading = ch.heading_path or "(no heading)"
            text_excerpt = (ch.text or "")[:500].replace("\n", " ")
            lines.append(f"  [{filename} > {heading}]")
            if text_excerpt:
                lines.append(f"    {text_excerpt}")

    lines.extend(
        [
            "",
            f"Generate a Module → Submodules → Steps tree for: {module_name}.",
            "Cite source_frd_codes from the FRD list above. Use [] if a node is "
            "your own composition rather than from a specific FRD.",
        ],
    )
    return "\n".join(lines)


def _persist_module_tree(
    db: Session,
    project_id: int,
    plan_id: int,
    requested_module_name: str,
    module_payload: dict[str, Any],
    available_frds: list[Requirement],
) -> int:
    """Persist one Module subtree. Returns total nodes added (module + sub + steps)."""
    code_to_id: dict[str, int] = {f.code: f.id for f in available_frds}

    def map_codes(codes: Any) -> list[int]:
        if not isinstance(codes, list):
            return []
        return [
            code_to_id[c]
            for c in codes
            if isinstance(c, str) and c in code_to_id
        ]

    module_title = (module_payload.get("title") or "").strip()
    if not module_title:
        module_title = requested_module_name

    module_desc = module_payload.get("description_md") or None
    module_source = map_codes(module_payload.get("source_frd_codes"))

    module_ordinal = _next_root_ordinal(db, plan_id)
    module = TcNode(
        project_id=project_id,
        plan_id=plan_id,
        parent_id=None,
        kind="module",
        ordinal=module_ordinal,
        depth=0,
        path_cached=module_title[:2048],
        title=module_title[:512],
        description_md=module_desc,
        source_requirement_ids=module_source,
    )
    db.add(module)
    db.flush()
    nodes_added = 1

    submodules = module_payload.get("submodules") or []
    for sm_idx, sm in enumerate(submodules):
        if not isinstance(sm, dict):
            continue
        sm_title = (sm.get("title") or "").strip()
        if not sm_title:
            continue

        sm_path = f"{module.path_cached} > {sm_title}"
        submodule = TcNode(
            project_id=project_id,
            plan_id=plan_id,
            parent_id=module.id,
            kind="submodule",
            ordinal=sm_idx,
            depth=1,
            path_cached=sm_path[:2048],
            title=sm_title[:512],
            description_md=sm.get("description_md") or None,
            source_requirement_ids=map_codes(sm.get("source_frd_codes")),
        )
        db.add(submodule)
        db.flush()
        nodes_added += 1

        steps = sm.get("steps") or []
        for st_idx, st in enumerate(steps):
            if not isinstance(st, dict):
                continue
            st_title = (st.get("title") or "").strip()
            if not st_title:
                continue
            st_path = f"{sm_path} > {st_title}"

            data_needs = st.get("data_needs")
            if not isinstance(data_needs, list):
                data_needs = []

            step = TcNode(
                project_id=project_id,
                plan_id=plan_id,
                parent_id=submodule.id,
                kind="step",
                ordinal=st_idx,
                depth=2,
                path_cached=st_path[:2048],
                title=st_title[:512],
                action_type=(st.get("action_type") or None),
                target_hint=(st.get("target_hint") or None),
                narrative=(st.get("narrative") or None),
                expected=(st.get("expected") or None),
                data_needs_json=data_needs,
                source_requirement_ids=map_codes(st.get("source_frd_codes")),
            )
            db.add(step)
            nodes_added += 1

    db.flush()
    return nodes_added


# ── Orchestrator entry point ──────────────────────────────────────


def synthesize_tc(
    db: Session,
    provider: LLMProvider,
    plan_id: int,
    *,
    cap_per_module_frds: int = 15,
    cap_per_module_chunks: int = 10,
    emit_event: Callable[[str, dict], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> TcSynthesisResult:
    """Synthesize a test-case tree for the given plan.

    Args:
        db: SQLAlchemy session. Caller commits/rolls back.
        provider: LLM provider supporting ``chat_structured``.
        plan_id: TestPlan row id.
        cap_per_module_frds: Max approved FRDs to include in each module's prompt.
        cap_per_module_chunks: Max linked-doc chunks per module's prompt.
        emit_event: Optional callback for SSE events.
        is_cancelled: Optional callback returning True to cancel.

    Returns:
        TcSynthesisResult with module + node counts and token usage.

    Raises:
        ValueError: invalid plan or insufficient signal (no FRDs/description/docs).
        AgentCancelled: user flagged cancellation between modules.
        RuntimeError: LLM returned invalid shape.
    """
    plan = db.get(TestPlan, plan_id)
    if not plan:
        raise ValueError(f"Plan {plan_id} not found")

    project_id = plan.project_id

    _emit(emit_event, "phase", {
        "phase": "validating",
        "message": f"Loading plan '{plan.name}'",
    })

    # Modules to generate
    raw_scope = plan.scope or []
    modules = [str(m).strip() for m in raw_scope if str(m).strip()]
    if not modules:
        modules = ["All test cases"]

    # Linked-doc ids for chunk filtering
    linked_doc_ids: list[int] = [link.document_id for link in plan.linked_docs]

    # Pre-flight: must have at least one signal source
    has_frds = _project_has_approved_frds(db, project_id)
    has_description = bool(plan.description and plan.description.strip())
    has_linked_docs = bool(linked_doc_ids)

    if not (has_frds or has_description or has_linked_docs):
        raise ValueError(
            "Plan needs scope, instructions, or linked docs to generate from. "
            "Approve at least one FRD on the Requirements tab, or add a "
            "description / link a doc to this plan.",
        )

    _emit(emit_event, "phase", {
        "phase": "loading",
        "message": (
            f"Will generate {len(modules)} module(s) "
            f"({len(linked_doc_ids)} linked doc(s))"
        ),
        "module_count": len(modules),
    })

    embedder = get_embedder()

    modules_generated = 0
    modules_skipped: list[str] = []
    total_nodes = 0
    total_in_tokens = 0
    total_out_tokens = 0

    for idx, module_name in enumerate(modules):
        _check_cancel(is_cancelled, f"before module {idx + 1}/{len(modules)}")

        _emit(emit_event, "module_started", {
            "module_name": module_name,
            "current": idx + 1,
            "total": len(modules),
        })

        # Retrieval — embed the query once, reuse for both namespaces
        query_text = (
            f"{module_name}: {plan.description.strip() if plan.description else plan.name}"
        )
        query_vec = embedder.embed_query(query_text)

        frds = _retrieve_frds(db, project_id, query_vec, cap_per_module_frds)
        chunks, chunks_by_doc = _retrieve_chunks(
            db, project_id, linked_doc_ids, query_vec, cap_per_module_chunks,
        )

        if not frds and not chunks and not has_description:
            modules_skipped.append(module_name)
            _emit(emit_event, "module_completed", {
                "module_name": module_name,
                "current": idx + 1,
                "total": len(modules),
                "skipped": True,
                "reason": "No relevant FRDs/chunks/description for this module.",
            })
            continue

        _check_cancel(is_cancelled, f"before LLM call for '{module_name}'")

        _emit(emit_event, "phase", {
            "phase": "calling_llm",
            "message": (
                f"Module {idx + 1}/{len(modules)}: {module_name} "
                f"({len(frds)} FRDs, {len(chunks)} chunks)"
            ),
            "current": idx + 1,
            "total": len(modules),
        })

        user_prompt = _build_user_prompt(
            module_name, plan, frds, chunks, chunks_by_doc,
        )

        try:
            chat_result = provider.chat_structured(
                messages=[
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=user_prompt),
                ],
                schema=PER_MODULE_SCHEMA,
                schema_name="tc_module",
                temperature=0.2,
                max_output_tokens=4096,
            )
        except Exception as e:
            raise RuntimeError(
                f"LLM call failed for module '{module_name}': "
                f"{type(e).__name__}: {str(e)[:300]}",
            ) from e

        if chat_result.input_tokens:
            total_in_tokens += chat_result.input_tokens
        if chat_result.output_tokens:
            total_out_tokens += chat_result.output_tokens

        parsed = chat_result.parsed
        if not isinstance(parsed, dict) or not isinstance(
            parsed.get("module"), dict,
        ):
            raise RuntimeError(
                f"LLM returned unexpected shape for module '{module_name}' — "
                f"expected top-level 'module' object",
            )

        _check_cancel(is_cancelled, f"before persist for '{module_name}'")

        nodes_added = _persist_module_tree(
            db, project_id, plan.id, module_name, parsed["module"], frds,
        )
        db.commit()  # commit per module so partial progress is durable

        total_nodes += nodes_added
        modules_generated += 1

        _emit(emit_event, "module_completed", {
            "module_name": module_name,
            "current": idx + 1,
            "total": len(modules),
            "nodes_added": nodes_added,
            "input_tokens": total_in_tokens,
            "output_tokens": total_out_tokens,
        })

    _emit(emit_event, "done", {
        "modules_generated": modules_generated,
        "modules_skipped": modules_skipped,
        "nodes_total": total_nodes,
        "input_tokens": total_in_tokens,
        "output_tokens": total_out_tokens,
    })

    return TcSynthesisResult(
        plan_id=plan.id,
        modules_requested=modules,
        modules_generated=modules_generated,
        modules_skipped=modules_skipped,
        nodes_total=total_nodes,
        input_tokens=total_in_tokens,
        output_tokens=total_out_tokens,
    )
