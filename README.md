# QAI Tester v2

Local agentic QA testing platform — fresh build.

End-to-end target flow:
**BRD/FRD → AI-generated test cases (HITL review) → Selected execution → in-app + Excel report**

Phase 1 ships the foundations only. Phase 2 layers on doc ingestion, the
BRD→FRD→TC pipeline, HITL gates, and execution.

---

## What Phase 1 delivers

| | |
|---|---|
| 🗂️ Multiple projects, each with isolated state on disk | ✅ |
| 🔌 LLM provider abstraction (Gemini / OpenAI / OpenAI-compatible) with live "Test connection" | ✅ |
| 🧠 Local English embeddings via **BAAI/bge-large-en-v1.5** (sentence-transformers, CPU) | ✅ |
| 🔍 Per-project, per-namespace FAISS store (`IndexIDMap` over `IndexFlatIP`) with disk persistence | ✅ |
| 📡 SSE event bus that bridges sync agent threads to async frontend listeners with replayable history | ✅ |
| 🎨 Professional UI: Next 16 + React 19 + Tailwind v4 + shadcn/ui (New York / neutral, OKLCH palette) + dark/light/system | ✅ |
| 🚪 First-run gate that blocks the app behind LLM configuration | ✅ |

Everything except the frontend can be exercised end-to-end via the `/api/_debug/*`
routes — useful for unit-testing the agent pipeline in Phase 2 without UI work.

---

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI + SQLAlchemy 2.0 + SQLite (WAL) + Alembic migrations |
| Embeddings | sentence-transformers · `BAAI/bge-large-en-v1.5` (1024-dim, L2-normalized, CPU) |
| Vector store | FAISS · per-project `IndexIDMap[IndexFlatIP]` |
| LLM | Gemini (`google-genai`) / OpenAI (`openai`) / OpenAI-compatible (custom `base_url`) |
| Live updates | Server-Sent Events (`sse-starlette`) with `Last-Event-ID` reconnect |
| Package manager | `uv` for backend, `npm` for frontend |
| Frontend | Next 16 · React 19 · Tailwind v4 · shadcn/ui · TanStack Query · Zustand · sonner · lucide |

---

## Setup

### Backend

```bash
cd v2/backend
uv sync                                    # ~3–5 min first time (PyTorch + transformers)
uv run alembic upgrade head                # create app_settings + projects tables
uv run uvicorn app.main:app --reload --port 8000
```

API docs: <http://localhost:8000/docs>

### Frontend

```bash
cd v2/frontend
npm install
npm run dev
```

App: <http://localhost:3000>

If you run the backend on a different host/port, set
`NEXT_PUBLIC_API_URL` in `v2/frontend/.env.local`.

---

## End-to-end demo (Phase 1)

1. **Start backend** (Terminal 1) and **frontend** (Terminal 2) per above.
2. Open <http://localhost:3000> — you'll see a **Welcome** card because no LLM is configured.
3. Click **Configure now →**. The Settings page loads.
4. Pick a provider:
   - **Gemini** → grab a key from <https://aistudio.google.com/app/apikey>
   - **OpenAI** → key from <https://platform.openai.com/api-keys>
   - **OpenAI-compatible** → e.g. **Ollama** at `http://localhost:11434/v1`
5. Choose a model (use the suggestions dropdown or type your own).
6. Click **Test connection** — green panel + toast confirms reachability.
7. Click **Save** — first-run gate flips, you're now inside the app.
8. Sidebar → **Projects** → **+ New project** → enter a name → **Create**.
9. Click the project card — you land on the detail shell with three "Phase 2" placeholder cards (Documents / Test Cases / Runs).
10. **Edit** or **Delete** the project from the detail header. Delete also wipes the project's FAISS dir at `data/faiss/<id>/`.
11. Theme toggle in the sidebar bottom — instant light/dark flip with no page flash.

---

## What's running where on disk

```
v2/backend/data/
├── qai.db              # projects, app_settings, future tables
├── faiss/<project_id>/ # per-project FAISS indices, one file per namespace
├── docs/<project_id>/  # uploaded BRDs/FRDs (Phase 2)
├── screenshots/<run_id>/
└── reports/<run_id>/
```

Wiping `data/` and re-running `alembic upgrade head` resets the world.
Nothing else is persisted (no `~/.qai/` files, no encrypted vaults — secrets
live only in memory at runtime per the local-MVP policy).

---

## Backend API map

| Group | Routes |
|---|---|
| Health | `GET /api/health` |
| Settings | `GET /api/settings` · `PUT /api/settings` · `DELETE /api/settings` · `POST /api/settings/test` |
| Projects | `POST /api/projects` · `GET /api/projects` · `GET /api/projects/{id}` · `PATCH /api/projects/{id}` · `DELETE /api/projects/{id}` |
| Documents | `POST /api/projects/{pid}/documents/upload` (multipart) · `POST /…/paste` · `POST /…/search` · `GET /…/events` (SSE) · `GET /…` (list) · `GET /…/{doc_id}` · `GET /…/{doc_id}/parsed` · `GET /…/{doc_id}/chunks` · `DELETE /…/{doc_id}` |
| Test Plans | `POST/GET /api/projects/{pid}/plans` · `GET/PATCH/DELETE /…/{plan_id}` · `GET /…/heading-suggestions?document_ids=…` · `GET/POST /…/{plan_id}/credentials` · `PATCH/DELETE /…/{plan_id}/credentials/{cred_id}` |
| Agent Runs | `POST /api/projects/{pid}/agent-runs/brd-to-frd` · `POST /…/frd-to-tc` · `GET /…` (list) · `GET /…/events` (SSE project-wide) · `GET /…/{run_id}` · `GET /…/{run_id}/events` (SSE per-run) · `POST /…/{run_id}/cancel` |
| Requirements | `GET /api/projects/{pid}/requirements` · `GET/PATCH/DELETE /…/{req_id}` · `POST /…/bulk-update` |
| TC Nodes | `GET /api/projects/{pid}/plans/{plan_id}/tc-nodes` (tree) · `GET/PATCH/DELETE /…/{node_id}` · `POST /…/bulk-update` (approve / archive / delete / select / deselect) |
| Execution | `POST /api/projects/{pid}/agent-runs/execute` (start) · `GET /…/{run_id}/steps` (per-step rows) · plus all `agent-runs` routes — same lifecycle and SSE topics |
| Static | `/static/screenshots/<run_id>/step_<step_id>.png` — per-step PNGs from execute runs |
| Debug · Parsers | `POST /api/_debug/parse` (md / paste) · `POST /api/_debug/parse-file` (pdf / docx / md) · `POST /api/_debug/chunk` |
| Debug · LLM | `POST /api/_debug/llm/structured` (round-trip a typed JSON LLM call) |
| Debug · Embedder | `POST /api/_debug/embed` · `GET /api/_debug/embed/status` |
| Debug · FAISS | `POST /api/_debug/faiss/add` · `POST /api/_debug/faiss/search` · `GET /api/_debug/faiss/info` · `DELETE /api/_debug/faiss` |
| Debug · SSE | `POST /api/_debug/sse/publish` · `POST /api/_debug/sse/demo` · `GET /api/_debug/sse/stream` · `GET /api/_debug/sse/topics` |
| Static | `/static/screenshots/*` · `/static/reports/*` |

Open <http://localhost:8000/docs> for the live Swagger UI.

---

## Phase 1 progress  ✅ shipped

- [x] 1. Scaffold `v2/` directory + pyproject + gitignore
- [x] 2. Backend boots (FastAPI + health + alembic)
- [x] 3. `app_settings` singleton + settings router
- [x] 4. LLM provider abstraction + factory + `/test` round-trip
- [x] 5. Embedding service (bge-large-en-v1.5)
- [x] 6. FAISS store (per-project IndexFlatIP, persisted)
- [x] 7. SSE event bus (sync→async, replayable)
- [x] 8. Projects model + CRUD router
- [x] 9. Frontend scaffold (Next 16 + Tailwind v4 + shadcn New York)
- [x] 10. First-run gate + Settings page
- [x] 11. Projects list + create dialog + detail shell
- [x] 12. Demo polish + README

---

## Phase 2 · Week 2 — Doc Ingest  ✅ shipped

- [x] 1. Documents + DocumentChunks models + migration `0003`
- [x] 2. Markdown + paste-text parsers (no new deps)
- [x] 3. PDF (PyMuPDF / `pymupdf4llm`) + DOCX (`python-docx`) parsers
- [x] 4. Heading-aware chunker (~800 chars / 100 overlap, code-fence aware)
- [x] 5. Ingest service — orchestrator + SSE events + FAISS upsert
- [x] 6. Documents router — upload / paste / list / detail / parsed / chunks / delete / search / events
- [x] 7. Project delete cascade — wipes `data/docs/<id>/` + chunk rows
- [x] 8. Frontend tabbed project layout (nested routes — Documents / Test Cases / Runs)
- [x] 9. Frontend `DocumentUploader` (drag-drop / file picker / paste dialog)
- [x] 10. Backend `INSTRUCTIONS` doc kind + frontend SSE hook (replaces polling)
- [x] 11. Frontend `DocumentDetail` — react-markdown render + chunks browser + raw view
- [x] 12. Frontend `SemanticSearch` panel + this README update

---

## Phase 2 · Week 3 — BRD → FRD agent + HITL review  ✅ shipped

The first agent goes online. Click a BRD → "Synthesize FRDs" → the agent
reads the chunks, derives functional requirements with traceability, and
streams progress live. Each generated FRD lands as `proposed`; user reviews
with Approve / Edit / Reject. Approving embeds the FRD into FAISS in a
dedicated `frd_requirements` namespace, ready for week 4's TC agent.

Steps shipped:

- [x] 1. `requirements` + `agent_runs` models + migration `0006`
- [x] 2. Cross-provider structured-JSON helper (`chat_structured`) — Gemini schema, OpenAI strict, compat parse-retry
- [x] 3. BRD→FRD orchestrator — pure function with phase events, schema-validated output, code auto-assignment
- [x] 4. Agent-run runtime — background task, status state machine, cancellation registry, dual SSE topics
- [x] 5. Agent-runs router — start/list/detail/cancel + project-wide and per-run SSE streams
- [x] 6. Requirements router — list/get/PATCH/bulk-update/delete with edit-demotes-to-edited semantics
- [x] 7. Approval pipeline — single + bulk + delete helpers wire approved FRDs into FAISS
- [x] 8. Frontend Requirements tab + nested route + tab nav
- [x] 9. Synthesize-FRD trigger modal with BRD picker + cap_chunks
- [x] 10. Live run progress card via SSE — phase narration, token counts, cancel button
- [x] 11. Review cards — collapsed preview / expanded body + rationale + source-chunk excerpts + actions
- [x] 12. Bulk approve/reject all proposed + filter chips + this README

End-to-end: **upload BRD → ingest → click Synthesize → watch live → review → approve → FAISS-ready for week 4**.

---

## Phase 2 · Week 4 — FRD → Test Case agent + tree HITL  ✅ shipped

The second agent goes online. Pick a Plan → "Generate test cases" → the
agent retrieves relevant approved FRDs (and optional linked-doc chunks) for
each module in `plan.scope`, runs **one LLM call per module**, and persists
a `Module → Submodule → Step` tree. Re-synthesis appends new module
siblings — never wipes existing trees, so partial progress survives cancel
or failure (commits are per-module).

The Test Cases tab is the HITL surface for the tree:

- **Recursive collapsible tree** — kind icons (folder / folder-tree /
  circle-dot), status pills, source-FRD count, action_type chip on steps
- **Hover row actions** — approve / undo-to-draft / archive / delete (CASCADE)
- **Detail side panel** (lg+) — narrative, target hint, expected, data needs,
  and a clickable list of source FRD codes that links back to the
  Requirements tab
- **Tri-state checkboxes** — selecting a parent flips every descendant; a
  partial fill renders the native indeterminate state. Persisted as
  `selectable_default` per row, so the run-time selection survives reloads
- **Filter chips** — All / Draft / Approved / Archived (parents stay visible
  if any descendant matches)
- **Bulk actions** — "Approve all N drafts" via the same bulk-update endpoint
  the Requirements tab uses; live run progress card with module-by-module
  progress bar reuses the week-3 SSE hook

Steps shipped:

- [x] 1. `tc_nodes` model + migration `0007` — self-FK with `ON DELETE CASCADE`, `selectable_default`, `path_cached`
- [x] 2. Pydantic schemas — `TcNodeTreeRead` self-reference + `TcNodeUpdate` + `TcNodeBulkUpdateRequest`
- [x] 3. FRD→TC orchestrator — module-by-module retrieval (`frd_requirements` + plan-linked chunks), per-module commits, source-FRD code mapping
- [x] 4. Runtime + `POST /agent-runs/frd-to-tc` — same lifecycle pattern as week 3, dual SSE topics
- [x] 5. TC nodes router — list (tree-shape) / get / PATCH (rebuilds `path_cached` subtree on rename) / DELETE / bulk-update
- [x] 6. Test Cases tab shell — plan picker + per-plan tree query
- [x] 7. Synthesize-TC dialog — plan summary + signal-source checks + caps
- [x] 8. Live run progress — module N/M phase ticks with progress bar, kind-aware completion summary
- [x] 9. Recursive `<TcTreeNode>` — caret, kind icons, status pills, hover actions
- [x] 10. Step detail side panel — read-only fields + source-FRD traceback links
- [x] 11. Tri-state selection checkboxes with parent/child propagation, backed by bulk `select`/`deselect` actions
- [x] 12. Bulk approve drafts + filter chips + this README

End-to-end: **plan with scope → generate test cases → live module ticks →
review tree → tri-state select what to run → ready for week 5's executor**.

---

## Phase 2 · Week 5 — Executor agent + live timeline  ✅ shipped

The third agent goes online. Click <strong>Start run</strong> on the Runs
tab → pick a plan → headed Chromium opens, walks the selected steps, and
streams per-step results in real time.

The executor consumes the tree the FRD→TC agent built (week 4): every step
the user ticked via the tri-state checkboxes runs in DFS order, with a
screenshot captured after each action. HITL data needs (`credentials`,
`otp`) short-circuit to ``blocked`` — the actual credential prefill + OTP
modal arrive in week 6.

Steps shipped:

- [x] 1. ``execution_steps`` table + migration ``0008`` — snapshot fields freeze title/path/action_type/target_hint/expected/narrative at run-time, so editing or deleting the source TcNode later doesn't mutate history. CASCADE on run + project; SET NULL on plan + tc_node
- [x] 2. ``ExecutionStepRead`` + ``ExecuteRunRequest`` + ``ExecutionRunSummary`` schemas
- [x] 3. Playwright wiring (``playwright>=1.48``) — ``app/executor/browser.py`` context-manager, ``chromium_installed()`` pre-flight probe, headed Chromium by default
- [x] 4. Selector waterfall — engine prefix → text marker → CSS (always tried) → plain text → role probe; ``count() == 0 → skip`` per stage
- [x] 5. Action dispatcher — one handler per ``action_type`` (navigate / click / type / select / verify / wait / submit / screenshot); ``data_needs.kind ∈ {credentials, otp}`` short-circuits to ``blocked``
- [x] 6. Execute orchestrator — DFS+ordinal walk, ancestor cut on ``selectable_default=False``, per-step screenshot to ``data/screenshots/<run_id>/``, fail-doesn't-cut-siblings semantics, cancel between steps with ``skipped`` for leftovers
- [x] 7. ``execute_run`` runtime — same lifecycle pattern as week 3/4, with ``BrowserNotInstalledError`` translated to a 4xx-style failure message
- [x] 8. ``POST /agent-runs/execute`` + ``GET /agent-runs/{id}/steps`` — pre-flights project, plan, target_url, chromium binary, selected_step_ids ownership, and "any selectable steps exist" before queueing
- [x] 9. Runs tab list + ``StartExecuteDialog`` (plan picker + headed/headless toggle); ``RunProgressCard`` extended with execute-kind summary (``N/M passed · K failed · K blocked · X.Xs``)
- [x] 10. ``<ExecutionTimeline>`` — SSE-driven live row updates, status icons + colored left-border, screenshot thumbnails with click-to-zoom lightbox, error message inlined for failed/blocked rows
- [x] 11. Run detail page — ``<RunHeader>`` (plan link + browser mode + wall-clock vs. stepwork duration), filter chips by status with live counts, "Re-run N failed steps" button that bundles failed ``tc_node_id``s into a new run via the ``selected_step_ids`` override
- [x] 12. Smoke test against live example.com (3 passed, 1 intentionally failing) + this README

End-to-end: **pick the steps you want to run on the Test Cases tab → Runs
tab → Start run → watch headed Chromium walk through them → click any
screenshot to zoom → re-run only the failed ones with one button**.

---

## Phase 2 · Week 6 — Visible cursor + narration overlay  ✅ shipped

The "watch the agent work" UX from agent-watching tools — every page the
executor visits gets two overlays injected via Playwright's init-script:

- **Cursor ring** — circular indicator that tracks `mousemove` and pulses
  on click. Lets a viewer see where the agent is looking, even between
  actions.
- **Narration banner** — translucent bottom pill with an action-type chip,
  the step title, and an *N/M* step counter on the right. Updated by the
  orchestrator at each `step_started` boundary.

Always-on (headed and headless) so per-step PNGs in
`data/screenshots/<run_id>/` capture the cursor position + narration that
was active when the screenshot fired — the run-detail timeline thumbnails
inherit it for free.

Implementation: [`app/executor/overlay.py`](backend/app/executor/overlay.py).
Idempotent across navigations via `page.context.add_init_script(...)`. JS
errors during `update_narration` are swallowed silently — pages can close
or block JS, and a missed banner update must never derail the run.

The other parts of the originally-scoped week 6 (credentials prefill from
`test_plan_credentials`, OTP HITL modal with 180s timeout) are captured
in [`futurescope.md`](futurescope.md) for a future session — explicitly
deferred per scope decision.

---

## Phase 2 · Week 2.5 — Test Plans  ✅ shipped

A `Project` owns many `TestPlan`s. A plan bundles execution config:

- **Target URL** — the app under test
- **Login credentials** — one or more rows (admin / user / etc.); plaintext per local-MVP policy
- **Scope** — module names ("Authentication", "Dashboard"); dropdown pre-populated from headings of linked docs
- **Instructions** — free-text guidance for the agent
- **Linked documents** (optional) — many-to-many to BRD/FRD/INSTRUCTIONS docs

OTP secrets are **not** stored — handled live via HITL intervention every time
(Phase 2 · Week 6).

Steps shipped:

- [x] A. `test_plans` + `test_plan_credentials` + `test_plan_documents` models + migration `0005`
- [x] B. TestPlan router — CRUD + credentials sub-CRUD + heading-suggestions
- [x] C. Frontend Plans tab + list view + quick-create dialog
- [x] D. Plan editor — basic-fields save + scope chips with live suggestions + linked-doc picker + credentials table + delete
- [x] E. Documents↔Plans cross-link callouts + README

---

## Coming up — Phase 2 weeks 7–8

1. **Intervention state machine** — credentials prefill from `test_plan_credentials` injected into `data_needs.kind='credentials'` steps; OTP / confirm modals via in-memory vault (180s timeout, never persisted). Pitched scope captured in [`futurescope.md`](futurescope.md).
2. **Accuracy + transparency** — replay timeline polish, "Why" explanations on each decision, cost meter
3. **Reports** — in-app table + Excel download, regen-on-doc-change with diff modal

---

## Notes / gotchas

- **Pyrefly "missing-import" warnings** mean VS Code is using your global Python interpreter. Run `uv sync` in `v2/backend/` and select `.venv\Scripts\python.exe` (`Ctrl+Shift+P` → "Python: Select Interpreter").
- **First request to `/api/_debug/embed` is slow** — the BGE model loads lazily (~5–10 s) and downloads ~1.3 GB to `~/.cache/huggingface/` on first use.
- **GPT-5 reasoning models reject `temperature` and `max_tokens`** — the OpenAI provider handles this by routing native OpenAI requests to `max_completion_tokens` and never sending `temperature` in test calls.
- **API keys in `qai.db` are plaintext** per the local-MVP "no master key" decision. Anyone with file access can read them. Surface this in any production-pivot conversation.
- **First execute run is slow** — the Chromium binary downloads on demand the first time it's missing (`uv run playwright install chromium`, ~290 MB across `chromium-*` and `chromium_headless_shell-*`). The runner pre-flight probe surfaces this as a 503 with the exact remedy command instead of crashing.
- **Headed runs can be "stolen"** — Chromium grabs window focus on launch (Windows + macOS). Switch to headless on the Start-run dialog when you don't want to watch.
