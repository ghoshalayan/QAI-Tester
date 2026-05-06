# QAI Tester v2 — Backend

FastAPI + SQLAlchemy 2.0 + SQLite (WAL) + Alembic + sentence-transformers + FAISS + SSE.

See [../README.md](../README.md) for the full Phase 1 demo and architecture.

## Run

```bash
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

API: <http://localhost:8000/docs>

## Module layout

```
app/
├── main.py                 # FastAPI entry, CORS, lifespan, static mounts
├── config.py               # Pydantic-settings (env_prefix=QAI_)
├── db.py                   # SQLAlchemy 2.0 DeclarativeBase + WAL pragmas + get_db
├── models/                 # ORM models — register new ones in alembic/env.py
│   ├── app_settings.py     # singleton: provider + model + api_key + base_url
│   └── project.py
├── schemas/                # Pydantic in/out DTOs
├── llm/
│   ├── base.py             # LLMProvider ABC + ChatMessage / ChatResult / TestConnectionResult
│   ├── gemini.py           # google-genai SDK
│   ├── openai_provider.py  # openai SDK; covers native + compat (Ollama, vLLM, …)
│   └── factory.py          # build_provider + cached get_provider + invalidate_cache
├── embeddings/
│   └── bge.py              # BAAI/bge-large-en-v1.5 singleton, lazy, L2-normalized
├── faiss_store/
│   └── store.py            # per-(project, namespace) IndexIDMap[IndexFlatIP], persisted
├── sse/
│   ├── bus.py              # in-memory pub/sub, sync producers + async consumers
│   └── response.py         # sse_for_topic() → EventSourceResponse
├── routers/
│   ├── health.py           # /api/health
│   ├── settings.py         # /api/settings (GET/PUT/DELETE/test)
│   ├── projects.py         # /api/projects (CRUD)
│   └── _debug.py           # /api/_debug/* — embed / faiss / sse smoke endpoints
└── services/               # (Phase 2: agent-facing helpers)
```

## Migrations

```bash
# Create a new migration after changing models
uv run alembic revision --autogenerate -m "describe the change"

# Apply
uv run alembic upgrade head

# Roll back one step
uv run alembic downgrade -1
```

When you add a new model file, add it to `alembic/env.py`:

```python
from app.models import app_settings, project, your_new_model
_REGISTERED_MODELS = (app_settings, project, your_new_model)
```

## Testing the foundations without a frontend

Each foundational service has debug endpoints under `/api/_debug/*`:

```bash
# Embedder
curl -X POST localhost:8000/api/_debug/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["sign in with email"], "is_query": true}'

# FAISS — add then search
curl -X POST localhost:8000/api/_debug/faiss/add \
  -H 'Content-Type: application/json' \
  -d '{"project_id":1,"namespace":"smoke","docs":[{"id":1,"text":"login form"}]}'
curl -X POST localhost:8000/api/_debug/faiss/search \
  -H 'Content-Type: application/json' \
  -d '{"project_id":1,"namespace":"smoke","query":"how do I log in","k":3}'

# SSE — terminal A subscribes, terminal B publishes
curl -N 'localhost:8000/api/_debug/sse/stream?topic=demo'
curl -X POST localhost:8000/api/_debug/sse/demo \
  -H 'Content-Type: application/json' \
  -d '{"topic":"demo","count":5,"interval_seconds":1.0}'
```
