"""FastAPI application entry point — QAI Tester v2."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import (
    _debug,
    agent_runs,
    documents,
    health,
    projects,
    recordings,
    requirements,
    settings as settings_router,
    sub_flow_modules,
    tc_nodes,
    test_plans,
)

# Windows: enable subprocess support in asyncio (needed by Playwright in later phases)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting QAI Tester v2 backend")
    for d in (
        settings.data_dir,
        settings.faiss_dir,
        settings.docs_dir,
        settings.screenshots_dir,
        settings.reports_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Data directories ready under %s", settings.data_dir.resolve())

    # Phase K.2 — reap orphaned agent runs from the previous process.
    # Any row left in (queued / running / paused) with no completed_at
    # is dead by construction: no Python thread exists to drive it.
    # Mark them as ``cancelled`` so the UI doesn't show ghost runs and
    # the duplicate-run guards don't refuse new starts.
    try:
        from app.services.agent_run_service import reap_orphaned_runs  # noqa: PLC0415
        result = reap_orphaned_runs(stale_after_seconds=0)
        if result.get("count"):
            logger.warning(
                "Startup reaper cancelled %d orphaned run(s): %s",
                result["count"], result.get("reaped"),
            )
    except Exception as e:
        logger.exception("startup reaper failed (non-fatal): %s", e)

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="QAI Tester v2",
    description="Agentic QA platform — local MVP",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Phase Y.1 — permissive CORS for recording-event ingest.
#
# The JS injected into the recording browser runs at the TARGET app's
# origin (e.g. some Cloudflare-tunnelled internal admin). It POSTs
# captured events back to ``http://localhost:8000/api/recordings/
# {run_id}/events``. The browser fires an OPTIONS preflight; the
# global CORSMiddleware above only allows the qai frontend origin, so
# OPTIONS returns 400 and the POST never fires — empty buffer, "0
# captured" forever.
#
# We can't widen the global CORS without weakening every other
# endpoint. Instead: a tiny ASGI middleware that intercepts requests
# to ``/api/recordings/<id>/events`` and either (a) short-circuits
# OPTIONS with a permissive 200, or (b) injects
# ``Access-Control-Allow-Origin: *`` on the POST response BEFORE
# CORSMiddleware sees it. ``add_middleware`` is a stack (last added
# runs first), so registering this AFTER CORSMiddleware means it
# fires FIRST and can hand back its own response.


class _RecordingIngestCorsMiddleware:
    """Permissive CORS for the recording-event ingest path only.

    Any other path passes through to the global CORSMiddleware
    unchanged.
    """

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        path = scope.get("path") or ""
        method = (scope.get("method") or "").upper()
        is_recording_ingest = (
            path.startswith("/api/recordings/")
            and path.endswith("/events")
        )
        if not is_recording_ingest:
            await self._app(scope, receive, send)
            return

        permissive_headers = [
            (b"access-control-allow-origin", b"*"),
            (b"access-control-allow-methods", b"POST, OPTIONS"),
            (b"access-control-allow-headers", b"content-type"),
            (b"access-control-max-age", b"3600"),
            (b"vary", b"origin"),
        ]

        if method == "OPTIONS":
            # Short-circuit preflight — bypass the strict global
            # CORSMiddleware entirely.
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": permissive_headers + [
                    (b"content-length", b"0"),
                ],
            })
            await send({"type": "http.response.body", "body": b""})
            return

        # For POST, intercept the response and inject ACAO:* if the
        # downstream handler didn't already set one. Keeps the
        # browser happy when reading the response from a cross-
        # origin XHR.
        async def patched_send(message):
            if message["type"] == "http.response.start":
                existing = [
                    (k, v) for (k, v) in (message.get("headers") or [])
                    if k.lower() != b"access-control-allow-origin"
                ]
                message["headers"] = existing + permissive_headers
            await send(message)

        await self._app(scope, receive, patched_send)


app.add_middleware(_RecordingIngestCorsMiddleware)

app.mount(
    "/static/screenshots",
    StaticFiles(directory=settings.screenshots_dir),
    name="screenshots",
)
app.mount(
    "/static/reports",
    StaticFiles(directory=settings.reports_dir),
    name="reports",
)

app.include_router(health.router)
app.include_router(settings_router.router)
app.include_router(projects.router)
app.include_router(documents.router)
app.include_router(test_plans.router)
app.include_router(agent_runs.router)
app.include_router(requirements.router)
app.include_router(tc_nodes.router)
app.include_router(sub_flow_modules.router)
# Phase W — recording ingest + lifecycle endpoints. ``public_router``
# accepts JS-posted event batches at /api/recordings/{run_id}/events;
# ``lifecycle_router`` exposes start / stop / discard under the
# existing project-scoped agent-runs path.
app.include_router(recordings.public_router)
app.include_router(recordings.lifecycle_router)
app.include_router(_debug.router)


@app.get("/")
def root():
    return {
        "name": "QAI Tester v2",
        "version": "0.1.0",
        "docs": "/docs",
    }
