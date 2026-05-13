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
app.include_router(_debug.router)


@app.get("/")
def root():
    return {
        "name": "QAI Tester v2",
        "version": "0.1.0",
        "docs": "/docs",
    }
