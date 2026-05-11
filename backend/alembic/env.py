"""Alembic environment.

Pulls metadata from ``app.db.Base`` so autogenerate sees every ORM model that
``app.models`` has registered. Models are imported below; add new model
modules to that import block as they're created.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the backend root importable so we can grab ``app.*``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import Base  # noqa: E402

# ── Register all model modules here so autogenerate sees them ──
# Imported for side effects only (the import populates Base.metadata).
# Add new model modules to the tuple below as they're created.
from app.models import (  # noqa: E402
    agent_run,
    app_knowledge,
    app_settings,
    document,
    execution_step,
    llm_call_log,
    project,
    requirement,
    tc_node,
    test_plan,
)

_REGISTERED_MODELS = (
    agent_run,
    app_knowledge,
    app_settings,
    document,
    execution_step,
    llm_call_log,
    project,
    requirement,
    tc_node,
    test_plan,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # SQLite needs batch mode for ALTER TABLE
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
