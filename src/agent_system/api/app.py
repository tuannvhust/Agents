"""FastAPI application factory and lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent_system import __version__
from agent_system.config import get_settings
from agent_system.logging import configure_logging
from agent_system.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
_tool_registry: ToolRegistry | None = None
_config_store = None   # AgentConfigStore — typed loosely to avoid circular import
_run_store = None      # RunStore — persists runs + tool calls + memory


def get_tool_registry() -> ToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def get_config_store():
    global _config_store
    if _config_store is None:
        from agent_system.storage.agent_config_store import AgentConfigStore
        _config_store = AgentConfigStore()
    return _config_store


def get_run_store():
    global _run_store
    if _run_store is None:
        from agent_system.storage.run_store import RunStore
        _run_store = RunStore()
    return _run_store


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown lifecycle hooks."""
    cfg = get_settings()

    configure_logging(cfg.app.log_level)
    logger.info("Starting Agent System v%s", __version__)

    # ── Database pool (asyncpg) ───────────────────────────────────────────────
    from agent_system.database import close_pool, init_pool
    await init_pool(cfg.agent_postgres_url)

    # Ensure agent_configs table exists (idempotent — harmless on every start)
    await _ensure_agent_configs_table()

    # ── Built-in tools + MCP tools ────────────────────────────────────────────
    registry = get_tool_registry()
    _register_builtin_tools(registry)
    await registry.load_mcp_tools()
    logger.info("Tool registry ready with %d tool(s).", len(registry))

    # ── Langfuse tracing ──────────────────────────────────────────────────────
    from agent_system.tracing import init_langfuse_handler
    init_langfuse_handler()

    # ── Restore agents from DB ────────────────────────────────────────────────
    await _restore_agents_from_db()

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Agent System shutting down.")
    await close_pool()


async def _ensure_agent_configs_table() -> None:
    """Create agent_configs if it does not exist (moved from AgentConfigStore._ensure_table)."""
    from agent_system.database import get_pool
    try:
        async with get_pool().acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_configs (
                    name        TEXT        PRIMARY KEY,
                    config      JSONB       NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        logger.debug("agent_configs table ensured.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not ensure agent_configs table: %s", exc)


async def _restore_agents_from_db() -> None:
    """Load all persisted AgentConfig rows and rebuild Agent objects in cache."""
    from agent_system.api.routes.agents import _agent_cache
    from agent_system.core.agent import Agent, AgentConfig

    store = get_config_store()
    registry = get_tool_registry()

    configs = await store.load_all()
    if not configs:
        logger.info("No persisted agent configs found — starting fresh.")
        return

    restored = 0
    for cfg_dict in configs:
        name = cfg_dict.get("name", "")
        try:
            config = AgentConfig(
                name=name,
                skill_name=cfg_dict["skill_name"],
                model=cfg_dict.get("model"),
                model_source=cfg_dict.get("model_source", "openrouter"),
                temperature=cfg_dict.get("temperature", 0.0),
                max_reflections=cfg_dict.get("max_reflections", 3),
                tools=cfg_dict.get("tools", []),
                extra_metadata=cfg_dict.get("extra_metadata", {}),
            )
            agent = await Agent.create(config, tool_registry=registry)
            _agent_cache[name] = agent
            restored += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not restore agent '%s': %s", name, exc)

    logger.info("Restored %d agent(s) from PostgreSQL.", restored)


def _register_builtin_tools(registry: ToolRegistry) -> None:
    from agent_system.tools.builtin_tools import ALL_BUILTIN_TOOLS

    registry.register_many(ALL_BUILTIN_TOOLS)
    logger.info(
        "Registered %d built-in tool(s): %s",
        len(ALL_BUILTIN_TOOLS),
        [t.name for t in ALL_BUILTIN_TOOLS],
    )


# ── Factory ───────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="Agent System",
        description=(
            "Production-ready multi-agent framework with LangGraph, MCP, MinIO, "
            "ElasticSearch and Langfuse."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    from agent_system.api.middleware import RequestIDMiddleware
    app.add_middleware(RequestIDMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if cfg.app.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    from agent_system.api.routes import agents_router, debug_router, files_router, health_router

    app.include_router(health_router)
    app.include_router(agents_router)
    app.include_router(debug_router)
    app.include_router(files_router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request, exc):  # noqa: ANN001
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    return app
