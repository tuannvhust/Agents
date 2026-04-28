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


def _cache_redis_active(cfg) -> bool:
    return cfg.cache_enabled and cfg.cache_type.lower().strip() == "redis"


def get_run_store():
    global _run_store
    if _run_store is None:
        from agent_system.storage.caching_run_store import CachingRunStore
        from agent_system.storage.run_store import RunStore

        cfg = get_settings()
        inner = RunStore()
        if _cache_redis_active(cfg):
            from agent_system.cache.redis_client import get_redis

            _run_store = CachingRunStore(
                inner,
                get_redis(),
                memory_ttl_seconds=cfg.cache_memory_ttl_seconds,
                conversation_ttl_seconds=cfg.cache_conversation_ttl_seconds,
                tool_messages_ttl_seconds=cfg.cache_tool_messages_ttl_seconds,
            )
            logger.info(
                "RunStore: Redis cache-aside enabled (memory TTL=%ds, run TTL=%ds, tool_calls TTL=%ds)",
                cfg.cache_memory_ttl_seconds,
                cfg.cache_conversation_ttl_seconds,
                cfg.cache_tool_messages_ttl_seconds,
            )
        else:
            _run_store = inner
            if cfg.cache_enabled:
                logger.warning(
                    "CACHE_ENABLED=true but CACHE_TYPE=%r is not supported; using Postgres only.",
                    cfg.cache_type,
                )
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

    if _cache_redis_active(cfg):
        from agent_system.cache.redis_client import init_redis

        logger.info("Connecting Redis cache at startup (CACHE_ENABLED=true, CACHE_TYPE=redis).")
        await init_redis(cfg.cache_redis_url)
    elif cfg.cache_enabled:
        logger.warning(
            "CACHE_ENABLED=true but CACHE_TYPE=%r is not 'redis'; skipping Redis startup.",
            cfg.cache_type,
        )

    # Ensure agent_configs table exists (idempotent — harmless on every start)
    await _ensure_agent_configs_table()

    # Older agent-postgres volumes may lack run_trace; app code expects it (see init-db/02_run_trace_column.sql)
    await _ensure_agent_runs_run_trace_column()

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
    if _cache_redis_active(cfg):
        from agent_system.cache.redis_client import close_redis

        await close_redis()
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


async def _ensure_agent_runs_run_trace_column() -> None:
    """Add run_trace to agent_runs if missing (existing DBs created before the column existed)."""
    from agent_system.database import get_pool
    try:
        async with get_pool().acquire() as conn:
            await conn.execute(
                """
                ALTER TABLE agent_runs
                    ADD COLUMN IF NOT EXISTS run_trace JSONB NOT NULL DEFAULT '{}'::jsonb
                """
            )
        logger.debug("agent_runs.run_trace column ensured.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not ensure agent_runs.run_trace column: %s", exc)


async def _restore_agents_from_db() -> None:
    """Load all persisted AgentConfig rows and rebuild Agent objects in cache.

    Two-pass restoration: sub-agents are built first so that coordinator agents
    can resolve their invoke_* tools against the already-populated cache.
    """
    from agent_system.api.routes.agents import _agent_cache
    from agent_system.core.agent import Agent, AgentConfig

    store = get_config_store()
    registry = get_tool_registry()

    configs = await store.load_all()
    if not configs:
        logger.info("No persisted agent configs found — starting fresh.")
        return

    # Sort so sub-agents are restored before coordinators
    sub_configs = [c for c in configs if c.get("role", "subagent") != "coordinator"]
    coord_configs = [c for c in configs if c.get("role", "subagent") == "coordinator"]
    ordered = sub_configs + coord_configs

    restored = 0
    for cfg_dict in ordered:
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
                tools_requiring_approval=cfg_dict.get("tools_requiring_approval", []),
                plugins=cfg_dict.get("plugins", []),
                extra_metadata=cfg_dict.get("extra_metadata", {}),
                role=cfg_dict.get("role", "subagent"),
                sub_agents=cfg_dict.get("sub_agents", []),
            )
            agent = await Agent.create(
                config, tool_registry=registry, agent_cache=_agent_cache
            )
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
    from agent_system.api.routes import (
        agents_router,
        debug_router,
        files_router,
        health_router,
        review_router,
    )

    app.include_router(health_router)
    app.include_router(agents_router)
    app.include_router(review_router)
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
