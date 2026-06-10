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

    from agent_system.runtime.bootstrap import bootstrap_shutdown, bootstrap_startup

    await bootstrap_startup(load_agents=True)

    if cfg.queue_enabled:
        from agent_system.queue import close_queue_pool, init_queue_pool

        await init_queue_pool(cfg.rabbitmq_url)
        logger.info("RabbitMQ job queue enabled (RABBITMQ_URL).")
    elif cfg.cache_enabled and cfg.cache_type.lower().strip() != "redis":
        logger.warning(
            "CACHE_ENABLED=true but CACHE_TYPE=%r is not 'redis'; skipping Redis startup.",
            cfg.cache_type,
        )

    yield

    logger.info("Agent System shutting down.")
    if cfg.queue_enabled:
        from agent_system.queue import close_queue_pool

        await close_queue_pool()
    await bootstrap_shutdown()


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
                enable_ocr=cfg_dict.get("enable_ocr", False),
                ocr_model=cfg_dict.get("ocr_model"),
                ocr_model_source=cfg_dict.get("ocr_model_source"),
                ocr_skill_name=cfg_dict.get("ocr_skill_name", "ocr"),
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
        runs_router,
    )

    app.include_router(health_router)
    app.include_router(agents_router)
    app.include_router(runs_router)
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
