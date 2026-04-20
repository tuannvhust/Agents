"""Bootstrap helper for the Chainlit chat UI.

Handles one-time async initialisation (DB pool, tool registry, agent cache) so
``app.py`` can be kept focused on UI logic.  Uses a module-level singleton so
multiple Chainlit sessions share the same in-process state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_system.config import get_settings
from agent_system.core.agent import Agent, AgentConfig

logger = logging.getLogger(__name__)

# ── Singleton state ────────────────────────────────────────────────────────────
_initialized = False
_init_lock = asyncio.Lock()
_agent_cache: dict[str, Agent] = {}
_tool_registry: Any = None


async def ensure_initialized() -> None:
    """Idempotent async init — safe to call on every chat start."""
    global _initialized
    if _initialized:
        return
    async with _init_lock:
        if _initialized:
            return
        await _do_init()
        _initialized = True


async def _do_init() -> None:
    cfg = get_settings()

    # ── Database pool ────────────────────────────────────────────────────────
    from agent_system.database import init_pool
    await init_pool(cfg.agent_postgres_url)
    logger.info("Chat: DB pool ready.")

    # ── Optional Redis cache ──────────────────────────────────────────────────
    if cfg.cache_enabled and cfg.cache_type.lower().strip() == "redis":
        try:
            from agent_system.cache.redis_client import init_redis
            await init_redis(cfg.cache_redis_url)
            logger.info("Chat: Redis cache connected.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chat: Redis cache unavailable (%s) — Postgres-only mode.", exc)

    # ── Tool registry ─────────────────────────────────────────────────────────
    global _tool_registry
    from agent_system.tools.registry import ToolRegistry
    from agent_system.tools.builtin_tools import ALL_BUILTIN_TOOLS

    _tool_registry = ToolRegistry()
    _tool_registry.register_many(ALL_BUILTIN_TOOLS)
    await _tool_registry.load_mcp_tools()
    logger.info("Chat: tool registry ready (%d tools).", len(_tool_registry))

    # ── Langfuse tracing ──────────────────────────────────────────────────────
    try:
        from agent_system.tracing import init_langfuse_handler
        init_langfuse_handler()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Chat: Langfuse init skipped: %s", exc)

    # ── Load agents from DB ───────────────────────────────────────────────────
    from agent_system.storage.agent_config_store import AgentConfigStore
    store = AgentConfigStore()
    configs = await store.load_all()

    for cfg_dict in configs:
        name = cfg_dict.get("name", "")
        try:
            config = AgentConfig(
                name=name,
                skill_name=cfg_dict["skill_name"],
                model=cfg_dict.get("model"),
                model_source=cfg_dict.get("model_source", "openrouter"),
                temperature=float(cfg_dict.get("temperature", 0.0)),
                max_reflections=int(cfg_dict.get("max_reflections", 3)),
                tools=cfg_dict.get("tools") or [],
                tools_requiring_approval=cfg_dict.get("tools_requiring_approval") or [],
                plugins=cfg_dict.get("plugins") or [],
                extra_metadata=cfg_dict.get("extra_metadata") or {},
            )
            agent = await Agent.create(config, tool_registry=_tool_registry)
            _agent_cache[name] = agent
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chat: could not load agent '%s': %s", name, exc)

    if _agent_cache:
        logger.info("Chat: loaded %d agent(s): %s", len(_agent_cache), list(_agent_cache))
    else:
        logger.warning("Chat: no agents found in DB — create one via the API first.")


async def reload_agents() -> None:
    """Reload agent configs from DB (call after creating a new agent via API)."""
    global _tool_registry
    if not _initialized or _tool_registry is None:
        return
    from agent_system.storage.agent_config_store import AgentConfigStore
    store = AgentConfigStore()
    configs = await store.load_all()
    for cfg_dict in configs:
        name = cfg_dict.get("name", "")
        if name in _agent_cache:
            continue
        try:
            config = AgentConfig(
                name=name,
                skill_name=cfg_dict["skill_name"],
                model=cfg_dict.get("model"),
                model_source=cfg_dict.get("model_source", "openrouter"),
                temperature=float(cfg_dict.get("temperature", 0.0)),
                max_reflections=int(cfg_dict.get("max_reflections", 3)),
                tools=cfg_dict.get("tools") or [],
                tools_requiring_approval=cfg_dict.get("tools_requiring_approval") or [],
                plugins=cfg_dict.get("plugins") or [],
                extra_metadata=cfg_dict.get("extra_metadata") or {},
            )
            agent = await Agent.create(config, tool_registry=_tool_registry)
            _agent_cache[name] = agent
            logger.info("Chat: reloaded agent '%s'.", name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chat: reload failed for '%s': %s", name, exc)


def list_agents() -> list[str]:
    """Sorted list of available agent names."""
    return sorted(_agent_cache.keys())


def get_agent(name: str) -> Agent | None:
    """Return the Agent instance or None."""
    return _agent_cache.get(name)
