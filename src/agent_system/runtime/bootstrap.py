"""Startup / shutdown shared by the FastAPI app and RabbitMQ worker."""

from __future__ import annotations

import logging
from pathlib import Path

from agent_system.config import get_settings

logger = logging.getLogger(__name__)


async def bootstrap_startup(*, load_agents: bool = True) -> None:
    """Initialise DB, Redis, checkpoints, tools, and optionally restore agents."""
    cfg = get_settings()

    from agent_system.database import init_pool

    await init_pool(
        cfg.agent_postgres_url,
        min_size=cfg.db_pool_min,
        max_size=cfg.db_pool_max,
    )

    redis_url = _resolve_redis_url(cfg)
    if redis_url:
        from agent_system.cache.redis_client import init_redis

        await init_redis(redis_url)

    from agent_system.core.interrupt_registry import configure_interrupt_registry

    configure_interrupt_registry(use_redis=bool(redis_url))

    from agent_system.core.checkpointing import init_checkpoint_saver

    await init_checkpoint_saver(cfg.agent_postgres_url if cfg.queue_enabled else None)

    await _ensure_schema_migrations()

    from agent_system.tools.registry import ToolRegistry
    from agent_system.api.app import get_tool_registry, _register_builtin_tools

    registry = get_tool_registry()
    if not registry.all():
        _register_builtin_tools(registry)
    await registry.load_mcp_tools()
    logger.info("Tool registry ready with %d tool(s).", len(registry))

    from agent_system.tracing import init_langfuse_handler

    init_langfuse_handler()

    if load_agents:
        from agent_system.api.app import _restore_agents_from_db

        await _restore_agents_from_db()


async def bootstrap_shutdown() -> None:
    """Release shared resources."""
    cfg = get_settings()

    from agent_system.core.checkpointing import close_checkpoint_saver

    await close_checkpoint_saver()

    if _resolve_redis_url(cfg):
        from agent_system.cache.redis_client import close_redis

        await close_redis()

    from agent_system.database import close_pool

    await close_pool()


def _resolve_redis_url(cfg) -> str | None:
    """Redis for cache-aside and HITL pending approvals (not the job queue)."""
    if cfg.cache_enabled and cfg.cache_type.lower().strip() == "redis":
        return cfg.cache_redis_url
    if cfg.queue_enabled:
        # Human-in-the-loop state is shared between API and workers via Redis.
        return cfg.cache_redis_url
    return None


def _init_db_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "init-db"


def _sql_statements(path: Path) -> list[str]:
    """Split a SQL file into executable statements (line comments stripped)."""
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("--")
    ]
    return [stmt.strip() for stmt in "\n".join(lines).split(";") if stmt.strip()]


async def _ensure_schema_migrations() -> None:
    from agent_system.database import get_pool

    upgrade_sql = _init_db_dir() / "02_upgrade.sql"
    if not upgrade_sql.is_file():
        logger.warning("Schema upgrade file not found: %s", upgrade_sql)
        return

    try:
        async with get_pool().acquire() as conn:
            for stmt in _sql_statements(upgrade_sql):
                await conn.execute(stmt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not ensure schema migrations: %s", exc)
