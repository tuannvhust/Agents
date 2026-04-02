"""Asyncpg connection pool — process-level singleton.

Usage
-----
Initialise once at application startup (inside the FastAPI lifespan)::

    from agent_system.database import init_pool
    await init_pool(dsn="postgresql://user:pass@host/db")

Acquire a connection anywhere::

    from agent_system.database import get_pool
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow("SELECT 1")

Shut down cleanly on application exit::

    from agent_system.database import close_pool
    await close_pool()
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level singleton — set by init_pool(), read by get_pool()
_pool: Any = None  # asyncpg.Pool


async def init_pool(
    dsn: str,
    min_size: int = 2,
    max_size: int = 10,
) -> None:
    """Create the asyncpg connection pool.

    Safe to call multiple times — subsequent calls are no-ops if the pool
    is already initialised.
    """
    global _pool

    if _pool is not None:
        logger.debug("DB pool already initialised — skipping.")
        return

    import asyncpg

    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=min_size,
        max_size=max_size,
        command_timeout=60,
        # Return dicts instead of asyncpg.Record where convenient
    )
    logger.info(
        "DB pool ready (asyncpg) — min=%d max=%d dsn=%s",
        min_size,
        max_size,
        _redact_dsn(dsn),
    )


async def close_pool() -> None:
    """Gracefully close all pool connections."""
    global _pool

    if _pool is None:
        return
    await _pool.close()
    _pool = None
    logger.info("DB pool closed.")


def get_pool():
    """Return the active asyncpg pool.

    Raises RuntimeError if ``init_pool()`` has not been called yet.
    """
    if _pool is None:
        raise RuntimeError(
            "DB pool is not initialised. "
            "Call agent_system.database.init_pool() at application startup."
        )
    return _pool


def _redact_dsn(dsn: str) -> str:
    """Replace the password in a DSN with *** for safe logging."""
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(dsn)
        if p.password:
            netloc = p.netloc.replace(p.password, "***")
            return urlunparse(p._replace(netloc=netloc))
    except Exception:  # noqa: BLE001
        pass
    return dsn
