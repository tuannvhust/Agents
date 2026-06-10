"""Shared LangGraph checkpointer for human-in-the-loop (interrupt / resume)."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

_saver: Any = None
_saver_cm: Any = None  # async context manager for Postgres saver shutdown


async def init_checkpoint_saver(postgres_dsn: str | None = None) -> None:
    """Initialise the process-wide checkpointer.

    When ``postgres_dsn`` is set (job-queue mode), checkpoints live in Postgres so
    any worker can resume a paused run.  Otherwise falls back to in-memory storage.
    """
    global _saver, _saver_cm
    if _saver is not None:
        return

    if postgres_dsn:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        _saver_cm = AsyncPostgresSaver.from_conn_string(postgres_dsn)
        _saver = await _saver_cm.__aenter__()
        logger.info("LangGraph checkpointer: Postgres (%s)", _redact_dsn(postgres_dsn))
        return

    _saver = MemorySaver()
    logger.info("LangGraph checkpointer: in-memory MemorySaver")


async def close_checkpoint_saver() -> None:
    """Release checkpoint resources on shutdown."""
    global _saver, _saver_cm
    if _saver_cm is not None:
        await _saver_cm.__aexit__(None, None, None)
        _saver_cm = None
    _saver = None


def get_checkpoint_saver():
    """Return the active checkpointer (call only after ``init_checkpoint_saver``)."""
    if _saver is None:
        _fallback = MemorySaver()
        logger.warning(
            "Checkpoint saver not initialised — using ephemeral MemorySaver. "
            "Call init_checkpoint_saver() at startup for production."
        )
        return _fallback
    return _saver


def _redact_dsn(dsn: str) -> str:
    if "@" not in dsn:
        return dsn
    head, tail = dsn.split("@", 1)
    if "://" in head:
        scheme, _ = head.split("://", 1)
        return f"{scheme}://***@{tail}"
    return "***@" + tail
