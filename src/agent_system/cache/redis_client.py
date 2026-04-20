"""Async Redis connection used when CACHE_ENABLED=true and CACHE_TYPE=redis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_client: Redis | None = None


async def init_redis(url: str) -> None:
    """Open a shared async Redis client and verify connectivity."""
    global _client
    if _client is not None:
        return
    import redis.asyncio as redis

    _client = redis.from_url(url, decode_responses=True, socket_connect_timeout=5)
    await _client.ping()
    logger.info("Redis cache connected (%s)", _mask_url(url))


async def close_redis() -> None:
    """Close the Redis client if it was opened."""
    global _client
    if _client is None:
        return
    try:
        await _client.aclose()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis close: %s", exc)
    finally:
        _client = None


def get_redis() -> Redis:
    """Return the live async Redis client (call only after ``init_redis``)."""
    if _client is None:
        raise RuntimeError("Redis is not initialised; check CACHE_ENABLED / startup.")
    return _client


def _mask_url(url: str) -> str:
    if "@" not in url:
        return url
    head, tail = url.split("@", 1)
    if "://" in head:
        scheme, _ = head.split("://", 1)
        return f"{scheme}://***@{tail}"
    return "***@" + tail
