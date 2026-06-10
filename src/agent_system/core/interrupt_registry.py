"""Registry of runs paused for human tool approval (Reviewer UI).

Uses Redis when the job queue is enabled so API and worker processes share state.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_PENDING_KEY_PREFIX = "agent:pending:"
_PENDING_TTL_SECONDS = 86_400  # 24 h

_pending: dict[str, "PendingApproval"] = {}
_use_redis = False


@dataclass
class PendingApproval:
    agent_name: str
    run_id: str
    task: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)


def configure_interrupt_registry(*, use_redis: bool) -> None:
    """Select backing store. Call once at process startup."""
    global _use_redis
    _use_redis = use_redis
    logger.info("Interrupt registry: %s", "Redis" if use_redis else "in-memory")


async def register_pending(
    agent_name: str,
    run_id: str,
    task: str,
    payload: dict[str, Any],
) -> None:
    entry = PendingApproval(
        agent_name=agent_name,
        run_id=run_id,
        task=task,
        payload=payload,
    )
    if _use_redis:
        from agent_system.cache.redis_client import get_redis

        await get_redis().setex(
            f"{_PENDING_KEY_PREFIX}{run_id}",
            _PENDING_TTL_SECONDS,
            json.dumps(
                {
                    "agent_name": entry.agent_name,
                    "run_id": entry.run_id,
                    "task": entry.task,
                    "payload": entry.payload,
                    "created_at": entry.created_at,
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        return
    _pending[run_id] = entry


async def clear_pending(run_id: str) -> None:
    if _use_redis:
        from agent_system.cache.redis_client import get_redis

        await get_redis().delete(f"{_PENDING_KEY_PREFIX}{run_id}")
        return
    _pending.pop(run_id, None)


async def get_pending(run_id: str) -> PendingApproval | None:
    if _use_redis:
        from agent_system.cache.redis_client import get_redis

        raw = await get_redis().get(f"{_PENDING_KEY_PREFIX}{run_id}")
        if not raw:
            return None
        return _decode_pending(raw)
    return _pending.get(run_id)


async def list_pending() -> list[PendingApproval]:
    if _use_redis:
        from agent_system.cache.redis_client import get_redis

        redis = get_redis()
        keys = [k async for k in redis.scan_iter(f"{_PENDING_KEY_PREFIX}*")]
        if not keys:
            return []
        values = await redis.mget(keys)
        out: list[PendingApproval] = []
        for raw in values:
            if raw:
                out.append(_decode_pending(raw))
        out.sort(key=lambda p: p.created_at)
        return out
    return list(_pending.values())


def _decode_pending(raw: str) -> PendingApproval:
    data = json.loads(raw)
    return PendingApproval(
        agent_name=data["agent_name"],
        run_id=data["run_id"],
        task=data["task"],
        payload=data.get("payload") or {},
        created_at=float(data.get("created_at") or time.time()),
    )
