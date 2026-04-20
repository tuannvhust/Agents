"""Redis cache-aside wrapper around :class:`RunStore`.

Postgres remains authoritative: writes always hit the inner store first, then
cache entries are updated or invalidated. Read paths try Redis first (TTL),
fall back to Postgres, and repopulate Redis on a hit from the database.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_system.storage.run_store import RunStore

logger = logging.getLogger(__name__)

_KEYP = "as:v1"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


class CachingRunStore:
    """Delegates to :class:`RunStore` and adds Redis read-through caching."""

    def __init__(
        self,
        inner: RunStore,
        redis: Any,
        *,
        memory_ttl_seconds: int,
        conversation_ttl_seconds: int,
        tool_messages_ttl_seconds: int,
    ) -> None:
        self._inner = inner
        self._r = redis
        self._ttl_mem = memory_ttl_seconds
        self._ttl_run = conversation_ttl_seconds
        self._ttl_tc = tool_messages_ttl_seconds

    def _k_run(self, run_id: str) -> str:
        return f"{_KEYP}:run:{run_id}"

    def _k_tc(self, run_id: str) -> str:
        return f"{_KEYP}:tc:{run_id}"

    def _k_mem(self, agent_name: str, key: str) -> str:
        return f"{_KEYP}:mem:{agent_name}:{key}"

    def _k_mem_list(self, agent_name: str) -> str:
        return f"{_KEYP}:meml:{agent_name}"

    async def _redis_get_json(self, key: str) -> Any | None:
        try:
            raw = await self._r.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis GET %s failed (falling back to PG): %s", key, exc)
            return None

    async def _redis_set_json(self, key: str, value: Any, ttl: int) -> None:
        try:
            await self._r.set(key, _json_dumps(value), ex=ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis SET %s failed: %s", key, exc)

    async def _redis_delete(self, *keys: str) -> None:
        if not keys:
            return
        try:
            await self._r.delete(*keys)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis DEL failed: %s", exc)

    # ── Reads (cache-aside) ─────────────────────────────────────────────────

    async def fetch_run(self, run_id: str) -> dict[str, Any] | None:
        key = self._k_run(run_id)
        cached = await self._redis_get_json(key)
        if isinstance(cached, dict):
            return cached
        row = await self._inner.fetch_run(run_id)
        if row is not None:
            await self._redis_set_json(key, row, self._ttl_run)
        return row

    async def fetch_tool_calls_for_run(self, run_id: str) -> list[dict[str, Any]]:
        key = self._k_tc(run_id)
        cached = await self._redis_get_json(key)
        if isinstance(cached, list):
            return cached
        rows = await self._inner.fetch_tool_calls_for_run(run_id)
        await self._redis_set_json(key, rows, self._ttl_tc)
        return rows

    async def memory_get(self, agent_name: str, key: str) -> str | None:
        rkey = self._k_mem(agent_name, key)
        try:
            hit = await self._r.get(rkey)
            if hit is not None:
                obj = json.loads(hit)
                if isinstance(obj, dict) and "v" in obj:
                    v = obj["v"]
                    return None if v is None else str(v)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis memory_get failed: %s", exc)
        val = await self._inner.memory_get(agent_name, key)
        try:
            await self._r.set(rkey, _json_dumps({"v": val}), ex=self._ttl_mem)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis memory_get cache fill failed: %s", exc)
        return val

    async def memory_list(self, agent_name: str) -> dict[str, str]:
        lkey = self._k_mem_list(agent_name)
        cached = await self._redis_get_json(lkey)
        if isinstance(cached, dict):
            return {str(k): str(v) for k, v in cached.items()}
        data = await self._inner.memory_list(agent_name)
        await self._redis_set_json(lkey, data, self._ttl_mem)
        return data

    # ── Writes (Postgres first, then cache maintenance) ───────────────────────

    async def memory_save(self, agent_name: str, key: str, value: str) -> None:
        await self._inner.memory_save(agent_name, key, value)
        try:
            await self._r.set(
                self._k_mem(agent_name, key),
                _json_dumps({"v": value}),
                ex=self._ttl_mem,
            )
            await self._r.delete(self._k_mem_list(agent_name))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis memory_save cache update failed: %s", exc)

    async def save_run(
        self,
        run_id: str,
        agent_name: str,
        task: str,
        final_answer: str,
        success: bool,
        reflection_count: int,
        minio_artifacts: list[str] | None = None,
        run_trace: dict[str, Any] | None = None,
    ) -> bool:
        ok = await self._inner.save_run(
            run_id,
            agent_name,
            task,
            final_answer,
            success,
            reflection_count,
            minio_artifacts,
            run_trace,
        )
        if ok:
            await self._redis_delete(self._k_run(run_id), self._k_tc(run_id))
        return ok

    async def save_tool_calls(self, run_id: str, records: list[dict[str, Any]]) -> bool:
        ok = await self._inner.save_tool_calls(run_id, records)
        if ok:
            await self._redis_delete(self._k_tc(run_id), self._k_run(run_id))
        return ok

    async def log_file_artifact(
        self,
        run_id: str,
        agent_name: str,
        file_path: str,
        file_size: int | None = None,
        content_type: str = "text/plain",
    ) -> None:
        await self._inner.log_file_artifact(
            run_id, agent_name, file_path, file_size, content_type
        )

    async def list_file_artifacts(
        self,
        run_id: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return await self._inner.list_file_artifacts(
            run_id=run_id, agent_name=agent_name, limit=limit
        )
