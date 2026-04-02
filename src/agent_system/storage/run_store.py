"""RunStore — persists agent runs, tool calls, file artifacts, and agent memory.

All methods are async and use the shared asyncpg connection pool.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RunStore:
    """Async repository for all run-related tables.

    Depends on the pool being initialised via ``agent_system.database.init_pool()``
    before any method is called.  A module-level singleton is exposed via
    ``get_run_store()`` in ``api/app.py``.
    """

    def _pool(self):
        from agent_system.database import get_pool
        return get_pool()

    # ── agent_runs ────────────────────────────────────────────────────────────

    async def save_run(
        self,
        run_id: str,
        agent_name: str,
        task: str,
        final_answer: str,
        success: bool,
        reflection_count: int,
        minio_artifacts: list[str] | None = None,
    ) -> None:
        """Insert or update a row in agent_runs."""
        try:
            async with self._pool().acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent_runs
                        (run_id, agent_name, task, final_answer, success, reflection_count, minio_artifacts)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (run_id) DO UPDATE
                        SET final_answer     = EXCLUDED.final_answer,
                            success          = EXCLUDED.success,
                            reflection_count = EXCLUDED.reflection_count,
                            minio_artifacts  = EXCLUDED.minio_artifacts,
                            updated_at       = NOW()
                    """,
                    run_id, agent_name, task, final_answer, success,
                    reflection_count,
                    json.dumps(minio_artifacts or []),
                )
            logger.debug(
                "RunStore: saved agent_run run_id=%s artifacts=%s",
                run_id, minio_artifacts or [],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("RunStore: failed to save run %s: %s", run_id, exc)

    # ── tool_calls ────────────────────────────────────────────────────────────

    async def save_tool_calls(self, records: list[dict[str, Any]]) -> None:
        """Bulk-insert tool call records.

        Each record dict must have:
            run_id, tool_name, input_args (dict), output (str),
            success (bool), error (str | None)
        """
        if not records:
            return
        try:
            rows = [
                (
                    r["run_id"],
                    r["tool_name"],
                    json.dumps(r.get("input_args") or {}),
                    r.get("output", ""),
                    r.get("success", True),
                    r.get("error"),
                )
                for r in records
            ]
            async with self._pool().acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO tool_calls (run_id, tool_name, input_args, output, success, error)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    rows,
                )
            logger.debug(
                "RunStore: saved %d tool_call record(s) for run_id=%s",
                len(records),
                records[0]["run_id"] if records else "—",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("RunStore: failed to save tool_calls: %s", exc)

    # ── agent_memory ──────────────────────────────────────────────────────────

    async def memory_save(self, agent_name: str, key: str, value: str) -> None:
        """Upsert a key-value pair in agent_memory."""
        try:
            async with self._pool().acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent_memory (agent_name, key, value)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (agent_name, key) DO UPDATE
                        SET value      = EXCLUDED.value,
                            updated_at = NOW()
                    """,
                    agent_name, key, value,
                )
            logger.debug("RunStore: memory_save agent=%s key=%s", agent_name, key)
        except Exception as exc:  # noqa: BLE001
            logger.error("RunStore: memory_save failed: %s", exc)

    async def memory_get(self, agent_name: str, key: str) -> str | None:
        """Retrieve a value from agent_memory. Returns None if not found."""
        try:
            async with self._pool().acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT value FROM agent_memory WHERE agent_name = $1 AND key = $2",
                    agent_name, key,
                )
            return row["value"] if row else None
        except Exception as exc:  # noqa: BLE001
            logger.error("RunStore: memory_get failed: %s", exc)
            return None

    async def memory_list(self, agent_name: str) -> dict[str, str]:
        """Return all key-value pairs stored for an agent."""
        try:
            async with self._pool().acquire() as conn:
                rows = await conn.fetch(
                    "SELECT key, value FROM agent_memory WHERE agent_name = $1 ORDER BY key",
                    agent_name,
                )
            return {row["key"]: row["value"] for row in rows}
        except Exception as exc:  # noqa: BLE001
            logger.error("RunStore: memory_list failed: %s", exc)
            return {}

    # ── file_artifacts ────────────────────────────────────────────────────────

    async def log_file_artifact(
        self,
        run_id: str,
        agent_name: str,
        file_path: str,
        file_size: int | None = None,
        content_type: str = "text/plain",
    ) -> None:
        """Insert a row into file_artifacts when a file is written to MinIO."""
        try:
            async with self._pool().acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO file_artifacts (run_id, agent_name, file_path, file_size, content_type)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    run_id, agent_name, file_path, file_size, content_type,
                )
            logger.debug(
                "RunStore: logged file_artifact run_id=%s path=%s size=%s",
                run_id, file_path, file_size,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("RunStore: log_file_artifact failed: %s", exc)

    async def list_file_artifacts(
        self,
        run_id: str | None = None,
        agent_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query file_artifacts filtered by run_id and/or agent_name."""
        try:
            conditions: list[str] = []
            params: list[Any] = []
            idx = 1
            if run_id:
                conditions.append(f"run_id = ${idx}")
                params.append(run_id)
                idx += 1
            if agent_name:
                conditions.append(f"agent_name = ${idx}")
                params.append(agent_name)
                idx += 1
            params.append(limit)
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

            async with self._pool().acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT id, run_id, agent_name, file_path, file_size, content_type, written_at
                    FROM file_artifacts
                    {where}
                    ORDER BY written_at DESC
                    LIMIT ${idx}
                    """,
                    *params,
                )
            cols = ["id", "run_id", "agent_name", "file_path", "file_size", "content_type", "written_at"]
            return [dict(zip(cols, row)) for row in rows]
        except Exception as exc:  # noqa: BLE001
            logger.error("RunStore: list_file_artifacts failed: %s", exc)
            return []
