"""PostgreSQL-backed store for agent configurations.

All methods are async and use the shared asyncpg connection pool.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class AgentConfigStore:
    """Async CRUD operations for agent configs backed by PostgreSQL.

    Depends on the pool being initialised via ``agent_system.database.init_pool()``
    before any method is called.
    """

    def _pool(self):
        from agent_system.database import get_pool
        return get_pool()

    # ── Public API ────────────────────────────────────────────────────────────

    async def save(self, name: str, config_dict: dict[str, Any]) -> None:
        """Insert or update an agent config."""
        try:
            async with self._pool().acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent_configs (name, config)
                    VALUES ($1, $2)
                    ON CONFLICT (name) DO UPDATE
                        SET config     = EXCLUDED.config,
                            updated_at = NOW()
                    """,
                    name, json.dumps(config_dict),
                )
            logger.debug("Saved agent config: %s", name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save agent config '%s': %s", name, exc)

    async def load_all(self) -> list[dict[str, Any]]:
        """Return all persisted agent configs as dicts."""
        try:
            async with self._pool().acquire() as conn:
                rows = await conn.fetch(
                    "SELECT config FROM agent_configs ORDER BY created_at"
                )
            return [json.loads(row["config"]) for row in rows]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load agent configs: %s", exc)
            return []

    async def delete(self, name: str) -> bool:
        """Delete an agent config. Returns True if a row was deleted."""
        try:
            async with self._pool().acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM agent_configs WHERE name = $1", name
                )
            # asyncpg returns e.g. "DELETE 1" or "DELETE 0"
            deleted = result.endswith("1") or (result.split()[-1].isdigit() and int(result.split()[-1]) > 0)
            logger.debug("Deleted agent config: %s (found=%s)", name, deleted)
            return deleted
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to delete agent config '%s': %s", name, exc)
            return False

    async def exists(self, name: str) -> bool:
        """Return True if a config with the given name exists."""
        try:
            async with self._pool().acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT 1 FROM agent_configs WHERE name = $1", name
                )
            return row is not None
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to check agent config '%s': %s", name, exc)
            return False
