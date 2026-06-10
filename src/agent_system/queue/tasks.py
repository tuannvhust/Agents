"""Job handlers executed by RabbitMQ worker processes."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def execute_agent_run(
    *,
    agent_name: str,
    run_id: str,
    task: str,
    image_url: str | None = None,
    include_trace: bool = False,
) -> dict[str, Any]:
    """Run an agent graph for a previously enqueued job."""
    from agent_system.api.app import get_run_store
    from agent_system.runtime.agent_loader import ensure_agent_loaded

    store = get_run_store()
    await store.update_run_status(run_id, "running")

    try:
        agent = await ensure_agent_loaded(agent_name)
        result = await agent.run(task=task, session_id=run_id, image_url=image_url)
        return {
            "run_id": result.run_id,
            "run_status": result.run_status,
            "success": result.success,
            "include_trace": include_trace,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Worker run failed run_id=%s: %s", run_id, exc)
        await store.update_run_status(run_id, "failed", error_message=str(exc))
        raise


async def execute_agent_resume(
    *,
    agent_name: str,
    run_id: str,
    decision: dict[str, Any],
    include_trace: bool = False,
) -> dict[str, Any]:
    """Resume a human-in-the-loop paused run."""
    from agent_system.api.app import get_run_store
    from agent_system.runtime.agent_loader import ensure_agent_loaded

    store = get_run_store()
    await store.update_run_status(run_id, "running")

    try:
        agent = await ensure_agent_loaded(agent_name)
        result = await agent.resume_run(run_id=run_id, decision=decision)
        return {
            "run_id": result.run_id,
            "run_status": result.run_status,
            "success": result.success,
            "include_trace": include_trace,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Worker resume failed run_id=%s: %s", run_id, exc)
        await store.update_run_status(run_id, "failed", error_message=str(exc))
        raise
