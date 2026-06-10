"""RabbitMQ consumer — processes agent run / resume jobs."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aio_pika

from agent_system.config import get_settings
from agent_system.queue.broker import QUEUE_JOBS, connect, declare_topology
from agent_system.queue.tasks import execute_agent_resume, execute_agent_run

logger = logging.getLogger(__name__)


async def _dispatch(payload: dict[str, Any]) -> None:
    job_type = payload.get("type")
    cfg = get_settings()

    if job_type == "run":
        coro = execute_agent_run(
            agent_name=payload["agent_name"],
            run_id=payload["run_id"],
            task=payload["task"],
            image_url=payload.get("image_url"),
            include_trace=bool(payload.get("include_trace")),
        )
    elif job_type == "resume":
        coro = execute_agent_resume(
            agent_name=payload["agent_name"],
            run_id=payload["run_id"],
            decision=payload["decision"],
            include_trace=bool(payload.get("include_trace")),
        )
    else:
        raise ValueError(f"Unknown job type: {job_type!r}")

    await asyncio.wait_for(coro, timeout=cfg.queue_job_timeout)


async def _handle_message(message: aio_pika.IncomingMessage, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        payload = json.loads(message.body.decode("utf-8"))
        job_id = payload.get("job_id", message.message_id or "?")
        run_id = payload.get("run_id", "?")
        logger.info("Processing job_id=%s run_id=%s type=%s", job_id, run_id, payload.get("type"))

        async with message.process(requeue=False):
            try:
                await _dispatch(payload)
            except asyncio.TimeoutError:
                logger.error("Job timed out job_id=%s run_id=%s", job_id, run_id)
                from agent_system.api.app import get_run_store

                store = get_run_store()
                await store.update_run_status(
                    run_id,
                    "failed",
                    error_message=f"Job exceeded timeout ({get_settings().queue_job_timeout}s)",
                )
                raise


async def run_consumer() -> None:
    """Start the RabbitMQ consumer loop (blocks until cancelled)."""
    from agent_system.logging import configure_logging
    from agent_system.runtime.bootstrap import bootstrap_shutdown, bootstrap_startup

    cfg = get_settings()
    configure_logging(cfg.app.log_level)
    await bootstrap_startup(load_agents=True)

    connection = await connect(cfg.rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=cfg.queue_max_jobs)
    _, queue = await declare_topology(channel)

    semaphore = asyncio.Semaphore(cfg.queue_max_jobs)
    logger.info(
        "RabbitMQ worker consuming queue=%s (max_jobs=%d, timeout=%ds)",
        QUEUE_JOBS,
        cfg.queue_max_jobs,
        cfg.queue_job_timeout,
    )

    try:
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                asyncio.create_task(_handle_message(message, semaphore))
    finally:
        await bootstrap_shutdown()
        await connection.close()
