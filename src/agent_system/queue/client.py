"""RabbitMQ publisher used by the API to enqueue agent jobs."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import aio_pika

from agent_system.queue.broker import (
    EXCHANGE_NAME,
    ROUTING_RESUME,
    ROUTING_RUN,
    connect,
    declare_topology,
)

logger = logging.getLogger(__name__)

_connection: aio_pika.abc.AbstractRobustConnection | None = None
_channel: aio_pika.abc.AbstractChannel | None = None
_exchange: aio_pika.abc.AbstractExchange | None = None


async def init_queue_pool(rabbitmq_url: str) -> None:
    """Open the shared publisher connection (kept for API compatibility)."""
    global _connection, _channel, _exchange
    if _connection is not None:
        return
    _connection = await connect(rabbitmq_url)
    _channel = await _connection.channel()
    _exchange, _ = await declare_topology(_channel)
    logger.info("RabbitMQ publisher ready")


async def close_queue_pool() -> None:
    global _connection, _channel, _exchange
    if _connection is None:
        return
    try:
        if _channel is not None and not _channel.is_closed:
            await _channel.close()
        await _connection.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("RabbitMQ publisher close: %s", exc)
    finally:
        _connection = None
        _channel = None
        _exchange = None


def _require_exchange() -> aio_pika.abc.AbstractExchange:
    if _exchange is None:
        raise RuntimeError(
            "RabbitMQ publisher is not initialised. Call init_queue_pool() at startup."
        )
    return _exchange


async def _publish(routing_key: str, payload: dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    body = json.dumps({**payload, "job_id": job_id}, ensure_ascii=False).encode("utf-8")
    exchange = _require_exchange()
    await exchange.publish(
        aio_pika.Message(
            body=body,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=job_id,
            content_type="application/json",
        ),
        routing_key=routing_key,
    )
    return job_id


async def enqueue_agent_run(
    *,
    agent_name: str,
    run_id: str,
    task: str,
    image_url: str | None = None,
    include_trace: bool = False,
) -> str:
    job_id = await _publish(
        ROUTING_RUN,
        {
            "type": "run",
            "agent_name": agent_name,
            "run_id": run_id,
            "task": task,
            "image_url": image_url,
            "include_trace": include_trace,
        },
    )
    logger.info("Enqueued agent run job_id=%s run_id=%s agent=%s", job_id, run_id, agent_name)
    return job_id


async def enqueue_agent_resume(
    *,
    agent_name: str,
    run_id: str,
    decision: dict[str, Any],
    include_trace: bool = False,
) -> str:
    job_id = await _publish(
        ROUTING_RESUME,
        {
            "type": "resume",
            "agent_name": agent_name,
            "run_id": run_id,
            "decision": decision,
            "include_trace": include_trace,
        },
    )
    logger.info("Enqueued agent resume job_id=%s run_id=%s", job_id, run_id)
    return job_id
