"""RabbitMQ topology and connection helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aio_pika

if TYPE_CHECKING:
    from aio_pika.abc import AbstractChannel, AbstractRobustConnection

logger = logging.getLogger(__name__)

EXCHANGE_NAME = "agent.direct"
QUEUE_JOBS = "agent.jobs"
ROUTING_RUN = "agent.run"
ROUTING_RESUME = "agent.resume"


async def connect(url: str) -> AbstractRobustConnection:
    """Open a robust (auto-reconnect) RabbitMQ connection."""
    connection = await aio_pika.connect_robust(url)
    logger.info("RabbitMQ connected (%s)", _mask_url(url))
    return connection


async def declare_topology(channel: AbstractChannel):
    """Declare durable exchange + queue used by API producers and workers."""
    exchange = await channel.declare_exchange(
        EXCHANGE_NAME,
        aio_pika.ExchangeType.DIRECT,
        durable=True,
    )
    queue = await channel.declare_queue(QUEUE_JOBS, durable=True)
    await queue.bind(exchange, routing_key=ROUTING_RUN)
    await queue.bind(exchange, routing_key=ROUTING_RESUME)
    logger.debug("RabbitMQ topology ready (exchange=%s, queue=%s)", EXCHANGE_NAME, QUEUE_JOBS)
    return exchange, queue


def _mask_url(url: str) -> str:
    if "@" not in url:
        return url
    head, tail = url.split("@", 1)
    if "://" in head:
        scheme, _ = head.split("://", 1)
        return f"{scheme}://***@{tail}"
    return "***@" + tail
