"""Run the RabbitMQ worker: ``agent-worker``."""

from __future__ import annotations

import asyncio


def run() -> None:
    from agent_system.queue.worker import run_consumer

    asyncio.run(run_consumer())


if __name__ == "__main__":
    run()
