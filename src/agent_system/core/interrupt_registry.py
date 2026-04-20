"""In-memory registry of runs paused for human tool approval (Reviewer UI)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PendingApproval:
    agent_name: str
    run_id: str
    task: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)


_pending: dict[str, PendingApproval] = {}


def register_pending(agent_name: str, run_id: str, task: str, payload: dict[str, Any]) -> None:
    _pending[run_id] = PendingApproval(
        agent_name=agent_name,
        run_id=run_id,
        task=task,
        payload=payload,
    )


def clear_pending(run_id: str) -> None:
    _pending.pop(run_id, None)


def get_pending(run_id: str) -> PendingApproval | None:
    return _pending.get(run_id)


def list_pending() -> list[PendingApproval]:
    return list(_pending.values())
