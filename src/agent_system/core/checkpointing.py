"""Shared LangGraph checkpointer for human-in-the-loop (interrupt / resume)."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

_saver: MemorySaver | None = None


def get_checkpoint_saver() -> MemorySaver:
    """Process-wide MemorySaver — thread_id (run_id) isolates concurrent runs.

    Pending approvals are lost on process restart; use a single API worker for consistent behavior.
    """
    global _saver
    if _saver is None:
        _saver = MemorySaver()
    return _saver
