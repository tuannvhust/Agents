"""Run-scoped context variables.

A lightweight way to carry the current ``run_id`` and ``agent_name`` into any
code that runs inside an agent graph invocation — including LangChain tools —
without threading them through every function signature.

Usage
-----
Set at the start of each run (in ``Agent.run()``)::

    from agent_system.core.run_context import run_ctx
    token = run_ctx.set(RunContext(run_id="...", agent_name="..."))
    try:
        ...  # invoke graph
    finally:
        run_ctx.reset(token)

Read anywhere (e.g. inside a tool)::

    from agent_system.core.run_context import get_run_context
    ctx = get_run_context()   # None if called outside an agent run
    if ctx:
        print(ctx.run_id, ctx.agent_name)
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RunContext:
    run_id: str
    agent_name: str


# Module-level ContextVar — each asyncio Task inherits a copy from its parent,
# so the value set in Agent.run() is visible inside every tool call.
run_ctx: ContextVar[RunContext | None] = ContextVar("run_ctx", default=None)


def get_run_context() -> RunContext | None:
    """Return the current RunContext, or None if not inside an agent run."""
    return run_ctx.get()
