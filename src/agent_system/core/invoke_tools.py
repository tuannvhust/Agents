"""Factory that creates ``invoke_<name>`` tools so a coordinator agent can
delegate sub-tasks to registered sub-agents via normal LangGraph tool calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_system.core.agent import Agent

logger = logging.getLogger(__name__)


class _InvokeAgentInput(BaseModel):
    task: str = Field(..., description="The specific sub-task to delegate to this agent.")


def make_invoke_agent_tools(agents: dict[str, "Agent"]) -> list[StructuredTool]:
    """Return one ``invoke_<name>`` StructuredTool per sub-agent in *agents*.

    Each tool asynchronously calls ``Agent.run(task)`` on the named sub-agent and
    returns its ``final_answer`` (or an error/status string on failure).  The tools
    are added to the coordinator agent's tool list so the coordinator LLM can delegate
    sub-tasks via standard tool calls.

    Args:
        agents: Mapping of agent name → Agent instance.  Only sub-agents should be
            passed here; the caller is responsible for filtering out the coordinator.

    Returns:
        A list of StructuredTool instances, one per entry in *agents*.
    """
    tools: list[StructuredTool] = []

    for agent_name, agent in agents.items():
        skill_name = agent.config.skill_name

        # Bind loop variables explicitly to avoid the classic Python closure-in-loop trap.
        async def _run(
            task: str,
            _agent: Agent = agent,
            _name: str = agent_name,
        ) -> str:
            logger.info(
                "[INVOKE AGENT] coordinator delegating to '%s' | task: %.200s",
                _name, task,
            )
            try:
                result = await _agent.run(task)
                if result.run_status == "awaiting_approval":
                    return (
                        f"Sub-agent '{_name}' paused — awaiting human approval. "
                        f"run_id={result.run_id!r}"
                    )
                if not result.success and result.error:
                    return f"Sub-agent '{_name}' failed: {result.error}"
                answer = result.final_answer or ""
                logger.info(
                    "[INVOKE AGENT] '%s' returned %d chars", _name, len(answer)
                )
                return answer or f"Sub-agent '{_name}' completed (no answer returned)."
            except Exception as exc:  # noqa: BLE001
                logger.error("[INVOKE AGENT] '%s' raised: %s", _name, exc, exc_info=True)
                return f"Sub-agent '{_name}' raised an exception: {exc}"

        tool = StructuredTool.from_function(
            coroutine=_run,
            name=f"invoke_{agent_name}",
            description=(
                f"Delegate a sub-task to the '{agent_name}' specialized agent "
                f"(skill: {skill_name!r}). "
                "Provide a clear, self-contained task description. "
                "The agent will execute the task and return its result."
            ),
            args_schema=_InvokeAgentInput,
        )
        tools.append(tool)
        logger.debug("Created invoke tool for sub-agent '%s'", agent_name)

    logger.info(
        "Wired %d invoke tool(s) for coordinator: %s",
        len(tools),
        [t.name for t in tools],
    )
    return tools
