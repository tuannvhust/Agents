"""ADK proxy agent for `analyst`."""

from __future__ import annotations

from agent_system.adk.common import resolve_adk_model, run_backend_agent

try:
    from google.adk.agents.llm_agent import Agent
except Exception:  # noqa: BLE001
    from google.adk.agents import Agent  # type: ignore


def run_analyst(task: str, session_id: str = "") -> str:
    """Run the backend `analyst` agent and return its final response text."""
    return run_backend_agent(agent_name="analyst", task=task, session_id=session_id)


root_agent = Agent(
    model=resolve_adk_model(),
    name="analyst_proxy",
    description="Routes prompts to the existing `analyst` backend agent.",
    instruction=(
        "You are a deterministic proxy for the backend `analyst` agent.\n"
        "Always call the `run_analyst` tool once with the user's request.\n"
        "Then return the tool output verbatim as your final response.\n"
        "Do not add commentary, prefacing text, or post-processing."
    ),
    tools=[run_analyst],
)

