"""ADK proxy agent for `researcher`."""

from __future__ import annotations

from agent_system.adk.common import resolve_adk_model, run_backend_agent

try:
    from google.adk.agents.llm_agent import Agent
except Exception:  # noqa: BLE001
    from google.adk.agents import Agent  # type: ignore


def run_researcher(task: str, session_id: str = "") -> str:
    """Run the backend `researcher` agent and return its final response text."""
    return run_backend_agent(agent_name="researcher", task=task, session_id=session_id)


root_agent = Agent(
    model=resolve_adk_model(),
    name="researcher_proxy",
    description="Routes prompts to the existing `researcher` backend agent.",
    instruction=(
        "You are a deterministic proxy for the backend `researcher` agent.\n"
        "Always call the `run_researcher` tool once with the user's request.\n"
        "Then return the tool output verbatim as your final response.\n"
        "Do not add commentary, prefacing text, or post-processing."
    ),
    tools=[run_researcher],
)

