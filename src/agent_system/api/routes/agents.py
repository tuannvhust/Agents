"""Agent management and execution routes."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from agent_system.api.security import require_api_key
from agent_system.api.schemas import (
    AgentConfigRequest,
    AgentListResponse,
    AgentRunRequest,
    AgentRunResponse,
    AgentSummary,
)
from agent_system.core.agent import Agent, AgentConfig

router = APIRouter(
    prefix="/agents",
    tags=["Agents"],
    dependencies=[Depends(require_api_key)],
)
logger = logging.getLogger(__name__)

# ── In-process cache ──────────────────────────────────────────────────────────
# Stores fully-initialised Agent objects keyed by name.
# Populated on startup from PostgreSQL and updated on create/delete.
# With workers=1 (Dockerfile) this is always consistent.
_agent_cache: dict[str, Agent] = {}


def get_cache() -> dict[str, Agent]:
    return _agent_cache


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=AgentSummary,
    summary="Register a new agent",
)
async def create_agent(
    payload: AgentConfigRequest,
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
) -> AgentSummary:
    """Create, persist, and register a new agent."""
    from agent_system.api.app import get_config_store, get_tool_registry

    store = get_config_store()

    # Check both cache and DB so we're consistent even after a partial restart
    if payload.name in cache or await store.exists(payload.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{payload.name}' already exists. Delete it first or use a different name.",
        )

    config = AgentConfig(
        name=payload.name,
        skill_name=payload.skill_name,
        model=payload.model,
        model_source=payload.model_source,
        temperature=payload.temperature,
        max_reflections=payload.max_reflections,
        tools=[t for t in payload.tools if t != "string"],  # strip swagger placeholder
        extra_metadata=payload.extra_metadata,
    )

    try:
        tool_registry = get_tool_registry()
        agent = await Agent.create(config, tool_registry=tool_registry)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create agent '%s': %s", payload.name, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent creation failed: {exc}",
        ) from exc

    # Persist config to DB first, then add to cache
    await store.save(payload.name, _config_to_dict(config))
    cache[payload.name] = agent
    logger.info("Registered agent '%s'", payload.name)

    return _agent_to_summary(agent)


@router.get("", response_model=AgentListResponse, summary="List all registered agents")
async def list_agents(
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
) -> AgentListResponse:
    summaries = [_agent_to_summary(a) for a in cache.values()]
    return AgentListResponse(agents=summaries, total=len(summaries))


@router.get("/{name}", response_model=AgentSummary, summary="Get a specific agent")
async def get_agent(
    name: str,
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
) -> AgentSummary:
    return _agent_to_summary(_get_or_404(name, cache))


@router.delete(
    "/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a registered agent",
)
async def delete_agent(
    name: str,
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
) -> None:
    from agent_system.api.app import get_config_store

    _get_or_404(name, cache)
    await get_config_store().delete(name)
    del cache[name]
    logger.info("Deleted agent '%s'", name)


@router.post(
    "/{name}/run",
    response_model=AgentRunResponse,
    summary="Run an agent on a task",
)
async def run_agent(
    name: str,
    payload: AgentRunRequest,
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
) -> AgentRunResponse:
    """Trigger a synchronous agent run and return the final answer."""
    agent = _get_or_404(name, cache)
    result = await agent.run(task=payload.task, session_id=payload.session_id)

    return AgentRunResponse(
        agent_name=result.agent_name,
        run_id=result.run_id,
        task=result.task,
        final_answer=result.final_answer,
        success=result.success,
        reflection_count=result.reflection_count,
        messages_count=result.messages_count,
        stored_artifacts=result.stored_artifacts,
        error=result.error,
    )


@router.get("/{name}/skills", summary="List available skills")
async def list_skills(
    name: str,
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
):
    _get_or_404(name, cache)
    from agent_system.core.skill_loader import SkillLoader

    return {"skills": SkillLoader().list_available()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_404(name: str, cache: dict[str, Agent]) -> Agent:
    if name not in cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{name}' not found.",
        )
    return cache[name]


def _agent_to_summary(agent: Agent) -> AgentSummary:
    return AgentSummary(
        name=agent.name,
        skill_name=agent.config.skill_name,
        model=agent.config.model,
        model_source=agent.config.model_source,
        tools=agent.config.tools,
    )


def _config_to_dict(config: AgentConfig) -> dict:
    return {
        "name": config.name,
        "skill_name": config.skill_name,
        "model": config.model,
        "model_source": config.model_source,
        "temperature": config.temperature,
        "max_reflections": config.max_reflections,
        "tools": config.tools,
        "extra_metadata": config.extra_metadata,
    }
