"""Load agents on demand in worker processes (handles agents created after worker start)."""

from __future__ import annotations

import logging

from agent_system.api.routes.agents import _agent_cache
from agent_system.core.agent import Agent, AgentConfig

logger = logging.getLogger(__name__)


async def ensure_agent_loaded(name: str) -> Agent:
    if name in _agent_cache:
        return _agent_cache[name]

    from agent_system.api.app import get_config_store, get_tool_registry

    store = get_config_store()
    cfg_dict = await store.load(name)
    if cfg_dict is None:
        raise ValueError(f"Agent '{name}' is not registered.")

    config = AgentConfig(
        name=name,
        skill_name=cfg_dict["skill_name"],
        model=cfg_dict.get("model"),
        model_source=cfg_dict.get("model_source", "openrouter"),
        temperature=cfg_dict.get("temperature", 0.0),
        max_reflections=cfg_dict.get("max_reflections", 3),
        tools=cfg_dict.get("tools", []),
        tools_requiring_approval=cfg_dict.get("tools_requiring_approval", []),
        plugins=cfg_dict.get("plugins", []),
        extra_metadata=cfg_dict.get("extra_metadata", {}),
        role=cfg_dict.get("role", "subagent"),
        sub_agents=cfg_dict.get("sub_agents", []),
        enable_ocr=cfg_dict.get("enable_ocr", False),
        ocr_model=cfg_dict.get("ocr_model"),
        ocr_model_source=cfg_dict.get("ocr_model_source"),
        ocr_skill_name=cfg_dict.get("ocr_skill_name", "ocr"),
    )
    agent = await Agent.create(
        config,
        tool_registry=get_tool_registry(),
        agent_cache=_agent_cache,
    )
    _agent_cache[name] = agent
    logger.info("Lazy-loaded agent '%s' in worker process.", name)
    return agent
