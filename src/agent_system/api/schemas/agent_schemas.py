"""Pydantic v2 request/response schemas for the Agent System API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentConfigRequest(BaseModel):
    """Payload for registering a new agent."""

    name: str = Field(..., description="Unique agent identifier", examples=["researcher"])
    skill_name: str = Field(
        ...,
        description="Name of the SKILLS.md file to load (without extension)",
        examples=["researcher"],
    )
    model: str | None = Field(
        None,
        description="LLM model override. Defaults to the configured default model.",
        examples=["qwen/qwen3-30b-a3b-thinking-2507"],
    )
    model_source: Literal["openrouter", "local"] = Field(
        "openrouter",
        description="Which LLM backend to use",
    )
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_reflections: int = Field(3, ge=0, le=10)
    tools: list[str] = Field(
        default_factory=list,
        description="Tool names to enable. Empty list = all registered tools.",
    )
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    """Payload for triggering an agent run."""

    task: str = Field(..., description="The task / instruction for the agent", min_length=1)
    session_id: str | None = Field(
        None,
        description="Optional session/run ID for traceability. Auto-generated if omitted.",
    )


class AgentRunResponse(BaseModel):
    """Response returned after an agent completes a run."""

    agent_name: str
    run_id: str
    task: str
    final_answer: str
    success: bool
    reflection_count: int
    messages_count: int
    stored_artifacts: list[str] = Field(default_factory=list)
    error: str | None = None


class AgentSummary(BaseModel):
    name: str
    skill_name: str
    model: str | None
    model_source: str
    tools: list[str]


class AgentListResponse(BaseModel):
    agents: list[AgentSummary]
    total: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    services: dict[str, str]
