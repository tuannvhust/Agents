"""Pydantic v2 request/response schemas for the Agent System API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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
    tools_requiring_approval: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names that pause the run for human approval before execution "
            "(Reviewer UI + POST .../resume). If any planned call matches, the whole batch waits."
        ),
    )
    plugins: list[str] = Field(
        default_factory=list,
        description=(
            "Plugin names to activate for this agent. "
            "Currently supported: 'safety' (prompt-injection classifier that runs "
            "before every LLM call). Leave empty to disable all plugins."
        ),
        examples=[["safety"]],
    )
    extra_metadata: dict[str, Any] = Field(default_factory=dict)
    role: Literal["subagent", "coordinator"] = Field(
        "subagent",
        description=(
            "'subagent' — focused worker without reflection (default). "
            "'coordinator' — orchestrator that delegates to sub-agents via invoke_* tools "
            "and reflects on the full workflow result."
        ),
    )
    sub_agents: list[str] = Field(
        default_factory=list,
        description=(
            "For coordinators only: names of already-registered sub-agents to wire as "
            "invoke_* tools. Empty list = all currently registered agents (except self). "
            "Sub-agents must be registered before the coordinator."
        ),
    )

    @field_validator("sub_agents", mode="before")
    @classmethod
    def _strip_swagger_placeholders(cls, v: list[str]) -> list[str]:
        return [n for n in (v or []) if n != "string"]


class AgentRunRequest(BaseModel):
    """Payload for triggering an agent run."""

    task: str = Field(..., description="The task / instruction for the agent", min_length=1)
    session_id: str | None = Field(
        None,
        description="Optional session/run ID for traceability. Auto-generated if omitted.",
    )
    include_trace: bool = Field(
        False,
        description="If true, include the full structured run trace in the response (can be large).",
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
    run_status: Literal["completed", "awaiting_approval"] = Field(
        "completed",
        description="awaiting_approval when the run stopped for human tool approval.",
    )
    approval_request: dict[str, Any] | None = Field(
        None,
        description="Payload for the Reviewer UI (planned tools, args, message digest). "
        "Set when run_status is awaiting_approval.",
    )
    trace: dict[str, Any] | None = Field(
        None,
        description="Structured trace (plan text, tool choices, exact args, reflection). "
        "Present when include_trace was true on the request.",
    )


class AgentResumeRequest(BaseModel):
    """Operator decision to resume a paused run."""

    action: Literal["approve", "reject"] = Field(
        ...,
        description="approve executes planned tools; reject sends rejection tool messages to the agent.",
    )
    reason: str | None = Field(
        None,
        description="Required context when action is reject (shown to the agent).",
    )


class AgentSummary(BaseModel):
    name: str
    skill_name: str
    model: str | None
    model_source: str
    tools: list[str]
    tools_requiring_approval: list[str] = Field(default_factory=list)
    plugins: list[str] = Field(default_factory=list)
    role: str = "subagent"
    sub_agents: list[str] = Field(default_factory=list)


class AgentListResponse(BaseModel):
    agents: list[AgentSummary]
    total: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    services: dict[str, str]
