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
    enable_ocr: bool = Field(
        False,
        description=(
            "When True, a vision LLM pre-processing node is added before the main agent "
            "loop.  The node fires only when image_url is provided at run time."
        ),
    )
    ocr_model: str | None = Field(
        None,
        description=(
            "Vision LLM model for the OCR node.  Falls back to OCR_MODEL env var when "
            "omitted.  Only used when enable_ocr=True."
        ),
        examples=["qwen/qwen2-vl-7b-instruct"],
    )
    ocr_model_source: Literal["openrouter", "local"] | None = Field(
        None,
        description="Backend for the OCR VLM.  Falls back to OCR_MODEL_SOURCE env var.",
    )
    ocr_skill_name: str = Field(
        "ocr",
        description=(
            "Langfuse prompt name (or local skills/<name>.md) used as the OCR system "
            "prompt.  Defaults to 'ocr'."
        ),
    )

    @field_validator("sub_agents", mode="before")
    @classmethod
    def _strip_swagger_placeholders(cls, v: list[str]) -> list[str]:
        return [n for n in (v or []) if n != "string"]


RunStatusLiteral = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "awaiting_approval",
]


class RunAcceptedResponse(BaseModel):
    """Returned immediately when a run is enqueued (HTTP 202)."""

    agent_name: str
    run_id: str
    task: str
    run_status: Literal["queued"] = "queued"
    job_id: str | None = None
    poll_url: str = Field(
        ...,
        description="Poll this path until run_status is completed, failed, or awaiting_approval.",
    )


class RunStatusResponse(BaseModel):
    """Poll result for an async agent run."""

    agent_name: str
    run_id: str
    task: str
    run_status: RunStatusLiteral
    job_id: str | None = None
    input_file: str | None = Field(
        None,
        description=(
            "MinIO object path of the original input file (image/PDF), if one was provided. "
            "Format: runs/{agent_name}/{run_id}/inputs/{filename}"
        ),
    )
    final_answer: str = ""
    success: bool = False
    reflection_count: int = 0
    messages_count: int = 0
    stored_artifacts: list[str] = Field(default_factory=list)
    error: str | None = None
    approval_request: dict[str, Any] | None = Field(
        None,
        description="Set when run_status is awaiting_approval.",
    )
    trace: dict[str, Any] | None = Field(
        None,
        description="Present when include_trace=true and the run has finished.",
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
    enable_ocr: bool = False
    ocr_skill_name: str = "ocr"


class AgentListResponse(BaseModel):
    agents: list[AgentSummary]
    total: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    services: dict[str, str]
