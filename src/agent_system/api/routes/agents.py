"""Agent management and execution routes."""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from agent_system.api.security import require_api_key
from agent_system.api.schemas import (
    AgentConfigRequest,
    AgentListResponse,
    AgentResumeRequest,
    AgentSummary,
    RunAcceptedResponse,
)
from agent_system.core.interrupt_registry import get_pending
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
        tools_requiring_approval=[
            t for t in (payload.tools_requiring_approval or []) if t != "string"
        ],
        plugins=list(payload.plugins or []),
        extra_metadata=payload.extra_metadata,
        role=payload.role,
        sub_agents=list(payload.sub_agents or []),
        enable_ocr=payload.enable_ocr,
        ocr_model=payload.ocr_model,
        ocr_model_source=payload.ocr_model_source,
        ocr_skill_name=payload.ocr_skill_name,
    )

    try:
        tool_registry = get_tool_registry()
        agent = await Agent.create(config, tool_registry=tool_registry, agent_cache=cache)
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
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunAcceptedResponse,
    summary="Enqueue an agent run (multipart/form-data)",
)
async def enqueue_agent_run_route(
    name: str,
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
    task: Annotated[str, Form(description="The task / instruction for the agent")],
    session_id: Annotated[str | None, Form()] = None,
    include_trace: Annotated[bool, Form()] = False,
    image_url: Annotated[
        str | None,
        Form(description="Remote image URL for OCR agents. Ignored when a file is uploaded."),
    ] = None,
    image: Annotated[
        UploadFile | None,
        File(description="Image or PDF for OCR agents. Converted to base64 JPEG by the API."),
    ] = None,
) -> RunAcceptedResponse:
    """Enqueue a run and return immediately (``multipart/form-data``).

    - Text-only agents: send ``task`` only.
    - OCR with a local file: send ``task`` + ``image`` file.
    - OCR with a remote URL: send ``task`` + ``image_url``.

    Poll ``GET /agents/{name}/runs/{run_id}`` for the result.
    """
    _get_or_404(name, cache)
    from agent_system.api.app import get_run_store
    from agent_system.config import get_settings
    from agent_system.queue import enqueue_agent_run

    cfg = get_settings()
    if not cfg.queue_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job queue is disabled (QUEUE_ENABLED=false).",
        )

    resolved_image_url: str | None = image_url.strip() if image_url else None
    input_file: str | None = None
    run_id = session_id or str(uuid.uuid4())

    if image is not None and image.filename:
        agent = cache[name]
        if not agent.config.enable_ocr:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Agent '{name}' does not have enable_ocr=True — image upload is not supported.",
            )
        from agent_system.core.graph import bytes_to_image_data_url
        from agent_system.storage.minio_client import MinIOClient
        content = await image.read()
        try:
            resolved_image_url = bytes_to_image_data_url(content, image.filename)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process uploaded image: {exc}",
            ) from exc
        logger.info(
            "enqueue run: converted '%s' (%d bytes) → data URL (%d chars)",
            image.filename, len(content), len(resolved_image_url),
        )
        # Store the original file in MinIO for auditing / replay
        try:
            minio = MinIOClient()
            content_type = image.content_type or "application/octet-stream"
            input_file = minio.upload_input_file(
                agent_name=name,
                run_id=run_id,
                filename=image.filename,
                data=content,
                content_type=content_type,
            )
            logger.info("enqueue run: input file stored at %s", input_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("enqueue run: could not upload input file to MinIO: %s", exc)

    store = get_run_store()
    ok = await store.save_queued_run(
        run_id=run_id,
        agent_name=name,
        task=task,
        input_file=input_file,
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist queued run.",
        )

    job_id = await enqueue_agent_run(
        agent_name=name,
        run_id=run_id,
        task=task,
        image_url=resolved_image_url,
        include_trace=include_trace,
    )
    await store.update_job_id(run_id, job_id)

    return RunAcceptedResponse(
        agent_name=name,
        run_id=run_id,
        task=task,
        job_id=job_id,
        poll_url=f"/agents/{name}/runs/{run_id}",
    )


@router.post(
    "/{name}/runs/{run_id}/resume",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunAcceptedResponse,
    summary="Enqueue resume for a run paused for human tool approval",
)
async def resume_agent_run(
    name: str,
    run_id: str,
    payload: AgentResumeRequest,
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
) -> RunAcceptedResponse:
    """Enqueue operator decision; poll GET /agents/{name}/runs/{run_id} for the outcome."""
    _get_or_404(name, cache)
    from agent_system.api.app import get_run_store
    from agent_system.config import get_settings
    from agent_system.queue import enqueue_agent_resume

    cfg = get_settings()
    if not cfg.queue_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job queue is disabled (QUEUE_ENABLED=false).",
        )

    pending = await get_pending(run_id)
    if pending is None or pending.agent_name != name:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pending tool approval for run_id '{run_id}' on agent '{name}'.",
        )
    if payload.action == "reject" and not (payload.reason or "").strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a non-empty 'reason' when action is 'reject'.",
        )

    decision: dict = (
        {"action": "approve"}
        if payload.action == "approve"
        else {"action": "reject", "reason": (payload.reason or "").strip()}
    )

    store = get_run_store()
    row = await store.fetch_run(run_id)
    task = (row or {}).get("task") or pending.task
    await store.update_run_status(run_id, "queued")

    job_id = await enqueue_agent_resume(
        agent_name=name,
        run_id=run_id,
        decision=decision,
    )

    return RunAcceptedResponse(
        agent_name=name,
        run_id=run_id,
        task=task,
        job_id=job_id,
        poll_url=f"/agents/{name}/runs/{run_id}",
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
        tools=agent.tool_names,
        tools_requiring_approval=list(agent.config.tools_requiring_approval or []),
        plugins=list(agent.config.plugins or []),
        role=agent.config.role,
        sub_agents=list(agent.config.sub_agents or []),
        enable_ocr=agent.config.enable_ocr,
        ocr_skill_name=agent.config.ocr_skill_name,
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
        "tools_requiring_approval": list(config.tools_requiring_approval or []),
        "plugins": list(config.plugins or []),
        "extra_metadata": config.extra_metadata,
        "role": config.role,
        "sub_agents": list(config.sub_agents or []),
        "enable_ocr": config.enable_ocr,
        "ocr_model": config.ocr_model,
        "ocr_model_source": config.ocr_model_source,
        "ocr_skill_name": config.ocr_skill_name,
    }
