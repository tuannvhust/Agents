"""Run status polling for async job-queue execution."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from agent_system.api.app import get_run_store
from agent_system.api.routes.agents import get_cache
from agent_system.api.schemas import RunStatusResponse
from agent_system.api.security import require_api_key
from agent_system.core.agent import Agent
from agent_system.core.interrupt_registry import get_pending

router = APIRouter(
    prefix="/agents",
    tags=["Runs"],
    dependencies=[Depends(require_api_key)],
)


@router.get(
    "/{name}/runs/{run_id}",
    response_model=RunStatusResponse,
    summary="Poll async run status and result",
)
async def get_run_status(
    name: str,
    run_id: str,
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
    include_trace: bool = Query(False, description="Include full run trace when completed."),
) -> RunStatusResponse:
    if name not in cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{name}' not found.",
        )

    store = get_run_store()
    row = await store.fetch_run(run_id)
    if row is None or row.get("agent_name") != name:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found for agent '{name}'.",
        )

    run_status = row.get("run_status") or "completed"
    approval_request = None
    if run_status == "awaiting_approval":
        pending = await get_pending(run_id)
        if pending is not None:
            approval_request = pending.payload

    artifacts = row.get("minio_artifacts") or []
    if isinstance(artifacts, str):
        import json

        artifacts = json.loads(artifacts) if artifacts else []

    trace = row.get("run_trace") if include_trace else None
    if isinstance(trace, str):
        import json

        trace = json.loads(trace) if trace else {}

    return RunStatusResponse(
        agent_name=name,
        run_id=run_id,
        task=row.get("task") or "",
        run_status=run_status,
        job_id=row.get("job_id"),
        input_file=row.get("input_file"),
        final_answer=row.get("final_answer") or "",
        success=bool(row.get("success")),
        reflection_count=int(row.get("reflection_count") or 0),
        messages_count=0,
        stored_artifacts=list(artifacts),
        error=row.get("error_message"),
        approval_request=approval_request,
        trace=trace if include_trace else None,
    )
