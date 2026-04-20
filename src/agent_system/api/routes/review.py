"""Operator Reviewer API — list pending tool approvals and decide without naming the agent in the URL."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse

from agent_system.api.routes.agents import get_cache
from agent_system.api.schemas import AgentResumeRequest, AgentRunResponse
from agent_system.api.security import require_api_key
from agent_system.core.agent import Agent
from agent_system.core.interrupt_registry import get_pending, list_pending

router = APIRouter(
    prefix="/review",
    tags=["Human review"],
    dependencies=[Depends(require_api_key)],
)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Serialize /decide per run_id so double-clicks or parallel tabs cannot run resume twice.
_decide_locks: dict[str, asyncio.Lock] = {}


def _lock_for_decide(run_id: str) -> asyncio.Lock:
    lock = _decide_locks.get(run_id)
    if lock is None:
        lock = asyncio.Lock()
        _decide_locks[run_id] = lock
    return lock


@router.get(
    "/ui",
    summary="Reviewer UI (static HTML)",
    include_in_schema=False,
)
async def reviewer_ui() -> FileResponse:
    path = _STATIC_DIR / "human_review.html"
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Reviewer UI file missing")
    return FileResponse(path, media_type="text/html")


def _no_store_headers(response: Response) -> None:
    """Avoid stale pending lists in browsers/proxies that cache GET requests."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"


@router.get("/pending", summary="List runs waiting for tool approval")
async def review_pending(response: Response) -> dict:
    _no_store_headers(response)
    pending = list_pending()
    return {
        "pending": [
            {
                "run_id": p.run_id,
                "agent_name": p.agent_name,
                "task": p.task,
                "created_at": p.created_at,
            }
            for p in pending
        ],
        "count": len(pending),
    }


@router.get("/{run_id}", summary="Get pending approval payload for a run")
async def review_detail(run_id: str, response: Response) -> dict:
    _no_store_headers(response)
    p = get_pending(run_id)
    if p is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"No pending approval for run_id '{run_id}'.",
        )
    return {
        "run_id": p.run_id,
        "agent_name": p.agent_name,
        "task": p.task,
        "created_at": p.created_at,
        "approval_request": p.payload,
    }


@router.post(
    "/{run_id}/decide",
    response_model=AgentRunResponse,
    summary="Approve or reject the paused tool batch",
)
async def review_decide(
    run_id: str,
    payload: AgentResumeRequest,
    response: Response,
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
) -> AgentRunResponse:
    _no_store_headers(response)
    async with _lock_for_decide(run_id):
        pending = get_pending(run_id)
        if pending is None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail=(
                    f"No pending approval for run_id '{run_id}' "
                    "(it may already have been decided)."
                ),
            )
        if payload.action == "reject" and not (payload.reason or "").strip():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="Provide a non-empty 'reason' when action is 'reject'.",
            )
        agent = cache.get(pending.agent_name)
        if agent is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{pending.agent_name}' is not loaded.",
            )
        decision: dict = (
            {"action": "approve"}
            if payload.action == "approve"
            else {"action": "reject", "reason": (payload.reason or "").strip()}
        )
        try:
            result = await agent.resume_run(run_id=run_id, decision=decision)
        except ValueError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

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
        run_status=result.run_status,
        approval_request=result.approval_request,
        trace=None,
    )
