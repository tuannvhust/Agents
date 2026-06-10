"""Operator Reviewer API — list pending tool approvals and decide without naming the agent in the URL."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse

from agent_system.api.routes.agents import get_cache
from agent_system.api.schemas import AgentResumeRequest, RunAcceptedResponse
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
    pending = await list_pending()
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
    p = await get_pending(run_id)
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
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunAcceptedResponse,
    summary="Approve or reject the paused tool batch (async)",
)
async def review_decide(
    run_id: str,
    payload: AgentResumeRequest,
    response: Response,
    cache: Annotated[dict[str, Agent], Depends(get_cache)],
) -> RunAcceptedResponse:
    _no_store_headers(response)
    async with _lock_for_decide(run_id):
        pending = await get_pending(run_id)
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
        if pending.agent_name not in cache:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{pending.agent_name}' is not loaded.",
            )
        decision: dict = (
            {"action": "approve"}
            if payload.action == "approve"
            else {"action": "reject", "reason": (payload.reason or "").strip()}
        )

        from agent_system.api.app import get_run_store
        from agent_system.config import get_settings
        from agent_system.queue import enqueue_agent_resume

        cfg = get_settings()
        if not cfg.queue_enabled:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Job queue is disabled (QUEUE_ENABLED=false).",
            )

        await get_run_store().update_run_status(run_id, "queued")
        job_id = await enqueue_agent_resume(
            agent_name=pending.agent_name,
            run_id=run_id,
            decision=decision,
        )

    return RunAcceptedResponse(
        agent_name=pending.agent_name,
        run_id=run_id,
        task=pending.task,
        job_id=job_id,
        poll_url=f"/agents/{pending.agent_name}/runs/{run_id}",
    )
