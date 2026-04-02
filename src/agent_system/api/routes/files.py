"""File artifacts API — query and download files written to MinIO by agents."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from agent_system.api.security import require_api_key
from pydantic import BaseModel

router = APIRouter(
    prefix="/files",
    tags=["files"],
    dependencies=[Depends(require_api_key)],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class FileArtifact(BaseModel):
    id: int
    run_id: str
    agent_name: str
    file_path: str
    file_size: int | None
    content_type: str
    written_at: datetime


class FileArtifactList(BaseModel):
    total: int
    items: list[FileArtifact]


class PresignedUrl(BaseModel):
    file_path: str
    url: str
    expires_in_seconds: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_store():
    from agent_system.api.app import get_run_store
    store = get_run_store()
    if store is None:
        raise HTTPException(status_code=503, detail="PostgreSQL store not available.")
    return store


def _row_to_artifact(row: dict[str, Any]) -> FileArtifact:
    return FileArtifact(
        id=row["id"],
        run_id=row["run_id"],
        agent_name=row["agent_name"],
        file_path=row["file_path"],
        file_size=row["file_size"],
        content_type=row["content_type"],
        written_at=row["written_at"],
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=FileArtifactList, summary="List all file artifacts")
async def list_files(
    agent_name: str | None = Query(None, description="Filter by agent name"),
    run_id: str | None = Query(None, description="Filter by run ID"),
    limit: int = Query(100, ge=1, le=500, description="Max rows to return"),
):
    """Return file artifacts written by agents, optionally filtered."""
    store = _get_store()
    rows = await store.list_file_artifacts(run_id=run_id, agent_name=agent_name, limit=limit)
    items = [_row_to_artifact(r) for r in rows]
    return FileArtifactList(total=len(items), items=items)


@router.get("/agents/{agent_name}", response_model=FileArtifactList,
            summary="List files written by a specific agent")
async def list_files_by_agent(
    agent_name: str,
    run_id: str | None = Query(None, description="Narrow down to a specific run"),
    limit: int = Query(100, ge=1, le=500),
):
    """Return all file artifacts written by the given agent."""
    store = _get_store()
    rows = await store.list_file_artifacts(run_id=run_id, agent_name=agent_name, limit=limit)
    items = [_row_to_artifact(r) for r in rows]
    return FileArtifactList(total=len(items), items=items)


@router.get("/download", response_model=PresignedUrl,
            summary="Get a presigned MinIO download URL for a file")
def get_download_url(
    file_path: str = Query(..., description="MinIO object key, e.g. stats_utils.py"),
    expires: int = Query(3600, ge=60, le=86400, description="URL expiry in seconds"),
):
    """Generate a time-limited presigned URL for downloading a file from MinIO."""
    try:
        from agent_system.storage.minio_client import MinIOClient
        client = MinIOClient()
        url = client.presigned_url(file_path, expires_seconds=expires)
        return PresignedUrl(file_path=file_path, url=url, expires_in_seconds=expires)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
