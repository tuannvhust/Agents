"""MinIO key layout for run/session-scoped workspace and framework exports."""

from __future__ import annotations

from agent_system.config import get_settings


def run_workspace_prefix(agent_name: str, run_id: str) -> str | None:
    """Return ``runs/{agent}/{run_id}/`` when tool path scoping is enabled; else ``None``."""
    if not get_settings().minio.scope_paths_to_run:
        return None
    return f"runs/{agent_name}/{run_id}/"


def run_exports_prefix(agent_name: str, run_id: str) -> str:
    """Directory prefix for ``result.json`` / ``trace.json`` (same session as ``run_id``).

    With workspace scoping: ``runs/{agent}/{run_id}/_exports/`` (avoids clashing with tool paths
    like ``result.json`` at workspace root).

    Without scoping: ``exports/{agent}/{run_id}/`` so exports stay isolated per session even when
    tool keys are flat.
    """
    if get_settings().minio.scope_paths_to_run:
        return f"runs/{agent_name}/{run_id}/_exports/"
    return f"exports/{agent_name}/{run_id}/"
