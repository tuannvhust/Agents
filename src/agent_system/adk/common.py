"""Shared helpers for ADK proxy agents."""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx


def run_backend_agent(agent_name: str, task: str, session_id: str = "") -> str:
    """Forward one task to the Agent System API and return final answer text."""
    base_url = os.getenv("ADK_TARGET_API_BASE_URL", "http://localhost:8080").rstrip("/")
    timeout_s = float(os.getenv("ADK_TARGET_TIMEOUT_SECONDS", "240"))
    api_key = os.getenv("ADK_TARGET_API_KEY") or os.getenv("API_KEY", "")
    if not session_id.strip():
        session_id = f"adk-{agent_name}-{uuid.uuid4()}"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload: dict[str, Any] = {"task": task, "session_id": session_id}
    endpoint = f"{base_url}/agents/{agent_name}/run"

    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        final_answer = (data.get("final_answer") or "").strip()
        if final_answer:
            return final_answer
        return (
            "Agent run succeeded but no final_answer was returned. "
            f"run_id={data.get('run_id', '')}"
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return (
            f"Error forwarding task to backend agent '{agent_name}': {exc}\n"
            "Check ADK_TARGET_API_BASE_URL, API availability, and API key settings."
        )


def resolve_adk_model(default: str = "gemini-2.0-flash") -> str:
    """Resolve ADK LLM model name from env with a sane default."""
    return os.getenv("ADK_MODEL", default)

