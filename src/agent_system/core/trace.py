"""Structured run traces — agent reasoning, tool intents, and execution records."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage

from agent_system.core.reflection import ReflectionResult

logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = 1


def _stringify_ai_content(msg: AIMessage) -> str:
    """Plain-text plan / reasoning from an assistant message (multimodal-safe)."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content is not None else ""


def build_agent_trace_step(
    msg: AIMessage,
    reflection_attempt: int,
) -> dict[str, Any]:
    """One agent node visit: visible assistant text plus planned tool calls with exact args."""
    tool_calls = getattr(msg, "tool_calls", None) or []
    planned: list[dict[str, Any]] = []
    for tc in tool_calls:
        raw_args = tc.get("args", {})
        planned.append(
            {
                "name": tc.get("name", ""),
                "id": tc.get("id", ""),
                "arguments": copy.deepcopy(raw_args) if isinstance(raw_args, dict) else raw_args,
            }
        )
    return {
        "type": "agent",
        "reflection_attempt": reflection_attempt,
        "assistant_text": _stringify_ai_content(msg),
        "tools_chosen": [p["name"] for p in planned],
        "tool_calls_planned": planned,
    }


def build_tools_trace_step(executions: list[dict[str, Any]]) -> dict[str, Any]:
    """One tools node visit: exact arguments and outcomes for each invocation."""
    return {
        "type": "tools",
        "executions": executions,
    }


def build_reflect_trace_step(result: ReflectionResult) -> dict[str, Any]:
    """Reflection / internal evaluation step."""
    return {
        "type": "reflect",
        "decision": result.decision.value,
        "reason": result.reason,
        "suggestions": result.suggestions or "",
    }


def assemble_run_trace_document(
    *,
    run_id: str,
    agent_name: str,
    task: str,
    trace_events: list[dict[str, Any]],
    tool_call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Full export payload: ordered steps plus a flat list of tool invocations."""
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": run_id,
        "agent_name": agent_name,
        "task": task,
        "steps": list(trace_events),
        "tool_invocations": [
            {
                "tool_name": r.get("tool_name"),
                "arguments": copy.deepcopy(r.get("input_args") or {}),
                "success": r.get("success"),
                "error": r.get("error"),
                "output": r.get("output", ""),
            }
            for r in tool_call_records
        ],
    }


def log_trace_summary(trace_doc: dict[str, Any]) -> None:
    """INFO: compact line; DEBUG: full JSON for operators who enable verbose logs."""
    steps = trace_doc.get("steps") or []
    inv = trace_doc.get("tool_invocations") or []
    names = [t.get("tool_name") for t in inv if t.get("tool_name")]
    logger.info(
        "run_trace run_id=%s agent=%s steps=%d tool_invocations=%d tools=%s",
        trace_doc.get("run_id"),
        trace_doc.get("agent_name"),
        len(steps),
        len(inv),
        names,
    )
    logger.debug(
        "run_trace_full %s",
        json.dumps(trace_doc, ensure_ascii=False, default=str),
    )
