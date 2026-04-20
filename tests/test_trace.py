"""Unit tests for run trace serialization."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agent_system.core.reflection import ReflectionDecision, ReflectionResult
from agent_system.core.trace import (
    assemble_run_trace_document,
    build_agent_trace_step,
    build_reflect_trace_step,
    build_tools_trace_step,
)


def test_build_agent_trace_step_text_and_tools():
    msg = AIMessage(
        content="I will search first.",
        tool_calls=[
            {"name": "web_search", "id": "call-1", "args": {"query": "python asyncio", "n": 3}},
        ],
    )
    step = build_agent_trace_step(msg, reflection_attempt=0)
    assert step["type"] == "agent"
    assert step["reflection_attempt"] == 0
    assert step["assistant_text"] == "I will search first."
    assert step["tools_chosen"] == ["web_search"]
    assert len(step["tool_calls_planned"]) == 1
    assert step["tool_calls_planned"][0]["name"] == "web_search"
    assert step["tool_calls_planned"][0]["arguments"] == {"query": "python asyncio", "n": 3}


def test_build_agent_trace_step_deep_copy_args():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "t", "id": "x", "args": {"nested": {"a": 1}}}],
    )
    step = build_agent_trace_step(msg, 0)
    step["tool_calls_planned"][0]["arguments"]["nested"]["a"] = 99
    # Original message args unchanged
    assert msg.tool_calls[0]["args"]["nested"]["a"] == 1


def test_build_tools_trace_step():
    step = build_tools_trace_step(
        [
            {
                "tool_name": "calc",
                "tool_call_id": "c1",
                "arguments": {"expression": "1+1"},
                "success": True,
                "error": None,
                "output": "2",
            }
        ]
    )
    assert step["type"] == "tools"
    assert step["executions"][0]["arguments"] == {"expression": "1+1"}


def test_build_reflect_trace_step():
    r = ReflectionResult(
        decision=ReflectionDecision.DONE,
        reason="ok",
        suggestions="",
    )
    step = build_reflect_trace_step(r)
    assert step["type"] == "reflect"
    assert step["decision"] == "DONE"
    assert step["reason"] == "ok"


def test_assemble_run_trace_document():
    doc = assemble_run_trace_document(
        run_id="r1",
        agent_name="a1",
        task="do thing",
        trace_events=[{"type": "agent", "assistant_text": "plan"}],
        tool_call_records=[
            {
                "run_id": "r1",
                "tool_name": "x",
                "input_args": {"k": "v"},
                "output": "out",
                "success": True,
                "error": None,
            }
        ],
    )
    assert doc["schema_version"] == 1
    assert doc["run_id"] == "r1"
    assert doc["steps"][0]["assistant_text"] == "plan"
    assert doc["tool_invocations"][0]["arguments"] == {"k": "v"}
    assert doc["tool_invocations"][0]["output"] == "out"
