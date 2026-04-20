"""Tests for the reflection engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_system.core.reflection import (
    ReflectionDecision,
    ReflectionEngine,
    ReflectionResult,
    _format_trace_events_for_prompt,
)


def _make_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    ai_response = MagicMock()
    ai_response.content = response_text
    llm.ainvoke = AsyncMock(return_value=ai_response)
    llm.invoke = MagicMock(return_value=ai_response)
    return llm


@pytest.mark.asyncio
async def test_reflect_done():
    llm = _make_llm("STATUS: DONE\nREASON: Task completed successfully.")
    engine = ReflectionEngine(llm=llm, max_retries=3)

    result = await engine.areflect(
        task="Write a hello-world function",
        agent_output="def hello(): return 'hello world'",
        tool_errors=[],
        attempt=0,
    )

    assert result.decision == ReflectionDecision.DONE
    assert "completed" in result.reason.lower()


@pytest.mark.asyncio
async def test_reflect_retry():
    llm = _make_llm(
        "STATUS: RETRY\nREASON: Output missing error handling.\nSUGGESTIONS: Add try/except."
    )
    engine = ReflectionEngine(llm=llm, max_retries=3)

    result = await engine.areflect(
        task="Write a robust function",
        agent_output="def f(): return 1/0",
        tool_errors=["ZeroDivisionError"],
        attempt=1,
    )

    assert result.decision == ReflectionDecision.RETRY
    assert "try/except" in result.suggestions


@pytest.mark.asyncio
async def test_reflect_fail_on_max_retries():
    llm = _make_llm("STATUS: DONE\nREASON: Should not be called.")
    engine = ReflectionEngine(llm=llm, max_retries=3)

    result = await engine.areflect(
        task="Impossible task",
        agent_output="",
        tool_errors=[],
        attempt=3,  # equals max_retries → should auto-fail
    )

    assert result.decision == ReflectionDecision.FAIL
    llm.ainvoke.assert_not_called()


def test_parse_response_handles_unknown_status():
    engine = ReflectionEngine(llm=MagicMock(), max_retries=3)
    result = engine._parse_response("STATUS: UNKNOWN\nREASON: Something weird.")
    # Falls back to RETRY for unknown statuses
    assert result.decision == ReflectionDecision.RETRY


def test_parse_response_process_review_merged_into_reason():
    engine = ReflectionEngine(llm=MagicMock(), max_retries=3)
    raw = """PROCESS_REVIEW:
1. Initial plan: Logical.
2. First tool: web_search — correct.
3. Tool arguments: Correct.

STATUS: DONE
REASON: Output is fine.
"""
    result = engine._parse_response(raw)
    assert result.decision == ReflectionDecision.DONE
    assert "Initial plan" in result.reason
    assert "Output is fine" in result.reason


def test_format_trace_events_for_prompt():
    text = _format_trace_events_for_prompt(
        [
            {
                "type": "agent",
                "reflection_attempt": 0,
                "assistant_text": "I will search.",
                "tool_calls_planned": [
                    {"name": "web_search", "id": "c1", "arguments": {"query": "x"}},
                ],
            },
            {
                "type": "tools",
                "executions": [
                    {
                        "tool_name": "web_search",
                        "success": True,
                        "arguments": {"query": "x"},
                    }
                ],
            },
        ]
    )
    assert "web_search" in text
    assert "query" in text


@pytest.mark.asyncio
async def test_areflect_passes_trace_to_user_message():
    llm = _make_llm(
        "PROCESS_REVIEW:\n1. Initial plan: ok\n2. First tool: N/A\n3. Tool arguments: N/A\n\n"
        "STATUS: DONE\nREASON: ok.\n"
    )
    engine = ReflectionEngine(llm=llm, max_retries=3)
    trace = [{"type": "agent", "reflection_attempt": 0, "assistant_text": "", "tool_calls_planned": []}]
    await engine.areflect(
        task="t",
        agent_output="out",
        tool_errors=[],
        attempt=0,
        trace_events=trace,
    )
    call_args = llm.ainvoke.call_args
    human = call_args[0][0][1]
    assert "Process trace" in human.content
    assert "[agent]" in human.content
