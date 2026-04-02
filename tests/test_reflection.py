"""Tests for the reflection engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_system.core.reflection import (
    ReflectionDecision,
    ReflectionEngine,
    ReflectionResult,
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
