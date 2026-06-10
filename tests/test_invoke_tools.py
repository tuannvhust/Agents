"""Tests for coordinator invoke_* tool wiring."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_system.core.invoke_tools import make_invoke_agent_tools
from agent_system.core.run_context import RunContext, run_ctx


@dataclass
class _FakeConfig:
    skill_name: str = "agents/cccd_processor"
    enable_ocr: bool = False


def _fake_agent(*, enable_ocr: bool = False) -> MagicMock:
    agent = MagicMock()
    agent.config = _FakeConfig(enable_ocr=enable_ocr)
    agent.run = AsyncMock(
        return_value=MagicMock(
            run_status="completed",
            success=True,
            error=None,
            final_answer="ok",
        )
    )
    return agent


@pytest.mark.asyncio
async def test_invoke_forwards_image_url_to_ocr_subagent():
    ocr_agent = _fake_agent(enable_ocr=True)
    tools = make_invoke_agent_tools({"cccd_agent": ocr_agent})
    invoke = next(t for t in tools if t.name == "invoke_cccd_agent")

    token = run_ctx.set(
        RunContext(
            run_id="parent-run",
            agent_name="coordinator",
            image_url="https://example.com/cccd.jpg",
        )
    )
    try:
        await invoke.ainvoke({"task": "Extract CCCD fields."})
    finally:
        run_ctx.reset(token)

    ocr_agent.run.assert_awaited_once_with(
        "Extract CCCD fields.",
        image_url="https://example.com/cccd.jpg",
    )


@pytest.mark.asyncio
async def test_invoke_skips_image_url_for_non_ocr_subagent():
    plain_agent = _fake_agent(enable_ocr=False)
    tools = make_invoke_agent_tools({"researcher": plain_agent})
    invoke = next(t for t in tools if t.name == "invoke_researcher")

    token = run_ctx.set(
        RunContext(
            run_id="parent-run",
            agent_name="coordinator",
            image_url="https://example.com/cccd.jpg",
        )
    )
    try:
        await invoke.ainvoke({"task": "Research Python versions."})
    finally:
        run_ctx.reset(token)

    plain_agent.run.assert_awaited_once_with(
        "Research Python versions.",
        image_url=None,
    )


@pytest.mark.asyncio
async def test_invoke_without_parent_image_url():
    ocr_agent = _fake_agent(enable_ocr=True)
    tools = make_invoke_agent_tools({"cccd_agent": ocr_agent})
    invoke = next(t for t in tools if t.name == "invoke_cccd_agent")

    token = run_ctx.set(RunContext(run_id="parent-run", agent_name="coordinator"))
    try:
        await invoke.ainvoke({"task": "Extract CCCD fields."})
    finally:
        run_ctx.reset(token)

    ocr_agent.run.assert_awaited_once_with(
        "Extract CCCD fields.",
        image_url=None,
    )
