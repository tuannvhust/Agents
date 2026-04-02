"""Tests for the tool registry."""

import pytest
from unittest.mock import MagicMock

from agent_system.tools.registry import ToolRegistry


def _make_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def test_register_and_retrieve():
    registry = ToolRegistry()
    tool = _make_tool("search")
    registry.register(tool)
    assert registry.get("search") is tool


def test_register_many():
    registry = ToolRegistry()
    tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
    registry.register_many(tools)
    assert len(registry) == 3
    assert set(registry.names()) == {"a", "b", "c"}


def test_get_missing_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_overwrite_warns(caplog):
    import logging

    registry = ToolRegistry()
    registry.register(_make_tool("dup"))
    with caplog.at_level(logging.WARNING):
        registry.register(_make_tool("dup"))
    assert any("Overwriting" in r.message for r in caplog.records)


def test_all_returns_copy():
    registry = ToolRegistry()
    registry.register(_make_tool("x"))
    all_tools = registry.all()
    assert len(all_tools) == 1
