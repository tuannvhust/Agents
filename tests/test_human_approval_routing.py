"""Graph compile checks for human-in-the-loop approval."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_system.core.graph import build_agent_graph


def test_build_graph_without_approval_has_no_checkpoint():
    llm = MagicMock()
    g = build_agent_graph(llm=llm, tools=[], tools_requiring_approval=frozenset())
    assert "human_approval" not in g.nodes
    assert getattr(g, "checkpointer", None) is None


def test_build_graph_with_approval_registers_human_node_and_checkpoint():
    llm = MagicMock()
    g = build_agent_graph(
        llm=llm,
        tools=[],
        tools_requiring_approval=frozenset({"danger"}),
    )
    assert "human_approval" in g.nodes
    assert g.checkpointer is not None
