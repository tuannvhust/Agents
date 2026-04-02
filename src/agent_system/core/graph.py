"""LangGraph agent graph — plan → execute → reflect → (retry | respond).

The graph implements a self-correcting ReAct-style loop:

  START
    │
    ▼
  [agent]  ──── tool calls? ──▶  [tools]  ─┐
    │                                        │
    │◀───────────────────────────────────────┘
    │
    ▼ (no more tool calls)
  [reflect] ── DONE ──▶ END
       │
       └── RETRY ──▶ [agent]  (inject reflection feedback)
       │
       └── FAIL  ──▶ END
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent_system.core.reflection import ReflectionDecision, ReflectionEngine

logger = logging.getLogger(__name__)

MAX_REFLECTIONS = 3

_SEP = "-" * 72


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """Shared state passed between all graph nodes."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: str
    skill_prompt: str
    tool_errors: list[str]
    reflection_count: int
    final_answer: str
    reflection_decision: str
    reflection_feedback: str
    # Accumulated tool call records — written to PostgreSQL at run end by Agent.run()
    tool_call_records: list[dict]
    run_id: str   # propagated from Agent.run() so tools_node can tag records


# ── Node implementations ──────────────────────────────────────────────────────

def make_agent_node(llm_with_tools: BaseChatModel):
    """Return the agent node function bound to a specific LLM."""

    async def agent_node(state: AgentState) -> dict[str, Any]:
        attempt = state.get("reflection_count", 0)
        task = state.get("task", "")
        messages = list(state.get("messages", []))

        logger.info(_SEP)
        logger.info("[AGENT NODE] attempt=%d | messages_in_history=%d", attempt, len(messages))
        logger.debug("[AGENT NODE] task: %s", task)

        # Prepend system prompt once
        if not messages or not isinstance(messages[0], SystemMessage):
            skill_prompt = state.get("skill_prompt", "")
            system_msg = SystemMessage(content=skill_prompt)
            messages = [system_msg] + messages
            logger.debug("[AGENT NODE] injected system prompt (%d chars)", len(skill_prompt))

        # Inject reflection feedback on retry
        feedback = state.get("reflection_feedback", "")
        if attempt > 0 and feedback:
            messages = messages + [HumanMessage(content=f"[Reflection feedback]: {feedback}")]
            logger.info("[AGENT NODE] injecting reflection feedback: %.300s", feedback)

        logger.info("[AGENT NODE] calling LLM with %d message(s) in context...", len(messages))

        response: AIMessage = await llm_with_tools.ainvoke(messages)

        tool_calls = getattr(response, "tool_calls", []) or []
        if tool_calls:
            names = [tc["name"] for tc in tool_calls]
            logger.info(
                "[AGENT NODE] LLM responded with %d tool call(s): %s",
                len(tool_calls),
                names,
            )
            for tc in tool_calls:
                logger.debug(
                    "[AGENT NODE]   tool=%s | args=%s",
                    tc["name"],
                    json.dumps(tc.get("args", {}), ensure_ascii=False)[:300],
                )
        else:
            content = str(response.content)
            logger.info(
                "[AGENT NODE] LLM responded with text (%d chars)",
                len(content),
            )
            logger.debug("[AGENT NODE] response preview: %.400s", content[:400])

        return {
            "messages": [response],
            "reflection_decision": "",
        }

    return agent_node


def make_tools_node(tools: list[BaseTool]):
    """Return a tool-execution node that runs all requested tool calls."""

    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    async def tools_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {}
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        n_calls = len(last_message.tool_calls)
        logger.info(_SEP)
        logger.info("[TOOLS NODE] executing %d tool call(s)", n_calls)

        tool_messages: list[ToolMessage] = []
        errors: list[str] = []
        call_records: list[dict] = []

        for i, tool_call in enumerate(last_message.tool_calls, start=1):
            tool_name: str = tool_call["name"]
            tool_args: dict[str, Any] = tool_call.get("args", {})
            call_id: str = tool_call["id"]

            args_preview = json.dumps(tool_args, ensure_ascii=False)[:300]
            logger.info(
                "[TOOLS NODE] (%d/%d) calling tool=%s | args=%s",
                i, n_calls, tool_name, args_preview,
            )

            if tool_name not in tool_map:
                error_msg = f"Tool '{tool_name}' not found in registry."
                errors.append(error_msg)
                tool_messages.append(ToolMessage(content=error_msg, tool_call_id=call_id))
                call_records.append({
                    "run_id": state.get("run_id", ""),
                    "tool_name": tool_name,
                    "input_args": tool_args,
                    "output": error_msg,
                    "success": False,
                    "error": error_msg,
                })
                logger.error(
                    "[TOOLS NODE] (%d/%d) tool=%s NOT FOUND in registry (available: %s)",
                    i, n_calls, tool_name, list(tool_map.keys()),
                )
                continue

            tool = tool_map[tool_name]
            try:
                if hasattr(tool, "arun"):
                    result = await tool.arun(tool_args)
                else:
                    result = tool.run(tool_args)

                result_str = str(result)
                tool_messages.append(ToolMessage(content=result_str, tool_call_id=call_id))
                call_records.append({
                    "run_id": state.get("run_id", ""),
                    "tool_name": tool_name,
                    "input_args": tool_args,
                    "output": result_str,
                    "success": True,
                    "error": None,
                })
                logger.info(
                    "[TOOLS NODE] (%d/%d) tool=%s SUCCESS (%d chars)",
                    i, n_calls, tool_name, len(result_str),
                )
                logger.debug(
                    "[TOOLS NODE] tool=%s result preview: %.500s",
                    tool_name, result_str[:500],
                )
            except Exception as exc:  # noqa: BLE001
                error_text = f"Tool '{tool_name}' failed: {exc}"
                errors.append(error_text)
                tool_messages.append(ToolMessage(content=error_text, tool_call_id=call_id))
                call_records.append({
                    "run_id": state.get("run_id", ""),
                    "tool_name": tool_name,
                    "input_args": tool_args,
                    "output": "",
                    "success": False,
                    "error": str(exc),
                })
                logger.error(
                    "[TOOLS NODE] (%d/%d) tool=%s FAILED: %s",
                    i, n_calls, tool_name, exc,
                    exc_info=True,
                )

        if errors:
            logger.warning("[TOOLS NODE] completed with %d error(s): %s", len(errors), errors)
        else:
            logger.info("[TOOLS NODE] all %d tool call(s) completed successfully", n_calls)

        existing_records = list(state.get("tool_call_records") or [])
        return {
            "messages": tool_messages,
            "tool_errors": (state.get("tool_errors") or []) + errors,
            "tool_call_records": existing_records + call_records,
        }

    return tools_node


def make_reflect_node(reflection_engine: ReflectionEngine):
    """Return the reflection node that decides: DONE | RETRY | FAIL."""

    async def reflect_node(state: AgentState) -> dict[str, Any]:
        attempt = state.get("reflection_count", 0)
        last_ai = _last_ai_text(state.get("messages", []))
        tool_errors = state.get("tool_errors") or []

        logger.info(_SEP)
        logger.info("[REFLECT NODE] attempt=%d | output_len=%d chars", attempt, len(last_ai))
        logger.debug("[REFLECT NODE] agent output preview: %.400s", last_ai[:400])

        if tool_errors:
            logger.warning(
                "[REFLECT NODE] %d tool error(s) carried into reflection: %s",
                len(tool_errors), tool_errors,
            )

        logger.info("[REFLECT NODE] calling reflection LLM...")

        result = await reflection_engine.areflect(
            task=state.get("task", ""),
            agent_output=last_ai,
            tool_errors=tool_errors,
            attempt=attempt,
        )

        logger.info(
            "[REFLECT NODE] decision=%s | reason=%s",
            result.decision.value,
            result.reason,
        )
        if result.suggestions:
            logger.info("[REFLECT NODE] suggestions: %s", result.suggestions)

        updates: dict[str, Any] = {
            "reflection_count": attempt + 1,
            "reflection_decision": result.decision.value,
            "reflection_feedback": result.suggestions or result.reason,
            "tool_errors": [],
        }

        if result.decision == ReflectionDecision.DONE:
            updates["final_answer"] = last_ai
            logger.info("[REFLECT NODE] final answer set (%d chars)", len(last_ai))

        return updates

    return reflect_node


# ── Routing conditions ────────────────────────────────────────────────────────

def route_after_agent(state: AgentState) -> Literal["tools", "reflect"]:
    """After agent runs: if tool calls present → tools; else reflect."""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            names = [tc["name"] for tc in last.tool_calls]
            logger.info("[ROUTE] agent → tools %s", names)
            return "tools"
    logger.info("[ROUTE] agent → reflect")
    return "reflect"


def route_after_reflection(state: AgentState) -> Literal["agent", "__end__"]:
    """After reflection: DONE/FAIL → END; RETRY → back to agent."""
    decision = state.get("reflection_decision", ReflectionDecision.RETRY.value)
    if decision in (ReflectionDecision.DONE.value, ReflectionDecision.FAIL.value):
        logger.info("[ROUTE] reflect → END (%s)", decision)
        return END
    logger.info("[ROUTE] reflect → agent (RETRY)")
    return "agent"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    max_reflections: int = MAX_REFLECTIONS,
) -> Any:
    """Assemble and compile the LangGraph agent graph."""
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    reflection_engine = ReflectionEngine(llm=llm, max_retries=max_reflections)

    graph = StateGraph(AgentState)

    graph.add_node("agent", make_agent_node(llm_with_tools))
    graph.add_node("tools", make_tools_node(tools))
    graph.add_node("reflect", make_reflect_node(reflection_engine))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "reflect": "reflect"}
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "reflect", route_after_reflection, {"agent": "agent", END: END}
    )

    return graph.compile()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _last_ai_text(messages: Sequence[BaseMessage]) -> str:
    """Extract text content from the last AIMessage."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str):
                return msg.content
            if isinstance(msg.content, list):
                return " ".join(
                    c.get("text", "") for c in msg.content if isinstance(c, dict)
                )
    return ""
