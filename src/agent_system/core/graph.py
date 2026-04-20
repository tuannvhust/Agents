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
from typing import Annotated, Any, Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from typing_extensions import TypedDict

from agent_system.core.checkpointing import get_checkpoint_saver
from agent_system.core.plugins import AgentPlugin, SafetyViolation
from agent_system.core.reflection import ReflectionDecision, ReflectionEngine
from agent_system.core.trace import (
    build_agent_trace_step,
    build_reflect_trace_step,
    build_tools_trace_step,
)

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
    agent_name: str  # set on initial state — tools_node re-binds RunContext (LangGraph task isolation)
    # Ordered narrative trace (agent plan + tool args + reflection) for export / DB
    trace_events: list[dict]
    # Set by human_approval node: next edge target ("tools" | "agent")
    human_approval_next: str


# ── Node implementations ──────────────────────────────────────────────────────

def make_agent_node(
    llm_with_tools: BaseChatModel,
    plugins: list[AgentPlugin] | None = None,
):
    """Return the agent node function bound to a specific LLM and plugin list."""
    _plugins: list[AgentPlugin] = plugins or []

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

        # ── Plugin before_model callbacks ──────────────────────────────────
        for plugin in _plugins:
            try:
                messages = await plugin.before_model(state, messages)
            except SafetyViolation:
                raise
            except FileNotFoundError:
                # Missing guardrail rule file — fail closed; do not run the main LLM without safety config.
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[AGENT NODE] plugin '%s' before_model raised unexpected error: %s",
                    getattr(plugin, "name", type(plugin).__name__),
                    exc,
                )

        logger.info("[AGENT NODE] calling LLM with %d message(s) in context...", len(messages))

        response: AIMessage = await llm_with_tools.ainvoke(messages)

        # ── Plugin after_model callbacks ───────────────────────────────────
        for plugin in _plugins:
            try:
                response = await plugin.after_model(state, response)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[AGENT NODE] plugin '%s' after_model raised unexpected error: %s",
                    getattr(plugin, "name", type(plugin).__name__),
                    exc,
                )

        tool_calls = getattr(response, "tool_calls", []) or []
        if tool_calls:
            names = [tc["name"] for tc in tool_calls]
            logger.info(
                "[AGENT NODE] LLM responded with %d tool call(s): %s",
                len(tool_calls),
                names,
            )
            for tc in tool_calls:
                logger.info(
                    "[AGENT NODE]   tool=%s | args=%s",
                    tc["name"],
                    json.dumps(tc.get("args", {}), ensure_ascii=False),
                )
        else:
            content = str(response.content)
            logger.info(
                "[AGENT NODE] LLM responded with text (%d chars)",
                len(content),
            )
            logger.debug("[AGENT NODE] response preview: %.400s", content[:400])

        prior_trace = list(state.get("trace_events") or [])
        agent_step = build_agent_trace_step(response, attempt)
        return {
            "messages": [response],
            "reflection_decision": "",
            "trace_events": prior_trace + [agent_step],
        }

    return agent_node


def make_tools_node(tools: list[BaseTool]):
    """Return a tool-execution node that runs all requested tool calls."""

    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    async def tools_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return {}
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        cfg = (config or {}).get("configurable") or {}
        run_id_resolved = state.get("run_id") or str(cfg.get("thread_id") or "")
        agent_name_resolved = state.get("agent_name") or str(cfg.get("agent_name") or "")

        from agent_system.core.run_context import RunContext, run_ctx

        ctx_token = None
        if run_id_resolved and agent_name_resolved:
            ctx_token = run_ctx.set(
                RunContext(run_id=run_id_resolved, agent_name=agent_name_resolved)
            )

        n_calls = len(last_message.tool_calls)
        logger.info(_SEP)
        logger.info("[TOOLS NODE] executing %d tool call(s)", n_calls)

        tool_messages: list[ToolMessage] = []
        errors: list[str] = []
        call_records: list[dict] = []
        trace_executions: list[dict[str, Any]] = []

        try:
            for i, tool_call in enumerate(last_message.tool_calls, start=1):
                tool_name: str = tool_call["name"]
                tool_args: dict[str, Any] = tool_call.get("args", {})
                call_id: str = tool_call["id"]

                args_json = json.dumps(tool_args, ensure_ascii=False)
                logger.info(
                    "[TOOLS NODE] (%d/%d) calling tool=%s | args=%s",
                    i, n_calls, tool_name, args_json,
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
                    trace_executions.append({
                        "tool_name": tool_name,
                        "tool_call_id": call_id,
                        "arguments": tool_args,
                        "success": False,
                        "error": error_msg,
                        "output": error_msg,
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
                    trace_executions.append({
                        "tool_name": tool_name,
                        "tool_call_id": call_id,
                        "arguments": tool_args,
                        "success": True,
                        "error": None,
                        "output": result_str,
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
                    trace_executions.append({
                        "tool_name": tool_name,
                        "tool_call_id": call_id,
                        "arguments": tool_args,
                        "success": False,
                        "error": str(exc),
                        "output": "",
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
            prior_trace = list(state.get("trace_events") or [])
            tools_step = build_tools_trace_step(trace_executions)
            return {
                "messages": tool_messages,
                "tool_errors": (state.get("tool_errors") or []) + errors,
                "tool_call_records": existing_records + call_records,
                "trace_events": prior_trace + [tools_step],
            }
        finally:
            if ctx_token is not None:
                run_ctx.reset(ctx_token)

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
            trace_events=list(state.get("trace_events") or []),
        )

        logger.info(
            "[REFLECT NODE] decision=%s | reason=%s",
            result.decision.value,
            result.reason,
        )
        if result.suggestions:
            logger.info("[REFLECT NODE] suggestions: %s", result.suggestions)

        prior_trace = list(state.get("trace_events") or [])
        reflect_step = build_reflect_trace_step(result)

        updates: dict[str, Any] = {
            "reflection_count": attempt + 1,
            "reflection_decision": result.decision.value,
            "reflection_feedback": result.suggestions or result.reason,
            "tool_errors": [],
            "trace_events": prior_trace + [reflect_step],
        }

        if result.decision == ReflectionDecision.DONE:
            updates["final_answer"] = last_ai
            logger.info("[REFLECT NODE] final answer set (%d chars)", len(last_ai))

        return updates

    return reflect_node


def make_human_approval_node():
    """Pause for operator approval before executing planned tool calls (high-stakes gate)."""

    async def human_approval_node(state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        if not messages:
            logger.warning("[HUMAN APPROVAL] no messages — routing back to agent")
            return {"human_approval_next": "agent"}
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"human_approval_next": "agent"}

        planned = [
            {"name": tc["name"], "id": tc["id"], "arguments": dict(tc.get("args", {}))}
            for tc in last.tool_calls
        ]
        payload = {
            "kind": "tool_approval",
            "run_id": state.get("run_id", ""),
            "task": state.get("task", ""),
            "planned_tools": planned,
            "assistant_plan_excerpt": _last_ai_text(messages)[:4000],
            "messages_digest": _messages_digest(messages),
            "trace_tail": list(state.get("trace_events") or [])[-16:],
        }
        logger.info(
            "[HUMAN APPROVAL] interrupting for operator review | tools=%s",
            [p["name"] for p in planned],
        )

        decision = interrupt(payload) or {}
        action = str(decision.get("action", "reject")).lower().strip()

        if action == "approve":
            logger.info("[HUMAN APPROVAL] approved — proceeding to tools node")
            return {"human_approval_next": "tools"}

        reason = str(decision.get("reason") or "Rejected by operator.").strip()
        logger.info("[HUMAN APPROVAL] rejected — %s", reason[:200])
        tool_messages = [
            ToolMessage(
                content=f"Tool call rejected by human operator: {reason}",
                tool_call_id=tc["id"],
            )
            for tc in last.tool_calls
        ]
        return {
            "human_approval_next": "agent",
            "messages": tool_messages,
            "tool_errors": (state.get("tool_errors") or []) + [f"Human rejection: {reason}"],
        }

    return human_approval_node


# ── Routing conditions ────────────────────────────────────────────────────────

def route_after_human_approval(state: AgentState) -> Literal["tools", "agent"]:
    nxt = state.get("human_approval_next", "agent")
    if nxt == "tools":
        logger.info("[ROUTE] human_approval → tools")
        return "tools"
    logger.info("[ROUTE] human_approval → agent")
    return "agent"


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
    tools_requiring_approval: frozenset[str] | None = None,
    plugins: list[AgentPlugin] | None = None,
) -> Any:
    """Assemble and compile the LangGraph agent graph.

    When ``tools_requiring_approval`` is non-empty, any agent step that plans at least one
    tool whose name is in that set routes through ``human_approval`` first. The graph is
    compiled with a checkpointer so LangGraph ``interrupt`` / ``Command(resume=...)`` works.
    """
    approval = frozenset(tools_requiring_approval or [])

    def route_after_agent(state: AgentState) -> Literal["tools", "reflect", "human_approval"]:
        """After agent runs: high-stakes tools → human gate; else tools; else reflect."""
        messages = state.get("messages", [])
        if not messages:
            logger.info("[ROUTE] agent → reflect")
            return "reflect"
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            logger.info("[ROUTE] agent → reflect")
            return "reflect"
        names = [tc["name"] for tc in last.tool_calls]
        if approval:
            planned = {tc["name"] for tc in last.tool_calls}
            if planned & approval:
                logger.info("[ROUTE] agent → human_approval %s (approval set hit)", names)
                return "human_approval"
        logger.info("[ROUTE] agent → tools %s", names)
        return "tools"

    llm_with_tools = llm.bind_tools(tools) if tools else llm
    reflection_engine = ReflectionEngine(llm=llm, max_retries=max_reflections)

    graph = StateGraph(AgentState)

    graph.add_node("agent", make_agent_node(llm_with_tools, plugins=plugins or []))
    graph.add_node("tools", make_tools_node(tools))
    graph.add_node("reflect", make_reflect_node(reflection_engine))

    graph.add_edge(START, "agent")
    if approval:
        graph.add_node("human_approval", make_human_approval_node())
        graph.add_conditional_edges(
            "agent",
            route_after_agent,
            {"tools": "tools", "reflect": "reflect", "human_approval": "human_approval"},
        )
        graph.add_conditional_edges(
            "human_approval",
            route_after_human_approval,
            {"tools": "tools", "agent": "agent"},
        )
    else:
        graph.add_conditional_edges(
            "agent",
            route_after_agent,
            {"tools": "tools", "reflect": "reflect"},
        )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "reflect", route_after_reflection, {"agent": "agent", END: END}
    )

    if approval:
        return graph.compile(checkpointer=get_checkpoint_saver())
    return graph.compile()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _messages_digest(messages: Sequence[BaseMessage], limit: int = 24) -> list[dict[str, Any]]:
    """Compact message list for reviewer UI (role + truncated content)."""
    out: list[dict[str, Any]] = []
    for m in messages[-limit:]:
        cls = type(m).__name__
        content: Any = getattr(m, "content", "")
        if isinstance(content, list):
            text = str(content)[:1200]
        else:
            text = str(content)[:1200]
        entry: dict[str, Any] = {"type": cls, "content": text}
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            entry["tool_calls"] = [
                {"name": tc.get("name"), "id": tc.get("id"), "args": tc.get("args")}
                for tc in (m.tool_calls or [])
            ]
        out.append(entry)
    return out


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
