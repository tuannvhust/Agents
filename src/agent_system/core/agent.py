"""Agent — high-level interface that wires together skill, LLM, tools, and graph."""

from __future__ import annotations

import logging
import time
import uuid
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from agent_system.config import get_settings
from agent_system.core.graph import AgentState, build_agent_graph
from agent_system.core.plugins import SafetyViolation
from agent_system.core.interrupt_registry import (
    clear_pending,
    get_pending,
    register_pending,
)
from agent_system.core.trace import assemble_run_trace_document, log_trace_summary
from agent_system.core.skill_loader import SkillDefinition, SkillLoader
from agent_system.models import get_llm
from agent_system.storage import MinIOClient
from agent_system.storage.session_paths import run_exports_prefix

logger = logging.getLogger(__name__)


def _process_graph_event(
    event: dict[str, Any],
    final_graph_output: dict[str, Any],
) -> dict[str, Any] | None:
    """Translate a single LangGraph ``astream_events`` dict into a chat stream event.

    Returns:
      - A dict with ``kind`` key to yield to the caller.
      - ``{"_capture": <state>}`` to update ``final_graph_output`` (internal sentinel).
      - ``None`` to skip the event.
    """
    ev_type: str = event["event"]
    ev_name: str = event.get("name", "")
    metadata: dict = event.get("metadata", {})
    node: str = metadata.get("langgraph_node", "")

    # ── Node start ────────────────────────────────────────────────────────────
    if ev_type == "on_chain_start":
        if ev_name == "agent":
            attempt = 0
            try:
                attempt = int(
                    (event.get("data", {}).get("input") or {}).get("reflection_count", 0)
                )
            except (TypeError, ValueError):
                pass
            prefix = "🔄 Retrying" if attempt > 0 else "🤔 Thinking"
            return {"kind": "status", "text": f"{prefix}…"}

        if ev_name == "tools":
            in_data = (event.get("data", {}).get("input") or {})
            msgs = in_data.get("messages", [])
            tool_names: list[str] = []
            if msgs:
                last = msgs[-1]
                if hasattr(last, "tool_calls") and last.tool_calls:
                    tool_names = [tc["name"] for tc in last.tool_calls]
            names_str = ", ".join(f"`{n}`" for n in tool_names) if tool_names else "tools"
            return {"kind": "status", "text": f"🔧 Calling {names_str}…"}

        if ev_name == "reflect":
            return {"kind": "status", "text": "🔄 Evaluating output…"}

        if ev_name == "human_approval":
            return {"kind": "status", "text": "⏸ Waiting for approval…"}

    # ── LLM token streaming (only fires when streaming=True LLM) ─────────────
    elif ev_type == "on_chat_model_stream" and node == "agent":
        chunk = (event.get("data") or {}).get("chunk")
        if chunk is not None:
            content = chunk.content if hasattr(chunk, "content") else ""
            if content:
                return {"kind": "token", "text": str(content)}

    # ── Tool results ──────────────────────────────────────────────────────────
    elif ev_type == "on_chain_end" and ev_name == "tools":
        output = (event.get("data") or {}).get("output") or {}
        records = output.get("tool_call_records") or []
        if records:
            results = []
            for rec in records:
                icon = "✅" if rec.get("success") else "❌"
                results.append({
                    "tool_name": rec.get("tool_name", "?"),
                    "success": rec.get("success", True),
                    "icon": icon,
                    "text": str(rec.get("output", ""))[:600],
                })
            return {"kind": "tool_results", "results": results}

    # ── Final graph output (capture for state inspection) ────────────────────
    elif ev_type == "on_chain_end" and ev_name == "LangGraph":
        out = (event.get("data") or {}).get("output") or {}
        if isinstance(out, dict):
            return {"_capture": out}

    return None


@dataclass
class AgentConfig:
    """Declarative configuration for a single agent instance."""

    name: str
    skill_name: str                              # maps to a SKILLS.md file
    model: str | None = None                     # overrides the default LLM model
    model_source: Literal["openrouter", "local"] = "openrouter"
    temperature: float = 0.0
    max_reflections: int = 3
    tools: list[str] = field(default_factory=list)  # tool names from registry
    # Tool names that require human approval in the Reviewer UI before execution
    tools_requiring_approval: list[str] = field(default_factory=list)
    # Plugin names to activate (resolved via PLUGIN_REGISTRY in safety_plugin.py).
    # Currently supported: "safety"
    plugins: list[str] = field(default_factory=list)
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    # Multi-agent orchestration role.
    # "coordinator" — orchestrates sub-agents, includes reflection, uses ORCHESTRATOR_MODEL.
    # "subagent"    — focused worker, no reflection node, uses SUBAGENT_MODEL.
    role: Literal["subagent", "coordinator"] = "subagent"
    # For coordinators: names of sub-agents to wire as invoke_* tools.
    # Empty list means all agents currently in the cache (excluding self).
    sub_agents: list[str] = field(default_factory=list)


@dataclass
class AgentRunResult:
    """Result returned after an agent completes a run."""

    agent_name: str
    run_id: str
    task: str
    final_answer: str
    success: bool
    reflection_count: int
    messages_count: int
    stored_artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    # Full structured trace (plan text, tool intents, exact args, reflection). Optional when omitted.
    trace: dict[str, Any] | None = None
    run_status: Literal["completed", "awaiting_approval"] = "completed"
    approval_request: dict[str, Any] | None = None


class Agent:
    """Encapsulates a fully-configured agent ready to execute tasks.

    Usage::

        config = AgentConfig(name="researcher", skill_name="researcher")
        agent = await Agent.create(config)
        result = await agent.run("Summarise the latest news on LLMs")
    """

    def __init__(
        self,
        config: AgentConfig,
        skill: SkillDefinition,
        tools: list[BaseTool],
        storage: MinIOClient | None = None,
    ) -> None:
        self._config = config
        self._skill = skill
        self._tools = tools
        self._storage = storage

        # Resolve effective model: explicit config > role-specific env default > global default
        _settings = get_settings()
        if config.model is not None:
            _effective_model: str | None = config.model
        elif config.role == "coordinator":
            _effective_model = _settings.orchestrator_model  # None → openrouter default
        else:
            _effective_model = _settings.subagent_model  # None → openrouter default

        # Resolve effective model_source: explicit config > role-specific env override
        if config.role == "coordinator" and _settings.orchestrator_model_source:
            _effective_source: str = _settings.orchestrator_model_source
        elif config.role == "subagent" and _settings.subagent_model_source:
            _effective_source = _settings.subagent_model_source
        else:
            _effective_source = config.model_source

        self._effective_model = _effective_model
        self._effective_source = _effective_source

        llm = get_llm(
            model=_effective_model,
            source=_effective_source,
            temperature=config.temperature,
        )

        # ── Build plugins ──────────────────────────────────────────────────────
        from agent_system.core.safety_plugin import PLUGIN_REGISTRY
        from agent_system.core.plugins import AgentPlugin

        plugin_instances: list[AgentPlugin] = []
        for plugin_name in (config.plugins or []):
            plugin_cls = PLUGIN_REGISTRY.get(plugin_name)
            if plugin_cls is None:
                logger.warning(
                    "Agent '%s': unknown plugin '%s' — skipped. "
                    "Available: %s",
                    config.name, plugin_name, list(PLUGIN_REGISTRY.keys()),
                )
                continue
            plugin_instances.append(plugin_cls(llm=llm))
            logger.info("Agent '%s': loaded plugin '%s'", config.name, plugin_name)

        _reflect = config.role == "coordinator"
        self._graph = build_agent_graph(
            llm=llm,
            tools=tools,
            max_reflections=config.max_reflections,
            tools_requiring_approval=frozenset(config.tools_requiring_approval or []),
            plugins=plugin_instances,
            include_reflection=_reflect,
        )

        # Streaming graph: same topology but LLM has streaming=True so token-level
        # events fire via astream_events.  Used by stream_run / resume_stream_run.
        streaming_llm = get_llm(
            model=_effective_model,
            source=_effective_source,
            temperature=config.temperature,
            streaming=True,
        )
        self._streaming_graph = build_agent_graph(
            llm=streaming_llm,
            tools=tools,
            max_reflections=config.max_reflections,
            tools_requiring_approval=frozenset(config.tools_requiring_approval or []),
            plugins=plugin_instances,
            include_reflection=_reflect,
        )

        logger.info(
            "Agent '%s' [role=%s] initialised | skill='%s' | tools=%s | model='%s' (%s) | "
            "reflection=%s | approval_tools=%s | plugins=%s",
            config.name,
            config.role,
            config.skill_name,
            [t.name for t in tools],
            _effective_model or _settings.openrouter.default_model,
            _effective_source,
            _reflect,
            list(config.tools_requiring_approval or []) or "—",
            [p.name for p in plugin_instances] or "—",
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        config: AgentConfig,
        tool_registry: Any | None = None,
        agent_cache: dict[str, "Agent"] | None = None,
    ) -> "Agent":
        """Async factory: loads skill, resolves tools, wires sub-agent invoke tools (coordinator).

        Args:
            config: Agent configuration.
            tool_registry: Registry of builtin / MCP tools to attach.
            agent_cache: Live agent cache; required for coordinators so invoke_* tools
                can be wired to already-registered sub-agents.
        """
        loader = SkillLoader()
        skill = loader.load(config.skill_name)

        tools: list[BaseTool] = []
        if tool_registry and config.tools:
            tools = tool_registry.get_many(config.tools)
        elif tool_registry and not config.tools:
            tools = tool_registry.all()

        # For coordinator agents: prepend invoke_<name> tools for each sub-agent.
        if config.role == "coordinator" and agent_cache:
            from agent_system.core.invoke_tools import make_invoke_agent_tools

            if config.sub_agents:
                missing = [n for n in config.sub_agents if n not in agent_cache]
                if missing:
                    logger.warning(
                        "Coordinator '%s': sub-agent(s) %s not found in cache and will be skipped. "
                        "Register sub-agents before the coordinator.",
                        config.name, missing,
                    )
                sub_agent_map = {
                    n: agent_cache[n]
                    for n in config.sub_agents
                    if n in agent_cache
                }
            else:
                # No explicit list → wire all currently registered agents (except self)
                sub_agent_map = {
                    n: a for n, a in agent_cache.items() if n != config.name
                }

            if sub_agent_map:
                invoke_tools = make_invoke_agent_tools(sub_agent_map)
                tools = invoke_tools + tools
            else:
                logger.warning(
                    "Coordinator '%s': no sub-agents available to wire. "
                    "Register sub-agents first.",
                    config.name,
                )

        storage: MinIOClient | None = None
        try:
            storage = MinIOClient()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MinIO not available — artifact storage disabled: %s", exc)

        return cls(config=config, skill=skill, tools=tools, storage=storage)

    # ── Public API ────────────────────────────────────────────────────────────

    _SEP_HEAVY = "=" * 72
    _SEP_LIGHT = "-" * 72

    async def run(self, task: str, session_id: str | None = None) -> AgentRunResult:
        """Execute the agent on a task and return a structured result."""
        run_id = session_id or str(uuid.uuid4())
        start_time = time.monotonic()

        logger.info(self._SEP_HEAVY)
        logger.info(
            "AGENT RUN START  |  agent=%-20s  |  run_id=%s",
            self._config.name, run_id,
        )
        logger.info("  task      : %s", task)
        logger.info(
            "  model     : %s (%s)",
            self._effective_model or get_settings().openrouter.default_model,
            self._effective_source,
        )
        logger.info(
            "  tools (%d) : %s",
            len(self._tools),
            [t.name for t in self._tools],
        )
        logger.info(
            "  skill     : %s (from %s)  |  max_reflections=%d",
            self._config.skill_name,
            getattr(self._skill, "source", "unknown"),
            self._config.max_reflections,
        )
        logger.info(self._SEP_LIGHT)

        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "skill_prompt": self._skill.system_prompt,
            "tool_errors": [],
            "reflection_count": 0,
            "final_answer": "",
            "reflection_decision": "",
            "reflection_feedback": "",
            "run_id": run_id,           # available to tools_node for tool_call_records
            "agent_name": self._config.name,  # tools_node re-binds RunContext (LangGraph task isolation)
            "tool_call_records": [],
            "trace_events": [],
        }

        try:
            from agent_system.tracing import get_langfuse_handler, is_tracing_enabled

            callbacks = []
            if is_tracing_enabled():
                # Reuse the global singleton handler initialised at startup.
                # Creating a new CallbackHandler(trace_context=...) per run attaches
                # an OTel context token that must be detached from the SAME asyncio
                # task — LangGraph's internal task group breaks that contract and
                # produces noisy "Failed to detach context" errors.
                # Instead we pass the run identity via `metadata` so Langfuse can
                # group and filter traces by run_id / session_id.
                handler = get_langfuse_handler()
                if handler is not None:
                    callbacks = [handler]
                    logger.debug("  Langfuse tracing attached for run %s", run_id)

            invoke_config: dict = {
                "configurable": {
                    "thread_id": run_id,
                    "agent_name": self._config.name,
                },
                "callbacks": callbacks,
                "metadata": {
                    # Langfuse reads these keys from LangChain metadata
                    "langfuse_session_id": run_id,
                    "langfuse_trace_name": f"{self._config.name}/{run_id}",
                    "langfuse_tags": [self._config.name, self._config.skill_name],
                    # standard LangChain keys
                    "session_id": run_id,
                    "agent_name": self._config.name,
                    "skill": self._config.skill_name,
                },
            }

            # Pre-insert the agent_runs row so file_artifacts FK is satisfied
            # during graph execution (write_file logs artifacts mid-run).
            # _persist_run() will UPDATE the same row with the final result.
            await self._pre_insert_run(run_id, task)

            # Set run context so tools (e.g. write_file) can log file_artifacts
            from agent_system.core.run_context import RunContext, run_ctx
            ctx_token = run_ctx.set(RunContext(run_id=run_id, agent_name=self._config.name))

            logger.info("  invoking LangGraph...")
            try:
                raw = await self._graph.ainvoke(
                    initial_state,
                    config=invoke_config,
                )
            finally:
                run_ctx.reset(ctx_token)
            logger.info("  LangGraph invocation returned")

            final_state = await self._finalize_invoke_state(invoke_config, raw)

            intr_payload = self._extract_interrupt_payload(final_state)
            if intr_payload is not None:
                register_pending(self._config.name, run_id, task, intr_payload)
                logger.info("  run paused — awaiting human approval for tool execution")
                result = AgentRunResult(
                    agent_name=self._config.name,
                    run_id=run_id,
                    task=task,
                    final_answer="",
                    success=False,
                    reflection_count=int(final_state.get("reflection_count") or 0),
                    messages_count=len(final_state.get("messages") or []),
                    error=None,
                    trace=None,
                    run_status="awaiting_approval",
                    approval_request=intr_payload,
                )
                await self._persist_paused_for_approval(run_id, task, final_state)
            else:
                result = await self._finalize_completed_run(run_id, task, final_state)
                clear_pending(run_id)
        except SafetyViolation as exc:
            # Expected guardrail outcome — not an application error; avoid ERROR/traceback noise.
            logger.info(
                "AGENT RUN BLOCKED BY GUARDRAIL  |  agent=%s  |  run_id=%s  |  %s",
                self._config.name,
                run_id,
                exc,
            )
            result = AgentRunResult(
                agent_name=self._config.name,
                run_id=run_id,
                task=task,
                final_answer="",
                success=False,
                reflection_count=0,
                messages_count=0,
                error=str(exc),
                run_status="completed",
                approval_request=None,
            )
            await self._persist_run(result, [], None)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "AGENT RUN FAILED  |  agent=%s  |  run_id=%s  |  error=%s",
                self._config.name, run_id, exc,
            )
            result = AgentRunResult(
                agent_name=self._config.name,
                run_id=run_id,
                task=task,
                final_answer="",
                success=False,
                reflection_count=0,
                messages_count=0,
                error=str(exc),
                run_status="completed",
                approval_request=None,
            )
            await self._persist_run(result, [], None)

        duration = time.monotonic() - start_time
        logger.info(self._SEP_LIGHT)
        logger.info(
            "AGENT RUN END    |  agent=%-20s  |  run_id=%s",
            self._config.name, run_id,
        )
        logger.info(
            "  success=%s  |  reflections=%d  |  messages=%d  |  duration=%.2fs",
            result.success,
            result.reflection_count,
            result.messages_count,
            duration,
        )
        logger.info(self._SEP_HEAVY)
        return result

    async def resume_run(self, run_id: str, decision: dict[str, Any]) -> AgentRunResult:
        """Resume after a human-in-the-loop interrupt. ``decision`` is
        ``{"action": "approve"}`` or ``{"action": "reject", "reason": "..."}``.
        """
        pending = get_pending(run_id)
        if pending is None or pending.agent_name != self._config.name:
            raise ValueError(
                f"No pending approval for run_id={run_id!r} and agent={self._config.name!r}."
            )
        task = pending.task
        # Drop from the in-memory reviewer queue immediately while the graph resumes.
        # If the run hits another approval gate, we register again below.
        clear_pending(run_id)

        start_time = time.monotonic()

        logger.info(self._SEP_HEAVY)
        logger.info(
            "AGENT RESUME      |  agent=%-20s  |  run_id=%s",
            self._config.name,
            run_id,
        )
        logger.info(self._SEP_LIGHT)

        try:
            from agent_system.tracing import get_langfuse_handler, is_tracing_enabled

            callbacks = []
            if is_tracing_enabled():
                handler = get_langfuse_handler()
                if handler is not None:
                    callbacks = [handler]

            invoke_config: dict = {
                "configurable": {
                    "thread_id": run_id,
                    "agent_name": self._config.name,
                },
                "callbacks": callbacks,
                "metadata": {
                    "langfuse_session_id": run_id,
                    "langfuse_trace_name": f"{self._config.name}/{run_id}",
                    "langfuse_tags": [self._config.name, self._config.skill_name],
                    "session_id": run_id,
                    "agent_name": self._config.name,
                    "skill": self._config.skill_name,
                },
            }

            from agent_system.core.run_context import RunContext, run_ctx
            ctx_token = run_ctx.set(RunContext(run_id=run_id, agent_name=self._config.name))

            logger.info("  resuming LangGraph with operator decision...")
            try:
                raw = await self._graph.ainvoke(
                    Command(resume=decision),
                    config=invoke_config,
                )
            finally:
                run_ctx.reset(ctx_token)

            final_state = await self._finalize_invoke_state(invoke_config, raw)

            intr_payload = self._extract_interrupt_payload(final_state)
            if intr_payload is not None:
                register_pending(self._config.name, run_id, task, intr_payload)
                logger.info("  run paused again — another approval gate")
                result = AgentRunResult(
                    agent_name=self._config.name,
                    run_id=run_id,
                    task=task,
                    final_answer="",
                    success=False,
                    reflection_count=int(final_state.get("reflection_count") or 0),
                    messages_count=len(final_state.get("messages") or []),
                    error=None,
                    trace=None,
                    run_status="awaiting_approval",
                    approval_request=intr_payload,
                )
                await self._persist_paused_for_approval(run_id, task, final_state)
            else:
                result = await self._finalize_completed_run(run_id, task, final_state)
                clear_pending(run_id)
        except SafetyViolation as exc:
            logger.info(
                "AGENT RESUME BLOCKED BY GUARDRAIL  |  agent=%s  |  run_id=%s  |  %s",
                self._config.name,
                run_id,
                exc,
            )
            result = AgentRunResult(
                agent_name=self._config.name,
                run_id=run_id,
                task=task,
                final_answer="",
                success=False,
                reflection_count=0,
                messages_count=0,
                error=str(exc),
                run_status="completed",
                approval_request=None,
            )
            await self._persist_run(result, [], None)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "AGENT RESUME FAILED  |  agent=%s  |  run_id=%s  |  error=%s",
                self._config.name,
                run_id,
                exc,
            )
            result = AgentRunResult(
                agent_name=self._config.name,
                run_id=run_id,
                task=task,
                final_answer="",
                success=False,
                reflection_count=0,
                messages_count=0,
                error=str(exc),
                run_status="completed",
                approval_request=None,
            )
            await self._persist_run(result, [], None)

        duration = time.monotonic() - start_time
        logger.info(self._SEP_LIGHT)
        logger.info(
            "AGENT RESUME END  |  run_id=%s  |  run_status=%s  |  duration=%.2fs",
            run_id,
            result.run_status,
            duration,
        )
        logger.info(self._SEP_HEAVY)
        return result

    # ── Streaming API (Chainlit / SSE) ────────────────────────────────────────

    async def stream_run(
        self,
        task: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator that streams execution events for a new agent run.

        Yields dicts with a ``kind`` key:
          - ``"status"``        text describing the current phase
          - ``"token"``         single LLM output token (only when streaming LLM fires)
          - ``"tool_result"``   per-tool outcome after the tools node finishes
          - ``"done"``          run completed — includes final_answer, artifacts, run_id
          - ``"approval_needed"`` run paused for human review — includes run_id + payload
          - ``"error"``         unrecoverable failure — includes text
        """
        run_id = session_id or str(uuid.uuid4())
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "skill_prompt": self._skill.system_prompt,
            "tool_errors": [],
            "reflection_count": 0,
            "final_answer": "",
            "reflection_decision": "",
            "reflection_feedback": "",
            "run_id": run_id,
            "agent_name": self._config.name,
            "tool_call_records": [],
            "trace_events": [],
        }
        async for ev in self._stream_graph(
            self._streaming_graph, initial_state, run_id, task
        ):
            yield ev

    async def resume_stream_run(
        self,
        run_id: str,
        decision: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Resume a paused run (human approval) and stream the remaining execution.

        ``decision`` is ``{"action": "approve"}`` or
        ``{"action": "reject", "reason": "…"}``.
        """
        pending = get_pending(run_id)
        if pending is None or pending.agent_name != self._config.name:
            yield {
                "kind": "error",
                "text": f"No pending approval for run_id={run_id!r}.",
            }
            return

        task = pending.task
        clear_pending(run_id)

        from agent_system.tracing import get_langfuse_handler, is_tracing_enabled

        callbacks = []
        if is_tracing_enabled():
            handler = get_langfuse_handler()
            if handler is not None:
                callbacks = [handler]
                logger.debug("  Langfuse tracing attached for resume stream run %s", run_id)

        invoke_config: dict = {
            "configurable": {
                "thread_id": run_id,
                "agent_name": self._config.name,
            },
            "callbacks": callbacks,
            "metadata": {
                "langfuse_session_id": run_id,
                "langfuse_trace_name": f"{self._config.name}/{run_id}",
                "langfuse_tags": [self._config.name, self._config.skill_name],
                "session_id": run_id,
                "agent_name": self._config.name,
                "skill": self._config.skill_name,
            },
        }

        from agent_system.core.run_context import RunContext, run_ctx
        ctx_token = run_ctx.set(RunContext(run_id=run_id, agent_name=self._config.name))
        final_graph_output: dict[str, Any] = {}

        try:
            from langgraph.types import Command as _Command

            async for event in self._streaming_graph.astream_events(
                _Command(resume=decision),
                config=invoke_config,
                version="v2",
            ):
                result = _process_graph_event(event, final_graph_output)
                if result is not None:
                    if isinstance(result, dict):
                        if result.get("_capture"):
                            final_graph_output = result["_capture"]
                        else:
                            yield result

            final_state = await self._finalize_invoke_state(invoke_config, final_graph_output)
            intr_payload = self._extract_interrupt_payload(final_state)
            if intr_payload is not None:
                register_pending(self._config.name, run_id, task, intr_payload)
                yield {"kind": "approval_needed", "run_id": run_id, "payload": intr_payload}
            else:
                result_obj = await self._finalize_completed_run(run_id, task, final_state)
                clear_pending(run_id)
                yield {
                    "kind": "done",
                    "run_id": run_id,
                    "final_answer": result_obj.final_answer,
                    "success": result_obj.success,
                    "reflection_count": result_obj.reflection_count,
                    "artifacts": result_obj.stored_artifacts,
                }
        except Exception as exc:  # noqa: BLE001
            logger.exception("resume_stream_run failed run_id=%s: %s", run_id, exc)
            yield {"kind": "error", "text": str(exc)}
        finally:
            run_ctx.reset(ctx_token)

    async def _stream_graph(
        self,
        graph: Any,
        initial_state: AgentState,
        run_id: str,
        task: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Core streaming logic shared by stream_run and resume_stream_run."""
        from agent_system.tracing import get_langfuse_handler, is_tracing_enabled

        callbacks = []
        if is_tracing_enabled():
            handler = get_langfuse_handler()
            if handler is not None:
                callbacks = [handler]
                logger.debug("  Langfuse tracing attached for stream run %s", run_id)

        invoke_config: dict = {
            "configurable": {
                "thread_id": run_id,
                "agent_name": self._config.name,
            },
            "callbacks": callbacks,
            "metadata": {
                "langfuse_session_id": run_id,
                "langfuse_trace_name": f"{self._config.name}/{run_id}",
                "langfuse_tags": [self._config.name, self._config.skill_name],
                "session_id": run_id,
                "agent_name": self._config.name,
                "skill": self._config.skill_name,
            },
        }

        await self._pre_insert_run(run_id, task)

        from agent_system.core.run_context import RunContext, run_ctx
        ctx_token = run_ctx.set(RunContext(run_id=run_id, agent_name=self._config.name))
        final_graph_output: dict[str, Any] = {}

        try:
            async for event in graph.astream_events(
                initial_state,
                config=invoke_config,
                version="v2",
            ):
                result = _process_graph_event(event, final_graph_output)
                if result is None:
                    continue
                if isinstance(result, dict) and result.get("_capture"):
                    final_graph_output = result["_capture"]
                    continue
                yield result

            final_state = await self._finalize_invoke_state(invoke_config, final_graph_output)
            intr_payload = self._extract_interrupt_payload(final_state)

            if intr_payload is not None:
                register_pending(self._config.name, run_id, task, intr_payload)
                await self._persist_paused_for_approval(run_id, task, final_state)
                yield {
                    "kind": "approval_needed",
                    "run_id": run_id,
                    "payload": intr_payload,
                }
            else:
                result_obj = await self._finalize_completed_run(run_id, task, final_state)
                clear_pending(run_id)
                yield {
                    "kind": "done",
                    "run_id": run_id,
                    "final_answer": result_obj.final_answer,
                    "success": result_obj.success,
                    "reflection_count": result_obj.reflection_count,
                    "artifacts": result_obj.stored_artifacts,
                }

        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "stream_run failed  |  agent=%s  |  run_id=%s: %s",
                self._config.name, run_id, exc,
            )
            yield {"kind": "error", "text": str(exc)}
        finally:
            run_ctx.reset(ctx_token)

    @staticmethod
    def _normalize_ainvoke_result(result: Any) -> dict[str, Any]:
        """LangGraph v1 returns a dict; v2 may return GraphOutput(value=..., interrupts=...)."""
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        value = getattr(result, "value", None)
        interrupts = getattr(result, "interrupts", None) or ()
        out: dict[str, Any] = dict(value) if isinstance(value, dict) else {}
        if interrupts and "__interrupt__" not in out:
            out["__interrupt__"] = list(interrupts)
        return out

    async def _merge_checkpoint_state(
        self, config: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        """Overlay ainvoke output on checkpoint values so we never drop channels (e.g. tool_call_records)."""
        ag = getattr(self._graph, "aget_state", None)
        if ag is None:
            return state
        try:
            snap = await ag(config)
            if snap is None:
                return state
            values = getattr(snap, "values", None)
            if not isinstance(values, dict):
                return state
            merged = dict(values)
            merged.update(state)
            return merged
        except Exception as exc:  # noqa: BLE001
            logger.debug("checkpoint state merge skipped: %s", exc)
            return state

    async def _finalize_invoke_state(self, config: dict[str, Any], raw: Any) -> dict[str, Any]:
        state = self._normalize_ainvoke_result(raw)
        return await self._merge_checkpoint_state(config, state)

    @staticmethod
    def _extract_interrupt_payload(state: dict[str, Any]) -> dict[str, Any] | None:
        intr = state.get("__interrupt__")
        if not intr:
            return None
        if isinstance(intr, (list, tuple)):
            if not intr:
                return None
            first = intr[0]
        else:
            first = intr
        val = getattr(first, "value", None)
        if isinstance(val, dict):
            return val
        if val is None:
            return {}
        return {"value": val}

    async def _finalize_completed_run(
        self, run_id: str, task: str, final_state: AgentState
    ) -> AgentRunResult:
        """Build result, artifacts, trace, and persist after a full graph completion."""
        logger.info("  LangGraph execution complete")

        final_answer = final_state.get("final_answer", "")
        if not final_answer:
            from agent_system.core.graph import _last_ai_text

            final_answer = _last_ai_text(final_state["messages"])
            logger.debug("  final_answer extracted from last AI message")

        logger.debug("  final answer (%d chars): %.300s", len(final_answer), final_answer[:300])

        artifacts = await self._store_artifacts(run_id, task, final_answer)

        reflection_count = final_state.get("reflection_count", 0)
        tool_call_records = list(final_state.get("tool_call_records") or [])
        trace_events = list(final_state.get("trace_events") or [])

        trace_doc = assemble_run_trace_document(
            run_id=run_id,
            agent_name=self._config.name,
            task=task,
            trace_events=trace_events,
            tool_call_records=tool_call_records,
        )
        log_trace_summary(trace_doc)

        trace_key = await self._store_trace_json(run_id, trace_doc)
        artifacts = list(artifacts)
        if trace_key:
            artifacts.append(trace_key)
        if artifacts:
            logger.info("  artifacts stored: %s", artifacts)

        result = AgentRunResult(
            agent_name=self._config.name,
            run_id=run_id,
            task=task,
            final_answer=final_answer,
            success=True,
            reflection_count=reflection_count,
            messages_count=len(final_state.get("messages") or []),
            stored_artifacts=artifacts,
            trace=trace_doc,
            run_status="completed",
            approval_request=None,
        )

        await self._persist_run(result, tool_call_records, trace_doc)
        return result

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def tool_names(self) -> list[str]:
        """Actual tool names attached to this agent (includes invoke_* for coordinators)."""
        return [t.name for t in self._tools]

    # ── Private ───────────────────────────────────────────────────────────────

    async def _pre_insert_run(self, run_id: str, task: str) -> None:
        """Insert a placeholder agent_runs row at run start.

        This satisfies the FK constraint on file_artifacts so write_file can
        log artifacts during graph execution, before the final result is known.
        The row is updated with the real result in _persist_run().
        """
        try:
            from agent_system.api.app import get_run_store
            store = get_run_store()
            if store is None:
                return
            ok = await store.save_run(
                run_id=run_id,
                agent_name=self._config.name,
                task=task,
                final_answer="",
                success=False,
                reflection_count=0,
                minio_artifacts=[],
            )
            if not ok:
                logger.error("  pre-insert agent_run failed for run_id=%s", run_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("  could not pre-insert agent_run row: %s", exc)

    async def _persist_run(
        self,
        result: AgentRunResult,
        tool_call_records: list[dict],
        run_trace: dict[str, Any] | None = None,
    ) -> None:
        """Write the run result, tool call records, and optional trace to PostgreSQL."""
        try:
            from agent_system.api.app import get_run_store
            store = get_run_store()
            ok_run = await store.save_run(
                run_id=result.run_id,
                agent_name=result.agent_name,
                task=result.task,
                final_answer=result.final_answer,
                success=result.success,
                reflection_count=result.reflection_count,
                minio_artifacts=result.stored_artifacts,
                run_trace=run_trace,
            )
            if not ok_run:
                logger.error(
                    "  persist failed: agent_runs not saved (run_id=%s) — skipped tool_calls",
                    result.run_id,
                )
                return
            ok_tc = await store.save_tool_calls(result.run_id, tool_call_records)
            if ok_tc:
                logger.info(
                    "  persisted: agent_runs + %d tool_call(s) → PostgreSQL",
                    len(tool_call_records),
                )
            else:
                logger.error(
                    "  persist partial: agent_runs saved but tool_calls failed (run_id=%s)",
                    result.run_id,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("  could not persist run to PostgreSQL: %s", exc)

    async def _persist_paused_for_approval(
        self,
        run_id: str,
        task: str,
        final_state: AgentState,
    ) -> None:
        """Update PostgreSQL with tool_calls and partial trace while a run awaits approval."""
        tool_call_records = list(final_state.get("tool_call_records") or [])
        trace_events = list(final_state.get("trace_events") or [])
        trace_doc = None
        if trace_events or tool_call_records:
            trace_doc = assemble_run_trace_document(
                run_id=run_id,
                agent_name=self._config.name,
                task=task,
                trace_events=trace_events,
                tool_call_records=tool_call_records,
            )
        result = AgentRunResult(
            agent_name=self._config.name,
            run_id=run_id,
            task=task,
            final_answer="",
            success=False,
            reflection_count=int(final_state.get("reflection_count") or 0),
            messages_count=len(final_state.get("messages") or []),
            stored_artifacts=[],
            error=None,
            trace=trace_doc,
            run_status="awaiting_approval",
            approval_request=None,
        )
        await self._persist_run(result, tool_call_records, trace_doc)

    async def _store_artifacts(
        self, run_id: str, task: str, answer: str
    ) -> list[str]:
        """Optionally persist the run result to MinIO."""
        if self._storage is None:
            return []

        artifacts: list[str] = []
        try:
            payload = json.dumps(
                {"run_id": run_id, "task": task, "answer": answer}, ensure_ascii=False
            ).encode()
            key = f"{run_exports_prefix(self._config.name, run_id)}result.json"
            self._storage.upload_bytes(key, payload, content_type="application/json")
            artifacts.append(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to store artifacts for run %s: %s", run_id, exc)

        return artifacts

    async def _store_trace_json(self, run_id: str, trace_doc: dict[str, Any]) -> str | None:
        """Upload the full run trace JSON to MinIO. Returns object key or None."""
        if self._storage is None:
            return None
        try:
            payload = json.dumps(trace_doc, ensure_ascii=False, default=str).encode()
            key = f"{run_exports_prefix(self._config.name, run_id)}trace.json"
            self._storage.upload_bytes(key, payload, content_type="application/json")
            logger.info("  run trace exported: %s", key)
            return key
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to store trace for run %s: %s", run_id, exc)
            return None
