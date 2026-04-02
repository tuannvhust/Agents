"""Agent — high-level interface that wires together skill, LLM, tools, and graph."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from agent_system.config import get_settings
from agent_system.core.graph import AgentState, build_agent_graph
from agent_system.core.skill_loader import SkillDefinition, SkillLoader
from agent_system.models import get_llm
from agent_system.storage import MinIOClient

logger = logging.getLogger(__name__)


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
    extra_metadata: dict[str, Any] = field(default_factory=dict)


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

        llm = get_llm(
            model=config.model,
            source=config.model_source,
            temperature=config.temperature,
        )
        self._graph = build_agent_graph(
            llm=llm,
            tools=tools,
            max_reflections=config.max_reflections,
        )
        logger.info(
            "Agent '%s' initialised with skill='%s', tools=%s, model='%s'",
            config.name,
            config.skill_name,
            [t.name for t in tools],
            config.model or get_settings().openrouter.default_model,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        config: AgentConfig,
        tool_registry: Any | None = None,
    ) -> "Agent":
        """Async factory: loads skill + resolves tools, then constructs Agent."""
        loader = SkillLoader()
        skill = loader.load(config.skill_name)

        tools: list[BaseTool] = []
        if tool_registry and config.tools:
            tools = tool_registry.get_many(config.tools)
        elif tool_registry and not config.tools:
            tools = tool_registry.all()

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
            self._config.model or get_settings().openrouter.default_model,
            self._config.model_source,
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
            "tool_call_records": [],
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
                "configurable": {"thread_id": run_id},
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
                final_state: AgentState = await self._graph.ainvoke(
                    initial_state,
                    config=invoke_config,
                )
            finally:
                run_ctx.reset(ctx_token)
            logger.info("  LangGraph execution complete")

            final_answer = final_state.get("final_answer", "")
            if not final_answer:
                from agent_system.core.graph import _last_ai_text
                final_answer = _last_ai_text(final_state["messages"])
                logger.debug("  final_answer extracted from last AI message")

            logger.debug("  final answer (%d chars): %.300s", len(final_answer), final_answer[:300])

            artifacts = await self._store_artifacts(run_id, task, final_answer)
            if artifacts:
                logger.info("  artifacts stored: %s", artifacts)

            reflection_count = final_state.get("reflection_count", 0)
            tool_call_records = list(final_state.get("tool_call_records") or [])

            result = AgentRunResult(
                agent_name=self._config.name,
                run_id=run_id,
                task=task,
                final_answer=final_answer,
                success=True,
                reflection_count=reflection_count,
                messages_count=len(final_state["messages"]),
                stored_artifacts=artifacts,
            )

            # Persist to PostgreSQL
            await self._persist_run(result, tool_call_records)
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
            )
            await self._persist_run(result, [])

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

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> AgentConfig:
        return self._config

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
            await store.save_run(
                run_id=run_id,
                agent_name=self._config.name,
                task=task,
                final_answer="",
                success=False,
                reflection_count=0,
                minio_artifacts=[],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("  could not pre-insert agent_run row: %s", exc)

    async def _persist_run(
        self, result: AgentRunResult, tool_call_records: list[dict]
    ) -> None:
        """Write the run result and tool call records to PostgreSQL."""
        try:
            from agent_system.api.app import get_run_store
            store = get_run_store()
            await store.save_run(
                run_id=result.run_id,
                agent_name=result.agent_name,
                task=result.task,
                final_answer=result.final_answer,
                success=result.success,
                reflection_count=result.reflection_count,
                minio_artifacts=result.stored_artifacts,
            )
            if tool_call_records:
                await store.save_tool_calls(tool_call_records)
            logger.info(
                "  persisted: agent_runs + %d tool_call(s) → PostgreSQL",
                len(tool_call_records),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("  could not persist run to PostgreSQL: %s", exc)

    async def _store_artifacts(
        self, run_id: str, task: str, answer: str
    ) -> list[str]:
        """Optionally persist the run result to MinIO."""
        if self._storage is None:
            return []

        artifacts: list[str] = []
        try:
            import json as _json

            payload = _json.dumps(
                {"run_id": run_id, "task": task, "answer": answer}, ensure_ascii=False
            ).encode()
            key = f"runs/{self._config.name}/{run_id}/result.json"
            self._storage.upload_bytes(key, payload, content_type="application/json")
            artifacts.append(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to store artifacts for run %s: %s", run_id, exc)

        return artifacts
