"""LangGraph agent graph — two topology variants.

Coordinator (include_reflection=True):
  START → [agent] → [tools] → [agent] → ... → [reflect] → END
                                                   └─ RETRY → [agent]

Sub-agent (include_reflection=False):
  START → [agent] → [tools] → [agent] → ... → END
  (terminates as soon as the agent produces a text response without tool calls)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
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
    # Optional image URL consumed by the ocr pre-processing node (enable_ocr=True agents)
    image_url: str


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

        image_url_resolved = state.get("image_url") or None
        ctx_token = None
        if run_id_resolved and agent_name_resolved:
            ctx_token = run_ctx.set(
                RunContext(
                    run_id=run_id_resolved,
                    agent_name=agent_name_resolved,
                    image_url=image_url_resolved,
                )
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


# ── Image resolution helper ───────────────────────────────────────────────────

def bytes_to_image_data_url(data: bytes, filename: str) -> str:
    """Convert raw file bytes (PDF, JPEG, PNG …) to a ``data:image/jpeg;base64,…`` URL.

    Intended for API upload endpoints where the client sends the file content
    directly so the worker does not need access to the client's filesystem.

    - PDF  → page 1 rendered at 200 DPI → JPEG base64
    - Other → opened with Pillow → JPEG base64
    """
    import io as _io

    _, ext = os.path.splitext(filename.lower())

    if ext == ".pdf":
        try:
            from pdf2image import convert_from_bytes  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pdf2image is required to process PDF files. "
                "Install it: pip install pdf2image"
            ) from exc
        pages = convert_from_bytes(data, dpi=200, first_page=1, last_page=1)
        if not pages:
            raise ValueError("[OCR] Could not render any pages from uploaded PDF.")
        img = pages[0]
        logger.info("[OCR] rendered uploaded PDF page 1 at 200 DPI (%dx%d)", img.width, img.height)
    else:
        try:
            from PIL import Image as _Image  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to process image files. Install it: pip install Pillow"
            ) from exc
        img = _Image.open(_io.BytesIO(data))
        img.load()
        logger.info("[OCR] opened uploaded image %s (%dx%d, mode=%s)", ext, img.width, img.height, img.mode)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    logger.info("[OCR] encoded uploaded image to base64 JPEG (%d bytes)", buf.tell())
    return f"data:image/jpeg;base64,{b64}"


def _resolve_image_to_data_url(image_input: str) -> str:
    """Normalise *image_input* to a value the vision LLM API will accept.

    Accepted inputs
    ---------------
    - ``https://`` / ``http://`` URL  → returned unchanged (provider fetches it)
    - ``data:image/...;base64,...``   → returned unchanged (already encoded)
    - Local file path (abs or rel)   → file is read, first page rendered at 200 DPI,
      JPEG-encoded, and returned as ``data:image/jpeg;base64,<b64>``

    PDF support requires ``pdf2image`` + system ``poppler-utils``.
    Other formats (JPEG, PNG, WEBP, TIFF …) require ``Pillow``.
    """
    if image_input.startswith(("http://", "https://", "data:")):
        return image_input

    if not os.path.exists(image_input):
        raise FileNotFoundError(
            f"[OCR] Local file not found: {image_input!r}. "
            "For Docker deployments use paths under /ocr_input/ (host ocr_input/ is mounted there)."
        )

    _, ext = os.path.splitext(image_input.lower())

    if ext == ".pdf":
        try:
            from pdf2image import convert_from_path  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pdf2image is required to process PDF files. "
                "Install it: pip install pdf2image  "
                "(also ensure poppler-utils is installed on the system)."
            ) from exc
        pages = convert_from_path(image_input, dpi=200, first_page=1, last_page=1)
        if not pages:
            raise ValueError(f"[OCR] Could not render any pages from PDF: {image_input!r}")
        img = pages[0]
        logger.info("[OCR] rendered PDF page 1 at 200 DPI (%dx%d)", img.width, img.height)
    else:
        try:
            from PIL import Image  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to process local image files. "
                "Install it: pip install Pillow"
            ) from exc
        img = Image.open(image_input)
        img.load()
        logger.info("[OCR] opened local image %s (%dx%d, mode=%s)", ext, img.width, img.height, img.mode)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    logger.info("[OCR] encoded image to base64 JPEG (%d bytes)", buf.tell())
    return f"data:image/jpeg;base64,{b64}"


async def _upload_local_input_to_minio(loop: asyncio.AbstractEventLoop, file_path: str) -> None:
    """Upload a local OCR input file to MinIO and persist the object path in the run row.

    Runs in a thread executor so blocking I/O does not block the event loop.
    Failures are logged but never raise — the OCR result is still returned.
    """
    try:
        from agent_system.core.run_context import get_run_context
        from agent_system.storage.minio_client import MinIOClient

        ctx = get_run_context()
        if not (ctx and ctx.run_id and ctx.agent_name):
            logger.warning("[OCR NODE] no RunContext — skipping MinIO upload for %s", file_path)
            return

        run_id = ctx.run_id
        agent_name = ctx.agent_name
        filename = os.path.basename(file_path)

        def _sync_upload() -> str:
            with open(file_path, "rb") as fh:
                data = fh.read()
            minio = MinIOClient()
            return minio.upload_input_file(
                agent_name=agent_name,
                run_id=run_id,
                filename=filename,
                data=data,
            )

        object_path: str = await loop.run_in_executor(None, _sync_upload)
        logger.info("[OCR NODE] input file uploaded to MinIO: %s", object_path)

        # Persist the MinIO path in the run row (non-fatal if it fails)
        try:
            from agent_system.api.app import get_run_store  # works in worker too
            store = get_run_store()
            await store.update_input_file(run_id, object_path)
            logger.info("[OCR NODE] input_file persisted in agent_runs for run %s", run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[OCR NODE] could not persist input_file in DB: %s", exc)

    except Exception as exc:  # noqa: BLE001
        logger.warning("[OCR NODE] could not upload input file to MinIO: %s", exc)


# ── OCR pre-processing node ───────────────────────────────────────────────────

def make_ocr_node(vlm: BaseChatModel, ocr_skill_name: str):
    """Return an async node that calls a vision LLM before the main agent loop.

    The node:
      1. Loads the OCR system prompt from Langfuse (TTL-cached via SkillLoader).
      2. Sends the image URL + task to the VLM in a multimodal message.
      3. Appends the extracted text as a HumanMessage so the agent node can
         reason over it without ever needing to call OCR as a tool.

    The node fires only when ``state["image_url"]`` is non-empty (enforced by
    ``route_start`` in ``build_agent_graph``).
    """
    from agent_system.core.skill_loader import SkillLoader

    _loader = SkillLoader()  # cheap — reuses the process-level Langfuse singleton

    async def ocr_node(state: AgentState) -> dict[str, Any]:
        image_url: str = state.get("image_url", "")
        task: str = state.get("task", "")

        logger.info(_SEP)
        logger.info("[OCR NODE] resolving image input: %.100s", image_url)

        # Resolve local paths / PDFs → base64 data URL (runs in thread to avoid blocking)
        loop = asyncio.get_event_loop()
        resolved_url: str = await loop.run_in_executor(
            None, _resolve_image_to_data_url, image_url
        )

        if resolved_url.startswith("data:"):
            logger.info("[OCR NODE] image resolved to base64 data URL (%d chars)", len(resolved_url))
        else:
            logger.info("[OCR NODE] image is remote URL, using as-is")

        # Upload the original local file to MinIO for auditing / replay
        if not image_url.startswith(("http://", "https://", "data:")):
            await _upload_local_input_to_minio(loop, image_url)

        skill = _loader.load(ocr_skill_name)
        messages: list[BaseMessage] = [
            SystemMessage(content=skill.system_prompt),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": resolved_url}},
                # Use a neutral extraction instruction — the user's task is reserved for
                # the reasoning agent node that runs after this node.  Passing `task` here
                # would cause the VLM to respond to the user's question instead of just
                # extracting data from the image.
                {"type": "text", "text": "Extract all information from this image."},
            ]),
        ]

        response: AIMessage = await vlm.ainvoke(messages)
        ocr_text = str(response.content)

        logger.info("[OCR NODE] extracted %d chars from image", len(ocr_text))
        logger.debug("[OCR NODE] preview: %.400s", ocr_text[:400])

        prior_trace = list(state.get("trace_events") or [])
        ocr_step: dict[str, Any] = {
            "type": "ocr",
            # Store original input (not the base64 blob) to keep trace readable
            "image_source": image_url if not image_url.startswith("data:") else "<base64 input>",
            "resolved_as": "data_url" if resolved_url.startswith("data:") else "remote_url",
            "skill": ocr_skill_name,
            "extracted_chars": len(ocr_text),
            "preview": ocr_text[:200],
        }

        return {
            "messages": [HumanMessage(content=f"[OCR Result]:\n{ocr_text}")],
            "trace_events": prior_trace + [ocr_step],
        }

    return ocr_node


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    max_reflections: int = MAX_REFLECTIONS,
    tools_requiring_approval: frozenset[str] | None = None,
    plugins: list[AgentPlugin] | None = None,
    include_reflection: bool = True,
    enable_ocr: bool = False,
    ocr_vlm: BaseChatModel | None = None,
    ocr_skill_name: str = "ocr",
) -> Any:
    """Assemble and compile the LangGraph agent graph.

    Args:
        include_reflection: When True (coordinator), a reflect node is added after
            each agent response.  When False (sub-agent), the graph terminates as
            soon as the agent produces text without tool calls — no reflect overhead.
        tools_requiring_approval: Tool names that pause the graph for human review
            (LangGraph interrupt).  Requires a checkpointer; compiled automatically.
        enable_ocr: When True, a vision LLM pre-processing node is inserted before
            the main agent node.  The node fires only when ``image_url`` is present
            in the initial state; otherwise the graph flows directly to ``agent``.
        ocr_vlm: The vision LLM to use in the OCR node.  Required when
            ``enable_ocr=True``; ignored otherwise.
        ocr_skill_name: Langfuse prompt name (or local skill file) that provides
            the OCR system prompt.  Defaults to ``"ocr"``.
    """
    approval = frozenset(tools_requiring_approval or [])

    if include_reflection:
        def route_after_agent(
            state: AgentState,
        ) -> Literal["tools", "reflect", "human_approval"]:
            """After agent: high-stakes tools → human gate; tool calls → tools; else → reflect."""
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
    else:
        def route_after_agent(  # type: ignore[misc]
            state: AgentState,
        ) -> Literal["tools", "__end__", "human_approval"]:
            """After agent: tool calls → tools; no tool calls → END (no reflection)."""
            messages = state.get("messages", [])
            if not messages:
                logger.info("[ROUTE] agent → END (sub-agent, no messages)")
                return END
            last = messages[-1]
            if not isinstance(last, AIMessage) or not last.tool_calls:
                logger.info("[ROUTE] agent → END (sub-agent, final response)")
                return END
            names = [tc["name"] for tc in last.tool_calls]
            if approval:
                planned = {tc["name"] for tc in last.tool_calls}
                if planned & approval:
                    logger.info("[ROUTE] agent → human_approval %s (approval set hit)", names)
                    return "human_approval"
            logger.info("[ROUTE] agent → tools %s", names)
            return "tools"

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(llm_with_tools, plugins=plugins or []))
    graph.add_node("tools", make_tools_node(tools))
    graph.add_edge("tools", "agent")

    # ── OCR pre-processing node (opt-in per agent) ─────────────────────────
    if enable_ocr and ocr_vlm is not None:
        graph.add_node("ocr", make_ocr_node(ocr_vlm, ocr_skill_name))
        graph.add_edge("ocr", "agent")

        def route_start(state: AgentState) -> Literal["ocr", "agent"]:
            """Route to ocr when an image is present; skip directly to agent otherwise."""
            if state.get("image_url", ""):
                logger.info("[ROUTE] START → ocr (image_url present)")
                return "ocr"
            logger.info("[ROUTE] START → agent (no image_url)")
            return "agent"

        graph.add_conditional_edges(START, route_start, {"ocr": "ocr", "agent": "agent"})
    else:
        graph.add_edge(START, "agent")

    if include_reflection:
        reflection_engine = ReflectionEngine(llm=llm, max_retries=max_reflections)
        graph.add_node("reflect", make_reflect_node(reflection_engine))
        graph.add_conditional_edges(
            "reflect", route_after_reflection, {"agent": "agent", END: END}
        )
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
    else:
        if approval:
            graph.add_node("human_approval", make_human_approval_node())
            graph.add_conditional_edges(
                "agent",
                route_after_agent,
                {"tools": "tools", END: END, "human_approval": "human_approval"},
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
                {"tools": "tools", END: END},
            )

    needs_checkpointer = bool(approval)
    if needs_checkpointer:
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
