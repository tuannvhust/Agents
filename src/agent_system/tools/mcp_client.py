"""MCP (Model Context Protocol) client — connects to MCP servers and wraps
their tools as LangChain-compatible ``BaseTool`` instances.

Each MCP server is managed inside a persistent ``asyncio.Task`` that keeps the
``stdio_client`` context manager alive for the lifetime of the process.  Tool
calls are forwarded to the background task via an ``asyncio.Queue`` so the
cancel-scope always belongs to the task that entered it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_core.tools import BaseTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Internal request/response types ──────────────────────────────────────────

@dataclass
class _ToolCallRequest:
    tool_name: str
    kwargs: dict[str, Any]
    result_future: asyncio.Future = field(default_factory=asyncio.Future)


# ── Per-server background worker ──────────────────────────────────────────────

async def _run_server_session(
    cfg: dict[str, Any],
    ready_event: asyncio.Event,
    tools_out: list[BaseTool],
    call_queue: asyncio.Queue[_ToolCallRequest | None],
    server_name: str,
) -> None:
    """Background task that owns the stdio_client lifecycle for one MCP server."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        logger.warning(
            "mcp package not installed — MCP tool support disabled. "
            "Run: pip install mcp"
        )
        ready_event.set()
        return

    params = StdioServerParameters(
        command=cfg["command"],
        args=cfg.get("args", []),
        env=cfg.get("env"),
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("[MCP] server '%s' connected and initialized", server_name)

                # Collect tools and register them
                tools_response = await session.list_tools()
                tool_names: list[str] = []
                for mcp_tool in tools_response.tools:
                    lc_tool = _wrap_tool(mcp_tool, session, call_queue)
                    tools_out.append(lc_tool)
                    tool_names.append(mcp_tool.name)

                logger.info(
                    "[MCP] server '%s' registered %d tool(s): %s",
                    server_name, len(tool_names), tool_names,
                )

                # Signal that we are ready
                ready_event.set()

                # Serve tool-call requests until cancelled or queue closed
                while True:
                    req = await call_queue.get()
                    if req is None:  # sentinel → shutdown
                        break
                    try:
                        response = await session.call_tool(req.tool_name, arguments=req.kwargs)
                        if hasattr(response, "content"):
                            parts = [
                                c.text for c in response.content if hasattr(c, "text")
                            ]
                            result: str = "\n".join(parts)
                        else:
                            result = str(response)
                        req.result_future.set_result(result)
                    except Exception as exc:  # noqa: BLE001
                        req.result_future.set_exception(exc)

    except asyncio.CancelledError:
        pass  # normal shutdown
    except Exception as exc:  # noqa: BLE001
        logger.error("[MCP] server '%s' session error: %s", server_name, exc, exc_info=True)
    finally:
        if not ready_event.is_set():
            ready_event.set()  # unblock connect() on failure


def _wrap_tool(
    mcp_tool: Any,
    session: Any,  # kept for schema info only; calls go through queue
    call_queue: asyncio.Queue[_ToolCallRequest | None],
) -> BaseTool:
    """Build a LangChain StructuredTool that forwards calls through the queue."""
    tool_name: str = mcp_tool.name
    tool_desc: str = getattr(mcp_tool, "description", tool_name)

    async def call_tool(**kwargs: Any) -> str:
        logger.info("[MCP TOOL] calling %s | args=%s", tool_name, kwargs)
        loop = asyncio.get_event_loop()
        req = _ToolCallRequest(
            tool_name=tool_name,
            kwargs=kwargs,
            result_future=loop.create_future(),
        )
        await call_queue.put(req)
        result = await req.result_future
        logger.info("[MCP TOOL] %s returned %d chars", tool_name, len(result))
        logger.debug("[MCP TOOL] %s result preview: %.400s", tool_name, result[:400])
        return result

    input_schema = getattr(mcp_tool, "inputSchema", {}) or {}
    properties = input_schema.get("properties", {})
    required_fields = set(input_schema.get("required", []))

    fields: dict[str, Any] = {}
    for field_name, _ in properties.items():
        fields[field_name] = (str, ... if field_name in required_fields else None)

    DynamicModel = type(
        f"{tool_name}_input",
        (BaseModel,),
        {"__annotations__": {k: str for k in fields}},
    )

    return StructuredTool(
        name=tool_name,
        description=tool_desc,
        args_schema=DynamicModel,
        coroutine=call_tool,
        func=lambda **kw: asyncio.run(call_tool(**kw)),
    )


# ── Public API ────────────────────────────────────────────────────────────────

class MCPClient:
    """Manages long-lived connections to one or more MCP servers."""

    def __init__(self, server_configs: list[dict[str, Any]]) -> None:
        self._configs = server_configs
        self._tools: list[BaseTool] = []
        self._tasks: list[asyncio.Task] = []
        self._queues: list[asyncio.Queue] = []

    async def connect(self) -> None:
        """Start a background task per server and wait until all are ready."""
        for cfg in self._configs:
            server_name = cfg.get("name", cfg.get("command", "unknown"))
            cmd_display = f"{cfg['command']} {' '.join(cfg.get('args', []))}"
            logger.info("[MCP] connecting to server '%s' | cmd: %s", server_name, cmd_display)

            ready_event: asyncio.Event = asyncio.Event()
            call_queue: asyncio.Queue = asyncio.Queue()
            tools_out: list[BaseTool] = []

            task = asyncio.create_task(
                _run_server_session(cfg, ready_event, tools_out, call_queue, server_name),
                name=f"mcp-{server_name}",
            )
            self._tasks.append(task)
            self._queues.append(call_queue)

            try:
                await asyncio.wait_for(ready_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                logger.error("[MCP] server '%s' did not become ready within 30s", server_name)
                continue

            if tools_out:
                self._tools.extend(tools_out)
            else:
                logger.error("[MCP] FAILED to connect to server '%s'", server_name)

    async def disconnect(self) -> None:
        """Shut down all background server tasks gracefully."""
        for q in self._queues:
            await q.put(None)  # send sentinel
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        self._queues.clear()
        self._tools.clear()

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)
