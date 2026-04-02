"""Tool registry — central store for all tools available to agents."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool

from agent_system.config import get_settings

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages tool discovery and retrieval.

    Tools can be registered:
      - Manually via ``register()``
      - In bulk from an MCP client via ``load_mcp_tools()``
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            logger.warning("Overwriting existing tool: '%s'", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: '%s'", tool.name)

    def register_many(self, tools: list[BaseTool]) -> None:
        for t in tools:
            self.register(t)

    async def load_mcp_tools(self) -> None:
        """Connect to all configured MCP servers and import their tools."""
        from agent_system.tools.mcp_client import MCPClient

        cfg = get_settings()
        servers = cfg.mcp_servers
        if not servers:
            logger.debug("No MCP servers configured.")
            return

        client = MCPClient(servers)
        await client.connect()
        self.register_many(client.tools)
        logger.info("Loaded %d tools from MCP servers.", len(client.tools))

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return self._tools[name]

    def get_many(self, names: list[str]) -> list[BaseTool]:
        return [self.get(n) for n in names]

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
