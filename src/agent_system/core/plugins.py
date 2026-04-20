"""Plugin protocol — base class and callback hooks for agent pipeline extensions.

A plugin is a stateless (or self-managing) object that opts in to one or more
lifecycle callbacks in the agent graph. Plugins run in registration order.

Supported callbacks
-------------------
before_model  — runs just before llm_with_tools.ainvoke() in the agent node.
                Receives the current state and the message list about to be
                sent; returns (possibly modified) messages or raises
                SafetyViolation to abort the LLM call entirely.
after_model   — runs after llm_with_tools.ainvoke() returns; receives the
                AIMessage and may return a (possibly modified) copy.

Extending
---------
Subclass AgentPlugin and override only the callbacks you need.  Register the
plugin by adding its name to ``AgentConfig.plugins`` when creating an agent.

Example
-------
    class LoggingPlugin(AgentPlugin):
        name = "logging"

        async def before_model(self, state, messages):
            logger.info("PRE-LLM task=%s msgs=%d", state.get("task"), len(messages))
            return messages
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SafetyViolation(RuntimeError):
    """Raised by a plugin to abort the current LLM call.

    The agent run will be marked failed with the exception message as the error.
    """


class AgentPlugin(ABC):
    """Base class for all agent pipeline plugins.

    Subclasses must declare a unique ``name`` class attribute used to
    reference the plugin in ``AgentConfig.plugins``.
    """

    name: str = ""

    async def before_model(
        self,
        state: dict[str, Any],
        messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        """Called just before the LLM is invoked in the agent node.

        Args:
            state:    Current AgentState dict (read-only — mutations not persisted).
            messages: Message list about to be sent to the LLM.

        Returns:
            The (possibly modified) message list to send to the LLM.

        Raises:
            SafetyViolation: Abort the LLM call and fail the current run.
        """
        return messages

    async def after_model(
        self,
        state: dict[str, Any],
        response: AIMessage,
    ) -> AIMessage:
        """Called immediately after the LLM returns a response.

        Args:
            state:    Current AgentState dict.
            response: The AIMessage returned by the LLM.

        Returns:
            The (possibly modified) AIMessage to continue with.
        """
        return response
