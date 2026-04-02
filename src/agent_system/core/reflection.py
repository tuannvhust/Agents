"""Self-reflection module — evaluates agent outputs and decides next action."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """You are a critical evaluator for an AI agent system.

Your job is to assess whether the agent has successfully completed its task or whether it
encountered an error / produced an unsatisfactory result and needs to retry.

Respond with EXACTLY one of:
  DONE      — the task is fully complete and the output is correct
  RETRY     — the agent made an error or the output is incomplete / wrong; it should try again
  FAIL      — the task cannot be completed (e.g. impossible request, max retries exceeded)

Then on a new line, provide a short explanation (1-3 sentences) and, if RETRY, concrete
suggestions for what the agent should do differently next time.

Format:
STATUS: <DONE|RETRY|FAIL>
REASON: <explanation>
SUGGESTIONS: <what to try differently — only when STATUS is RETRY>
"""


class ReflectionDecision(str, Enum):
    DONE = "DONE"
    RETRY = "RETRY"
    FAIL = "FAIL"


class ReflectionResult:
    def __init__(
        self,
        decision: ReflectionDecision,
        reason: str,
        suggestions: str = "",
    ) -> None:
        self.decision = decision
        self.reason = reason
        self.suggestions = suggestions

    def __repr__(self) -> str:
        return f"<ReflectionResult decision={self.decision} reason={self.reason!r}>"


class ReflectionEngine:
    """Uses the LLM to reflect on the agent's last action and decide next step."""

    def __init__(self, llm: BaseChatModel, max_retries: int = 3) -> None:
        self._llm = llm
        self._max_retries = max_retries

    def reflect(
        self,
        task: str,
        agent_output: str,
        tool_errors: list[str],
        attempt: int,
    ) -> ReflectionResult:
        """Synchronous reflection."""
        if attempt >= self._max_retries:
            logger.warning(
                "[REFLECTION] max retries reached (%d/%d) → FAIL",
                attempt, self._max_retries,
            )
            return ReflectionResult(
                decision=ReflectionDecision.FAIL,
                reason=f"Maximum retry attempts ({self._max_retries}) exceeded.",
            )

        logger.debug("[REFLECTION] task: %.200s", task)
        logger.debug("[REFLECTION] agent_output (%.200s...)", agent_output[:200])
        if tool_errors:
            logger.warning("[REFLECTION] tool_errors: %s", tool_errors)

        messages: list[BaseMessage] = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(
                content=self._build_user_message(task, agent_output, tool_errors, attempt)
            ),
        ]

        try:
            logger.info("[REFLECTION] calling LLM for evaluation (attempt %d)...", attempt + 1)
            response = self._llm.invoke(messages)
            raw = str(response.content)
            logger.debug("[REFLECTION] raw LLM response: %.400s", raw[:400])
            result = self._parse_response(raw)
            logger.info(
                "[REFLECTION] parsed → decision=%s | reason=%.200s",
                result.decision.value, result.reason,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("[REFLECTION] LLM call failed: %s", exc, exc_info=True)
            return ReflectionResult(
                decision=ReflectionDecision.RETRY,
                reason=f"Reflection step failed with error: {exc}",
                suggestions="Review tool inputs and try a simpler approach.",
            )

    async def areflect(
        self,
        task: str,
        agent_output: str,
        tool_errors: list[str],
        attempt: int,
    ) -> ReflectionResult:
        """Asynchronous reflection."""
        if attempt >= self._max_retries:
            logger.warning(
                "[REFLECTION] max retries reached (%d/%d) → FAIL",
                attempt, self._max_retries,
            )
            return ReflectionResult(
                decision=ReflectionDecision.FAIL,
                reason=f"Maximum retry attempts ({self._max_retries}) exceeded.",
            )

        logger.debug("[REFLECTION] task: %.200s", task)
        logger.debug("[REFLECTION] agent_output preview: %.200s", agent_output[:200])
        if tool_errors:
            logger.warning("[REFLECTION] tool_errors: %s", tool_errors)

        messages: list[BaseMessage] = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(
                content=self._build_user_message(task, agent_output, tool_errors, attempt)
            ),
        ]

        try:
            logger.info("[REFLECTION] calling LLM for evaluation (attempt %d)...", attempt + 1)
            response = await self._llm.ainvoke(messages)
            raw = str(response.content)
            logger.debug("[REFLECTION] raw LLM response: %.400s", raw[:400])
            result = self._parse_response(raw)
            logger.info(
                "[REFLECTION] parsed → decision=%s | reason=%.200s",
                result.decision.value, result.reason,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("[REFLECTION] async LLM call failed: %s", exc, exc_info=True)
            return ReflectionResult(
                decision=ReflectionDecision.RETRY,
                reason=f"Reflection step failed with error: {exc}",
                suggestions="Review tool inputs and try a simpler approach.",
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_user_message(
        task: str, agent_output: str, tool_errors: list[str], attempt: int
    ) -> str:
        errors_section = ""
        if tool_errors:
            errors_section = "\n\nTool Errors:\n" + "\n".join(f"- {e}" for e in tool_errors)

        return (
            f"Original Task:\n{task}\n\n"
            f"Agent Output (attempt {attempt + 1}):\n{agent_output}"
            f"{errors_section}"
        )

    @staticmethod
    def _parse_response(raw: str) -> ReflectionResult:
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        decision = ReflectionDecision.RETRY
        reason = raw
        suggestions = ""

        for line in lines:
            if line.upper().startswith("STATUS:"):
                status_str = line.split(":", 1)[1].strip().upper()
                try:
                    decision = ReflectionDecision(status_str)
                except ValueError:
                    decision = ReflectionDecision.RETRY

            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

            elif line.upper().startswith("SUGGESTIONS:"):
                suggestions = line.split(":", 1)[1].strip()

        return ReflectionResult(decision=decision, reason=reason, suggestions=suggestions)
