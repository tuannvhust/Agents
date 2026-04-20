"""Self-reflection module — evaluates agent outputs and decides next action."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """You are a critical evaluator for an AI agent system.

Your job is to assess whether the agent has successfully completed its task or whether it
encountered an error / produced an unsatisfactory result and needs to retry.

You are given the original task, the agent's latest output, any tool errors, and a chronological
**process trace** (assistant text, tools planned, tools executed with exact arguments). Use that
trace to judge *how* the agent worked, not only the final text.

You MUST explicitly answer these process questions (use the trace; cite concrete tools/args):

1. **Initial plan** — Was the agent's initial approach logical for the task (reasoning + first
   actions, including which tools it chose to invoke first)?
2. **First tool choice** — Identify the **first tool actually executed** in the trace (if any).
   Was that tool the right first step, or would another tool have been more appropriate?
   If no tool ran, say "N/A — no tool invoked" and judge whether staying text-only was right.
3. **Tool arguments** — For that first executed tool (or each failed tool if errors are present):
   were the arguments correct, complete, and properly formatted for that tool?

Then decide overall status:

  DONE      — the task is fully complete and the output is correct
  RETRY     — the agent made an error or the output is incomplete / wrong; it should try again
  FAIL      — the task cannot be completed (e.g. impossible request, max retries exceeded)

Output format (strict order):

PROCESS_REVIEW:
1. Initial plan: <1-4 sentences>
2. First tool: <tool name or N/A> — <correct / wrong — which tool should have been used if wrong>
3. Tool arguments: <correct / issues — be specific>

STATUS: <DONE|RETRY|FAIL>
REASON: <1-3 sentences synthesizing output quality and the process review above>
SUGGESTIONS: <concrete next steps — required when STATUS is RETRY; otherwise leave empty or brief>
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
        trace_events: list[dict[str, Any]] | None = None,
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
                content=self._build_user_message(
                    task, agent_output, tool_errors, attempt, trace_events
                )
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
        trace_events: list[dict[str, Any]] | None = None,
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
                content=self._build_user_message(
                    task, agent_output, tool_errors, attempt, trace_events
                )
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
        task: str,
        agent_output: str,
        tool_errors: list[str],
        attempt: int,
        trace_events: list[dict[str, Any]] | None = None,
    ) -> str:
        errors_section = ""
        if tool_errors:
            errors_section = "\n\nTool Errors:\n" + "\n".join(f"- {e}" for e in tool_errors)

        trace_section = ""
        if trace_events:
            trace_section = (
                "\n\nProcess trace (chronological — use for questions 1–3):\n"
                + _format_trace_events_for_prompt(trace_events)
            )

        return (
            f"Original Task:\n{task}\n\n"
            f"Agent Output (attempt {attempt + 1}):\n{agent_output}"
            f"{errors_section}"
            f"{trace_section}"
        )

    @staticmethod
    def _parse_response(raw: str) -> ReflectionResult:
        decision = ReflectionDecision.RETRY
        reason_body = ""
        suggestions = ""
        process_lines: list[str] = []
        in_process = False

        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            upper = s.upper()
            if upper.startswith("PROCESS_REVIEW:"):
                in_process = True
                rest = s.split(":", 1)[1].strip()
                if rest:
                    process_lines.append(rest)
                continue
            if upper.startswith("STATUS:"):
                in_process = False
                status_str = s.split(":", 1)[1].strip().upper()
                try:
                    decision = ReflectionDecision(status_str)
                except ValueError:
                    decision = ReflectionDecision.RETRY
                continue
            if in_process:
                process_lines.append(s)
                continue
            if upper.startswith("REASON:"):
                reason_body = s.split(":", 1)[1].strip()
                continue
            if upper.startswith("SUGGESTIONS:"):
                suggestions = s.split(":", 1)[1].strip()
                continue

        proc = "\n".join(process_lines).strip()
        if proc and reason_body:
            reason = f"{proc}\n\n{reason_body}"
        elif proc:
            reason = proc
        elif reason_body:
            reason = reason_body
        else:
            reason = raw.strip()

        return ReflectionResult(decision=decision, reason=reason, suggestions=suggestions)


def _format_trace_events_for_prompt(events: list[dict[str, Any]]) -> str:
    """Compact, reflection-friendly view of graph trace_events."""
    if not events:
        return "(No structured trace recorded yet.)"

    parts: list[str] = []
    for idx, ev in enumerate(events, start=1):
        et = ev.get("type")
        if et == "agent":
            txt = (ev.get("assistant_text") or "").strip()
            excerpt = (txt[:500] + "…") if len(txt) > 500 else txt
            parts.append(f"{idx}. [agent] reflection_attempt={ev.get('reflection_attempt', 0)}")
            if excerpt:
                parts.append(f"   Assistant text (excerpt): {excerpt}")
            planned = ev.get("tool_calls_planned") or []
            if planned:
                parts.append("   Planned tool calls:")
                for p in planned:
                    args = p.get("arguments")
                    arg_s = json.dumps(args, ensure_ascii=False) if args is not None else "{}"
                    parts.append(f"     - {p.get('name')} | id={p.get('id')} | arguments={arg_s}")
            else:
                parts.append("   Planned tool calls: (none)")
        elif et == "tools":
            parts.append(f"{idx}. [tools]")
            for ex in ev.get("executions") or []:
                args = ex.get("arguments")
                arg_s = json.dumps(args, ensure_ascii=False) if args is not None else "{}"
                parts.append(
                    f"     executed: {ex.get('tool_name')} | success={ex.get('success')} | "
                    f"arguments={arg_s}"
                )
        elif et == "reflect":
            rsn = str(ev.get("reason", ""))[:400]
            parts.append(
                f"{idx}. [reflect] decision={ev.get('decision')} | reason_excerpt: {rsn}"
            )
        else:
            parts.append(f"{idx}. [{et}] {json.dumps(ev, ensure_ascii=False, default=str)[:300]}")

    return "\n".join(parts)
