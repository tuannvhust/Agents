"""SafetyPlugin — prompt-injection / input-safety gate.

Registers with the ``before_model`` callback so it runs just before every LLM
call in the agent node.  The safety rule (classifier prompt + action) is loaded
from a local guardrails/*.md file or from Langfuse, using the same hybrid/TTL
loading strategy as agent skills.

Rule loading
------------
Rules are loaded via SafetyRuleLoader which reads from:
  • guardrails/<rule_name>.md   (local)
  • Langfuse prompt named <rule_name>  (langfuse / hybrid)

The default rule name is "prompt_injection".

Actions
-------
block  (default) — raise SafetyViolation; the run fails with an error message.
warn              — log a warning but let the run continue.
log               — same as warn; alias for future extensibility.

Classifier
----------
The plugin makes a single additional LLM call using the rule's
``## classifier_prompt`` as the system message and the user task as the
human message.  If the response contains any of the configured unsafe keywords
(``unsafe``, ``injection``, ``noprocess``, etc.) the action is triggered.

Because the classifier uses the same BaseChatModel that the agent uses, no extra
model configuration or credentials are required.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from agent_system.core.plugins import AgentPlugin, SafetyViolation
from agent_system.core.safety_rule_loader import SafetyRuleDefinition, SafetyRuleLoader

logger = logging.getLogger(__name__)

# Substrings matched case-insensitively against the classifier's one-word reply.
# NOPROCESS = greetings / dummy / filler — do not run the main agent.
_UNSAFE_KEYWORDS = frozenset(
    {"unsafe", "injection", "blocked", "malicious", "noprocess", "no_process"}
)


class SafetyPlugin(AgentPlugin):
    """Prompt-injection and input-safety gate.

    Usage (in AgentConfig.plugins)::

        plugins=["safety"]

    The plugin name "safety" is resolved to SafetyPlugin in Agent.create().

    Args:
        rule_name: Name of the guardrails rule file (without extension).
                   Defaults to "prompt_injection".
        llm:       The BaseChatModel to use for classification.
                   Injected by Agent.__init__ so no extra config is needed.
        unsafe_keywords: Strings that, when found in the classifier response,
                         trigger the configured action. Case-insensitive.
    """

    name = "safety"

    def __init__(
        self,
        rule_name: str = "prompt_injection",
        llm: BaseChatModel | None = None,
        unsafe_keywords: frozenset[str] | None = None,
    ) -> None:
        self._rule_name = rule_name
        self._llm = llm
        self._loader = SafetyRuleLoader()
        self._unsafe_keywords = unsafe_keywords if unsafe_keywords is not None else _UNSAFE_KEYWORDS

    # ── AgentPlugin callbacks ──────────────────────────────────────────────────

    async def before_model(
        self,
        state: dict[str, Any],
        messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        """Run the safety classifier before every LLM call.

        Retrieves the current rule (TTL-cached), classifies the task, and either
        raises SafetyViolation (block), logs a warning (warn/log), or passes
        through silently (safe).
        """
        if self._llm is None:
            logger.warning("[SafetyPlugin] no LLM configured — skipping safety check")
            return messages

        task = state.get("task", "")
        if not task:
            return messages

        rule = self._get_rule()
        verdict = await self._classify(task, rule)
        verdict_lower = verdict.lower()

        is_unsafe = any(kw in verdict_lower for kw in self._unsafe_keywords)

        if not is_unsafe:
            logger.debug(
                "[SafetyPlugin] rule=%s verdict=SAFE (response=%.100s)",
                rule.name, verdict,
            )
            return messages

        action = rule.action
        msg = (
            f"[SafetyPlugin] rule={rule.name!r} action={action!r} "
            f"verdict={verdict[:200]!r}"
        )

        if action == "block":
            if "noprocess" in verdict_lower or "no_process" in verdict_lower:
                logger.info(
                    "[SafetyPlugin] rule=%s verdict=NOPROCESS — skipping run (no substantive task)",
                    rule.name,
                )
                raise SafetyViolation(
                    "This input was not processed: it looks like a greeting, placeholder, "
                    "or non-task message. Send a clear task or question if you need help."
                )
            logger.warning("%s → BLOCKED", msg)
            raise SafetyViolation(
                f"Input blocked by safety rule '{rule.name}': {verdict[:200]}"
            )

        # warn / log — pass through but surface the concern
        logger.warning("%s → passing through (action=%s)", msg, action)
        return messages

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_rule(self) -> SafetyRuleDefinition:
        """Load rule (TTL-cached via SafetyRuleLoader)."""
        return self._loader.load(self._rule_name)

    async def _classify(self, task: str, rule: SafetyRuleDefinition) -> str:
        """Run a single LLM call to classify the task.

        Returns the raw text response from the classifier.
        """
        try:
            response: AIMessage = await self._llm.ainvoke(  # type: ignore[union-attr]
                [
                    SystemMessage(content=rule.classifier_prompt),
                    HumanMessage(content=task),
                ]
            )
            content = response.content
            if isinstance(content, list):
                text = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            else:
                text = str(content) if content else ""
            logger.debug(
                "[SafetyPlugin] classifier response (rule=%s): %.200s",
                rule.name, text,
            )
            return text
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[SafetyPlugin] classifier LLM call failed (rule=%s): %s — treating as SAFE",
                rule.name, exc,
            )
            return "safe"


# ── Plugin registry ────────────────────────────────────────────────────────────
# Maps the string names used in AgentConfig.plugins to their factory functions.
# Factory receives the BaseChatModel as a keyword arg.

PLUGIN_REGISTRY: dict[str, type[AgentPlugin]] = {
    "safety": SafetyPlugin,
}
