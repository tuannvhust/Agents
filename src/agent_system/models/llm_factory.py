"""LLM factory — creates ChatOpenAI instances for Openrouter or local models."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_openai import ChatOpenAI

from agent_system.config import get_settings


class LLMFactory:
    """Builds LangChain-compatible LLM instances.

    Openrouter exposes an OpenAI-compatible API, so ``ChatOpenAI`` works
    seamlessly by overriding ``base_url`` and ``api_key``.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def create(
        self,
        model: str | None = None,
        source: Literal["openrouter", "local"] = "openrouter",
        temperature: float | None = None,
        max_tokens: int | None = None,
        streaming: bool = False,
    ) -> ChatOpenAI:
        cfg_or = self._settings.openrouter
        cfg_local = self._settings.local_model

        if source == "local":
            resolved_model = model or cfg_local.model_name or cfg_or.default_model
            return ChatOpenAI(
                model=resolved_model,
                base_url=cfg_local.base_url,
                api_key=cfg_local.api_key,  # type: ignore[arg-type]
                temperature=temperature if temperature is not None else cfg_or.temperature,
                max_tokens=max_tokens or cfg_or.max_tokens,
                streaming=streaming,
            )

        resolved_model = model or cfg_or.default_model
        return ChatOpenAI(
            model=resolved_model,
            base_url=cfg_or.base_url,
            api_key=cfg_or.api_key,  # type: ignore[arg-type]
            temperature=temperature if temperature is not None else cfg_or.temperature,
            max_tokens=max_tokens or cfg_or.max_tokens,
            streaming=streaming,
            default_headers={
                # Openrouter requires this header for rate-limiting analytics
                "HTTP-Referer": "https://github.com/agent-system",
                "X-Title": "Agent System",
            },
        )


@lru_cache(maxsize=8)
def get_llm(
    model: str | None = None,
    source: Literal["openrouter", "local"] = "openrouter",
    temperature: float = 0.0,
    streaming: bool = False,
) -> ChatOpenAI:
    """Cached LLM accessor — avoids re-instantiating for the same config."""
    factory = LLMFactory()
    return factory.create(
        model=model,
        source=source,
        temperature=temperature,
        streaming=streaming,
    )
