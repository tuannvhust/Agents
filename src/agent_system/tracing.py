"""Langfuse tracing — process-level singleton for the LangChain CallbackHandler.

Usage
-----
Initialise once at application startup::

    from agent_system.tracing import init_langfuse_handler
    init_langfuse_handler()

Retrieve in any module that needs to attach the handler to a LangChain call::

    from agent_system.tracing import get_langfuse_handler
    handler = get_langfuse_handler()   # None if Langfuse is not configured
    callbacks = [handler] if handler else []
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Module-level singleton — set once by init_langfuse_handler()
_handler = None


def init_langfuse_handler() -> bool:
    """Create and store the Langfuse CallbackHandler singleton.

    Returns True if the handler was successfully initialised, False otherwise.
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _handler

    if _handler is not None:
        return True  # already initialised

    from agent_system.config import get_settings

    cfg = get_settings().langfuse

    if not cfg.public_key or not cfg.secret_key:
        logger.info(
            "Langfuse tracing disabled — LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            "not set in .env."
        )
        return False

    try:
        # Langfuse v4 uses a global OTel-based client.
        # Initialise it first with credentials, then CallbackHandler picks it up.
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        Langfuse(
            public_key=cfg.public_key,
            secret_key=cfg.secret_key,
            host=cfg.url,
        )
        _handler = CallbackHandler()   # uses the global client — no key args in v4
        logger.info(
            "Langfuse tracing enabled (v4) — traces will be sent to %s", cfg.url
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse CallbackHandler init failed: %s", exc)
        return False


def get_langfuse_handler():
    """Return the active CallbackHandler, or None if not initialised."""
    return _handler


def is_tracing_enabled() -> bool:
    return _handler is not None
