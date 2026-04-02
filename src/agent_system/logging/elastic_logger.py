"""ElasticSearch logging handler + structured logging configuration."""

from __future__ import annotations

import logging
import logging.config
import socket
import traceback
from datetime import datetime, timezone
from typing import Any

from agent_system.config import get_settings

logger = logging.getLogger(__name__)


# ── Filters ───────────────────────────────────────────────────────────────────

class HealthCheckFilter(logging.Filter):
    """Suppress noisy uvicorn access log lines for /health and /debug/skills."""

    _SILENT_PATHS = ("/health", "/debug/skills", "/debug/tracing", "/favicon.ico")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._SILENT_PATHS)


# ── ES Handler ────────────────────────────────────────────────────────────────

class ElasticSearchHandler(logging.Handler):
    """Python logging handler that ships log records to ElasticSearch.

    Each record becomes a JSON document indexed in the configured index.
    The handler is non-blocking — failures are silently swallowed so they
    never interrupt the agent's main execution path.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cfg = get_settings().elasticsearch
        self._client = self._build_client()
        self._hostname = socket.gethostname()

    def _build_client(self):  # type: ignore[return]
        try:
            from elasticsearch import Elasticsearch

            kwargs: dict[str, Any] = {"hosts": [self._cfg.url]}
            if self._cfg.username and self._cfg.password:
                kwargs["basic_auth"] = (self._cfg.username, self._cfg.password)

            return Elasticsearch(**kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"[ElasticSearchHandler] Could not connect: {exc}")  # noqa: T201
            return None

    def emit(self, record: logging.LogRecord) -> None:
        if self._client is None:
            return
        try:
            doc = self._build_document(record)
            self._client.index(index=self._cfg.index, document=doc)
        except Exception:  # noqa: BLE001
            pass

    def _build_document(self, record: logging.LogRecord) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "@timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "hostname": self._hostname,
            "service": "agent-system",
        }

        if record.exc_info:
            doc["exception"] = "".join(traceback.format_exception(*record.exc_info))

        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            }:
                doc[key] = value

        return doc


# ── Configuration ─────────────────────────────────────────────────────────────

def configure_logging(level: str = "INFO") -> None:
    """Configure structured logging for the whole application.

    Sets up:
      - Console handler with a readable format aligned for docker logs
      - ElasticSearch handler for remote log storage
      - Patches uvicorn.access to use the same format + health-check filter
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # ── Console handler ───────────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-44s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    console.setFormatter(fmt)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(console)

    # ── ElasticSearch handler ─────────────────────────────────────────────────
    es_handler = ElasticSearchHandler()
    es_handler.setLevel(numeric_level)
    root.addHandler(es_handler)

    # ── Silence noisy third-party loggers ─────────────────────────────────────
    for noisy in (
        "httpx", "httpcore", "openai", "urllib3",
        "elasticsearch", "elastic_transport",   # prevent ES→log→ES loop
        "mcp",                                  # MCP internal protocol noise
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # OTel emits ERROR-level "Failed to detach context" when LangGraph tasks cross
    # asyncio task boundaries. The spans ARE still sent to Langfuse correctly.
    # Raise the threshold to CRITICAL so these spurious errors never appear.
    logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
    logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

    # ── Patch uvicorn.access ──────────────────────────────────────────────────
    # By default uvicorn uses its own handler/format. Redirect it through ours
    # so access logs look identical to application logs in docker logs.
    hc_filter = HealthCheckFilter()
    _patch_uvicorn_access(fmt, numeric_level, hc_filter)


def _patch_uvicorn_access(
    fmt: logging.Formatter,
    level: int,
    access_filter: logging.Filter,
) -> None:
    """Replace uvicorn.access handlers with our formatter + health-check filter."""
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers.clear()

    access_handler = logging.StreamHandler()
    access_handler.setLevel(level)
    access_handler.setFormatter(fmt)
    access_handler.addFilter(access_filter)

    uvicorn_access.addHandler(access_handler)
    uvicorn_access.propagate = False  # don't double-print via root
