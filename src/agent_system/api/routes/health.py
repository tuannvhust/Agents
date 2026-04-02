"""Health check route."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from agent_system import __version__
from agent_system.api.schemas import HealthResponse
from agent_system.config import get_settings

router = APIRouter(tags=["Health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    """Return the health status of all backing services."""
    services: dict[str, str] = {}
    cfg = get_settings()

    # MinIO
    try:
        from minio import Minio

        client = Minio(
            cfg.minio.endpoint,
            access_key=cfg.minio.access_key,
            secret_key=cfg.minio.secret_key,
            secure=cfg.minio.secure,
        )
        client.list_buckets()
        services["minio"] = "ok"
    except Exception as exc:  # noqa: BLE001
        services["minio"] = f"error: {exc}"
        logger.warning("MinIO health check failed: %s", exc)

    # ElasticSearch
    try:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(hosts=[cfg.elasticsearch.url])
        info = es.info()
        services["elasticsearch"] = f"ok (v{info['version']['number']})"
    except Exception as exc:  # noqa: BLE001
        services["elasticsearch"] = f"error: {exc}"
        logger.warning("ElasticSearch health check failed: %s", exc)

    # Langfuse
    try:
        import httpx

        resp = httpx.get(f"{cfg.langfuse.url}/api/public/health", timeout=3)
        services["langfuse"] = "ok" if resp.is_success else f"http {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        services["langfuse"] = f"error: {exc}"

    # PostgreSQL (agent database)
    pg_url = _build_postgres_url(cfg)
    if pg_url:
        try:
            import asyncio
            import socket

            host, port = _parse_pg_host_port(pg_url)
            sock = socket.create_connection((host, port), timeout=3)
            sock.close()
            services["postgres"] = f"ok ({host}:{port})"
        except Exception as exc:  # noqa: BLE001
            services["postgres"] = f"error: {exc}"
            logger.warning("PostgreSQL health check failed: %s", exc)
    else:
        services["postgres"] = "not configured"

    # MCP servers — just report which are configured
    mcp_servers = cfg.mcp_servers
    if mcp_servers:
        names = ", ".join(s.get("name", "?") for s in mcp_servers)
        services["mcp_servers"] = f"configured ({names})"
    else:
        services["mcp_servers"] = "none configured"

    _non_error = {"not configured", "none configured"}
    overall = "ok" if all(
        v.startswith("ok") or v.startswith("configured") or v in _non_error
        for v in services.values()
    ) else "degraded"
    return HealthResponse(status=overall, version=__version__, services=services)


def _build_postgres_url(cfg) -> str | None:  # type: ignore[return]
    """Extract postgres URL from the MCP_SERVERS config if a postgres server is present."""
    for server in cfg.mcp_servers:
        if server.get("name") == "postgres":
            args: list[str] = server.get("args", [])
            for arg in args:
                if arg.startswith("postgresql://") or arg.startswith("postgres://"):
                    return arg
    return None


def _parse_pg_host_port(url: str) -> tuple[str, int]:
    """Parse host and port from a postgres connection URL."""
    # postgresql://user:pass@host:port/db
    after_at = url.split("@", 1)[-1]
    host_port_db = after_at.split("/")[0]
    if ":" in host_port_db:
        host, port_str = host_port_db.rsplit(":", 1)
        return host, int(port_str)
    return host_port_db, 5432
