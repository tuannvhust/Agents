"""Integration-style tests for the FastAPI routes using TestClient."""

from __future__ import annotations

import os

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from agent_system.api.app import create_app
from agent_system.config.settings import get_settings


@pytest.fixture(scope="module")
def client():
    """TestClient with mocked external services."""
    os.environ["QUEUE_ENABLED"] = "false"
    get_settings.cache_clear()
    with (
        patch("agent_system.logging.elastic_logger.ElasticSearchHandler._build_client", return_value=None),
        patch("agent_system.tools.registry.ToolRegistry.load_mcp_tools", new_callable=AsyncMock),
        patch("agent_system.core.checkpointing.init_checkpoint_saver", new_callable=AsyncMock),
        patch("agent_system.cache.redis_client.init_redis", new_callable=AsyncMock),
        patch("agent_system.tracing.init_langfuse_handler"),
    ):
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    get_settings.cache_clear()


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "version" in data


def test_list_agents_empty(client: TestClient):
    resp = client.get("/agents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["agents"] == []


def test_create_agent_skill_not_found(client: TestClient):
    """Creating an agent with a non-existent skill should return 404."""
    resp = client.post(
        "/agents",
        json={
            "name": "test-agent",
            "skill_name": "nonexistent_skill_xyz",
        },
    )
    assert resp.status_code == 404


def test_get_nonexistent_agent(client: TestClient):
    resp = client.get("/agents/ghost")
    assert resp.status_code == 404


def test_delete_nonexistent_agent(client: TestClient):
    resp = client.delete("/agents/ghost")
    assert resp.status_code == 404
