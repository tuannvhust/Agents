"""Sanity checks for cache wiring (no live Redis required)."""

from __future__ import annotations

import pytest


def test_settings_cache_defaults():
    from agent_system.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.cache_enabled is False
    assert s.cache_type == "redis"
    assert "redis://" in s.cache_redis_url


@pytest.fixture
def reset_run_store_singleton():
    import agent_system.api.app as app_mod

    prev = app_mod._run_store
    app_mod._run_store = None
    yield
    app_mod._run_store = prev


def test_get_run_store_is_plain_run_store_when_cache_disabled(reset_run_store_singleton, monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "false")
    from agent_system.config import get_settings
    from agent_system.api.app import get_run_store
    from agent_system.storage.run_store import RunStore

    get_settings.cache_clear()
    store = get_run_store()
    assert isinstance(store, RunStore)


def test_caching_run_store_delegates_queue_lifecycle_methods():
    from unittest.mock import AsyncMock, MagicMock

    from agent_system.storage.caching_run_store import CachingRunStore

    inner = MagicMock()
    inner.save_queued_run = AsyncMock(return_value=True)
    inner.update_run_status = AsyncMock(return_value=True)
    inner.ensure_run_row = AsyncMock(return_value=True)
    redis = MagicMock()
    redis.delete = AsyncMock()

    store = CachingRunStore(
        inner,
        redis,
        memory_ttl_seconds=60,
        conversation_ttl_seconds=60,
        tool_messages_ttl_seconds=60,
    )
    assert hasattr(store, "save_queued_run")
    assert hasattr(store, "update_run_status")
    assert hasattr(store, "ensure_run_row")
