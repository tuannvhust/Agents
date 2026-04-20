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
