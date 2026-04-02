"""Tests for configuration loading."""

import os
import pytest
from unittest.mock import patch


def test_default_model_is_qwen():
    from agent_system.config.settings import OpenrouterSettings

    cfg = OpenrouterSettings()
    assert cfg.default_model == "qwen/qwen3-30b-a3b-thinking-2507"


def test_settings_override_via_env():
    with patch.dict(os.environ, {"OPENROUTER_DEFAULT_MODEL": "gpt-4o"}):
        from importlib import reload
        import agent_system.config.settings as s

        reload(s)
        cfg = s.OpenrouterSettings()
        assert cfg.default_model == "gpt-4o"


def test_mcp_servers_default_empty():
    from agent_system.config.settings import Settings

    s = Settings()
    assert s.mcp_servers == []


def test_mcp_servers_parsed_from_json():
    raw = '[{"name":"test","command":"npx","args":["-y","@test/server"]}]'
    with patch.dict(os.environ, {"MCP_SERVERS": raw}):
        from importlib import reload
        import agent_system.config.settings as s

        reload(s)
        cfg = s.Settings()
        assert len(cfg.mcp_servers) == 1
        assert cfg.mcp_servers[0]["name"] == "test"
