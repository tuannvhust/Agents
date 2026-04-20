"""Session-scoped MinIO path helpers."""

from __future__ import annotations

from unittest.mock import patch

from agent_system.storage.session_paths import run_exports_prefix, run_workspace_prefix


def test_run_workspace_prefix_respects_scope_flag():
    with patch("agent_system.storage.session_paths.get_settings") as m:
        m.return_value.minio.scope_paths_to_run = True
        assert run_workspace_prefix("ag", "r1") == "runs/ag/r1/"
    with patch("agent_system.storage.session_paths.get_settings") as m:
        m.return_value.minio.scope_paths_to_run = False
        assert run_workspace_prefix("ag", "r1") is None


def test_run_exports_prefix_scoped():
    with patch("agent_system.storage.session_paths.get_settings") as m:
        m.return_value.minio.scope_paths_to_run = True
        assert run_exports_prefix("researcher", "sess-1") == "runs/researcher/sess-1/_exports/"


def test_run_exports_prefix_unscoped():
    with patch("agent_system.storage.session_paths.get_settings") as m:
        m.return_value.minio.scope_paths_to_run = False
        assert run_exports_prefix("researcher", "sess-1") == "exports/researcher/sess-1/"
