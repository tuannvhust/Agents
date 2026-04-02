"""Tests for the three-mode skill loader (local, langfuse, hybrid) and TTL cache."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_system.core.skill_loader import (
    CachedLangfuseSkillSource,
    LocalSkillSource,
    SkillDefinition,
    SkillLoader,
    _CacheEntry,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def skill_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "researcher.md").write_text("# Researcher\n\n## Description\nDoes research.", encoding="utf-8")
        (p / "coder.md").write_text("# Coder\n\n## Description\nWrites code.", encoding="utf-8")
        yield p


def _mock_langfuse_client(skill_content: str = "# From Langfuse\n\n## Description\nLangfuse skill."):
    """Build a mock Langfuse client that returns a fixed skill content."""
    mock_prompt = MagicMock()
    mock_prompt.compile.return_value = skill_content

    # MagicMock(name=...) sets the mock's repr, NOT its .name attribute.
    # Create plain objects with an explicit .name attribute instead.
    researcher_entry = MagicMock(); researcher_entry.name = "researcher"
    analyst_entry = MagicMock(); analyst_entry.name = "analyst"

    mock_prompts_list = MagicMock()
    mock_prompts_list.data = [researcher_entry, analyst_entry]

    mock_client = MagicMock()
    mock_client.get_prompt.return_value = mock_prompt
    mock_client.client.prompts.list.return_value = mock_prompts_list
    return mock_client


# ── LocalSkillSource ──────────────────────────────────────────────────────────

class TestLocalSkillSource:
    def test_load_existing_skill(self, skill_dir):
        src = LocalSkillSource(skill_dir)
        skill = src.load("researcher")
        assert skill.name == "researcher"
        assert skill.source == "local"
        assert "Does research" in skill.description

    def test_load_missing_skill_raises(self, skill_dir):
        src = LocalSkillSource(skill_dir)
        with pytest.raises(FileNotFoundError, match="researcher_xyz"):
            src.load("researcher_xyz")

    def test_list_skills(self, skill_dir):
        src = LocalSkillSource(skill_dir)
        names = src.list_skills()
        assert "researcher" in names
        assert "coder" in names


# ── CachedLangfuseSkillSource ─────────────────────────────────────────────────

class TestCachedLangfuseSkillSource:
    def test_load_fetches_from_langfuse(self):
        src = CachedLangfuseSkillSource(expiry_seconds=60)
        src._client = _mock_langfuse_client()

        skill = src.load("researcher")
        assert skill.source == "langfuse"
        src._client.get_prompt.assert_called_once_with("researcher")

    def test_cache_hit_skips_langfuse_call(self):
        src = CachedLangfuseSkillSource(expiry_seconds=60)
        src._client = _mock_langfuse_client()

        src.load("researcher")
        src.load("researcher")  # second call should hit cache
        src._client.get_prompt.assert_called_once()  # only one actual fetch

    def test_cache_expired_refetches(self):
        src = CachedLangfuseSkillSource(expiry_seconds=0.05)  # 50ms TTL
        src._client = _mock_langfuse_client()

        src.load("researcher")
        time.sleep(0.1)          # wait for TTL to expire
        src.load("researcher")   # should trigger a re-fetch

        assert src._client.get_prompt.call_count == 2

    def test_cache_info_shows_entries(self):
        src = CachedLangfuseSkillSource(expiry_seconds=60)
        src._client = _mock_langfuse_client()
        src.load("researcher")

        info = src.cache_info()
        assert "researcher" in info["entries"]
        assert info["expiry_seconds"] == 60

    def test_invalidate_single_entry(self):
        src = CachedLangfuseSkillSource(expiry_seconds=60)
        src._client = _mock_langfuse_client()
        src.load("researcher")

        src.invalidate("researcher")
        src.load("researcher")  # must refetch
        assert src._client.get_prompt.call_count == 2

    def test_invalidate_all(self):
        src = CachedLangfuseSkillSource(expiry_seconds=60)
        src._client = _mock_langfuse_client()
        src.load("researcher")
        src.load("coder")  # also cache a second one

        src.invalidate()   # clear all
        assert src.cache_info()["entries"] == {}

    def test_unavailable_when_no_client(self):
        src = CachedLangfuseSkillSource(expiry_seconds=60)
        src._client = None
        assert not src.is_available

    def test_load_raises_when_unavailable(self):
        src = CachedLangfuseSkillSource(expiry_seconds=60)
        src._client = None
        with pytest.raises(RuntimeError, match="not available"):
            src.load("researcher")

    def test_list_skills_returns_empty_when_unavailable(self):
        src = CachedLangfuseSkillSource(expiry_seconds=60)
        src._client = None
        assert src.list_skills() == []

    def test_list_skills_from_langfuse(self):
        src = CachedLangfuseSkillSource(expiry_seconds=60)
        src._client = _mock_langfuse_client()
        names = src.list_skills()
        assert len(names) == 2


# ── SkillLoader — local mode ──────────────────────────────────────────────────

class TestSkillLoaderLocalMode:
    def _make_loader(self, skill_dir):
        loader = object.__new__(SkillLoader)
        loader._mode = "local"
        loader._local = LocalSkillSource(skill_dir)
        loader._langfuse = CachedLangfuseSkillSource(expiry_seconds=60)
        loader._langfuse._client = None   # Langfuse unavailable
        return loader

    def test_local_mode_loads_from_disk(self, skill_dir):
        loader = self._make_loader(skill_dir)
        skill = loader.load("researcher")
        assert skill.source == "local"

    def test_local_mode_never_calls_langfuse(self, skill_dir):
        loader = self._make_loader(skill_dir)
        loader._langfuse._client = _mock_langfuse_client()
        loader.load("researcher")
        loader._langfuse._client.get_prompt.assert_not_called()

    def test_local_mode_lists_only_local(self, skill_dir):
        loader = self._make_loader(skill_dir)
        names = loader.list_available()
        assert "researcher" in names
        assert "coder" in names


# ── SkillLoader — langfuse mode ───────────────────────────────────────────────

class TestSkillLoaderLangfuseMode:
    def _make_loader(self, skill_dir):
        loader = object.__new__(SkillLoader)
        loader._mode = "langfuse"
        loader._local = LocalSkillSource(skill_dir)
        loader._langfuse = CachedLangfuseSkillSource(expiry_seconds=60)
        loader._langfuse._client = _mock_langfuse_client()
        return loader

    def test_langfuse_mode_fetches_from_langfuse(self, skill_dir):
        loader = self._make_loader(skill_dir)
        skill = loader.load("researcher")
        assert skill.source == "langfuse"

    def test_langfuse_mode_raises_when_unavailable(self, skill_dir):
        loader = self._make_loader(skill_dir)
        loader._langfuse._client = None
        with pytest.raises(RuntimeError):
            loader.load("researcher")

    def test_langfuse_mode_never_falls_back_to_local(self, skill_dir):
        loader = self._make_loader(skill_dir)
        loader._langfuse._client = None
        # Even though local file exists, strict langfuse mode should NOT fall back
        with pytest.raises(RuntimeError):
            loader.load("researcher")


# ── SkillLoader — hybrid mode ─────────────────────────────────────────────────

class TestSkillLoaderHybridMode:
    def _make_loader(self, skill_dir, langfuse_available: bool = True):
        loader = object.__new__(SkillLoader)
        loader._mode = "hybrid"
        loader._local = LocalSkillSource(skill_dir)
        loader._langfuse = CachedLangfuseSkillSource(expiry_seconds=60)
        if langfuse_available:
            loader._langfuse._client = _mock_langfuse_client()
        else:
            loader._langfuse._client = None
        return loader

    def test_hybrid_prefers_langfuse_when_available(self, skill_dir):
        loader = self._make_loader(skill_dir, langfuse_available=True)
        skill = loader.load("researcher")
        assert skill.source == "langfuse"

    def test_hybrid_falls_back_to_local_when_langfuse_unavailable(self, skill_dir):
        loader = self._make_loader(skill_dir, langfuse_available=False)
        skill = loader.load("researcher")
        assert skill.source == "local"

    def test_hybrid_falls_back_when_langfuse_raises(self, skill_dir):
        loader = self._make_loader(skill_dir, langfuse_available=True)
        loader._langfuse._client.get_prompt.side_effect = Exception("network error")
        skill = loader.load("researcher")
        assert skill.source == "local"

    def test_hybrid_raises_when_both_fail(self, skill_dir):
        loader = self._make_loader(skill_dir, langfuse_available=False)
        with pytest.raises(FileNotFoundError, match="nonexistent_xyz"):
            loader.load("nonexistent_xyz")

    def test_hybrid_list_merges_langfuse_and_local(self, skill_dir):
        loader = self._make_loader(skill_dir, langfuse_available=True)
        # Langfuse returns: researcher, analyst
        # Local has: researcher, coder
        # Merged (langfuse priority): researcher, analyst, coder
        names = loader.list_available()
        assert "researcher" in names
        assert "analyst" in names    # only in langfuse
        assert "coder" in names      # only in local
        # researcher should not appear twice
        assert names.count("researcher") == 1

    def test_hybrid_list_falls_back_to_local_only(self, skill_dir):
        loader = self._make_loader(skill_dir, langfuse_available=False)
        names = loader.list_available()
        assert "researcher" in names
        assert "coder" in names

    def test_hybrid_mode_property(self, skill_dir):
        loader = self._make_loader(skill_dir)
        assert loader.mode == "hybrid"

    def test_cache_info_accessible(self, skill_dir):
        loader = self._make_loader(skill_dir)
        info = loader.cache_info()
        assert "entries" in info
        assert "expiry_seconds" in info
