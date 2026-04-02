"""Tests for the skill loader."""

import tempfile
from pathlib import Path

import pytest

from agent_system.core.skill_loader import LocalSkillSource, SkillDefinition


def test_skill_definition_parses_sections():
    content = """# Test Skill

## Description
A test skill for unit testing.

## Instructions
Do the test.

## Constraints
- No side effects.
"""
    skill = SkillDefinition(name="test", content=content)
    assert "test" in skill.description.lower()
    assert "Do the test" in skill.instructions
    assert "No side effects" in skill.constraints


def test_skill_definition_system_prompt_is_full_content():
    content = "# My Skill\n\n## Description\nDoes stuff."
    skill = SkillDefinition(name="my_skill", content=content)
    assert skill.system_prompt == content


def test_local_source_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_file = Path(tmpdir) / "researcher.md"
        skill_file.write_text("# Researcher\n\n## Description\nDoes research.", encoding="utf-8")

        source = LocalSkillSource(tmpdir)
        skill = source.load("researcher")
        assert skill.name == "researcher"
        assert "Does research" in skill.description


def test_local_source_list_skills():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "alpha.md").write_text("# Alpha", encoding="utf-8")
        (Path(tmpdir) / "beta.md").write_text("# Beta", encoding="utf-8")

        source = LocalSkillSource(tmpdir)
        names = source.list_skills()
        assert "alpha" in names
        assert "beta" in names


def test_local_source_raises_on_missing_skill():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = LocalSkillSource(tmpdir)
        with pytest.raises(FileNotFoundError):
            source.load("nonexistent")
