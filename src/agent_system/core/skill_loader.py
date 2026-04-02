"""Skill loader — reads SKILLS.md definitions with three operating modes.

Modes
-----
local    — reads from local ``skills/`` directory only.
langfuse — fetches from Langfuse prompts only; results are cached and
           refreshed every ``LANGFUSE_EXPIRY_TIME`` seconds (default 100 s).
hybrid   — Langfuse is tried first; on any failure (unavailable, key not
           configured, prompt not found) the loader silently falls back to
           the local file.  This is the **default** mode.

Cache behaviour (langfuse / hybrid)
------------------------------------
Each skill fetched from Langfuse is stored in an in-process dict together
with the timestamp of the fetch.  On the next ``load()`` call the cache
entry is returned immediately if it is younger than ``LANGFUSE_EXPIRY_TIME``.
Once the TTL expires the next call triggers a fresh Langfuse fetch and
updates the cache.  The cache is a process-level singleton so it is shared
by every ``SkillLoader`` instance — only one HTTP call is made per TTL window
regardless of how many agents are running.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Literal

from agent_system.config import get_settings

logger = logging.getLogger(__name__)

SkillMode = Literal["local", "langfuse", "hybrid"]

# ── Skill definition ──────────────────────────────────────────────────────────

class SkillDefinition:
    """Parsed representation of a SKILLS.md file."""

    def __init__(self, name: str, content: str, source: str = "unknown") -> None:
        self.name = name
        self.raw = content
        self.source = source          # "local" | "langfuse" — for logging/debugging
        self._sections = self._parse_sections(content)

    @property
    def system_prompt(self) -> str:
        return self.raw

    @property
    def description(self) -> str:
        return self._sections.get("description", "")

    @property
    def instructions(self) -> str:
        return self._sections.get("instructions", "")

    @property
    def constraints(self) -> str:
        return self._sections.get("constraints", "")

    @property
    def examples(self) -> str:
        return self._sections.get("examples", "")

    def __repr__(self) -> str:
        return f"<SkillDefinition name={self.name!r} source={self.source!r}>"

    @staticmethod
    def _parse_sections(content: str) -> dict[str, str]:
        """Split markdown by H2 headings into named sections."""
        sections: dict[str, str] = {}
        current_key: str | None = None
        buffer: list[str] = []

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                if current_key is not None:
                    sections[current_key] = "\n".join(buffer).strip()
                current_key = stripped[3:].strip().lower()
                buffer = []
            else:
                buffer.append(line)

        if current_key is not None:
            sections[current_key] = "\n".join(buffer).strip()

        return sections


# ── Local source ──────────────────────────────────────────────────────────────

class LocalSkillSource:
    """Reads skills from ``*.md`` files in a local directory."""

    def __init__(self, skills_dir: str | Path) -> None:
        self._dir = Path(skills_dir)

    def load(self, name: str) -> SkillDefinition:
        candidate = self._dir / f"{name}.md"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Skill '{name}' not found locally. "
                f"Expected: {candidate}"
            )
        content = candidate.read_text(encoding="utf-8")
        logger.debug("Loaded skill '%s' from local file %s", name, candidate)
        return SkillDefinition(name=name, content=content, source="local")

    def list_skills(self) -> list[str]:
        return [p.stem for p in sorted(self._dir.glob("*.md"))]


# ── Langfuse source (with TTL cache) ─────────────────────────────────────────

class _CacheEntry:
    __slots__ = ("skill", "fetched_at")

    def __init__(self, skill: SkillDefinition, fetched_at: float) -> None:
        self.skill = skill
        self.fetched_at = fetched_at


class CachedLangfuseSkillSource:
    """Fetches skills from Langfuse prompts with an in-process TTL cache.

    Thread-safe: a ``threading.Lock`` serialises cache reads/writes so
    concurrent agent calls never result in a thundering-herd of Langfuse
    requests for the same skill.
    """

    def __init__(self, expiry_seconds: float = 100.0) -> None:
        self._expiry = expiry_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._client = self._build_client()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def load(self, name: str) -> SkillDefinition:
        """Return skill — from cache if still fresh, otherwise re-fetch."""
        with self._lock:
            entry = self._cache.get(name)
            if entry and (time.monotonic() - entry.fetched_at) < self._expiry:
                logger.debug(
                    "Skill '%s' served from Langfuse cache (age=%.1fs ttl=%.1fs)",
                    name,
                    time.monotonic() - entry.fetched_at,
                    self._expiry,
                )
                return entry.skill

        # Fetch outside the lock so other skills are not blocked
        skill = self._fetch_from_langfuse(name)

        with self._lock:
            self._cache[name] = _CacheEntry(skill=skill, fetched_at=time.monotonic())

        return skill

    def list_skills(self) -> list[str]:
        if not self.is_available:
            return []
        try:
            prompts = self._client.client.prompts.list()  # type: ignore[union-attr]
            return [p.name for p in prompts.data]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not list Langfuse prompts: %s", exc)
            return []

    def invalidate(self, name: str | None = None) -> None:
        """Evict one entry (or the entire cache if name is None)."""
        with self._lock:
            if name is None:
                self._cache.clear()
                logger.debug("Langfuse skill cache cleared.")
            elif name in self._cache:
                del self._cache[name]
                logger.debug("Langfuse skill cache invalidated for '%s'.", name)

    def cache_info(self) -> dict[str, object]:
        """Return cache statistics for inspection / debugging."""
        with self._lock:
            now = time.monotonic()
            return {
                "entries": {
                    name: {
                        "age_seconds": round(now - e.fetched_at, 1),
                        "ttl_remaining": round(self._expiry - (now - e.fetched_at), 1),
                        "source": e.skill.source,
                    }
                    for name, e in self._cache.items()
                },
                "expiry_seconds": self._expiry,
                "available": self.is_available,
            }

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_client(self):  # type: ignore[return]
        cfg = get_settings()
        lf = cfg.langfuse
        if not (lf.public_key and lf.secret_key):
            logger.info(
                "Langfuse keys not configured — Langfuse skill source unavailable."
            )
            return None
        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=lf.public_key,
                secret_key=lf.secret_key,
                host=lf.url,
            )
            logger.info("Langfuse skill source connected at %s", lf.url)
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise Langfuse client: %s", exc)
            return None

    def _fetch_from_langfuse(self, name: str) -> SkillDefinition:
        if not self.is_available:
            raise RuntimeError(
                "Langfuse client is not available. "
                "Check LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env."
            )
        try:
            prompt = self._client.get_prompt(name)  # type: ignore[union-attr]
            content: str = prompt.compile()
            logger.info("Fetched skill '%s' from Langfuse (will cache for %.0fs).", name, self._expiry)
            return SkillDefinition(name=name, content=content, source="langfuse")
        except Exception as exc:
            raise RuntimeError(f"Langfuse fetch failed for skill '{name}': {exc}") from exc


# ── Process-level singleton (shared cache) ────────────────────────────────────
# One instance per process so the TTL cache is shared across all SkillLoaders.

_langfuse_source: CachedLangfuseSkillSource | None = None
_langfuse_source_lock = threading.Lock()


def _get_langfuse_source() -> CachedLangfuseSkillSource:
    global _langfuse_source
    with _langfuse_source_lock:
        if _langfuse_source is None:
            cfg = get_settings().skills
            _langfuse_source = CachedLangfuseSkillSource(
                expiry_seconds=cfg.langfuse_expiry_time
            )
    return _langfuse_source


# ── Skill loader facade ───────────────────────────────────────────────────────

class SkillLoader:
    """Public interface for loading skills.

    Selects the active source based on ``SKILLS_SOURCE`` in ``.env``:

    +---------+---------------------------------------------------------------+
    | Mode    | Behaviour                                                     |
    +=========+===============================================================+
    | local   | Always reads ``skills/<name>.md`` from disk.                  |
    +---------+---------------------------------------------------------------+
    | langfuse| Always fetches from Langfuse (TTL-cached).                    |
    |         | Raises if Langfuse is unavailable or the prompt doesn't exist.|
    +---------+---------------------------------------------------------------+
    | hybrid  | Tries Langfuse first (TTL-cached). On any failure —           |
    | (default| unavailable, key missing, prompt not found — silently falls   |
    |         | back to the local file.                                       |
    +---------+---------------------------------------------------------------+
    """

    def __init__(self) -> None:
        cfg = get_settings().skills
        self._mode: SkillMode = cfg.source  # type: ignore[assignment]
        self._local = LocalSkillSource(cfg.local_dir)
        self._langfuse = _get_langfuse_source()

        if self._mode not in ("local", "langfuse", "hybrid"):
            logger.warning(
                "Unknown SKILLS_SOURCE='%s', defaulting to 'hybrid'.", self._mode
            )
            self._mode = "hybrid"

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, name: str) -> SkillDefinition:
        if self._mode == "local":
            return self._local.load(name)

        if self._mode == "langfuse":
            return self._langfuse.load(name)

        # hybrid — prefer Langfuse, fall back to local
        return self._load_hybrid(name)

    def list_available(self) -> list[str]:
        if self._mode == "local":
            return self._local.list_skills()

        if self._mode == "langfuse":
            return self._langfuse.list_skills()

        # hybrid — merge both; Langfuse names take precedence (deduplicated)
        lf_names = self._langfuse.list_skills() if self._langfuse.is_available else []
        local_names = self._local.list_skills()
        seen = set(lf_names)
        extra = [n for n in local_names if n not in seen]
        return lf_names + extra

    @property
    def mode(self) -> str:
        return self._mode

    def cache_info(self) -> dict[str, object]:
        """Return Langfuse cache statistics (useful for /health or debugging)."""
        return self._langfuse.cache_info()

    def invalidate_cache(self, name: str | None = None) -> None:
        """Force a cache eviction so the next load always hits Langfuse."""
        self._langfuse.invalidate(name)

    # ── Hybrid logic ──────────────────────────────────────────────────────────

    def _load_hybrid(self, name: str) -> SkillDefinition:
        if self._langfuse.is_available:
            try:
                return self._langfuse.load(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Langfuse failed for skill '%s' (%s) — falling back to local.",
                    name,
                    exc,
                )

        # Langfuse unavailable or fetch failed — try local
        try:
            skill = self._local.load(name)
            logger.info("Skill '%s' loaded from local fallback.", name)
            return skill
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Skill '{name}' not found in Langfuse or local files. "
                f"Create 'skills/{name}.md' or add a Langfuse prompt named '{name}'."
            )
