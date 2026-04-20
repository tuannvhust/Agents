"""Safety rule loader — reads guardrails/*.md definitions with three operating modes.

Mirrors the design of skill_loader.py exactly so guardrail rules enjoy the same
local / langfuse / hybrid loading and TTL-caching behaviour as agent skills.

Modes
-----
local    — reads from the local ``guardrails/`` directory only.
langfuse — fetches from Langfuse prompts only; results are cached and
           refreshed every ``GUARDRAILS_EXPIRY_TIME`` seconds.
hybrid   — Langfuse is tried first; on any failure silently falls back to
           the local file.  This is the **default** mode.

Rule file format (guardrails/<name>.md)
---------------------------------------
The file is plain markdown split by H2 headings:

    ## classifier_prompt
    You are a safety classifier ...

    ## action
    block

    ## description
    (optional human-readable notes)

Sections are accessed via properties on SafetyRuleDefinition.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Literal

from agent_system.config import get_settings

logger = logging.getLogger(__name__)

SafetyRuleMode = Literal["local", "langfuse", "hybrid"]


# ── Rule definition ────────────────────────────────────────────────────────────

class SafetyRuleDefinition:
    """Parsed representation of a guardrails/*.md file."""

    def __init__(self, name: str, content: str, source: str = "unknown") -> None:
        self.name = name
        self.raw = content
        self.source = source
        self._sections = self._parse_sections(content)

    @property
    def classifier_prompt(self) -> str:
        """System prompt to send to the classifier LLM.

        Falls back to the whole file content if no ``## classifier_prompt``
        section is present.
        """
        return self._sections.get("classifier_prompt", self.raw).strip()

    @property
    def action(self) -> str:
        """What to do when the classifier flags the input.

        Values: ``block`` (default) | ``warn`` | ``log``
        """
        return self._sections.get("action", "block").strip().lower()

    @property
    def description(self) -> str:
        return self._sections.get("description", "").strip()

    def __repr__(self) -> str:
        return f"<SafetyRuleDefinition name={self.name!r} source={self.source!r} action={self.action!r}>"

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


# ── Local source ───────────────────────────────────────────────────────────────

class LocalSafetyRuleSource:
    """Reads safety rules from ``*.md`` files in a local directory."""

    def __init__(self, rules_dir: str | Path) -> None:
        self._dir = Path(rules_dir)

    def load(self, name: str) -> SafetyRuleDefinition:
        candidate = self._dir / f"{name}.md"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Safety rule '{name}' not found locally. "
                f"Expected: {candidate}"
            )
        content = candidate.read_text(encoding="utf-8")
        logger.debug("Loaded safety rule '%s' from local file %s", name, candidate)
        return SafetyRuleDefinition(name=name, content=content, source="local")

    def list_rules(self) -> list[str]:
        return [p.stem for p in sorted(self._dir.glob("*.md"))]


# ── Langfuse source (with TTL cache) ──────────────────────────────────────────

class _CacheEntry:
    __slots__ = ("rule", "fetched_at")

    def __init__(self, rule: SafetyRuleDefinition, fetched_at: float) -> None:
        self.rule = rule
        self.fetched_at = fetched_at


class CachedLangfuseSafetyRuleSource:
    """Fetches safety rules from Langfuse prompts with an in-process TTL cache.

    Thread-safe: a ``threading.Lock`` serialises cache reads/writes.
    """

    def __init__(self, expiry_seconds: float = 100.0) -> None:
        self._expiry = expiry_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._client = self._build_client()

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def load(self, name: str) -> SafetyRuleDefinition:
        """Return rule — from cache if still fresh, otherwise re-fetch."""
        with self._lock:
            entry = self._cache.get(name)
            if entry and (time.monotonic() - entry.fetched_at) < self._expiry:
                logger.debug(
                    "Safety rule '%s' served from Langfuse cache (age=%.1fs ttl=%.1fs)",
                    name,
                    time.monotonic() - entry.fetched_at,
                    self._expiry,
                )
                return entry.rule

        rule = self._fetch_from_langfuse(name)

        with self._lock:
            self._cache[name] = _CacheEntry(rule=rule, fetched_at=time.monotonic())

        return rule

    def list_rules(self) -> list[str]:
        if not self.is_available:
            return []
        try:
            prompts = self._client.client.prompts.list()  # type: ignore[union-attr]
            return [p.name for p in prompts.data]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not list Langfuse prompts (safety rules): %s", exc)
            return []

    def invalidate(self, name: str | None = None) -> None:
        """Evict one entry (or the entire cache if name is None)."""
        with self._lock:
            if name is None:
                self._cache.clear()
                logger.debug("Langfuse safety rule cache cleared.")
            elif name in self._cache:
                del self._cache[name]
                logger.debug("Langfuse safety rule cache invalidated for '%s'.", name)

    def cache_info(self) -> dict[str, object]:
        with self._lock:
            now = time.monotonic()
            return {
                "entries": {
                    name: {
                        "age_seconds": round(now - e.fetched_at, 1),
                        "ttl_remaining": round(self._expiry - (now - e.fetched_at), 1),
                        "source": e.rule.source,
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
                "Langfuse keys not configured — Langfuse safety rule source unavailable."
            )
            return None
        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=lf.public_key,
                secret_key=lf.secret_key,
                host=lf.url,
            )
            logger.info("Langfuse safety rule source connected at %s", lf.url)
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise Langfuse client for safety rules: %s", exc)
            return None

    def _fetch_from_langfuse(self, name: str) -> SafetyRuleDefinition:
        if not self.is_available:
            raise RuntimeError(
                "Langfuse client is not available. "
                "Check LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env."
            )
        try:
            prompt = self._client.get_prompt(name)  # type: ignore[union-attr]
            content: str = prompt.compile()
            logger.info(
                "Fetched safety rule '%s' from Langfuse (will cache for %.0fs).",
                name,
                self._expiry,
            )
            return SafetyRuleDefinition(name=name, content=content, source="langfuse")
        except Exception as exc:
            raise RuntimeError(
                f"Langfuse fetch failed for safety rule '{name}': {exc}"
            ) from exc


# ── Process-level singleton (shared cache) ─────────────────────────────────────

_langfuse_source: CachedLangfuseSafetyRuleSource | None = None
_langfuse_source_lock = threading.Lock()


def _get_langfuse_source() -> CachedLangfuseSafetyRuleSource:
    global _langfuse_source
    with _langfuse_source_lock:
        if _langfuse_source is None:
            cfg = get_settings().guardrails
            _langfuse_source = CachedLangfuseSafetyRuleSource(
                expiry_seconds=cfg.langfuse_expiry_time
            )
    return _langfuse_source


# ── Safety rule loader facade ──────────────────────────────────────────────────

class SafetyRuleLoader:
    """Public interface for loading safety rules.

    Selects the active source based on ``GUARDRAILS_SOURCE`` in ``.env``:

    +---------+---------------------------------------------------------------+
    | Mode    | Behaviour                                                     |
    +=========+===============================================================+
    | local   | Always reads ``guardrails/<name>.md`` from disk.              |
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
        cfg = get_settings().guardrails
        self._mode: SafetyRuleMode = cfg.source  # type: ignore[assignment]
        self._local = LocalSafetyRuleSource(cfg.local_dir)
        self._langfuse = _get_langfuse_source()

        if self._mode not in ("local", "langfuse", "hybrid"):
            logger.warning(
                "Unknown GUARDRAILS_SOURCE='%s', defaulting to 'hybrid'.", self._mode
            )
            self._mode = "hybrid"

    def load(self, name: str) -> SafetyRuleDefinition:
        if self._mode == "local":
            return self._local.load(name)
        if self._mode == "langfuse":
            return self._langfuse.load(name)
        return self._load_hybrid(name)

    def list_available(self) -> list[str]:
        if self._mode == "local":
            return self._local.list_rules()
        if self._mode == "langfuse":
            return self._langfuse.list_rules()
        lf_names = self._langfuse.list_rules() if self._langfuse.is_available else []
        local_names = self._local.list_rules()
        seen = set(lf_names)
        extra = [n for n in local_names if n not in seen]
        return lf_names + extra

    @property
    def mode(self) -> str:
        return self._mode

    def cache_info(self) -> dict[str, object]:
        return self._langfuse.cache_info()

    def invalidate_cache(self, name: str | None = None) -> None:
        self._langfuse.invalidate(name)

    # ── Hybrid logic ──────────────────────────────────────────────────────────

    def _load_hybrid(self, name: str) -> SafetyRuleDefinition:
        if self._langfuse.is_available:
            try:
                return self._langfuse.load(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Langfuse failed for safety rule '%s' (%s) — falling back to local.",
                    name,
                    exc,
                )
        try:
            rule = self._local.load(name)
            logger.info("Safety rule '%s' loaded from local fallback.", name)
            return rule
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Safety rule '{name}' not found in Langfuse or local files. "
                f"Create 'guardrails/{name}.md' or add a Langfuse prompt named '{name}'."
            )
