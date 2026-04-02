"""Debug / diagnostic endpoints — not for production use."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from agent_system.api.security import require_api_key
from agent_system.config import get_settings

router = APIRouter(
    prefix="/debug",
    tags=["Debug"],
    dependencies=[Depends(require_api_key)],
)
logger = logging.getLogger(__name__)


@router.get("/skills", summary="Show skill-loader mode and available skills")
async def debug_skills():
    """Returns which mode is active, skills from each source, and cache state.

    Use this to confirm whether skills are being served from Langfuse or local.
    """
    cfg = get_settings().skills
    mode = cfg.source

    # ── local skills ──────────────────────────────────────────────────────────
    from pathlib import Path
    local_dir = Path(cfg.local_dir)
    local_skills = [p.stem for p in sorted(local_dir.glob("*.md"))] if local_dir.exists() else []

    # ── Langfuse skills + cache ───────────────────────────────────────────────
    from agent_system.core.skill_loader import _get_langfuse_source
    lf_source = _get_langfuse_source()

    langfuse_available = lf_source.is_available
    langfuse_skills: list[str] = []
    langfuse_error: str | None = None

    if langfuse_available:
        try:
            langfuse_skills = lf_source.list_skills()
        except Exception as exc:  # noqa: BLE001
            langfuse_error = str(exc)

    cache_info = lf_source.cache_info()

    return {
        "mode": mode,
        "langfuse_expiry_seconds": cfg.langfuse_expiry_time,
        "local": {
            "dir": str(local_dir.resolve()),
            "skills": local_skills,
        },
        "langfuse": {
            "available": langfuse_available,
            "skills": langfuse_skills,
            "error": langfuse_error,
            "cache": cache_info,
        },
        "effective_priority": (
            "langfuse → local (fallback)" if mode == "hybrid"
            else mode
        ),
    }


@router.get("/tracing", summary="Show Langfuse tracing status")
async def debug_tracing():
    """Returns whether the Langfuse CallbackHandler is active.

    If 'enabled' is false, no LLM traces will appear in the Langfuse UI.
    """
    from agent_system.tracing import get_langfuse_handler, is_tracing_enabled

    handler = get_langfuse_handler()
    cfg = get_settings().langfuse

    public_key = cfg.public_key or ""
    secret_key = cfg.secret_key or ""

    key_issues: list[str] = []
    if not public_key:
        key_issues.append("LANGFUSE_PUBLIC_KEY is not set")
    elif not public_key.startswith("pk-lf-"):
        key_issues.append(
            f"LANGFUSE_PUBLIC_KEY looks wrong — expected 'pk-lf-…', got '{public_key[:12]}…'"
        )
    if not secret_key:
        key_issues.append("LANGFUSE_SECRET_KEY is not set")
    elif not secret_key.startswith("sk-lf-"):
        key_issues.append(
            f"LANGFUSE_SECRET_KEY looks wrong — expected 'sk-lf-…', got '{secret_key[:12]}…'"
        )
    if public_key and secret_key and public_key == secret_key:
        key_issues.append(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are identical — "
            "copy the correct keys from the Langfuse UI (Settings → API Keys)"
        )

    return {
        "tracing_enabled": is_tracing_enabled(),
        "handler_type": type(handler).__name__ if handler else None,
        "langfuse_host": cfg.url,
        "key_issues": key_issues,
        "hint": (
            "Tracing is working correctly." if is_tracing_enabled() and not key_issues
            else "Fix the key_issues above, then rebuild: docker compose build --no-cache app && docker compose up -d app"
        ),
    }


@router.post("/skills/invalidate-cache", summary="Force a Langfuse cache refresh")
async def invalidate_skill_cache(skill_name: str | None = None):
    """Evict one skill (or all) from the Langfuse cache.

    The next load() will re-fetch from Langfuse immediately instead of waiting
    for the TTL to expire.
    """
    from agent_system.core.skill_loader import _get_langfuse_source
    _get_langfuse_source().invalidate(skill_name)
    return {
        "invalidated": skill_name or "all",
        "message": (
            f"Cache cleared for skill '{skill_name}'."
            if skill_name
            else "Entire Langfuse skill cache cleared."
        ),
    }
