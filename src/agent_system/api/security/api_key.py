"""API key authentication dependency.

Usage in a router::

    from agent_system.api.security import require_api_key
    from fastapi import Depends

    router = APIRouter(..., dependencies=[Depends(require_api_key)])

Behaviour
---------
* If ``API_KEY`` is **not** set (empty string) in ``.env``, authentication is
  **disabled** and every request is allowed.  This is intentional for local
  development.
* If ``API_KEY`` is set, every request to a protected route must include the
  header ``X-API-Key: <value>``.  Requests with an incorrect or missing key
  receive a 403 response.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """FastAPI dependency that enforces API key authentication.

    Inject via ``dependencies=[Depends(require_api_key)]`` on a router or
    individual route.  Returns ``None`` on success (FastAPI ignores the return
    value for dependency-only side-effects).
    """
    from agent_system.config import get_settings

    expected = get_settings().app.api_key
    if not expected:
        # Auth disabled — no API_KEY configured
        return

    if not key or key != expected:
        logger.warning("Rejected request — invalid or missing X-API-Key header.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )
