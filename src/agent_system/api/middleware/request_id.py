"""Request ID and timing middleware.

For every HTTP request this middleware:

1. Reads or generates a ``X-Request-ID`` UUID.
2. Measures wall-clock processing time.
3. Logs one structured line at ``INFO`` level::

       REQUEST GET /agents/researcher/run  200  142.3ms  req-id=<uuid>

4. Adds ``X-Request-ID`` and ``X-Response-Time`` headers to every response.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request ID and log each request with its duration."""

    async def dispatch(self, request: Request, call_next) -> Response:  # noqa: ANN001
        # Reuse caller-supplied ID or generate a fresh one
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        start = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"

        logger.info(
            "REQUEST %s %s  %d  %.1fms  req-id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            req_id,
        )

        return response
