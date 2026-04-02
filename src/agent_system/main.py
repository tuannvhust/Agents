"""Application entry point."""

from __future__ import annotations

import uvicorn

from agent_system.api.app import create_app
from agent_system.config import get_settings

app = create_app()


def main() -> None:
    cfg = get_settings()
    uvicorn.run(
        "agent_system.main:app",
        host=cfg.app.host,
        port=cfg.app.port,
        reload=cfg.app.debug,
        log_level=cfg.app.log_level.lower(),
        workers=1 if cfg.app.debug else 2,
    )


if __name__ == "__main__":
    main()
