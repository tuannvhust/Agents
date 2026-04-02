from .agents import router as agents_router
from .debug import router as debug_router
from .files import router as files_router
from .health import router as health_router

__all__ = ["agents_router", "debug_router", "files_router", "health_router"]
