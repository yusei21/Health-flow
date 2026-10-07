from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies.container import Container, build_container
from app.api.errors import register_error_handlers
from app.api.middleware import RequestIdMiddleware
from app.api.routes import health, patients, routing
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # The ML model is loaded (never trained) at startup.
        app.state.container = container or build_container(resolved)
        yield

    app = FastAPI(
        title="Health-flow",
        version="0.1.0",
        description="PROTÓTIPO ACADÊMICO — NÃO UTILIZAR PARA DECISÕES CLÍNICAS REAIS.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(routing.router, prefix="/api/v1")
    app.include_router(patients.router, prefix="/api/v1")
    return app


app = create_app()
