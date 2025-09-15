from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.admin import admin
from app.api.router import router
from app.infra.logging import setup_logging
from app.infra.metrics import PrometheusMiddleware, get_metrics
from app.infra.rate_limit import RateLimitMiddleware
from app.infra.redis import get_redis
from app.infra.tracing import instrument_app, setup_tracing
from app.ui.router import ui as ui_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    setup_logging()
    setup_tracing()
    app.state.redis = await get_redis()
    yield
    # Shutdown - cleanup if needed


def create_app() -> FastAPI:
    """Create FastAPI application"""

    app = FastAPI(
        title="RAG Patient API",
        description="Therapeutic conversation simulation API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add middleware (order matters - RateLimit before Prometheus)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(PrometheusMiddleware)

    # Include routers
    app.include_router(router)
    app.include_router(admin)
    app.include_router(ui_router)

    # Add metrics endpoint
    app.get("/metrics")(get_metrics)

    # Instrument FastAPI for OpenTelemetry tracing
    instrument_app(app)

    return app


# Export app instance
app = create_app()
