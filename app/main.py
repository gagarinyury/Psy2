from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.router import router
from app.infra.logging import setup_logging
from app.infra.metrics import PrometheusMiddleware, get_metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    setup_logging()
    yield
    # Shutdown - cleanup if needed


def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="RAG Patient API",
        description="Therapeutic conversation simulation API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(PrometheusMiddleware)
    
    # Include routers
    app.include_router(router)
    
    # Add metrics endpoint
    app.get("/metrics")(get_metrics)
    
    return app


# Export app instance
app = create_app()