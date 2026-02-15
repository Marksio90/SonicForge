from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from .api.routes import router
from .api.websocket import ws_router
from .core.config import get_settings
from .core.logging import setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    setup_logging()

    # Initialize connection pools
    from .core.connection_pool import pool_manager
    await pool_manager.initialize(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
    )

    # Initialize cache
    from .core.cache import cache
    await cache.initialize(settings.redis_url)

    # Initialize S3 buckets
    try:
        from .core.storage import ensure_buckets
        ensure_buckets()
    except Exception:
        pass  # S3 may not be available in dev

    yield

    # Cleanup
    await pool_manager.close()


app = FastAPI(
    title="SonicForge",
    description="AI-Powered 24/7 Music Radio Platform — Orchestrator API",
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# API routes
app.include_router(router, prefix="/api/v1")
app.include_router(ws_router)


@app.get("/", tags=["health"])
async def root():
    return {
        "name": "SonicForge",
        "version": settings.app_version,
        "status": "operational",
        "description": "AI-Powered 24/7 Music Radio Platform",
    }


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "healthy"}
