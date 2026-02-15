from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from slowapi.errors import RateLimitExceeded

from .api.routes import router
from .api.auth_routes import router as auth_router
from .api.phase5_routes import router as phase5_router
from .api.webhook_routes import router as webhook_router
from .api.websocket import ws_router
from .core.config import get_settings
from .core.logging import setup_logging
from .security.rate_limiter import limiter, rate_limit_exceeded_handler
from .security.middleware import SecurityHeadersMiddleware, RequestLoggingMiddleware

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    setup_logging()

    # Initialize distributed tracing
    try:
        from .core.tracing import setup_tracing, instrument_fastapi
        setup_tracing()
        instrument_fastapi(app)
    except Exception as e:
        print(f"Tracing initialization failed: {e}")

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

    # Initialize payment service (Phase 5)
    try:
        from .features.payments import payment_service
        await payment_service.initialize()
    except Exception as e:
        print(f"Payment service initialization failed: {e}")

    yield

    # Cleanup
    await pool_manager.close()


app = FastAPI(
    title="SonicForge",
    description="AI-Powered 24/7 Music Radio Platform — Orchestrator API",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Rate limiter state
app.state.limiter = limiter

# Rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# CORS with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-API-Key",
        "X-Request-ID",
    ],
    expose_headers=[
        "X-Request-ID",
        "X-Response-Time",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ],
)

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# API routes
app.include_router(router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(ws_router)


@app.get("/", tags=["health"])
async def root():
    return {
        "name": "SonicForge",
        "version": settings.app_version,
        "status": "operational",
        "description": "AI-Powered 24/7 Music Radio Platform",
        "security": "Phase 4 - Security & Reliability enabled",
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Enhanced health check with connection pool status."""
    from .core.connection_pool import pool_manager
    from .core.cache import cache
    
    pool_health = await pool_manager.health_check()
    
    return {
        "status": "healthy",
        "connections": pool_health,
        "cache_stats": cache.get_stats(),
        "pool_stats": pool_manager.get_stats(),
    }


@app.get("/health/detailed", tags=["health"])
async def detailed_health_check():
    """Comprehensive health check with all components."""
    from .security.health_checks import run_all_health_checks
    
    health = await run_all_health_checks()
    return health.to_dict()


@app.get("/ready", tags=["health"])
async def readiness_probe():
    """Kubernetes readiness probe."""
    from .security.health_checks import readiness_check
    
    result = await readiness_check()
    status_code = 200 if result["ready"] else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/live", tags=["health"])
async def liveness_probe():
    """Kubernetes liveness probe."""
    from .security.health_checks import liveness_check
    
    return await liveness_check()
