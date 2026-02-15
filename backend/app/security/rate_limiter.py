"""
Rate Limiting & DDoS Protection (Phase 4 - Item 7.2)

Implements API rate limiting using SlowAPI with:
- Per-IP rate limiting
- Per-user rate limiting (authenticated)
- Per-endpoint rate limiting
- Custom rate limit exceeded handler
- Fallback to memory storage when Redis unavailable
"""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ..core.config import get_settings

settings = get_settings()


def get_user_identifier(request: Request) -> str:
    """Get rate limit key based on user or IP."""
    # Try to get user ID from JWT token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            from .auth import verify_token
            token = auth_header.split(" ")[1]
            payload = verify_token(token)
            return f"user:{payload.sub}"
        except Exception:
            pass
    
    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


# Try to use Redis, fallback to memory storage
def get_storage_uri() -> str:
    """Get storage URI for rate limiter."""
    try:
        import redis
        r = redis.Redis.from_url(settings.redis_url, socket_timeout=1)
        r.ping()
        return settings.redis_url
    except Exception:
        # Fallback to memory storage
        return "memory://"


# Initialize rate limiter with appropriate backend
limiter = Limiter(
    key_func=get_user_identifier,
    default_limits=["200/minute", "1000/hour"],
    storage_uri=get_storage_uri(),
    strategy="fixed-window",
    headers_enabled=True,
)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Custom handler for rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": f"Rate limit exceeded: {exc.detail}",
            "retry_after": getattr(exc, "retry_after", 60),
        },
        headers={
            "Retry-After": str(getattr(exc, "retry_after", 60)),
            "X-RateLimit-Limit": request.headers.get("X-RateLimit-Limit", ""),
            "X-RateLimit-Remaining": "0",
        },
    )


# Predefined rate limits for different endpoint types
class RateLimits:
    """Predefined rate limits for different operations."""
    
    # Public endpoints
    PUBLIC_READ = "60/minute"
    PUBLIC_WRITE = "20/minute"
    
    # Authenticated endpoints
    AUTH_READ = "120/minute"
    AUTH_WRITE = "60/minute"
    
    # Resource-intensive operations
    GENERATE = "10/minute"
    BATCH = "5/minute"
    
    # Admin operations
    ADMIN = "300/minute"
    
    # Authentication
    LOGIN = "10/minute"
    REGISTER = "5/minute"
    TOKEN_REFRESH = "30/minute"
    
    # Critical/sensitive operations
    CRITICAL = "5/minute"


# Decorator shortcuts for common rate limits
def limit_public_read(func):
    """Apply public read rate limit."""
    return limiter.limit(RateLimits.PUBLIC_READ)(func)


def limit_public_write(func):
    """Apply public write rate limit."""
    return limiter.limit(RateLimits.PUBLIC_WRITE)(func)


def limit_auth_read(func):
    """Apply authenticated read rate limit."""
    return limiter.limit(RateLimits.AUTH_READ)(func)


def limit_auth_write(func):
    """Apply authenticated write rate limit."""
    return limiter.limit(RateLimits.AUTH_WRITE)(func)


def limit_generate(func):
    """Apply generation rate limit."""
    return limiter.limit(RateLimits.GENERATE)(func)


def limit_login(func):
    """Apply login rate limit."""
    return limiter.limit(RateLimits.LOGIN)(func)
