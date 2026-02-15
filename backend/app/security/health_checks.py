"""
Health Checks & Reliability (Phase 4 - Item 7.4 extension)

Implements comprehensive health checks for:
- Database connectivity
- Redis connectivity
- External services (S3, MusicGen, etc.)
- System resources (CPU, memory, disk)
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import psutil

from ..core.config import get_settings

settings = get_settings()


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Optional[dict] = None
    checked_at: datetime = None
    
    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    components: list[ComponentHealth]
    system_info: dict
    checked_at: datetime = None
    
    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "components": [c.to_dict() for c in self.components],
            "system_info": self.system_info,
            "checked_at": self.checked_at.isoformat(),
        }


async def check_database() -> ComponentHealth:
    """Check database connectivity."""
    start = time.time()
    try:
        from ..core.connection_pool import pool_manager
        
        async with pool_manager.get_db_connection() as conn:
            await conn.fetchval("SELECT 1")
        
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY if latency < 100 else HealthStatus.DEGRADED,
            latency_ms=round(latency, 2),
            message="Connected successfully",
        )
    except Exception as e:
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            latency_ms=(time.time() - start) * 1000,
            message=f"Connection failed: {str(e)}",
        )


async def check_redis() -> ComponentHealth:
    """Check Redis connectivity."""
    start = time.time()
    try:
        from ..core.connection_pool import pool_manager
        
        redis = await pool_manager.get_redis_client()
        await redis.ping()
        info = await redis.info("memory")
        
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY if latency < 50 else HealthStatus.DEGRADED,
            latency_ms=round(latency, 2),
            message="Connected successfully",
            details={
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
            },
        )
    except Exception as e:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            latency_ms=(time.time() - start) * 1000,
            message=f"Connection failed: {str(e)}",
        )


async def check_s3() -> ComponentHealth:
    """Check S3/MinIO connectivity."""
    start = time.time()
    try:
        from ..core.storage import s3_client
        
        s3_client.list_buckets()
        
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="s3",
            status=HealthStatus.HEALTHY if latency < 200 else HealthStatus.DEGRADED,
            latency_ms=round(latency, 2),
            message="Connected successfully",
        )
    except Exception as e:
        return ComponentHealth(
            name="s3",
            status=HealthStatus.UNHEALTHY,
            latency_ms=(time.time() - start) * 1000,
            message=f"Connection failed: {str(e)}",
        )


async def check_celery() -> ComponentHealth:
    """Check Celery worker status."""
    start = time.time()
    try:
        from ..core.celery_app import celery_app
        
        # Check if any workers are active
        inspect = celery_app.control.inspect()
        active = inspect.active()
        
        if active:
            worker_count = len(active)
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name="celery",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 2),
                message=f"{worker_count} worker(s) active",
                details={"workers": list(active.keys())},
            )
        else:
            return ComponentHealth(
                name="celery",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - start) * 1000,
                message="No active workers found",
            )
    except Exception as e:
        return ComponentHealth(
            name="celery",
            status=HealthStatus.UNKNOWN,
            latency_ms=(time.time() - start) * 1000,
            message=f"Check failed: {str(e)}",
        )


def check_system_resources() -> dict:
    """Get system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    
    return {
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count(),
            "status": "healthy" if cpu_percent < 80 else "degraded" if cpu_percent < 95 else "critical",
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent,
            "status": "healthy" if memory.percent < 80 else "degraded" if memory.percent < 95 else "critical",
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "percent": disk.percent,
            "status": "healthy" if disk.percent < 80 else "degraded" if disk.percent < 95 else "critical",
        },
    }


async def run_all_health_checks() -> SystemHealth:
    """Run all health checks in parallel."""
    # Run component checks in parallel
    checks = await asyncio.gather(
        check_database(),
        check_redis(),
        check_s3(),
        check_celery(),
        return_exceptions=True,
    )
    
    # Handle any exceptions
    components = []
    for check in checks:
        if isinstance(check, Exception):
            components.append(ComponentHealth(
                name="unknown",
                status=HealthStatus.UNKNOWN,
                message=str(check),
            ))
        else:
            components.append(check)
    
    # Get system resources
    system_info = check_system_resources()
    
    # Determine overall status
    statuses = [c.status for c in components]
    if all(s == HealthStatus.HEALTHY for s in statuses):
        overall_status = HealthStatus.HEALTHY
    elif any(s == HealthStatus.UNHEALTHY for s in statuses):
        overall_status = HealthStatus.UNHEALTHY
    else:
        overall_status = HealthStatus.DEGRADED
    
    return SystemHealth(
        status=overall_status,
        components=components,
        system_info=system_info,
    )


# Readiness probe (is the service ready to accept traffic?)
async def readiness_check() -> dict:
    """Check if service is ready to accept traffic."""
    health = await run_all_health_checks()
    
    is_ready = health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    return {
        "ready": is_ready,
        "status": health.status.value,
        "message": "Service is ready" if is_ready else "Service is not ready",
    }


# Liveness probe (is the service alive?)
async def liveness_check() -> dict:
    """Check if service is alive."""
    return {
        "alive": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": time.time() - _start_time,
    }


# Track service start time
_start_time = time.time()
