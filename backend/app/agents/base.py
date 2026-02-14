import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum

import structlog
from redis import ConnectionPool, Redis

from ..core.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)

# Shared Redis connection pool — avoids creating new connections per agent
_redis_pool = ConnectionPool.from_url(
    settings.redis_url,
    max_connections=settings.redis_max_connections,
    decode_responses=True,
)


def get_shared_redis() -> Redis:
    """Return a Redis client backed by the shared connection pool."""
    return Redis(connection_pool=_redis_pool)


class AgentStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    STOPPED = "stopped"


class BaseAgent(ABC):
    """Base class for all SonicForge agents with shared connection pooling and metrics."""

    def __init__(self, name: str):
        self.name = name
        self.agent_id = str(uuid.uuid4())
        self.status = AgentStatus.IDLE
        self.redis = get_shared_redis()
        self.logger = logger.bind(agent=name, agent_id=self.agent_id)
        self._task_count = 0
        self._error_count = 0
        self._total_duration_ms = 0.0

    @abstractmethod
    async def execute(self, task: dict) -> dict:
        """Execute the agent's primary task."""

    async def publish_status(self):
        """Publish current agent status to Redis for monitoring."""
        status_data = {
            "agent": self.name,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "tasks_completed": str(self._task_count),
            "errors": str(self._error_count),
            "avg_duration_ms": str(
                round(self._total_duration_ms / max(self._task_count, 1), 1)
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.redis.hset(f"agent:status:{self.name}", mapping=status_data)
        self.redis.publish("agent:status", f"{self.name}:{self.status.value}")

    async def send_message(self, target_agent: str, message: dict):
        """Send a message to another agent via Redis pub/sub."""
        self.redis.publish(
            f"agent:messages:{target_agent}",
            json.dumps({"from": self.name, "payload": message}),
        )

    async def log_activity(self, action: str, details: dict | None = None):
        """Log agent activity for the dashboard."""
        activity = {
            "agent": self.name,
            "action": action,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.redis.lpush("agent:activity_log", json.dumps(activity))
        self.redis.ltrim("agent:activity_log", 0, 999)

    async def run(self, task: dict) -> dict:
        """Run the agent with status tracking, metrics, and error handling."""
        self.status = AgentStatus.WORKING
        await self.publish_status()
        task_type = task.get("type", "unknown")
        await self.log_activity("task_started", {"task": task_type})
        start_time = time.monotonic()

        try:
            result = await self.execute(task)
            duration_ms = (time.monotonic() - start_time) * 1000
            self._task_count += 1
            self._total_duration_ms += duration_ms
            self.status = AgentStatus.IDLE
            await self.publish_status()
            await self.log_activity(
                "task_completed",
                {"task": task_type, "duration_ms": round(duration_ms, 1)},
            )
            return result
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self._error_count += 1
            self.status = AgentStatus.ERROR
            await self.publish_status()
            await self.log_activity(
                "task_failed",
                {"error": str(e), "task": task_type, "duration_ms": round(duration_ms, 1)},
            )
            self.logger.error("agent_task_failed", error=str(e), task=task)
            raise
