import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

import structlog
from redis import Redis

from ..core.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class AgentStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    STOPPED = "stopped"


class BaseAgent(ABC):
    """Base class for all SonicForge agents."""

    def __init__(self, name: str):
        self.name = name
        self.agent_id = str(uuid.uuid4())
        self.status = AgentStatus.IDLE
        self.redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self.logger = logger.bind(agent=name, agent_id=self.agent_id)

    @abstractmethod
    async def execute(self, task: dict) -> dict:
        """Execute the agent's primary task."""

    async def publish_status(self):
        """Publish current agent status to Redis for monitoring."""
        status_data = {
            "agent": self.name,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.redis.hset(f"agent:status:{self.name}", mapping=status_data)
        self.redis.publish("agent:status", f"{self.name}:{self.status.value}")

    async def send_message(self, target_agent: str, message: dict):
        """Send a message to another agent via Redis pub/sub."""
        import json

        self.redis.publish(
            f"agent:messages:{target_agent}",
            json.dumps({"from": self.name, "payload": message}),
        )

    async def log_activity(self, action: str, details: dict | None = None):
        """Log agent activity for the dashboard."""
        import json

        activity = {
            "agent": self.name,
            "action": action,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.redis.lpush("agent:activity_log", json.dumps(activity))
        self.redis.ltrim("agent:activity_log", 0, 999)  # keep last 1000 entries

    async def run(self, task: dict) -> dict:
        """Run the agent with status tracking and error handling."""
        self.status = AgentStatus.WORKING
        await self.publish_status()
        await self.log_activity("task_started", {"task": task.get("type", "unknown")})

        try:
            result = await self.execute(task)
            self.status = AgentStatus.IDLE
            await self.publish_status()
            await self.log_activity("task_completed", {"result_summary": str(result)[:200]})
            return result
        except Exception as e:
            self.status = AgentStatus.ERROR
            await self.publish_status()
            await self.log_activity("task_failed", {"error": str(e)})
            self.logger.error("agent_task_failed", error=str(e), task=task)
            raise
