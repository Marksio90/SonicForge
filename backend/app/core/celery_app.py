from celery import Celery, Task
from celery.schedules import crontab
from kombu import Queue, Exchange
import structlog

from .config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)

# Define exchanges and queues with priorities
default_exchange = Exchange("default", type="direct")
priority_exchange = Exchange("priority", type="direct")

celery_app = Celery("sonicforge")

celery_app.conf.update(
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    
    # Task routing with priorities
    task_queues=(
        Queue("critical", exchange=priority_exchange, routing_key="critical", priority=10),
        Queue("high", exchange=priority_exchange, routing_key="high", priority=7),
        Queue("default", exchange=default_exchange, routing_key="default", priority=5),
        Queue("low", exchange=default_exchange, routing_key="low", priority=2),
        Queue("batch", exchange=default_exchange, routing_key="batch", priority=1),
    ),
    
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Performance tuning
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    
    # Compression
    task_compression="gzip",
    result_compression="gzip",
    
    # Rate limiting (prevent API overload)
    task_annotations={
        "app.services.tasks.generate_track": {
            "rate_limit": "10/m",  # 10 generations per minute
            "priority": 7,
            "queue": "high",
        },
        "app.services.tasks.analyze_trends": {
            "rate_limit": "60/h",  # 60 per hour (expensive LLM calls)
            "priority": 5,
        },
        "app.services.tasks.evaluate_track": {
            "rate_limit": "30/m",
            "priority": 7,
            "queue": "high",
        },
        "app.services.tasks.cleanup_old_tracks": {
            "rate_limit": "1/h",
            "priority": 1,
            "queue": "low",
        },
    },
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "refill-queue-buffer": {
            "task": "app.services.tasks.refill_queue",
            "schedule": crontab(minute="*/5"),  # Every 5 minutes
        },
        "cleanup-old-tracks": {
            "task": "app.services.tasks.cleanup_old_tracks",
            "schedule": crontab(hour=3, minute=0),  # Daily at 3 AM
        },
        "analytics-report": {
            "task": "app.services.tasks.generate_analytics_report",
            "schedule": crontab(hour="*/6", minute=0),  # Every 6 hours
        },
        "trend-analysis": {
            "task": "app.services.tasks.analyze_trends_task",
            "schedule": crontab(hour="*/1", minute=0),  # Every hour
        },
        "stream-health-check": {
            "task": "app.services.tasks.stream_health_check",
            "schedule": 30.0,  # Every 30 seconds
        },
    },
)


class RateLimitedTask(Task):
    """Base task with rate limiting and retries."""
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 5}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("task_failed", task=self.name, task_id=task_id, error=str(exc))
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info("task_success", task=self.name, task_id=task_id)
