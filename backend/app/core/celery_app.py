from celery import Celery

from .config import get_settings

settings = get_settings()

celery_app = Celery(
    "sonicforge",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "sonicforge.tasks.compose.*": {"queue": "compose"},
        "sonicforge.tasks.produce.*": {"queue": "produce"},
        "sonicforge.tasks.critic.*": {"queue": "critic"},
        "sonicforge.tasks.schedule.*": {"queue": "schedule"},
        "sonicforge.tasks.stream.*": {"queue": "stream"},
        "sonicforge.tasks.analytics.*": {"queue": "analytics"},
        "sonicforge.tasks.visual.*": {"queue": "visual"},
    },
    beat_schedule={
        "check-queue-buffer": {
            "task": "sonicforge.tasks.schedule.check_buffer",
            "schedule": 300.0,  # every 5 minutes
        },
        "stream-health-check": {
            "task": "sonicforge.tasks.stream.health_check",
            "schedule": 30.0,  # every 30 seconds
        },
        "analytics-snapshot": {
            "task": "sonicforge.tasks.analytics.snapshot",
            "schedule": 600.0,  # every 10 minutes
        },
        "trend-analysis": {
            "task": "sonicforge.tasks.compose.analyze_trends",
            "schedule": 3600.0,  # every hour
        },
    },
)
