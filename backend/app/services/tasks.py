"""Celery tasks for async pipeline execution."""
import asyncio

from ..core.celery_app import celery_app
from .orchestrator import Orchestrator


def _run_async(coro):
    """Run an async coroutine from a sync Celery task using a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(name="sonicforge.tasks.compose.run_pipeline", queue="compose")
def run_pipeline_task(genre: str | None = None, energy: int | None = None) -> dict:
    """Run the full production pipeline as a Celery task."""
    orchestrator = Orchestrator()
    return _run_async(orchestrator.run_full_pipeline(genre=genre, energy=energy))


@celery_app.task(name="sonicforge.tasks.produce.batch_generate", queue="produce")
def batch_generate_task(count: int = 5, genre: str | None = None) -> dict:
    """Batch-generate multiple tracks as a Celery task."""
    orchestrator = Orchestrator()
    return _run_async(orchestrator.run_batch_pipeline(count=count, genre=genre))


@celery_app.task(name="sonicforge.tasks.schedule.check_buffer", queue="schedule")
def check_buffer_task() -> dict:
    """Periodic buffer check — trigger generation if queue is low."""
    from ..agents.scheduler import SchedulerAgent
    scheduler = SchedulerAgent()
    return _run_async(scheduler.run({"type": "check_buffer"}))


@celery_app.task(name="sonicforge.tasks.stream.health_check", queue="stream")
def stream_health_check_task() -> dict:
    """Periodic stream health check."""
    from ..agents.stream_master import StreamMasterAgent
    stream = StreamMasterAgent()
    return _run_async(stream.run({"type": "health_check"}))


@celery_app.task(name="sonicforge.tasks.analytics.snapshot", queue="analytics")
def analytics_snapshot_task() -> dict:
    """Periodic analytics snapshot."""
    from ..agents.analytics import AnalyticsAgent
    analytics = AnalyticsAgent()
    return _run_async(analytics.run({"type": "snapshot"}))


@celery_app.task(name="sonicforge.tasks.compose.trend_analysis", queue="compose")
def trend_analysis_task(genre: str | None = None) -> dict:
    """Periodic trend analysis using OpenAI."""
    from ..agents.composer import ComposerAgent
    composer = ComposerAgent()
    return _run_async(composer.run({"type": "analyze_trends", "genre": genre}))
