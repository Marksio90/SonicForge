"""Celery tasks for async pipeline execution."""
import asyncio

from ..core.celery_app import celery_app, RateLimitedTask
from .orchestrator import Orchestrator


def _run_async(coro):
    """Run an async coroutine from a sync Celery task using a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    name="sonicforge.tasks.compose.run_pipeline",
    base=RateLimitedTask,
    queue="high",
    priority=7,
)
def run_pipeline_task(genre: str | None = None, energy: int | None = None) -> dict:
    """Run the full production pipeline as a Celery task."""
    orchestrator = Orchestrator()
    return _run_async(orchestrator.run_full_pipeline(genre=genre, energy=energy))


@celery_app.task(
    name="sonicforge.tasks.produce.batch_generate",
    base=RateLimitedTask,
    queue="batch",
    priority=1,
)
def run_batch_pipeline_task(genre: str | None = None, count: int = 5) -> dict:
    """Run batch pipeline as a Celery task."""
    from .batch_processor import PipelineBatchOrchestrator
    
    batch_orchestrator = PipelineBatchOrchestrator()
    return _run_async(batch_orchestrator.run_batch_pipeline(genre=genre, count=count))


@celery_app.task(
    name="app.services.tasks.analyze_trends_task",
    base=RateLimitedTask,
    queue="default",
    priority=5,
)
def analyze_trends_task(genre: str | None = None) -> dict:
    """Periodic trend analysis task."""
    from ..agents.composer import ComposerAgent
    
    composer = ComposerAgent()
    return _run_async(composer.analyze_trends(genre=genre))


@celery_app.task(
    name="app.services.tasks.cleanup_old_tracks",
    base=RateLimitedTask,
    queue="low",
    priority=1,
)
def cleanup_old_tracks() -> dict:
    """Cleanup old unapproved tracks."""
    # TODO: Implement cleanup logic
    return {"status": "success", "deleted": 0}


@celery_app.task(
    name="app.services.tasks.refill_queue",
    base=RateLimitedTask,
    queue="high",
    priority=7,
)
def refill_queue() -> dict:
    """Refill queue if running low."""
    from ..agents.scheduler import SchedulerAgent
    
    scheduler = SchedulerAgent()
    return _run_async(scheduler.run({"type": "check_buffer"}))


@celery_app.task(
    name="app.services.tasks.stream_health_check",
    base=RateLimitedTask,
    queue="default",
    priority=5,
)
def stream_health_check() -> dict:
    """Stream health check."""
    from ..agents.stream_master import StreamMasterAgent
    
    stream = StreamMasterAgent()
    return _run_async(stream.run({"type": "health_check"}))


@celery_app.task(
    name="app.services.tasks.generate_analytics_report",
    base=RateLimitedTask,
    queue="low",
    priority=2,
)
def generate_analytics_report() -> dict:
    """Generate analytics report."""
    from ..agents.analytics import AnalyticsAgent
    
    analytics = AnalyticsAgent()
    return _run_async(analytics.run({"type": "snapshot"}))
    analytics = AnalyticsAgent()
    return _run_async(analytics.run({"type": "snapshot"}))


@celery_app.task(name="sonicforge.tasks.compose.trend_analysis", queue="compose")
def trend_analysis_task(genre: str | None = None) -> dict:
    """Periodic trend analysis using OpenAI."""
    from ..agents.composer import ComposerAgent
    composer = ComposerAgent()
    return _run_async(composer.run({"type": "analyze_trends", "genre": genre}))
