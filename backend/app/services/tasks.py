"""Celery tasks for async pipeline execution."""
import asyncio

from ..core.celery_app import celery_app
from .orchestrator import Orchestrator


def _run_async(coro):
    """Helper to run async code in Celery tasks."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(name="sonicforge.tasks.pipeline.run_full")
def run_full_pipeline(genre: str | None = None, energy: int | None = None) -> dict:
    """Run the complete production pipeline as a Celery task."""
    orchestrator = Orchestrator()
    return _run_async(orchestrator.run_full_pipeline(genre=genre, energy=energy))


@celery_app.task(name="sonicforge.tasks.schedule.check_buffer")
def check_buffer() -> dict:
    """Periodic task: ensure stream queue buffer is sufficient."""
    orchestrator = Orchestrator()
    return _run_async(orchestrator.ensure_buffer())


@celery_app.task(name="sonicforge.tasks.stream.health_check")
def stream_health_check() -> dict:
    """Periodic task: check stream health every 30s."""
    orchestrator = Orchestrator()
    return _run_async(orchestrator.stream_master.run({"type": "health_check"}))


@celery_app.task(name="sonicforge.tasks.analytics.snapshot")
def analytics_snapshot() -> dict:
    """Periodic task: take analytics snapshot every 10 minutes."""
    orchestrator = Orchestrator()
    return _run_async(orchestrator.analytics.run({"type": "snapshot"}))


@celery_app.task(name="sonicforge.tasks.compose.analyze_trends")
def analyze_trends() -> dict:
    """Periodic task: analyze music trends every hour."""
    orchestrator = Orchestrator()
    return _run_async(orchestrator.composer.run({"type": "analyze_trends"}))


@celery_app.task(name="sonicforge.tasks.pipeline.generate_batch")
def generate_batch(count: int = 5, genre: str | None = None) -> list[dict]:
    """Generate a batch of tracks."""
    orchestrator = Orchestrator()
    results = []
    for _ in range(count):
        result = _run_async(orchestrator.run_full_pipeline(genre=genre))
        results.append(result)
    return results
