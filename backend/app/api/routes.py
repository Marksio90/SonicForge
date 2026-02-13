import json

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from ..core.config import Genre, get_settings
from ..schemas.analytics import (
    AgentStatusResponse,
    AnalyticsSnapshotResponse,
    DailyReportResponse,
    DashboardOverview,
    GenrePerformanceResponse,
)
from ..schemas.stream import (
    ListenerRequest,
    QueueResponse,
    ScheduleDecision,
    StreamControlRequest,
    StreamHealthResponse,
    StreamStatus,
)
from ..schemas.track import (
    TrackConceptRequest,
    TrackConceptResponse,
    TrackEvaluation,
    TrackGenerateRequest,
    TrackGenerateResponse,
)
from ..services.orchestrator import Orchestrator

settings = get_settings()

router = APIRouter()
orchestrator = Orchestrator()


# === Pipeline & Track Generation ===


@router.post("/pipeline/run", tags=["pipeline"])
async def run_pipeline(
    background_tasks: BackgroundTasks,
    genre: str | None = None,
    energy: int | None = Query(None, ge=1, le=5),
):
    """Trigger the full production pipeline (concept → generate → evaluate → queue)."""
    from ..services.tasks import run_full_pipeline

    task = run_full_pipeline.delay(genre=genre, energy=energy)
    return {"task_id": task.id, "status": "started", "genre": genre}


@router.post("/pipeline/run-sync", tags=["pipeline"])
async def run_pipeline_sync(
    genre: str | None = None,
    energy: int | None = Query(None, ge=1, le=5),
):
    """Run the full pipeline synchronously (for testing/debugging)."""
    result = await orchestrator.run_full_pipeline(genre=genre, energy=energy)
    return result


@router.post("/pipeline/batch", tags=["pipeline"])
async def generate_batch(
    count: int = Query(5, ge=1, le=20),
    genre: str | None = None,
):
    """Generate a batch of tracks asynchronously."""
    from ..services.tasks import generate_batch as generate_batch_task

    task = generate_batch_task.delay(count=count, genre=genre)
    return {"task_id": task.id, "status": "started", "count": count}


@router.post("/tracks/concept", response_model=TrackConceptResponse, tags=["tracks"])
async def create_concept(request: TrackConceptRequest):
    """Create a new track concept without generating audio."""
    result = await orchestrator.composer.run({
        "type": "create_concept",
        "genre": request.genre,
        "energy": request.energy,
        "mood": request.mood,
    })
    return result


@router.post("/tracks/evaluate/{track_id}", response_model=TrackEvaluation, tags=["tracks"])
async def evaluate_track(track_id: str, genre: str | None = None):
    """Evaluate a specific track's quality."""
    result = await orchestrator.critic.run({
        "type": "evaluate",
        "track_id": track_id,
        "genre": genre,
    })
    return result


# === Stream Control ===


@router.get("/stream/status", response_model=StreamStatus, tags=["stream"])
async def get_stream_status():
    """Get current stream status."""
    status = orchestrator.stream_master.get_status()
    return StreamStatus(
        is_streaming=status.get("is_streaming", False),
        platform=status.get("stream", {}).get("platform"),
        current_track=status.get("current_track"),
        queue_length=status.get("queue_length", 0),
    )


@router.post("/stream/control", tags=["stream"])
async def control_stream(request: StreamControlRequest):
    """Control the stream: start, stop, restart, or skip."""
    action_map = {
        "start": "start_stream",
        "stop": "stop_stream",
        "restart": "restart_stream",
        "skip": "play_next",
    }
    task_type = action_map.get(request.action)
    if not task_type:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    result = await orchestrator.stream_master.run({"type": task_type})
    return result


@router.get("/stream/health", response_model=StreamHealthResponse, tags=["stream"])
async def stream_health():
    """Get stream health status."""
    result = await orchestrator.stream_master.run({"type": "health_check"})
    return result


@router.get("/stream/queue", response_model=QueueResponse, tags=["stream"])
async def get_queue():
    """Get the current playback queue."""
    result = await orchestrator.scheduler.run({"type": "get_queue"})
    return result


@router.post("/stream/request", tags=["stream"])
async def listener_request(request: ListenerRequest):
    """Submit a listener request (genre, mood, energy change)."""
    result = await orchestrator.scheduler.run({
        "type": "process_request",
        "request": {
            "type": request.request_type,
            "value": request.value,
            "username": request.username,
            "source": request.source,
        },
    })
    return result


@router.post("/stream/override/{track_id}", tags=["stream"])
async def override_next_track(track_id: str):
    """Force a specific track to play next (manual override)."""
    result = await orchestrator.scheduler.run({
        "type": "override",
        "track_id": track_id,
    })
    return result


# === Schedule ===


@router.post("/schedule/next", response_model=ScheduleDecision, tags=["schedule"])
async def schedule_next():
    """Get the scheduling decision for the next track."""
    result = await orchestrator.scheduler.run({"type": "schedule_next"})
    return result


@router.get("/schedule/buffer", tags=["schedule"])
async def check_buffer():
    """Check the track buffer status."""
    result = await orchestrator.scheduler.run({"type": "check_buffer"})
    return result


# === Analytics ===


@router.get("/analytics/snapshot", response_model=AnalyticsSnapshotResponse, tags=["analytics"])
async def analytics_snapshot():
    """Get current analytics snapshot."""
    result = await orchestrator.analytics.run({"type": "snapshot"})
    return result


@router.get("/analytics/genre-performance", response_model=GenrePerformanceResponse, tags=["analytics"])
async def genre_performance():
    """Get genre performance analysis."""
    result = await orchestrator.analytics.run({"type": "genre_analysis"})
    return result


@router.get("/analytics/daily-report", response_model=DailyReportResponse, tags=["analytics"])
async def daily_report():
    """Generate daily analytics report."""
    result = await orchestrator.analytics.run({"type": "daily_report"})
    return result


# === Agents & Dashboard ===


@router.get("/agents/status", tags=["agents"])
async def agent_statuses():
    """Get status of all agents."""
    return await orchestrator.get_all_agent_statuses()


@router.get("/agents/activity", tags=["agents"])
async def agent_activity(limit: int = Query(50, ge=1, le=200)):
    """Get recent agent activity log."""
    return await orchestrator.get_activity_log(limit=limit)


@router.get("/dashboard/overview", response_model=DashboardOverview, tags=["dashboard"])
async def dashboard_overview():
    """Get complete dashboard overview."""
    stream_status = orchestrator.stream_master.get_status()
    agents = await orchestrator.get_all_agent_statuses()
    recent_activity = await orchestrator.get_activity_log(limit=10)

    # Get recent analytics
    analytics_data = {}
    try:
        analytics_data = await orchestrator.analytics.run({"type": "snapshot"})
    except Exception:
        pass

    return DashboardOverview(
        stream_status=stream_status.get("stream", {}),
        current_track=stream_status.get("current_track"),
        queue_length=stream_status.get("queue_length", 0),
        agents=[AgentStatusResponse(**a) for a in agents],
        recent_tracks=recent_activity,
        analytics=analytics_data,
    )


# === Genres ===


@router.get("/genres", tags=["genres"])
async def list_genres():
    """List all available genres with their profiles."""
    from ..core.config import GENRE_PROFILES

    return {
        genre.value: profile
        for genre, profile in GENRE_PROFILES.items()
    }


# === Visual ===


@router.post("/visuals/generate/{track_id}", tags=["visuals"])
async def generate_visual(track_id: str, genre: str | None = None, bpm: int | None = None):
    """Generate visual configuration for a track."""
    result = await orchestrator.visual.run({
        "type": "generate_visual",
        "track_id": track_id,
        "genre": genre,
        "bpm": bpm,
    })
    return result


@router.post("/visuals/thumbnail", tags=["visuals"])
async def generate_thumbnail(title: str = "SonicForge Radio", genre: str | None = None):
    """Generate a YouTube thumbnail."""
    result = await orchestrator.visual.run({
        "type": "generate_thumbnail",
        "title": title,
        "genre": genre,
    })
    return result
