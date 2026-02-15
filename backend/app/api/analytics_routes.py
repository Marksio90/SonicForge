"""
Phase 6 API Routes: Data & Analytics

Implements endpoints for:
- Event tracking
- A/B testing
- Dashboard metrics
- Analytics queries
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ..analytics.pipeline import (
    analytics_pipeline,
    AnalyticsEvent,
)
from ..analytics.ab_testing import (
    ab_testing,
    Experiment,
    ExperimentResult,
)
from ..analytics.dashboard import (
    dashboard_metrics,
    SystemMetrics,
    BusinessMetrics,
    DashboardWidget,
)
from ..security.auth import get_current_user, get_current_active_user, TokenPayload
from ..security.rbac import Role, RoleChecker

router = APIRouter(tags=["analytics"])


# ==================== EVENT TRACKING ====================

class TrackEventRequest(BaseModel):
    """Event tracking request."""
    event_type: str
    event_name: str
    properties: dict = {}
    metadata: dict = {}


@router.post("/track")
async def track_event(
    request: TrackEventRequest,
    http_request: Request,
    user: Optional[TokenPayload] = Depends(get_current_user),
):
    """Track an analytics event."""
    session_id = http_request.headers.get("X-Session-ID")
    
    event_id = await analytics_pipeline.track_event(
        event_type=request.event_type,
        event_name=request.event_name,
        user_id=user.sub if user else None,
        session_id=session_id,
        properties=request.properties,
        metadata=request.metadata,
    )
    
    return {"tracked": True, "event_id": event_id}


@router.get("/events")
async def get_events(
    event_type: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    user: TokenPayload = Depends(RoleChecker(Role.ADMIN)),
):
    """Query analytics events (admin only)."""
    events = await analytics_pipeline.get_events(
        event_type=event_type,
        limit=limit,
    )
    return {"events": [e.model_dump() for e in events], "count": len(events)}


@router.get("/stats/hourly")
async def get_hourly_stats(
    hours: int = Query(default=24, le=168),
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """Get hourly statistics."""
    stats = await analytics_pipeline.get_hourly_stats(hours)
    return {"hourly_stats": stats}


@router.get("/summary")
async def get_analytics_summary(
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """Get analytics summary."""
    return await analytics_pipeline.get_summary()


# ==================== A/B TESTING ====================

class CreateExperimentRequest(BaseModel):
    """Create experiment request."""
    experiment_id: str
    name: str
    description: Optional[str] = None
    variants: list[dict]
    target_sample_size: int = 1000


@router.post("/experiments", response_model=Experiment)
async def create_experiment(
    request: CreateExperimentRequest,
    user: TokenPayload = Depends(RoleChecker(Role.ADMIN)),
):
    """Create a new A/B test experiment (admin only)."""
    return ab_testing.create_experiment(
        experiment_id=request.experiment_id,
        name=request.name,
        description=request.description,
        variants=request.variants,
        target_sample_size=request.target_sample_size,
    )


@router.get("/experiments")
async def list_experiments(
    status: Optional[str] = None,
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """List all experiments."""
    experiments = ab_testing.list_experiments(status)
    return {"experiments": [e.model_dump() for e in experiments]}


@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """Get experiment details."""
    experiment = ab_testing.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    user: TokenPayload = Depends(RoleChecker(Role.ADMIN)),
):
    """Start an experiment (admin only)."""
    if not ab_testing.start_experiment(experiment_id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"message": "Experiment started", "experiment_id": experiment_id}


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    user: TokenPayload = Depends(RoleChecker(Role.ADMIN)),
):
    """Stop an experiment (admin only)."""
    if not ab_testing.stop_experiment(experiment_id):
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"message": "Experiment stopped", "experiment_id": experiment_id}


@router.get("/experiments/{experiment_id}/variant")
async def get_user_variant(
    experiment_id: str,
    user: Optional[TokenPayload] = Depends(get_current_user),
):
    """Get the variant assignment for current user."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    variant = ab_testing.get_variant(experiment_id, user.sub)
    if not variant:
        raise HTTPException(status_code=404, detail="Experiment not found or not running")
    
    return {"experiment_id": experiment_id, "variant": variant}


@router.post("/experiments/{experiment_id}/convert")
async def track_conversion(
    experiment_id: str,
    value: float = 1.0,
    user: TokenPayload = Depends(get_current_active_user),
):
    """Track a conversion for an experiment."""
    if not ab_testing.track_conversion(experiment_id, user.sub, value):
        raise HTTPException(status_code=400, detail="Could not track conversion")
    
    return {"converted": True, "experiment_id": experiment_id}


@router.get("/experiments/{experiment_id}/results", response_model=list[ExperimentResult])
async def get_experiment_results(
    experiment_id: str,
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """Get experiment results."""
    results = ab_testing.get_results(experiment_id)
    if not results:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return results


@router.get("/experiments/{experiment_id}/winner")
async def get_experiment_winner(
    experiment_id: str,
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """Get the winning variant for an experiment."""
    winner = ab_testing.get_winner(experiment_id)
    return {"experiment_id": experiment_id, "winner": winner, "decided": winner is not None}


# ==================== DASHBOARD & METRICS ====================

@router.get("/dashboard")
async def get_dashboard_data(
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """Get all dashboard data."""
    return dashboard_metrics.get_dashboard_data()


@router.get("/dashboard/widgets", response_model=list[DashboardWidget])
async def get_dashboard_widgets(
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """Get dashboard widgets."""
    return dashboard_metrics.get_widgets()


@router.get("/metrics/system", response_model=SystemMetrics)
async def get_system_metrics(
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """Get current system metrics."""
    return dashboard_metrics.get_system_metrics()


@router.get("/metrics/business", response_model=BusinessMetrics)
async def get_business_metrics(
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """Get current business metrics."""
    return dashboard_metrics.get_business_metrics()


@router.get("/metrics/history/{metric_name}")
async def get_metric_history(
    metric_name: str,
    minutes: int = Query(default=60, le=1440),
    user: TokenPayload = Depends(RoleChecker(Role.MODERATOR)),
):
    """Get historical values for a metric."""
    history = dashboard_metrics.get_metric_history(metric_name, minutes)
    return {"metric_name": metric_name, "history": [h.model_dump() for h in history]}


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Export metrics in Prometheus format."""
    return dashboard_metrics.export_prometheus()


# ==================== PUBLIC STATS ====================

@router.get("/public/stats")
async def get_public_stats():
    """Get public statistics (no auth required)."""
    business = dashboard_metrics.get_business_metrics()
    
    return {
        "total_tracks": business.total_tracks,
        "total_plays": business.total_plays,
        "total_users": business.total_users,
    }
