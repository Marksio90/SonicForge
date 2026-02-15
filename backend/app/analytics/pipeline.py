"""
Analytics Pipeline (Phase 6 - Data & Analytics)

Implements comprehensive event tracking and analytics:
- Event ingestion
- Real-time aggregation
- Time-series storage
- Query interface
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class AnalyticsEvent(BaseModel):
    """Single analytics event."""
    event_type: str
    event_name: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    properties: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class AggregatedMetric(BaseModel):
    """Aggregated metric data."""
    metric_name: str
    value: float
    count: int
    min_value: float
    max_value: float
    avg_value: float
    period_start: datetime
    period_end: datetime


class TimeSeriesPoint(BaseModel):
    """Single time-series data point."""
    timestamp: datetime
    value: float
    labels: dict = Field(default_factory=dict)


# In-memory storage (use ClickHouse/TimescaleDB in production)
_events: list[AnalyticsEvent] = []
_metrics: dict[str, list[TimeSeriesPoint]] = defaultdict(list)
_aggregations: dict[str, dict] = defaultdict(dict)
_hourly_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))


class AnalyticsPipeline:
    """Central analytics pipeline for event processing."""
    
    def __init__(self):
        self._buffer: list[AnalyticsEvent] = []
        self._buffer_size = 100
        self._flush_interval = 60  # seconds
    
    async def track_event(
        self,
        event_type: str,
        event_name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: dict = None,
        metadata: dict = None,
    ) -> str:
        """Track a single analytics event."""
        event = AnalyticsEvent(
            event_type=event_type,
            event_name=event_name,
            user_id=user_id,
            session_id=session_id,
            properties=properties or {},
            metadata=metadata or {},
        )
        
        # Add to buffer
        self._buffer.append(event)
        
        # Real-time aggregation
        await self._update_real_time_stats(event)
        
        # Flush if buffer is full
        if len(self._buffer) >= self._buffer_size:
            await self._flush_buffer()
        
        logger.debug(
            "event_tracked",
            event_type=event_type,
            event_name=event_name,
            user_id=user_id,
        )
        
        return f"{event.timestamp.isoformat()}:{event_type}:{event_name}"
    
    async def _update_real_time_stats(self, event: AnalyticsEvent):
        """Update real-time statistics."""
        hour_key = event.timestamp.strftime("%Y-%m-%d-%H")
        
        # Count by event type
        _hourly_stats[hour_key][f"events:{event.event_type}"] += 1
        _hourly_stats[hour_key]["events:total"] += 1
        
        # Count unique users
        if event.user_id:
            _hourly_stats[hour_key][f"users:{event.event_type}"] += 1
        
        # Track specific events
        if event.event_type == "track":
            if event.event_name == "generated":
                _hourly_stats[hour_key]["tracks:generated"] += 1
            elif event.event_name == "played":
                _hourly_stats[hour_key]["tracks:played"] += 1
            elif event.event_name == "voted":
                _hourly_stats[hour_key]["tracks:voted"] += 1
        
        elif event.event_type == "user":
            if event.event_name == "signup":
                _hourly_stats[hour_key]["users:signups"] += 1
            elif event.event_name == "login":
                _hourly_stats[hour_key]["users:logins"] += 1
        
        elif event.event_type == "payment":
            if event.event_name == "checkout_started":
                _hourly_stats[hour_key]["payments:started"] += 1
            elif event.event_name == "checkout_completed":
                _hourly_stats[hour_key]["payments:completed"] += 1
                amount = event.properties.get("amount", 0)
                _hourly_stats[hour_key]["revenue:total"] += int(amount * 100)
    
    async def _flush_buffer(self):
        """Flush event buffer to storage."""
        if not self._buffer:
            return
        
        # Move to persistent storage
        _events.extend(self._buffer)
        
        # Keep only last 10000 events in memory
        if len(_events) > 10000:
            _events[:] = _events[-10000:]
        
        logger.info("buffer_flushed", count=len(self._buffer))
        self._buffer.clear()
    
    async def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: dict = None,
    ):
        """Record a metric value."""
        point = TimeSeriesPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {},
        )
        
        _metrics[metric_name].append(point)
        
        # Keep only last 1000 points per metric
        if len(_metrics[metric_name]) > 1000:
            _metrics[metric_name] = _metrics[metric_name][-1000:]
    
    async def get_events(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AnalyticsEvent]:
        """Query events with filters."""
        await self._flush_buffer()
        
        results = []
        for event in reversed(_events):
            if event_type and event.event_type != event_type:
                continue
            if user_id and event.user_id != user_id:
                continue
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            
            results.append(event)
            if len(results) >= limit:
                break
        
        return results
    
    async def get_metric_series(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[TimeSeriesPoint]:
        """Get time-series data for a metric."""
        points = _metrics.get(metric_name, [])
        
        if start_time or end_time:
            filtered = []
            for point in points:
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                filtered.append(point)
            return filtered
        
        return points
    
    async def get_hourly_stats(
        self,
        hours: int = 24,
    ) -> dict[str, dict[str, int]]:
        """Get hourly statistics for the last N hours."""
        now = datetime.now(timezone.utc)
        result = {}
        
        for i in range(hours):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime("%Y-%m-%d-%H")
            if hour_key in _hourly_stats:
                result[hour_key] = dict(_hourly_stats[hour_key])
            else:
                result[hour_key] = {}
        
        return result
    
    async def get_summary(self) -> dict:
        """Get analytics summary."""
        await self._flush_buffer()
        
        now = datetime.now(timezone.utc)
        today_key = now.strftime("%Y-%m-%d")
        
        # Aggregate today's stats
        today_stats = defaultdict(int)
        for hour_key, stats in _hourly_stats.items():
            if hour_key.startswith(today_key):
                for key, value in stats.items():
                    today_stats[key] += value
        
        return {
            "total_events": len(_events),
            "total_metrics": len(_metrics),
            "buffer_size": len(self._buffer),
            "today": dict(today_stats),
            "last_event": _events[-1].model_dump() if _events else None,
        }


# Global instance
analytics_pipeline = AnalyticsPipeline()


# Convenience tracking functions
async def track_page_view(user_id: str, page: str, session_id: str = None):
    """Track a page view."""
    await analytics_pipeline.track_event(
        event_type="page",
        event_name="view",
        user_id=user_id,
        session_id=session_id,
        properties={"page": page},
    )


async def track_track_generated(user_id: str, track_id: str, genre: str):
    """Track a track generation."""
    await analytics_pipeline.track_event(
        event_type="track",
        event_name="generated",
        user_id=user_id,
        properties={"track_id": track_id, "genre": genre},
    )


async def track_track_played(track_id: str, user_id: str = None, duration: int = 0):
    """Track a track play."""
    await analytics_pipeline.track_event(
        event_type="track",
        event_name="played",
        user_id=user_id,
        properties={"track_id": track_id, "duration": duration},
    )


async def track_payment(user_id: str, event_name: str, amount: float = 0, plan: str = None):
    """Track a payment event."""
    await analytics_pipeline.track_event(
        event_type="payment",
        event_name=event_name,
        user_id=user_id,
        properties={"amount": amount, "plan": plan},
    )
