"""
Real-time Dashboard Metrics (Phase 6 - Data & Analytics)

Implements real-time metrics and dashboard data:
- System metrics (CPU, memory, latency)
- Business metrics (users, revenue, tracks)
- Custom KPIs
- Grafana-compatible exports
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class MetricValue(BaseModel):
    """Single metric value."""
    name: str
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    labels: dict = Field(default_factory=dict)


class DashboardWidget(BaseModel):
    """Dashboard widget data."""
    widget_id: str
    title: str
    type: str  # counter, gauge, chart, table
    value: Optional[float] = None
    data: Optional[list] = None
    config: dict = Field(default_factory=dict)


class SystemMetrics(BaseModel):
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_connections: int
    request_rate: float
    error_rate: float
    avg_latency_ms: float
    p99_latency_ms: float


class BusinessMetrics(BaseModel):
    """Business KPIs."""
    total_users: int
    active_users_24h: int
    new_users_today: int
    total_tracks: int
    tracks_generated_today: int
    total_plays: int
    plays_today: int
    total_revenue: float
    revenue_today: float
    conversion_rate: float
    churn_rate: float


# In-memory metric storage
_metric_history: dict[str, list[MetricValue]] = defaultdict(list)
_counters: dict[str, int] = defaultdict(int)
_gauges: dict[str, float] = defaultdict(float)
_business_stats: dict[str, int] = defaultdict(int)
_request_times: list[float] = []


class DashboardMetrics:
    """Real-time dashboard metrics provider."""
    
    def __init__(self):
        self._start_time = time.time()
    
    def increment(self, counter_name: str, value: int = 1):
        """Increment a counter."""
        _counters[counter_name] += value
    
    def set_gauge(self, gauge_name: str, value: float):
        """Set a gauge value."""
        _gauges[gauge_name] = value
        
        # Store in history
        _metric_history[gauge_name].append(MetricValue(
            name=gauge_name,
            value=value,
            unit="",
        ))
        
        # Keep only last 1000 values
        if len(_metric_history[gauge_name]) > 1000:
            _metric_history[gauge_name] = _metric_history[gauge_name][-1000:]
    
    def record_request(self, latency_ms: float, success: bool = True):
        """Record a request for metrics."""
        _request_times.append(latency_ms)
        self.increment("requests:total")
        
        if not success:
            self.increment("requests:errors")
        
        # Keep only last 10000 requests
        if len(_request_times) > 10000:
            _request_times[:] = _request_times[-10000:]
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            import psutil
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
        except ImportError:
            cpu = memory = disk = 0.0
        
        # Calculate request stats
        total_requests = _counters.get("requests:total", 0)
        error_requests = _counters.get("requests:errors", 0)
        
        uptime_seconds = time.time() - self._start_time
        request_rate = total_requests / uptime_seconds if uptime_seconds > 0 else 0
        error_rate = error_requests / total_requests if total_requests > 0 else 0
        
        # Calculate latencies
        if _request_times:
            sorted_times = sorted(_request_times)
            avg_latency = sum(_request_times) / len(_request_times)
            p99_index = int(len(sorted_times) * 0.99)
            p99_latency = sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1]
        else:
            avg_latency = p99_latency = 0.0
        
        return SystemMetrics(
            cpu_percent=cpu,
            memory_percent=memory,
            disk_percent=disk,
            active_connections=_gauges.get("connections:active", 0),
            request_rate=round(request_rate, 2),
            error_rate=round(error_rate, 4),
            avg_latency_ms=round(avg_latency, 2),
            p99_latency_ms=round(p99_latency, 2),
        )
    
    def get_business_metrics(self) -> BusinessMetrics:
        """Get current business metrics."""
        total_users = _business_stats.get("users:total", 0)
        active_24h = _business_stats.get("users:active_24h", 0)
        new_today = _business_stats.get("users:new_today", 0)
        
        total_tracks = _business_stats.get("tracks:total", 0)
        tracks_today = _business_stats.get("tracks:generated_today", 0)
        
        total_plays = _business_stats.get("plays:total", 0)
        plays_today = _business_stats.get("plays:today", 0)
        
        total_revenue = _business_stats.get("revenue:total", 0) / 100  # Stored as cents
        revenue_today = _business_stats.get("revenue:today", 0) / 100
        
        # Calculate rates
        checkouts_started = _business_stats.get("checkouts:started", 1)
        checkouts_completed = _business_stats.get("checkouts:completed", 0)
        conversion_rate = checkouts_completed / checkouts_started if checkouts_started > 0 else 0
        
        churned = _business_stats.get("users:churned", 0)
        churn_rate = churned / total_users if total_users > 0 else 0
        
        return BusinessMetrics(
            total_users=total_users,
            active_users_24h=active_24h,
            new_users_today=new_today,
            total_tracks=total_tracks,
            tracks_generated_today=tracks_today,
            total_plays=total_plays,
            plays_today=plays_today,
            total_revenue=round(total_revenue, 2),
            revenue_today=round(revenue_today, 2),
            conversion_rate=round(conversion_rate, 4),
            churn_rate=round(churn_rate, 4),
        )
    
    def update_business_stat(self, stat_name: str, value: int = 1, absolute: bool = False):
        """Update a business statistic."""
        if absolute:
            _business_stats[stat_name] = value
        else:
            _business_stats[stat_name] += value
    
    def get_metric_history(
        self,
        metric_name: str,
        minutes: int = 60,
    ) -> list[MetricValue]:
        """Get historical values for a metric."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        return [
            m for m in _metric_history.get(metric_name, [])
            if m.timestamp >= cutoff
        ]
    
    def get_dashboard_data(self) -> dict:
        """Get all dashboard data."""
        system = self.get_system_metrics()
        business = self.get_business_metrics()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - self._start_time),
            "system": system.model_dump(),
            "business": business.model_dump(),
            "counters": dict(_counters),
            "gauges": dict(_gauges),
        }
    
    def get_widgets(self) -> list[DashboardWidget]:
        """Get dashboard widgets data."""
        system = self.get_system_metrics()
        business = self.get_business_metrics()
        
        return [
            # System widgets
            DashboardWidget(
                widget_id="cpu",
                title="CPU Usage",
                type="gauge",
                value=system.cpu_percent,
                config={"max": 100, "unit": "%", "thresholds": [70, 90]},
            ),
            DashboardWidget(
                widget_id="memory",
                title="Memory Usage",
                type="gauge",
                value=system.memory_percent,
                config={"max": 100, "unit": "%", "thresholds": [70, 90]},
            ),
            DashboardWidget(
                widget_id="latency",
                title="Avg Latency",
                type="counter",
                value=system.avg_latency_ms,
                config={"unit": "ms"},
            ),
            DashboardWidget(
                widget_id="requests",
                title="Request Rate",
                type="counter",
                value=system.request_rate,
                config={"unit": "req/s"},
            ),
            
            # Business widgets
            DashboardWidget(
                widget_id="users",
                title="Total Users",
                type="counter",
                value=business.total_users,
                config={"unit": ""},
            ),
            DashboardWidget(
                widget_id="active_users",
                title="Active Users (24h)",
                type="counter",
                value=business.active_users_24h,
                config={"unit": ""},
            ),
            DashboardWidget(
                widget_id="tracks",
                title="Tracks Generated Today",
                type="counter",
                value=business.tracks_generated_today,
                config={"unit": ""},
            ),
            DashboardWidget(
                widget_id="revenue",
                title="Revenue Today",
                type="counter",
                value=business.revenue_today,
                config={"unit": "$", "format": "currency"},
            ),
            DashboardWidget(
                widget_id="conversion",
                title="Conversion Rate",
                type="gauge",
                value=business.conversion_rate * 100,
                config={"max": 100, "unit": "%"},
            ),
        ]
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # System metrics
        system = self.get_system_metrics()
        lines.append(f'sonicforge_cpu_percent {system.cpu_percent}')
        lines.append(f'sonicforge_memory_percent {system.memory_percent}')
        lines.append(f'sonicforge_disk_percent {system.disk_percent}')
        lines.append(f'sonicforge_request_rate {system.request_rate}')
        lines.append(f'sonicforge_error_rate {system.error_rate}')
        lines.append(f'sonicforge_latency_avg_ms {system.avg_latency_ms}')
        lines.append(f'sonicforge_latency_p99_ms {system.p99_latency_ms}')
        
        # Business metrics
        business = self.get_business_metrics()
        lines.append(f'sonicforge_users_total {business.total_users}')
        lines.append(f'sonicforge_users_active_24h {business.active_users_24h}')
        lines.append(f'sonicforge_tracks_total {business.total_tracks}')
        lines.append(f'sonicforge_plays_total {business.total_plays}')
        lines.append(f'sonicforge_revenue_total {business.total_revenue}')
        
        # Counters
        for name, value in _counters.items():
            clean_name = name.replace(":", "_").replace("-", "_")
            lines.append(f'sonicforge_counter_{clean_name} {value}')
        
        return "\n".join(lines)


# Global instance
dashboard_metrics = DashboardMetrics()


# Convenience functions
def track_user_signup():
    """Track a new user signup."""
    dashboard_metrics.update_business_stat("users:total")
    dashboard_metrics.update_business_stat("users:new_today")


def track_track_generation():
    """Track a track generation."""
    dashboard_metrics.update_business_stat("tracks:total")
    dashboard_metrics.update_business_stat("tracks:generated_today")


def track_play():
    """Track a play."""
    dashboard_metrics.update_business_stat("plays:total")
    dashboard_metrics.update_business_stat("plays:today")


def track_revenue(amount_cents: int):
    """Track revenue."""
    dashboard_metrics.update_business_stat("revenue:total", amount_cents)
    dashboard_metrics.update_business_stat("revenue:today", amount_cents)
