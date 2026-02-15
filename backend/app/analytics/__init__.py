# Analytics module for Phase 6: Data & Analytics

from .pipeline import (
    AnalyticsPipeline,
    analytics_pipeline,
    AnalyticsEvent,
    AggregatedMetric,
    TimeSeriesPoint,
    track_page_view,
    track_track_generated,
    track_track_played,
    track_payment,
)

from .ab_testing import (
    ABTestingFramework,
    ab_testing,
    Experiment,
    Variant,
    ExperimentResult,
    ConversionEvent,
    setup_default_experiments,
)

from .dashboard import (
    DashboardMetrics,
    dashboard_metrics,
    MetricValue,
    DashboardWidget,
    SystemMetrics,
    BusinessMetrics,
    track_user_signup,
    track_track_generation,
    track_play,
    track_revenue,
)

__all__ = [
    # Pipeline
    "AnalyticsPipeline",
    "analytics_pipeline",
    "AnalyticsEvent",
    "AggregatedMetric",
    "TimeSeriesPoint",
    "track_page_view",
    "track_track_generated",
    "track_track_played",
    "track_payment",
    # A/B Testing
    "ABTestingFramework",
    "ab_testing",
    "Experiment",
    "Variant",
    "ExperimentResult",
    "ConversionEvent",
    "setup_default_experiments",
    # Dashboard
    "DashboardMetrics",
    "dashboard_metrics",
    "MetricValue",
    "DashboardWidget",
    "SystemMetrics",
    "BusinessMetrics",
    "track_user_signup",
    "track_track_generation",
    "track_play",
    "track_revenue",
]
