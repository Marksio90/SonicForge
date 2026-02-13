from pydantic import BaseModel


class AnalyticsSnapshotResponse(BaseModel):
    timestamp: str
    concurrent_viewers: int
    total_views: int
    chat_messages: int = 0
    likes: int = 0
    new_subscribers: int = 0
    current_genre: str | None = None
    current_track: dict | None = None


class GenrePerformanceResponse(BaseModel):
    genre_performance: dict[str, dict]


class DailyReportResponse(BaseModel):
    date: str
    tracks_played: int
    peak_viewers: int
    avg_viewers: float
    total_snapshots: int
    genre_breakdown: dict


class AgentStatusResponse(BaseModel):
    agent: str
    status: str
    agent_id: str | None = None
    timestamp: str | None = None


class DashboardOverview(BaseModel):
    stream_status: dict
    current_track: dict | None = None
    queue_length: int
    agents: list[AgentStatusResponse]
    recent_tracks: list[dict]
    analytics: dict
