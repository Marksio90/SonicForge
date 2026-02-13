from pydantic import BaseModel


class StreamStatus(BaseModel):
    is_streaming: bool
    platform: str | None = None
    current_track: dict | None = None
    queue_length: int = 0
    uptime_seconds: float | None = None


class StreamControlRequest(BaseModel):
    action: str  # start, stop, restart, skip


class StreamHealthResponse(BaseModel):
    is_streaming: bool
    ffmpeg_alive: bool
    needs_restart: bool
    timestamp: str
    pid: int | None = None


class ScheduleDecision(BaseModel):
    genre: str
    energy: int
    energy_level: str
    hour: int
    preferred_genres: list[str]
    has_listener_request: bool


class QueueItem(BaseModel):
    track_id: str
    genre: str | None = None
    title: str | None = None
    position: int | None = None


class QueueResponse(BaseModel):
    queue: list[dict]
    total_length: int


class ListenerRequest(BaseModel):
    request_type: str  # genre, mood, energy, skip
    value: str
    username: str | None = None
    source: str = "dashboard"
