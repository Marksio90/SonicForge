import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class AnalyticsSnapshot(Base):
    __tablename__ = "analytics_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    concurrent_viewers: Mapped[int] = mapped_column(Integer, default=0)
    total_views: Mapped[int] = mapped_column(Integer, default=0)
    chat_messages_count: Mapped[int] = mapped_column(Integer, default=0)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    new_subscribers: Mapped[int] = mapped_column(Integer, default=0)
    current_genre: Mapped[str | None] = mapped_column(String(50))
    current_track_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    platform: Mapped[str] = mapped_column(String(50), default="youtube")
    extra_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class GenrePerformance(Base):
    __tablename__ = "genre_performance"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    genre: Mapped[str] = mapped_column(String(50), index=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    hour: Mapped[int] = mapped_column(Integer)
    avg_viewers: Mapped[float] = mapped_column(Float, default=0.0)
    avg_retention: Mapped[float] = mapped_column(Float, default=0.0)
    chat_engagement: Mapped[float] = mapped_column(Float, default=0.0)
    tracks_played: Mapped[int] = mapped_column(Integer, default=0)
    avg_track_score: Mapped[float] = mapped_column(Float, default=0.0)


class ListenerRequest(Base):
    __tablename__ = "listener_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_type: Mapped[str] = mapped_column(String(50))  # genre, mood, energy, skip
    value: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(50))  # chat, poll, superchat
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    fulfilled: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
