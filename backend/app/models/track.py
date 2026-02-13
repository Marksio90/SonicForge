import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255))
    genre: Mapped[str] = mapped_column(String(50), index=True)
    subgenre: Mapped[str | None] = mapped_column(String(50))
    bpm: Mapped[int] = mapped_column(Integer)
    key: Mapped[str] = mapped_column(String(10))  # e.g. "D minor", "A major"
    energy: Mapped[int] = mapped_column(Integer)  # 1-5 scale
    mood: Mapped[str | None] = mapped_column(String(100))
    duration_seconds: Mapped[float] = mapped_column(Float)
    instruments: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)

    # Production metadata
    prompt_used: Mapped[str] = mapped_column(Text)
    generation_engine: Mapped[str] = mapped_column(String(50))  # suno, udio, elevenlabs
    generation_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    variant_number: Mapped[int] = mapped_column(Integer, default=1)
    concept_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    # Quality
    critic_score: Mapped[float] = mapped_column(Float, index=True)
    critic_feedback: Mapped[str | None] = mapped_column(Text)
    spectral_analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    has_artifacts: Mapped[bool] = mapped_column(Boolean, default=False)
    approved: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Storage
    s3_key_wav: Mapped[str | None] = mapped_column(String(500), nullable=True)
    s3_key_mp3: Mapped[str | None] = mapped_column(String(500))
    s3_key_flac: Mapped[str | None] = mapped_column(String(500), nullable=True)
    visual_s3_key: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Broadcast history
    play_count: Mapped[int] = mapped_column(Integer, default=0)
    last_played_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_listeners: Mapped[int] = mapped_column(Integer, default=0)
    popularity_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Track {self.title} [{self.genre}] score={self.critic_score}>"


class TrackConcept(Base):
    __tablename__ = "track_concepts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    genre: Mapped[str] = mapped_column(String(50))
    subgenre: Mapped[str | None] = mapped_column(String(50))
    target_bpm: Mapped[int] = mapped_column(Integer)
    target_key: Mapped[str] = mapped_column(String(10))
    target_energy: Mapped[int] = mapped_column(Integer)
    mood_description: Mapped[str] = mapped_column(Text)
    structure: Mapped[str] = mapped_column(Text)  # e.g. "8-bar intro → 16-bar build → drop"
    reference_tracks: Mapped[list | None] = mapped_column(ARRAY(String), nullable=True)
    trend_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    prompt: Mapped[str] = mapped_column(Text)
    variants_generated: Mapped[int] = mapped_column(Integer, default=0)
    best_variant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, generating, completed, failed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PlayHistory(Base):
    __tablename__ = "play_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    concurrent_viewers: Mapped[int] = mapped_column(Integer, default=0)
    chat_reactions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    skip_requested: Mapped[bool] = mapped_column(Boolean, default=False)
