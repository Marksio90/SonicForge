import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class TrackConceptRequest(BaseModel):
    genre: str | None = None
    energy: int | None = Field(None, ge=1, le=5)
    mood: str | None = None


class TrackConceptResponse(BaseModel):
    genre: str
    subgenre: str
    bpm: int
    key: str
    energy: int
    mood: str
    instruments: list[str]
    structure: str
    visual_theme: str
    prompt: str


class TrackGenerateRequest(BaseModel):
    concept: TrackConceptResponse | None = None
    genre: str | None = None
    variants: int = Field(3, ge=1, le=5)


class TrackVariantResponse(BaseModel):
    track_id: str
    concept_id: str
    variant_number: int
    engine: str
    s3_key: str
    genre: str | None = None
    bpm: int | None = None
    key: str | None = None


class TrackGenerateResponse(BaseModel):
    concept_id: str
    variants: list[TrackVariantResponse]
    concept: dict


class TrackEvaluation(BaseModel):
    track_id: str
    overall_score: float
    approved: bool
    scores: dict[str, float]
    has_artifacts: bool
    feedback: str
    threshold: float


class TrackResponse(BaseModel):
    id: uuid.UUID
    title: str
    genre: str
    subgenre: str | None = None
    bpm: int
    key: str
    energy: int
    mood: str | None = None
    duration_seconds: float
    critic_score: float
    approved: bool
    play_count: int
    popularity_score: float
    created_at: datetime

    model_config = {"from_attributes": True}


class TrackListResponse(BaseModel):
    tracks: list[TrackResponse]
    total: int
    page: int
    page_size: int
