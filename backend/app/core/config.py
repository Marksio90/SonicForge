from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Genre(str, Enum):
    DRUM_AND_BASS = "drum_and_bass"
    LIQUID_DNB = "liquid_dnb"
    DUBSTEP_MELODIC = "dubstep_melodic"
    HOUSE_DEEP = "house_deep"
    HOUSE_PROGRESSIVE = "house_progressive"
    TRANCE_UPLIFTING = "trance_uplifting"
    TRANCE_PSY = "trance_psy"
    TECHNO_MELODIC = "techno_melodic"
    BREAKBEAT = "breakbeat"
    AMBIENT = "ambient"
    DOWNTEMPO = "downtempo"


GENRE_PROFILES = {
    Genre.DRUM_AND_BASS: {"bpm_range": (170, 180), "energy": "high", "visual_theme": "cyberpunk_neon_fractals"},
    Genre.LIQUID_DNB: {"bpm_range": (172, 176), "energy": "medium", "visual_theme": "flowing_liquid_colors"},
    Genre.DUBSTEP_MELODIC: {"bpm_range": (138, 142), "energy": "high", "visual_theme": "glitch_art"},
    Genre.HOUSE_DEEP: {"bpm_range": (120, 124), "energy": "low", "visual_theme": "warm_geometric_shapes"},
    Genre.HOUSE_PROGRESSIVE: {"bpm_range": (124, 128), "energy": "medium", "visual_theme": "evolving_patterns"},
    Genre.TRANCE_UPLIFTING: {"bpm_range": (136, 142), "energy": "high", "visual_theme": "cosmic_landscapes"},
    Genre.TRANCE_PSY: {"bpm_range": (140, 148), "energy": "high", "visual_theme": "psychedelic_fractals"},
    Genre.TECHNO_MELODIC: {"bpm_range": (124, 130), "energy": "medium", "visual_theme": "minimal_dark_geometry"},
    Genre.BREAKBEAT: {"bpm_range": (130, 140), "energy": "medium", "visual_theme": "retro_breakbeat_vhs"},
    Genre.AMBIENT: {"bpm_range": (60, 90), "energy": "low", "visual_theme": "ethereal_nature_scapes"},
    Genre.DOWNTEMPO: {"bpm_range": (80, 110), "energy": "low", "visual_theme": "dreamy_slow_motion"},
}

# Time-of-day energy mapping for contextual music
TIME_ENERGY_MAP = {
    # hour_range: (preferred_genres, energy_level)
    (6, 9): ([Genre.AMBIENT, Genre.DOWNTEMPO], "low"),
    (9, 12): ([Genre.HOUSE_DEEP, Genre.LIQUID_DNB], "low-medium"),
    (12, 15): ([Genre.HOUSE_PROGRESSIVE, Genre.BREAKBEAT], "medium"),
    (15, 18): ([Genre.HOUSE_PROGRESSIVE, Genre.TECHNO_MELODIC, Genre.TRANCE_UPLIFTING], "medium-high"),
    (18, 21): ([Genre.DRUM_AND_BASS, Genre.TRANCE_UPLIFTING, Genre.TRANCE_PSY], "high"),
    (21, 0): ([Genre.DUBSTEP_MELODIC, Genre.TECHNO_MELODIC, Genre.DRUM_AND_BASS], "high"),
    (0, 3): ([Genre.TECHNO_MELODIC, Genre.TRANCE_PSY, Genre.DUBSTEP_MELODIC], "high"),
    (3, 6): ([Genre.AMBIENT, Genre.DOWNTEMPO, Genre.HOUSE_DEEP], "low"),
}


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Application
    app_name: str = "SonicForge"
    app_version: str = "0.1.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "postgresql+asyncpg://sonicforge:sonicforge@localhost:5432/sonicforge"
    database_url_sync: str = "postgresql://sonicforge:sonicforge@localhost:5432/sonicforge"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # MinIO / S3 Storage
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_tracks: str = "sonicforge-tracks"
    s3_bucket_visuals: str = "sonicforge-visuals"

    # AI Music Generation
    suno_api_key: str = ""
    suno_api_url: str = "https://api.suno.ai/v1"
    udio_api_key: str = ""
    udio_api_url: str = "https://api.udio.com/v1"
    elevenlabs_api_key: str = ""

    # LLM APIs
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"  # "anthropic" or "openai"
    llm_model: str = "claude-sonnet-4-5-20250929"

    # YouTube Streaming
    youtube_stream_key: str = ""
    youtube_rtmp_url: str = "rtmps://a.rtmp.youtube.com/live2"
    youtube_api_key: str = ""
    youtube_channel_id: str = ""

    # Streaming
    stream_bitrate_video: str = "4500k"
    stream_bitrate_audio: str = "192k"
    stream_resolution: str = "1920x1080"
    stream_fps: int = 30
    crossfade_duration: int = 12  # seconds
    buffer_min_tracks: int = 20  # minimum tracks in queue buffer

    # Quality Gate
    quality_threshold: float = 8.5
    max_generation_attempts: int = 5
    variants_per_concept: int = 3

    # Gyre Backup
    gyre_api_key: str = ""
    gyre_enabled: bool = False

    # Restream
    restream_enabled: bool = False
    restream_key: str = ""

    # Monitoring
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    sentry_dsn: str = ""

    # Dashboard
    dashboard_url: str = "http://localhost:3000"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
