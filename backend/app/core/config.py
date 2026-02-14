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
    Genre.DRUM_AND_BASS: {
        "bpm_range": (170, 180),
        "energy": "high",
        "visual_theme": "cyberpunk_neon_fractals",
        "weight": 1.2,
    },
    Genre.LIQUID_DNB: {
        "bpm_range": (172, 176),
        "energy": "medium",
        "visual_theme": "flowing_liquid_colors",
        "weight": 1.0,
    },
    Genre.DUBSTEP_MELODIC: {
        "bpm_range": (138, 142),
        "energy": "high",
        "visual_theme": "glitch_art",
        "weight": 1.1,
    },
    Genre.HOUSE_DEEP: {
        "bpm_range": (120, 124),
        "energy": "low",
        "visual_theme": "warm_geometric_shapes",
        "weight": 1.0,
    },
    Genre.HOUSE_PROGRESSIVE: {
        "bpm_range": (124, 128),
        "energy": "medium",
        "visual_theme": "evolving_patterns",
        "weight": 1.1,
    },
    Genre.TRANCE_UPLIFTING: {
        "bpm_range": (136, 142),
        "energy": "high",
        "visual_theme": "cosmic_landscapes",
        "weight": 1.15,
    },
    Genre.TRANCE_PSY: {
        "bpm_range": (140, 148),
        "energy": "high",
        "visual_theme": "psychedelic_fractals",
        "weight": 1.0,
    },
    Genre.TECHNO_MELODIC: {
        "bpm_range": (124, 130),
        "energy": "medium",
        "visual_theme": "minimal_dark_geometry",
        "weight": 1.15,
    },
    Genre.BREAKBEAT: {
        "bpm_range": (130, 140),
        "energy": "medium",
        "visual_theme": "retro_breakbeat_vhs",
        "weight": 0.9,
    },
    Genre.AMBIENT: {
        "bpm_range": (60, 90),
        "energy": "low",
        "visual_theme": "ethereal_nature_scapes",
        "weight": 0.8,
    },
    Genre.DOWNTEMPO: {
        "bpm_range": (80, 110),
        "energy": "low",
        "visual_theme": "dreamy_slow_motion",
        "weight": 0.85,
    },
}

# Time-of-day energy mapping for contextual music scheduling
TIME_ENERGY_MAP = {
    # hour_range: (preferred_genres, energy_level)
    (6, 9): ([Genre.AMBIENT, Genre.DOWNTEMPO], "low"),
    (9, 12): ([Genre.HOUSE_DEEP, Genre.LIQUID_DNB, Genre.DOWNTEMPO], "low-medium"),
    (12, 15): ([Genre.HOUSE_PROGRESSIVE, Genre.BREAKBEAT, Genre.TECHNO_MELODIC], "medium"),
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
    app_version: str = "2.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "postgresql+asyncpg://sonicforge:sonicforge@localhost:5432/sonicforge"
    database_url_sync: str = "postgresql://sonicforge:sonicforge@localhost:5432/sonicforge"
    db_pool_size: int = 30
    db_max_overflow: int = 20
    db_pool_recycle: int = 1800

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 50
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # MinIO / S3 Storage
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_tracks: str = "sonicforge-tracks"
    s3_bucket_visuals: str = "sonicforge-visuals"

    # === Open-Source AI Music Generation (MusicGen) ===
    model_source: str = "local"  # "local" (GPU on host) or "runpod" (remote GPU)
    musicgen_model_version: str = "facebook/musicgen-stereo-large"
    musicgen_api_url: str = "http://musicgen:8001"  # local MusicGen service
    musicgen_segment_duration: int = 30  # seconds per generation segment
    musicgen_segments_count: int = 4  # segments per track (total = segment_duration * count)
    musicgen_continuation_overlap: int = 5  # seconds of overlap for audio continuation

    # RunPod (optional remote GPU): https://runpod.io
    runpod_api_key: str = ""
    runpod_endpoint_id: str = ""

    # Legacy paid APIs (optional fallback — set keys to enable)
    suno_api_key: str = ""
    suno_api_url: str = "https://api.suno.ai/v1"
    udio_api_key: str = ""
    udio_api_url: str = "https://api.udio.com/v1"
    elevenlabs_api_key: str = ""
    replicate_api_token: str = ""

    # === Open-Source Visuals (Stable Diffusion / ComfyUI) ===
    stable_diffusion_url: str = "http://comfyui:7860"  # A1111 or ComfyUI API
    sd_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    animatediff_enabled: bool = True  # Generate looping video instead of static images
    visual_loop_duration: int = 8  # seconds for looping video

    # LLM APIs — OpenAI is the primary provider for all LLM operations
    openai_api_key: str = ""
    anthropic_api_key: str = ""  # legacy fallback only
    llm_provider: str = "openai"  # "openai" (primary) or "anthropic" (legacy fallback)
    llm_model: str = "gpt-4o"  # primary model for prompt crafting
    llm_model_fast: str = "gpt-4o-mini"  # fast model for lightweight tasks
    llm_temperature: float = 0.85  # creativity level for music prompt generation
    llm_max_retries: int = 3  # retry count for LLM API calls

    # === Audio Mastering ===
    mastering_target_lufs: float = -14.0  # YouTube loudness standard
    mastering_true_peak: float = -1.0  # dBTP limiter ceiling
    mastering_stereo_width: float = 1.2  # stereo widening factor (1.0 = no change)

    # === Multi-Platform RTMP ===
    rtmp_proxy_enabled: bool = False  # Use NGINX RTMP proxy for multi-platform
    rtmp_proxy_url: str = "rtmp://rtmp-proxy:1935/live"
    twitch_stream_key: str = ""
    kick_stream_key: str = ""

    # YouTube Streaming
    youtube_stream_key: str = ""
    youtube_rtmp_url: str = "rtmps://a.rtmp.youtube.com/live2"
    youtube_api_key: str = ""
    youtube_channel_id: str = ""

    # Streaming
    stream_bitrate_video: str = "6000k"
    stream_bitrate_audio: str = "320k"
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
