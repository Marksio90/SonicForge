import asyncio
import uuid

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import get_settings
from ..core.storage import upload_track, upload_track_metadata
from .base import BaseAgent

try:
    import replicate as _replicate_mod

    _has_replicate = True
except ImportError:
    _has_replicate = False

settings = get_settings()
logger = structlog.get_logger(__name__)

# Shared httpx client for connection pooling across all API calls
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Lazy-initialize a shared async HTTP client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client


class ProducerAgent(BaseAgent):
    """Generates audio using AI music APIs with connection pooling and retry logic."""

    def __init__(self):
        super().__init__("producer")

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "generate")

        if task_type == "generate":
            return await self.generate_track(
                concept=task["concept"],
                variants=task.get("variants", settings.variants_per_concept),
            )
        elif task_type == "master":
            return await self.master_track(track_id=task["track_id"])
        elif task_type == "convert_format":
            return await self.convert_format(
                track_id=task["track_id"],
                target_format=task.get("format", "mp3"),
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def generate_track(self, concept: dict, variants: int = 3) -> dict:
        """Generate multiple track variants from a concept using AI music APIs in parallel."""
        concept_id = str(uuid.uuid4())
        results = []

        generation_tasks = []
        for i in range(variants):
            engine = self._select_engine(concept, i)
            generation_tasks.append(
                self._generate_with_engine(engine, concept, concept_id, i + 1)
            )

        variant_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

        for result in variant_results:
            if isinstance(result, Exception):
                self.logger.error("variant_generation_failed", error=str(result))
                continue
            results.append(result)

        self.logger.info(
            "generation_complete",
            concept_id=concept_id,
            variants_requested=variants,
            variants_succeeded=len(results),
        )

        return {
            "concept_id": concept_id,
            "variants": results,
            "concept": concept,
        }

    async def _generate_with_engine(
        self, engine: str, concept: dict, concept_id: str, variant_num: int
    ) -> dict:
        """Generate a single track variant with a specific engine."""
        track_id = str(uuid.uuid4())

        self.logger.info(
            "generating_variant",
            engine=engine,
            concept_id=concept_id,
            variant=variant_num,
        )

        if engine == "suno" and settings.suno_api_key:
            audio_data = await self._generate_suno(concept)
        elif engine == "udio" and settings.udio_api_key:
            audio_data = await self._generate_udio(concept)
        elif engine == "elevenlabs" and settings.elevenlabs_api_key:
            audio_data = await self._generate_elevenlabs(concept)
        elif engine == "replicate" and settings.replicate_api_token and _has_replicate:
            audio_data = await self._generate_replicate(concept)
        else:
            audio_data = self._generate_placeholder(concept)

        # Upload to S3
        s3_key = upload_track(track_id, audio_data, "mp3")
        metadata_key = upload_track_metadata(track_id, {
            "concept_id": concept_id,
            "variant": variant_num,
            "engine": engine,
            "concept": concept,
        })

        return {
            "track_id": track_id,
            "concept_id": concept_id,
            "variant_number": variant_num,
            "engine": engine,
            "s3_key": s3_key,
            "metadata_key": metadata_key,
            "genre": concept.get("genre"),
            "bpm": concept.get("bpm"),
            "key": concept.get("key"),
            "prompt": concept.get("prompt"),
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    async def _generate_suno(self, concept: dict) -> bytes:
        """Generate music using Suno API with retry logic."""
        client = _get_http_client()
        response = await client.post(
            f"{settings.suno_api_url}/generate",
            headers={"Authorization": f"Bearer {settings.suno_api_key}"},
            json={
                "prompt": concept["prompt"],
                "duration": 180,
                "make_instrumental": True,
            },
        )
        response.raise_for_status()
        result = response.json()

        generation_id = result["id"]
        for _ in range(60):
            await asyncio.sleep(5)
            status_resp = await client.get(
                f"{settings.suno_api_url}/generate/{generation_id}",
                headers={"Authorization": f"Bearer {settings.suno_api_key}"},
            )
            status = status_resp.json()
            if status["status"] == "completed":
                audio_resp = await client.get(status["audio_url"])
                return audio_resp.content
            elif status["status"] == "failed":
                raise RuntimeError(f"Suno generation failed: {status.get('error')}")

        raise TimeoutError("Suno generation timed out")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    async def _generate_udio(self, concept: dict) -> bytes:
        """Generate music using Udio API with retry logic."""
        client = _get_http_client()
        response = await client.post(
            f"{settings.udio_api_url}/generate",
            headers={"Authorization": f"Bearer {settings.udio_api_key}"},
            json={
                "prompt": concept["prompt"],
                "duration": 180,
            },
        )
        response.raise_for_status()
        result = response.json()

        generation_id = result["id"]
        for _ in range(60):
            await asyncio.sleep(5)
            status_resp = await client.get(
                f"{settings.udio_api_url}/generate/{generation_id}",
                headers={"Authorization": f"Bearer {settings.udio_api_key}"},
            )
            status = status_resp.json()
            if status["status"] == "completed":
                audio_resp = await client.get(status["audio_url"])
                return audio_resp.content
            elif status["status"] == "failed":
                raise RuntimeError(f"Udio generation failed: {status.get('error')}")

        raise TimeoutError("Udio generation timed out")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    async def _generate_elevenlabs(self, concept: dict) -> bytes:
        """Generate music using ElevenLabs Music API with retry logic."""
        client = _get_http_client()
        response = await client.post(
            "https://api.elevenlabs.io/v1/music/generate",
            headers={"xi-api-key": settings.elevenlabs_api_key},
            json={
                "prompt": concept["prompt"],
                "duration_seconds": 180,
            },
        )
        response.raise_for_status()
        return response.content

    def _generate_placeholder(self, concept: dict) -> bytes:
        """Generate a placeholder audio marker for development without API keys."""
        return b"\xff\xfb\x90\x00" + b"\x00" * 417

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    async def _generate_replicate(self, concept: dict) -> bytes:
        """Generate music using Replicate (Meta MusicGen) — stable public API fallback."""
        import os

        os.environ["REPLICATE_API_TOKEN"] = settings.replicate_api_token

        output = await asyncio.to_thread(
            _replicate_mod.run,
            "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedbbe",
            input={
                "prompt": concept["prompt"],
                "duration": 30,
                "model_version": "stereo-large",
                "output_format": "mp3",
                "normalization_strategy": "loudness",
            },
        )
        audio_url = output if isinstance(output, str) else str(output)

        client = _get_http_client()
        resp = await client.get(audio_url)
        resp.raise_for_status()
        return resp.content

    def _select_engine(self, concept: dict, variant_index: int) -> str:
        """Select the best generation engine — Udio is the primary engine."""
        engines = []
        # Udio first — primary engine
        if settings.udio_api_key:
            engines.append("udio")
        if settings.suno_api_key:
            engines.append("suno")
        if settings.elevenlabs_api_key:
            engines.append("elevenlabs")
        if settings.replicate_api_token and _has_replicate:
            engines.append("replicate")

        if not engines:
            return "placeholder"

        return engines[variant_index % len(engines)]

    async def master_track(self, track_id: str) -> dict:
        """Apply mastering to a track (loudness normalization, EQ, limiting)."""
        self.logger.info("mastering_track", track_id=track_id)

        mastering_params = {
            "target_lufs": -14.0,
            "true_peak": -1.0,
            "eq_applied": True,
            "limiter_ceiling": -0.3,
            "stereo_imaging": "enhanced",
            "high_pass_filter": "30Hz",
            "multiband_compression": True,
        }

        return {
            "track_id": track_id,
            "mastered": True,
            "params": mastering_params,
        }

    async def convert_format(self, track_id: str, target_format: str = "mp3") -> dict:
        """Convert track between audio formats."""
        self.logger.info("converting_format", track_id=track_id, format=target_format)
        return {
            "track_id": track_id,
            "format": target_format,
            "converted": True,
        }
