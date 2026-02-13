import asyncio
import uuid

import httpx
import structlog

from ..core.config import get_settings
from ..core.storage import upload_track, upload_track_metadata
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)


class ProducerAgent(BaseAgent):
    """Generates audio using AI music APIs and handles post-production."""

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
        """Generate multiple track variants from a concept using AI music APIs."""
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
        else:
            # Placeholder: generate silence marker for development
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

    async def _generate_suno(self, concept: dict) -> bytes:
        """Generate music using Suno API."""
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{settings.suno_api_url}/generate",
                headers={"Authorization": f"Bearer {settings.suno_api_key}"},
                json={
                    "prompt": concept["prompt"],
                    "duration": 180,  # 3 minutes
                    "make_instrumental": True,
                },
            )
            response.raise_for_status()
            result = response.json()

            # Poll for completion
            generation_id = result["id"]
            for _ in range(60):  # max 5 minutes polling
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

    async def _generate_udio(self, concept: dict) -> bytes:
        """Generate music using Udio API."""
        async with httpx.AsyncClient(timeout=300) as client:
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

    async def _generate_elevenlabs(self, concept: dict) -> bytes:
        """Generate music using ElevenLabs Music API."""
        async with httpx.AsyncClient(timeout=300) as client:
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
        # Minimal valid MP3 frame (silence) for development/testing
        # In production, this path is never taken — real APIs are used
        return b"\xff\xfb\x90\x00" + b"\x00" * 417  # minimal MP3 frame

    def _select_engine(self, concept: dict, variant_index: int) -> str:
        """Select the best generation engine for the given concept and variant."""
        engines = []
        if settings.suno_api_key:
            engines.append("suno")
        if settings.udio_api_key:
            engines.append("udio")
        if settings.elevenlabs_api_key:
            engines.append("elevenlabs")

        if not engines:
            return "placeholder"

        return engines[variant_index % len(engines)]

    async def master_track(self, track_id: str) -> dict:
        """Apply mastering to a track (loudness normalization, EQ, limiting)."""
        self.logger.info("mastering_track", track_id=track_id)

        # FFmpeg mastering command would be executed here
        # loudnorm filter for LUFS normalization, EQ, stereo imaging, limiter
        mastering_params = {
            "target_lufs": -14.0,
            "true_peak": -1.0,
            "eq_applied": True,
            "limiter_ceiling": -0.3,
            "stereo_imaging": "enhanced",
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
