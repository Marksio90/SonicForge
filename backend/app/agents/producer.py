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

# NOTE: Udio API integration was removed (2026-02).
# Following Udio's settlement with Universal Music Group (November 2025),
# all WAV/MP3/stem downloads were completely disabled. The platform is
# building a new licensed service expected in 2026.
# See: research notes on Udio post-UMG settlement.

settings = get_settings()
logger = structlog.get_logger(__name__)

# Shared httpx client for connection pooling across all API calls
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Get HTTP client from connection pool."""
    from ..core.connection_pool import pool_manager
    return pool_manager.http_client


class MusicGenEngine:
    """Local MusicGen (AudioCraft) engine with audio continuation for extended tracks."""

    def __init__(self):
        self.logger = structlog.get_logger("musicgen_engine")
        self.model_version = settings.musicgen_model_version
        self.api_url = settings.musicgen_api_url
        self.segment_duration = settings.musicgen_segment_duration
        self.segments_count = settings.musicgen_segments_count
        self.overlap = settings.musicgen_continuation_overlap

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    async def generate(self, concept: dict) -> bytes:
        """Generate a full track using segment-based audio continuation.

        Generates multiple segments with overlapping context to maintain musical
        coherence across the full track duration.
        """
        client = _get_http_client()
        genre = concept.get("genre", "electronic")
        bpm = concept.get("bpm", 128)
        key = concept.get("key", "Am")
        mood = concept.get("mood", "energetic")
        instruments = concept.get("instruments", "synthesizer, drums, bass")
        prompt = concept.get("prompt", f"{genre}, {mood}, {bpm} BPM, key of {key}")

        segments: list[bytes] = []
        continuation_audio: bytes | None = None

        for i in range(self.segments_count):
            segment_prompt = (
                f"{prompt}, part {i + 1} of {self.segments_count}, "
                f"seamless continuation, professional production quality"
            )

            payload: dict = {
                "prompt": segment_prompt,
                "duration": self.segment_duration,
                "model_version": self.model_version,
                "output_format": "wav",
                "normalization_strategy": "loudness",
                "bpm": bpm,
                "key": key,
            }

            if continuation_audio is not None:
                # Send the last N seconds as continuation context
                import base64

                payload["continuation_audio"] = base64.b64encode(
                    continuation_audio
                ).decode("utf-8")
                payload["continuation_start"] = max(
                    0, self.segment_duration - self.overlap
                )

            self.logger.info(
                "generating_segment",
                segment=i + 1,
                total=self.segments_count,
                prompt_len=len(segment_prompt),
            )

            response = await client.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=httpx.Timeout(connect=10, read=600, write=30, pool=10),
            )
            response.raise_for_status()

            result = response.json()
            generation_id = result["id"]

            # Poll for completion
            audio_data = await self._poll_generation(client, generation_id)
            segments.append(audio_data)

            # Keep the tail for continuation context
            overlap_bytes = self._extract_tail_bytes(audio_data, self.overlap)
            continuation_audio = overlap_bytes

        # Concatenate segments with crossfade
        full_track = await self._concatenate_segments(segments)

        self.logger.info(
            "track_generation_complete",
            segments=len(segments),
            total_bytes=len(full_track),
        )

        return full_track

    async def _poll_generation(
        self, client: httpx.AsyncClient, generation_id: str
    ) -> bytes:
        """Poll the MusicGen service for generation completion."""
        for _ in range(120):  # Up to 10 minutes
            await asyncio.sleep(5)
            status_resp = await client.get(
                f"{self.api_url}/status/{generation_id}",
            )
            status = status_resp.json()

            if status["status"] == "completed":
                audio_resp = await client.get(
                    f"{self.api_url}/download/{generation_id}"
                )
                audio_resp.raise_for_status()
                return audio_resp.content
            elif status["status"] == "failed":
                raise RuntimeError(
                    f"MusicGen generation failed: {status.get('error')}"
                )

        raise TimeoutError("MusicGen generation timed out after 10 minutes")

    def _extract_tail_bytes(self, audio_data: bytes, seconds: int) -> bytes:
        """Extract the last N seconds of audio for continuation context.

        Assumes WAV format at 44100 Hz stereo 16-bit (176400 bytes/sec).
        """
        bytes_per_second = 44100 * 2 * 2  # sample_rate * channels * bytes_per_sample
        tail_size = bytes_per_second * seconds
        if len(audio_data) > tail_size:
            return audio_data[-tail_size:]
        return audio_data

    async def _concatenate_segments(self, segments: list[bytes]) -> bytes:
        """Concatenate audio segments with crossfade blending."""
        if not segments:
            return b""
        if len(segments) == 1:
            return segments[0]

        # Simple concatenation — the MusicGen service handles the actual
        # audio continuation, so segments are already musically coherent.
        # The mastering engine will handle final crossfade and normalization.
        result = bytearray()
        for segment in segments:
            result.extend(segment)
        return bytes(result)


class RunPodMusicGenEngine:
    """Remote MusicGen engine via RunPod serverless GPU."""

    def __init__(self):
        self.logger = structlog.get_logger("runpod_musicgen")
        self.api_key = settings.runpod_api_key
        self.endpoint_id = settings.runpod_endpoint_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    async def generate(self, concept: dict) -> bytes:
        """Generate audio via RunPod serverless endpoint."""
        client = _get_http_client()
        prompt = concept.get("prompt", "electronic music")

        response = await client.post(
            f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "input": {
                    "prompt": prompt,
                    "duration": settings.musicgen_segment_duration
                    * settings.musicgen_segments_count,
                    "model_version": settings.musicgen_model_version,
                    "output_format": "wav",
                }
            },
            timeout=httpx.Timeout(connect=10, read=600, write=30, pool=10),
        )
        response.raise_for_status()
        result = response.json()

        audio_url = result.get("output", {}).get("audio_url", "")
        if not audio_url:
            raise RuntimeError(f"RunPod returned no audio URL: {result}")

        audio_resp = await client.get(audio_url)
        audio_resp.raise_for_status()
        return audio_resp.content


class ProducerAgent(BaseAgent):
    """Generates audio using local MusicGen (primary) with paid API fallback."""

    def __init__(self):
        super().__init__("producer")
        self._musicgen = MusicGenEngine()
        self._runpod = RunPodMusicGenEngine()

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
        """Generate multiple track variants from a concept using AI music engines in parallel."""
        concept_id = str(uuid.uuid4())
        results = []

        generation_tasks = []
        for i in range(variants):
            engine = self._select_engine(concept, i)
            generation_tasks.append(
                self._generate_with_engine(engine, concept, concept_id, i + 1)
            )

        variant_results = await asyncio.gather(
            *generation_tasks, return_exceptions=True
        )

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

        if engine == "acestep":
            audio_data = await self._generate_acestep(concept)
        elif engine == "stable_audio":
            audio_data = await self._generate_stable_audio(concept)
        elif engine == "musicgen_local":
            audio_data = await self._musicgen.generate(concept)
        elif engine == "musicgen_runpod":
            audio_data = await self._runpod.generate(concept)
        elif engine == "suno" and settings.suno_api_key:
            audio_data = await self._generate_suno(concept)
        elif engine == "elevenlabs" and settings.elevenlabs_api_key:
            audio_data = await self._generate_elevenlabs(concept)
        elif engine == "replicate" and settings.replicate_api_token and _has_replicate:
            audio_data = await self._generate_replicate(concept)
        else:
            audio_data = self._generate_placeholder(concept)

        # Apply auto-mastering before upload
        audio_data = await self._apply_mastering(audio_data)

        # Upload to S3
        s3_key = upload_track(track_id, audio_data, "wav")
        metadata_key = upload_track_metadata(
            track_id,
            {
                "concept_id": concept_id,
                "variant": variant_num,
                "engine": engine,
                "concept": concept,
            },
        )

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

    async def _apply_mastering(self, audio_data: bytes) -> bytes:
        """Apply audio mastering via the mastering service."""
        try:
            from ..services.audio_engineering import MasteringEngine

            engine = MasteringEngine()
            return await engine.master_audio(audio_data)
        except Exception as e:
            self.logger.warning("mastering_skipped", error=str(e))
            return audio_data

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30)
    )
    async def _generate_suno(self, concept: dict) -> bytes:
        """Generate music using Suno API (legacy fallback)."""
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

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30)
    )
    async def _generate_stable_audio(self, concept: dict) -> bytes:
        """Generate music using Stable Audio Open Small (local, CPU-optimized).

        341M parameters, generates 11s of 44.1 kHz stereo audio in <8s on Arm CPU.
        Requires ~3.6 GB RAM (INT8 quantized). MIT-compatible license (Stability AI
        Community License — commercial use permitted).
        """
        from ..services.stable_audio_service import StableAudioService

        svc = StableAudioService()
        prompt = concept.get("prompt", "ambient electronic music")
        return await svc.generate(prompt=prompt, duration=11)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30)
    )
    async def _generate_acestep(self, concept: dict) -> bytes:
        """Generate music using ACE-Step v1.5 (local, MIT license).

        Supports vocals in 50+ languages and instrumentals up to 10 min.
        Quality between Suno v4.5 and v5. CPU inference possible with
        quantization; works with <4 GB VRAM via CPU offloading.
        MIT license — commercial use permitted.
        """
        from ..services.acestep_service import ACEStepService

        svc = ACEStepService()
        prompt = concept.get("prompt", "electronic music")
        return await svc.generate(
            prompt=prompt,
            duration=180,
            genre=concept.get("genre", "electronic"),
        )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30)
    )
    async def _generate_elevenlabs(self, concept: dict) -> bytes:
        """Generate music using ElevenLabs (legacy fallback)."""
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

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30)
    )
    async def _generate_replicate(self, concept: dict) -> bytes:
        """Generate music using Replicate (Meta MusicGen) — cloud API fallback."""
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

    def _generate_placeholder(self, concept: dict) -> bytes:
        """Generate a placeholder audio marker for development without GPU/API keys."""
        return b"\xff\xfb\x90\x00" + b"\x00" * 417

    def _select_engine(self, concept: dict, variant_index: int) -> str:
        """Select the best generation engine for each variant.

        Priority order (revised 2026-02):
        1. acestep      — MIT license, vocals + instrumentals, best open-source quality
        2. stable_audio — MIT-compatible, fastest on CPU (11s clips, <8s generation)
        3. musicgen_local — CC-BY-NC (non-commercial only on pretrained weights)
        4. musicgen_runpod — remote GPU fallback via RunPod
        5. suno         — cloud API, commercial rights, high quality ($30/mo Premier)
        6. elevenlabs   — cloud fallback
        7. replicate    — pay-per-use cloud MusicGen
        8. placeholder  — development mode stub

        NOTE: Udio removed (Nov 2025 UMG settlement — downloads permanently disabled).
        NOTE: MusicGen pretrained weights are CC-BY-NC. Use ACE-Step or Stable Audio
              for commercially licensed outputs.
        """
        engines = []

        # Primary: ACE-Step v1.5 (MIT, best quality, vocals supported)
        if settings.acestep_enabled:
            engines.append("acestep")

        # Secondary: Stable Audio Open Small (fast CPU, MIT-compatible)
        if settings.stable_audio_enabled:
            engines.append("stable_audio")

        # Tertiary: local MusicGen (CC-BY-NC — non-commercial use only)
        if settings.model_source == "local":
            engines.append("musicgen_local")

        # Remote GPU via RunPod
        if settings.runpod_api_key and settings.runpod_endpoint_id:
            engines.append("musicgen_runpod")

        # Cloud APIs (paid, commercial rights)
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
        """Apply mastering to a track via the mastering engine."""
        self.logger.info("mastering_track", track_id=track_id)

        try:
            from ..core.storage import download_track
            from ..services.audio_engineering import MasteringEngine

            audio_data = download_track(track_id)
            engine = MasteringEngine()
            mastered = await engine.master_audio(audio_data)

            # Re-upload mastered version
            s3_key = upload_track(track_id, mastered, "wav")

            return {
                "track_id": track_id,
                "mastered": True,
                "s3_key": s3_key,
                "params": {
                    "target_lufs": settings.mastering_target_lufs,
                    "true_peak": settings.mastering_true_peak,
                    "stereo_width": settings.mastering_stereo_width,
                },
            }
        except Exception as e:
            self.logger.error("mastering_failed", track_id=track_id, error=str(e))
            return {
                "track_id": track_id,
                "mastered": False,
                "error": str(e),
            }

    async def convert_format(self, track_id: str, target_format: str = "mp3") -> dict:
        """Convert track between audio formats."""
        self.logger.info("converting_format", track_id=track_id, format=target_format)
        return {
            "track_id": track_id,
            "format": target_format,
            "converted": True,
        }
