"""ACE-Step v1.5 Music Generation Service.

ACE-Step v1.5 (released January 2026) is the most capable open-source music
generation model available as of early 2026:
- Supports vocals in 50+ languages and fully instrumental tracks
- Generates up to 10-minute tracks
- Quality between Suno v4.5 and v5
- MIT license — commercial use permitted
- CPU inference supported via quantization and CPU offloading (<4 GB VRAM)
- Built-in REST API: `uv run acestep-api`

Architecture: Wraps the ACE-Step REST API (default port 7860) similar to how
the MusicGen service wraps AudioCraft.

Reference: https://github.com/ace-step/ACE-Step
"""

import asyncio
import os
import uuid

import httpx
import structlog

logger = structlog.get_logger(__name__)

ACESTEP_API_URL = os.environ.get("ACESTEP_API_URL", "http://acestep:7860")
ACESTEP_TIMEOUT_SECONDS = int(os.environ.get("ACESTEP_TIMEOUT", "600"))


class ACEStepService:
    """Client for the ACE-Step v1.5 REST API.

    The ACE-Step service is expected to run as a separate Docker container
    (or locally via `uv run acestep-api`) and exposes a simple REST API.
    """

    def __init__(self, api_url: str = ACESTEP_API_URL):
        self.api_url = api_url
        self.logger = logger.bind(service="acestep")

    async def generate(
        self,
        prompt: str,
        duration: int = 180,
        genre: str = "electronic",
        lyrics: str | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Generate audio via ACE-Step v1.5.

        Args:
            prompt: Music description / style prompt
            duration: Target duration in seconds (ACE-Step supports up to 600s)
            genre: Genre tag for improved generation quality
            lyrics: Optional lyrics for vocal tracks (None = instrumental)
            seed: Random seed for reproducibility (None = random)

        Returns:
            WAV audio bytes
        """
        if seed is None:
            seed = int(uuid.uuid4().int % (2**31))

        payload: dict = {
            "prompt": prompt,
            "duration": min(duration, 600),
            "genre": genre,
            "seed": seed,
        }

        if lyrics:
            payload["lyrics"] = lyrics
        else:
            payload["instrumental"] = True

        self.logger.info(
            "acestep_generate_start",
            duration=duration,
            genre=genre,
            has_lyrics=bool(lyrics),
        )

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=ACESTEP_TIMEOUT_SECONDS,
                write=30.0,
                pool=10.0,
            )
        ) as client:
            # Submit generation job
            response = await client.post(
                f"{self.api_url}/generate",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            job_id = result.get("id") or result.get("job_id")

            if not job_id:
                # Synchronous API — audio returned directly
                audio_url = result.get("audio_url") or result.get("url")
                if audio_url:
                    audio_resp = await client.get(audio_url)
                    audio_resp.raise_for_status()
                    self.logger.info("acestep_generate_complete", bytes=len(audio_resp.content))
                    return audio_resp.content
                raise RuntimeError(f"ACE-Step returned unexpected response: {result}")

            # Async API — poll for completion
            audio_data = await self._poll_job(client, job_id)
            self.logger.info("acestep_generate_complete", bytes=len(audio_data))
            return audio_data

    async def _poll_job(self, client: httpx.AsyncClient, job_id: str) -> bytes:
        """Poll ACE-Step job until complete and return audio bytes."""
        max_polls = ACESTEP_TIMEOUT_SECONDS // 5
        for _ in range(max_polls):
            await asyncio.sleep(5)
            status_resp = await client.get(f"{self.api_url}/status/{job_id}")
            status_resp.raise_for_status()
            status = status_resp.json()

            job_status = status.get("status", "pending")
            if job_status == "completed":
                audio_url = status.get("audio_url") or status.get("url")
                if audio_url:
                    audio_resp = await client.get(audio_url)
                    audio_resp.raise_for_status()
                    return audio_resp.content

                # Inline bytes fallback
                audio_b64 = status.get("audio_base64")
                if audio_b64:
                    import base64
                    return base64.b64decode(audio_b64)

                raise RuntimeError("ACE-Step job completed but no audio found")

            elif job_status in ("failed", "error"):
                raise RuntimeError(
                    f"ACE-Step generation failed: {status.get('error', 'unknown')}"
                )

        raise TimeoutError(f"ACE-Step job {job_id} timed out after {ACESTEP_TIMEOUT_SECONDS}s")

    async def health(self) -> bool:
        """Check if the ACE-Step service is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.api_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
