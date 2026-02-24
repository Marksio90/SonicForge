"""TTS (Text-to-Speech) Service for Station Announcements.

Two-tier strategy (research findings, 2026-02):

1. **Piper TTS** (local, primary) — ONNX-based, generates speech 10x faster
   than real-time on CPU. ~50–100 MB per voice model, 1 GB RAM total.
   Zero API cost. Quality is good for station IDs and short announcements.
   Install: `pip install piper-tts`

2. **OpenAI TTS-1** (cloud, high-quality) — $15 per million characters,
   ~$0.08/hour for occasional announcements. Studio quality with 13 voices.
   Used for featured announcements when Piper quality is insufficient.

Typical announcement schedule:
- Every 20 minutes: "Now playing: [genre] from SonicForge"
- On stream start: "Welcome to SonicForge — 24/7 AI-generated electronic music"
- On listener milestone: "Thanks for listening! [N] people tuned in"
"""

import asyncio
import io
import os
import time
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)

TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "piper")  # "piper" | "openai" | "none"
OPENAI_TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "alloy")
OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
PIPER_VOICE = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")
PIPER_SPEED = float(os.environ.get("PIPER_SPEED", "1.0"))


class TTSService:
    """Unified TTS service with Piper (local) and OpenAI TTS (cloud) backends."""

    def __init__(self, provider: str = TTS_PROVIDER):
        self.provider = provider
        self.logger = logger.bind(service="tts", provider=provider)
        self._piper_model = None

    async def generate(
        self,
        text: str,
        provider: str | None = None,
    ) -> bytes:
        """Generate speech audio from text.

        Args:
            text: Text to convert to speech (max ~500 chars for announcements)
            provider: Override provider for this call ("piper", "openai", "none")

        Returns:
            WAV audio bytes at 22.05 kHz (Piper) or 24 kHz (OpenAI)
        """
        effective_provider = provider or self.provider

        if effective_provider == "none":
            return b""

        if effective_provider == "openai":
            return await self._generate_openai(text)

        # Default: Piper (local)
        return await self._generate_piper(text)

    async def _generate_piper(self, text: str) -> bytes:
        """Generate speech with Piper TTS (local, CPU-native ONNX).

        Piper generates speech ~10x faster than real-time on CPU.
        Voice models are 50–100 MB each and cached after first load.
        """
        return await asyncio.to_thread(self._generate_piper_sync, text)

    def _generate_piper_sync(self, text: str) -> bytes:
        """Synchronous Piper inference (runs in thread pool)."""
        start = time.monotonic()
        try:
            from piper.voice import PiperVoice
            import wave

            if self._piper_model is None:
                self.logger.info("loading_piper_voice", voice=PIPER_VOICE)
                self._piper_model = PiperVoice.load(PIPER_VOICE)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                self._piper_model.synthesize(
                    text,
                    wav_file,
                    length_scale=1.0 / PIPER_SPEED,
                )

            elapsed = time.monotonic() - start
            self.logger.info(
                "piper_generated",
                chars=len(text),
                elapsed_ms=round(elapsed * 1000),
            )
            return buf.getvalue()

        except ImportError:
            self.logger.warning(
                "piper_not_installed",
                hint="pip install piper-tts",
            )
            return b""
        except Exception as e:
            self.logger.error("piper_generation_failed", error=str(e))
            return b""

    async def _generate_openai(self, text: str) -> bytes:
        """Generate speech using OpenAI TTS-1 API.

        Cost: ~$15 per million characters (~$0.08/hour for occasional use).
        Quality: studio-grade, 13 voices, 24 kHz MP3 output.
        """
        try:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                self.logger.warning("openai_tts_no_api_key")
                return await self._generate_piper(text)

            client = openai.AsyncOpenAI(api_key=api_key)
            start = time.monotonic()

            response = await client.audio.speech.create(
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=text,
                response_format="wav",
            )
            audio_data = response.content

            elapsed = time.monotonic() - start
            self.logger.info(
                "openai_tts_generated",
                chars=len(text),
                bytes=len(audio_data),
                elapsed_ms=round(elapsed * 1000),
            )
            return audio_data

        except Exception as e:
            self.logger.error("openai_tts_failed", error=str(e))
            # Fall back to Piper
            return await self._generate_piper(text)

    def build_now_playing_text(
        self,
        genre: str,
        bpm: int | None = None,
        mood: str | None = None,
        track_num: int | None = None,
    ) -> str:
        """Build a natural station announcement for the current track."""
        parts = [f"Now playing on SonicForge:"]

        genre_readable = genre.replace("_", " ").title()
        if mood:
            parts.append(f"{mood.title()} {genre_readable}")
        else:
            parts.append(genre_readable)

        if bpm:
            parts.append(f"at {bpm} BPM")

        if track_num and track_num % 10 == 0:
            parts.append(f"— track number {track_num} on our 24/7 AI music stream")

        return " ".join(parts) + "."

    def build_station_id(self) -> str:
        """Build a station identification announcement."""
        return (
            "You're listening to SonicForge — the 24/7 AI-generated electronic music stream. "
            "All music is created in real time by artificial intelligence. "
            "Sit back and enjoy the sound."
        )

    async def generate_now_playing(
        self,
        genre: str,
        bpm: int | None = None,
        mood: str | None = None,
    ) -> bytes:
        """Convenience method: generate a 'Now Playing' announcement clip."""
        text = self.build_now_playing_text(genre=genre, bpm=bpm, mood=mood)
        return await self.generate(text)

    async def generate_station_id(self) -> bytes:
        """Convenience method: generate a station ID clip (high quality)."""
        # Use OpenAI for station IDs if available — these are the most prominent
        text = self.build_station_id()
        provider = "openai" if os.environ.get("OPENAI_API_KEY") else "piper"
        return await self.generate(text, provider=provider)
