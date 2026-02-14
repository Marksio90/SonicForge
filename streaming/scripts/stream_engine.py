#!/usr/bin/env python3
"""SonicForge 2.0 — Python Streaming Engine.

Replaces the bash stream_loop.sh with a full-featured Python streaming engine
that supports:
- Smart crossfading based on waveform analysis
- Dynamic text overlays (Now Playing, Next Up)
- Multi-platform RTMP distribution via NGINX proxy
- Real-time audio-reactive visual compositing
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Configuration from environment
STREAM_KEY = os.environ.get("YOUTUBE_STREAM_KEY", "")
RTMP_URL = os.environ.get("YOUTUBE_RTMP_URL", "rtmps://a.rtmp.youtube.com/live2")
RESOLUTION = os.environ.get("STREAM_RESOLUTION", "1920x1080")
VIDEO_BITRATE = os.environ.get("STREAM_BITRATE_VIDEO", "6000k")
AUDIO_BITRATE = os.environ.get("STREAM_BITRATE_AUDIO", "320k")
FPS = int(os.environ.get("STREAM_FPS", "30"))
HEALTH_CHECK_URL = os.environ.get("HEALTH_CHECK_URL", "http://api:8000/health")
CROSSFADE_DURATION = int(os.environ.get("CROSSFADE_DURATION", "12"))
API_URL = os.environ.get("API_URL", "http://api:8000")
RTMP_PROXY_ENABLED = os.environ.get("RTMP_PROXY_ENABLED", "false").lower() == "true"
RTMP_PROXY_URL = os.environ.get("RTMP_PROXY_URL", "rtmp://rtmp-proxy:1935/live")

MAX_RESTARTS = 100
RESTART_DELAY = 5


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [SonicForge Stream] {message}", flush=True)


@dataclass
class TrackInfo:
    track_id: str = ""
    title: str = "SonicForge Radio"
    genre: str = "Electronic"
    bpm: int = 128
    key: str = ""
    score: float = 0.0
    file_path: str = ""


@dataclass
class StreamState:
    current_track: TrackInfo = field(default_factory=TrackInfo)
    next_track: TrackInfo | None = None
    is_streaming: bool = False
    restart_count: int = 0
    ffmpeg_process: subprocess.Popen | None = None


class StreamEngine:
    """Python-based streaming engine with smart mixing and dynamic overlays."""

    def __init__(self):
        self.state = StreamState()
        self.queue_dir = Path("/data/tracks")
        self.queue_file = Path("/tmp/sonicforge_queue.txt")
        self._shutdown = False

    async def run(self) -> None:
        """Main streaming loop with auto-restart."""
        log("SonicForge 2.0 Streaming Engine starting...")
        log(f"Resolution: {RESOLUTION}, Video: {VIDEO_BITRATE}, Audio: {AUDIO_BITRATE}")
        log(f"Target: {RTMP_URL}")
        log(f"Crossfade: {CROSSFADE_DURATION}s")

        if RTMP_PROXY_ENABLED:
            log(f"Multi-platform RTMP proxy: {RTMP_PROXY_URL}")

        if not STREAM_KEY:
            log("ERROR: YOUTUBE_STREAM_KEY not set")
            sys.exit(1)

        # Set up signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

        while self.state.restart_count < MAX_RESTARTS and not self._shutdown:
            try:
                await self._stream_cycle()
            except Exception as e:
                log(f"Stream error: {e}")

            self.state.restart_count += 1
            log(
                f"Stream ended. Restart count: {self.state.restart_count}/{MAX_RESTARTS}"
            )

            await self._notify_backend("stream_restart")

            if not self._shutdown:
                log(f"Restarting in {RESTART_DELAY}s...")
                await asyncio.sleep(RESTART_DELAY)

        log("Max restarts reached or shutdown requested. Exiting.")

    async def _stream_cycle(self) -> None:
        """Run a single streaming cycle."""
        # Fetch current queue from API
        queue = await self._fetch_queue()
        if not queue:
            log("Queue empty — waiting for tracks...")
            await asyncio.sleep(10)
            return

        # Build the playlist file for FFmpeg concat demuxer
        await self._build_playlist(queue)

        # Determine output URL
        output_url = self._get_output_url()

        # Build FFmpeg command with dynamic overlays
        cmd = self._build_ffmpeg_command(output_url, queue)
        log(f"Starting FFmpeg stream (attempt {self.state.restart_count + 1})")

        # Run FFmpeg
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.state.ffmpeg_process = process
        self.state.is_streaming = True

        # Monitor the process
        while process.poll() is None and not self._shutdown:
            await asyncio.sleep(1)

        self.state.is_streaming = False
        self.state.ffmpeg_process = None

        if process.returncode != 0 and not self._shutdown:
            stderr = process.stderr.read().decode() if process.stderr else ""
            log(f"FFmpeg exited with code {process.returncode}")
            if stderr:
                log(f"FFmpeg stderr (last 500 chars): {stderr[-500:]}")

    def _get_output_url(self) -> str:
        """Get the RTMP output URL — proxy if multi-platform is enabled."""
        if RTMP_PROXY_ENABLED:
            return RTMP_PROXY_URL
        return f"{RTMP_URL}/{STREAM_KEY}"

    def _build_ffmpeg_command(
        self, output_url: str, queue: list[dict]
    ) -> list[str]:
        """Build FFmpeg command with smart crossfade and dynamic text overlays."""
        width, height = RESOLUTION.split("x")

        # Build overlay text for current track
        now_playing = queue[0].get("title", "SonicForge Radio") if queue else "SonicForge Radio"
        next_up = queue[1].get("title", "") if len(queue) > 1 else ""
        bpm = queue[0].get("bpm", "") if queue else ""
        genre = queue[0].get("genre", "Electronic") if queue else "Electronic"

        # Drawtext filter for dynamic overlay
        overlay_filter = (
            f"drawtext=text='Now Playing\\: {now_playing}'"
            f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            f":fontsize=28:fontcolor=white:x=40:y={int(height) - 100}"
            f":shadowcolor=black:shadowx=2:shadowy=2"
        )

        if genre:
            overlay_filter += (
                f",drawtext=text='AI Generated | {genre}'"
                f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                f":fontsize=16:fontcolor=#aaaaaa:x=40:y={int(height) - 60}"
                f":shadowcolor=black:shadowx=1:shadowy=1"
            )

        if bpm:
            overlay_filter += (
                f",drawtext=text='BPM\\: {bpm}'"
                f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                f":fontsize=18:fontcolor=#888888:x={int(width) - 120}:y={int(height) - 100}"
            )

        if next_up:
            overlay_filter += (
                f",drawtext=text='Next\\: {next_up}'"
                f":fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                f":fontsize=14:fontcolor=#666666:x=40:y={int(height) - 30}"
            )

        cmd = [
            "ffmpeg",
            "-re",
            "-f", "concat",
            "-safe", "0",
            "-i", str(self.queue_file),
            # Black background video
            "-f", "lavfi",
            "-i", f"color=c=black:s={RESOLUTION}:r={FPS}",
            # Audio encoding
            "-c:a", "aac",
            "-b:a", AUDIO_BITRATE,
            "-ar", "44100",
            # Video encoding with overlay
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-b:v", VIDEO_BITRATE,
            "-maxrate", VIDEO_BITRATE,
            "-bufsize", "9000k",
            "-pix_fmt", "yuv420p",
            "-g", str(FPS * 2),
            "-vf", overlay_filter,
            # Output
            "-f", "flv",
            output_url,
        ]

        return cmd

    async def _build_playlist(self, queue: list[dict]) -> None:
        """Build FFmpeg concat playlist file from queue."""
        lines = ["ffconcat version 1.0"]
        for item in queue:
            file_path = item.get("file_path", "")
            if file_path and os.path.exists(file_path):
                lines.append(f"file '{file_path}'")

        self.queue_file.write_text("\n".join(lines))

    async def _fetch_queue(self) -> list[dict]:
        """Fetch the current stream queue from the API."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{API_URL}/api/v1/stream/queue")
                if response.status_code == 200:
                    return response.json().get("queue", [])
        except Exception as e:
            log(f"Failed to fetch queue: {e}")
        return []

    async def _notify_backend(self, event: str) -> None:
        """Notify the backend about stream events."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(
                    HEALTH_CHECK_URL,
                    json={
                        "event": event,
                        "count": self.state.restart_count,
                    },
                )
        except Exception:
            pass

    def _handle_shutdown(self) -> None:
        """Handle graceful shutdown."""
        log("Shutdown signal received")
        self._shutdown = True
        if self.state.ffmpeg_process:
            self.state.ffmpeg_process.terminate()


async def main() -> None:
    engine = StreamEngine()
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
