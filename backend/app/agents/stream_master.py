import asyncio
import json
import signal
import subprocess
from datetime import datetime, timezone

import structlog

from ..core.config import get_settings
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)


class StreamMasterAgent(BaseAgent):
    """Manages the 24/7 live stream via FFmpeg with auto-recovery and crossfade support."""

    def __init__(self):
        super().__init__("stream_master")
        self._ffmpeg_process: subprocess.Popen | None = None
        self._is_streaming = False
        self._restart_count = 0
        self._max_restart_attempts = 10

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "health_check")

        if task_type == "start_stream":
            return await self.start_stream()
        elif task_type == "stop_stream":
            return await self.stop_stream()
        elif task_type == "health_check":
            return await self.health_check()
        elif task_type == "restart_stream":
            return await self.restart_stream()
        elif task_type == "play_next":
            return await self.play_next_track()
        elif task_type == "get_status":
            return self.get_status()
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def start_stream(self) -> dict:
        """Start the FFmpeg RTMP stream to YouTube."""
        if self._is_streaming:
            return {"status": "already_streaming"}

        self.logger.info("starting_stream")

        queue_file = await self._build_queue_file()
        cmd = self._build_ffmpeg_command(queue_file)

        try:
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._is_streaming = True
            self._restart_count = 0

            self.redis.hset("stream:status", mapping={
                "active": "true",
                "pid": str(self._ffmpeg_process.pid),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "platform": "youtube",
                "restart_count": str(self._restart_count),
            })

            self.logger.info("stream_started", pid=self._ffmpeg_process.pid)

            return {
                "status": "started",
                "pid": self._ffmpeg_process.pid,
            }

        except Exception as e:
            self.logger.error("stream_start_failed", error=str(e))
            self._is_streaming = False
            raise

    async def stop_stream(self) -> dict:
        """Gracefully stop the stream."""
        if self._ffmpeg_process:
            self._ffmpeg_process.send_signal(signal.SIGTERM)
            try:
                self._ffmpeg_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._ffmpeg_process.kill()

        self._is_streaming = False
        self._ffmpeg_process = None

        self.redis.hset("stream:status", mapping={
            "active": "false",
            "stopped_at": datetime.now(timezone.utc).isoformat(),
        })

        self.logger.info("stream_stopped")
        return {"status": "stopped"}

    async def health_check(self) -> dict:
        """Check stream health — called every 30 seconds."""
        health = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_streaming": self._is_streaming,
            "ffmpeg_alive": False,
            "needs_restart": False,
        }

        if self._ffmpeg_process:
            poll_result = self._ffmpeg_process.poll()
            health["ffmpeg_alive"] = poll_result is None
            health["pid"] = self._ffmpeg_process.pid

            if poll_result is not None:
                health["needs_restart"] = True
                health["exit_code"] = poll_result
                self.logger.error("ffmpeg_crashed", exit_code=poll_result)

                # Auto-restart with backoff protection
                if self._restart_count < self._max_restart_attempts:
                    backoff = min(2 ** self._restart_count, 30)
                    self.logger.info(
                        "auto_restart_scheduled",
                        attempt=self._restart_count + 1,
                        backoff_seconds=backoff,
                    )
                    await asyncio.sleep(backoff)
                    await self.restart_stream()
                    health["auto_restarted"] = True
                else:
                    self.logger.error("max_restart_attempts_reached")
                    health["auto_restarted"] = False

        self.redis.lpush("stream:health_log", json.dumps(health))
        self.redis.ltrim("stream:health_log", 0, 999)

        return health

    async def restart_stream(self) -> dict:
        """Restart the stream (stop then start)."""
        self.logger.info("restarting_stream", attempt=self._restart_count + 1)
        self._restart_count += 1
        await self.stop_stream()
        await asyncio.sleep(2)
        result = await self.start_stream()
        result["restarted"] = True
        result["restart_count"] = self._restart_count
        return result

    async def play_next_track(self) -> dict:
        """Advance to the next track in queue."""
        next_item = self.redis.lpop("stream:queue")
        if not next_item:
            self.logger.warning("queue_empty")
            return {"status": "queue_empty"}

        track_data = json.loads(next_item)

        track_data["played_at"] = datetime.now(timezone.utc).isoformat()
        self.redis.lpush("stream:history", json.dumps(track_data))
        self.redis.ltrim("stream:history", 0, 499)

        self.redis.hset("stream:current_track", mapping={
            "track_id": track_data.get("track_id", ""),
            "genre": track_data.get("genre", ""),
            "title": track_data.get("title", "Unknown"),
            "started_at": datetime.now(timezone.utc).isoformat(),
        })

        self.logger.info("playing_next", track_id=track_data.get("track_id"))
        return {"status": "playing", "track": track_data}

    def get_status(self) -> dict:
        """Get current stream status."""
        status_data = self.redis.hgetall("stream:status")
        current_track = self.redis.hgetall("stream:current_track")
        queue_length = self.redis.llen("stream:queue")

        return {
            "stream": status_data,
            "current_track": current_track,
            "queue_length": queue_length,
            "is_streaming": self._is_streaming,
        }

    async def _build_queue_file(self) -> str:
        """Build FFmpeg concat file from the track queue."""
        queue_file = "/tmp/sonicforge_queue.txt"
        queue_items = self.redis.lrange("stream:queue", 0, -1)

        lines = []
        for item in queue_items:
            data = json.loads(item)
            track_path = data.get("file_path", data.get("s3_key", ""))
            if track_path:
                lines.append(f"file '{track_path}'")

        with open(queue_file, "w") as f:
            f.write("\n".join(lines))

        return queue_file

    def _build_ffmpeg_command(self, queue_file: str) -> list[str]:
        """Build optimized FFmpeg command for RTMP streaming.

        Uses a background video loop if available at /app/assets/background_loop.mp4,
        otherwise falls back to a lavfi color source (black screen).
        """
        import os

        stream_url = f"{settings.youtube_rtmp_url}/{settings.youtube_stream_key}"
        bg_video = "/app/assets/background_loop.mp4"

        if os.path.isfile(bg_video):
            return [
                "ffmpeg",
                "-re",
                "-stream_loop", "-1",
                "-i", bg_video,
                "-f", "concat",
                "-safe", "0",
                "-i", queue_file,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-b:v", settings.stream_bitrate_video,
                "-maxrate", settings.stream_bitrate_video,
                "-bufsize", "12000k",
                "-pix_fmt", "yuv420p",
                "-g", str(settings.stream_fps * 2),
                "-c:a", "aac",
                "-b:a", settings.stream_bitrate_audio,
                "-ar", "48000",
                "-ac", "2",
                "-threads", "0",
                "-f", "flv",
                stream_url,
            ]

        # Fallback: generate a solid-color video source when no background video exists
        return [
            "ffmpeg",
            "-re",
            "-f", "lavfi",
            "-i", f"color=c=0x0a0a0f:s={settings.stream_resolution}:r={settings.stream_fps}",
            "-f", "concat",
            "-safe", "0",
            "-i", queue_file,
            "-map", "0:v",
            "-map", "1:a",
            "-shortest",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-b:v", settings.stream_bitrate_video,
            "-maxrate", settings.stream_bitrate_video,
            "-bufsize", "12000k",
            "-pix_fmt", "yuv420p",
            "-g", str(settings.stream_fps * 2),
            "-c:a", "aac",
            "-b:a", settings.stream_bitrate_audio,
            "-ar", "48000",
            "-ac", "2",
            "-threads", "0",
            "-f", "flv",
            stream_url,
        ]
