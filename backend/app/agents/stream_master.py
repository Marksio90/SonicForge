import asyncio
import json
import os
import signal
import subprocess
from datetime import datetime, timezone

import structlog

from ..core.config import get_settings
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)


class StreamMasterAgent(BaseAgent):
    """Manages the 24/7 live stream via FFmpeg → RTMP to YouTube."""

    def __init__(self):
        super().__init__("stream_master")
        self._ffmpeg_process: subprocess.Popen | None = None
        self._is_streaming = False

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

        # Build the concat file from queue
        queue_file = await self._build_queue_file()

        # Build FFmpeg command
        cmd = self._build_ffmpeg_command(queue_file)

        try:
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._is_streaming = True

            # Record stream start in Redis
            self.redis.hset("stream:status", mapping={
                "active": "true",
                "pid": str(self._ffmpeg_process.pid),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "platform": "youtube",
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
                # FFmpeg crashed
                health["needs_restart"] = True
                health["exit_code"] = poll_result
                self.logger.error("ffmpeg_crashed", exit_code=poll_result)

                # Auto-restart
                await self.restart_stream()
                health["auto_restarted"] = True

        # Record health check
        self.redis.lpush("stream:health_log", json.dumps(health))
        self.redis.ltrim("stream:health_log", 0, 999)

        return health

    async def restart_stream(self) -> dict:
        """Restart the stream (stop then start)."""
        self.logger.info("restarting_stream")
        await self.stop_stream()
        await asyncio.sleep(2)  # brief pause
        result = await self.start_stream()
        result["restarted"] = True
        return result

    async def play_next_track(self) -> dict:
        """Advance to the next track in queue."""
        # Pop from Redis queue
        next_item = self.redis.lpop("stream:queue")
        if not next_item:
            self.logger.warning("queue_empty")
            return {"status": "queue_empty"}

        track_data = json.loads(next_item)

        # Record in history
        track_data["played_at"] = datetime.now(timezone.utc).isoformat()
        self.redis.lpush("stream:history", json.dumps(track_data))
        self.redis.ltrim("stream:history", 0, 499)

        # Update current track
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
        """Build the FFmpeg command for RTMP streaming."""
        stream_url = f"{settings.youtube_rtmp_url}/{settings.youtube_stream_key}"

        return [
            "ffmpeg",
            "-re",
            "-f", "concat",
            "-safe", "0",
            "-i", queue_file,
            # Audio encoding
            "-c:a", "aac",
            "-b:a", settings.stream_bitrate_audio,
            "-ar", "44100",
            # Video (from visual pipeline or generated)
            "-f", "lavfi",
            "-i", f"color=c=black:s={settings.stream_resolution}:r={settings.stream_fps}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-b:v", settings.stream_bitrate_video,
            "-maxrate", settings.stream_bitrate_video,
            "-bufsize", "9000k",
            "-pix_fmt", "yuv420p",
            "-g", str(settings.stream_fps * 2),  # keyframe interval
            # Output
            "-f", "flv",
            stream_url,
        ]
