"""Multi-Platform Streaming Service.

Supports simultaneous streaming to:
- YouTube Live
- Twitch
- Kick
- Custom RTMP endpoints
"""

import asyncio
import subprocess
from typing import List, Dict, Optional
import structlog

from ..core.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class StreamPlatform:
    """Stream platform configuration."""
    
    def __init__(self, name: str, rtmp_url: str, stream_key: str, enabled: bool = True):
        self.name = name
        self.rtmp_url = rtmp_url
        self.stream_key = stream_key
        self.enabled = enabled
        self.process: Optional[subprocess.Popen] = None


class MultiPlatformStreamer:
    """Stream to multiple platforms simultaneously using FFmpeg."""
    
    def __init__(self):
        self.platforms: List[StreamPlatform] = []
        self.input_file: Optional[str] = None
        self.is_streaming = False
        
        self._load_platforms()
    
    def _load_platforms(self):
        """Load platform configurations from settings."""
        # YouTube
        if hasattr(settings, 'youtube_rtmp_url') and hasattr(settings, 'youtube_stream_key'):
            if settings.youtube_stream_key:
                self.platforms.append(StreamPlatform(
                    name="youtube",
                    rtmp_url=settings.youtube_rtmp_url,
                    stream_key=settings.youtube_stream_key,
                ))
        
        # Twitch
        if hasattr(settings, 'twitch_stream_key'):
            if settings.twitch_stream_key:
                self.platforms.append(StreamPlatform(
                    name="twitch",
                    rtmp_url="rtmp://live.twitch.tv/app",
                    stream_key=settings.twitch_stream_key,
                ))
        
        # Kick
        if hasattr(settings, 'kick_stream_key'):
            if settings.kick_stream_key:
                self.platforms.append(StreamPlatform(
                    name="kick",
                    rtmp_url="rtmp://live.kick.com/app",
                    stream_key=settings.kick_stream_key,
                ))
        
        logger.info("platforms_loaded", count=len(self.platforms))
    
    async def start_multistream(self, input_source: str, video_file: Optional[str] = None):
        """Start streaming to all enabled platforms.
        
        Args:
            input_source: Audio input source (file path or stream URL)
            video_file: Optional video/image for visual stream
        """
        if self.is_streaming:
            logger.warning("stream_already_running")
            return
        
        if not self.platforms:
            logger.error("no_platforms_configured")
            raise ValueError("No streaming platforms configured")
        
        self.input_file = input_source
        
        # Start stream for each platform
        for platform in self.platforms:
            if platform.enabled:
                await self._start_platform_stream(platform, input_source, video_file)
        
        self.is_streaming = True
        logger.info("multistream_started", platforms=[p.name for p in self.platforms if p.enabled])
    
    async def _start_platform_stream(self, platform: StreamPlatform, audio_input: str, video_file: Optional[str]):
        """Start stream for a single platform."""
        output_url = f"{platform.rtmp_url}/{platform.stream_key}"
        
        # FFmpeg command for streaming
        if video_file:
            # Audio + Video stream
            cmd = [
                "ffmpeg",
                "-re",  # Real-time
                "-loop", "1",  # Loop video
                "-i", video_file,  # Video input
                "-i", audio_input,  # Audio input
                "-c:v", "libx264",  # Video codec
                "-preset", "veryfast",
                "-maxrate", "3000k",
                "-bufsize", "6000k",
                "-pix_fmt", "yuv420p",
                "-g", "50",
                "-c:a", "aac",  # Audio codec
                "-b:a", "192k",
                "-ar", "44100",
                "-f", "flv",  # Format
                output_url,
            ]
        else:
            # Audio-only stream with static image
            cmd = [
                "ffmpeg",
                "-re",
                "-f", "lavfi",
                "-i", "color=c=black:s=1920x1080:r=30",  # Black background
                "-i", audio_input,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-maxrate", "3000k",
                "-bufsize", "6000k",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "44100",
                "-f", "flv",
                output_url,
            ]
        
        # Start FFmpeg process
        try:
            platform.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
            )
            logger.info("platform_stream_started", platform=platform.name, pid=platform.process.pid)
        except Exception as e:
            logger.error("platform_stream_failed", platform=platform.name, error=str(e))
    
    async def stop_multistream(self):
        """Stop all platform streams."""
        if not self.is_streaming:
            return
        
        for platform in self.platforms:
            if platform.process:
                platform.process.terminate()
                try:
                    platform.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    platform.process.kill()
                
                logger.info("platform_stream_stopped", platform=platform.name)
                platform.process = None
        
        self.is_streaming = False
        logger.info("multistream_stopped")
    
    def get_status(self) -> Dict:
        """Get streaming status for all platforms."""
        status = {
            "is_streaming": self.is_streaming,
            "platforms": [],
        }
        
        for platform in self.platforms:
            platform_status = {
                "name": platform.name,
                "enabled": platform.enabled,
                "running": platform.process is not None and platform.process.poll() is None,
            }
            
            if platform.process:
                platform_status["pid"] = platform.process.pid
            
            status["platforms"].append(platform_status)
        
        return status
    
    async def health_check(self) -> Dict:
        """Check health of all streams."""
        health = {"healthy": True, "platforms": []}
        
        for platform in self.platforms:
            if not platform.enabled:
                continue
            
            platform_health = {
                "name": platform.name,
                "healthy": False,
                "message": "not_running",
            }
            
            if platform.process:
                returncode = platform.process.poll()
                if returncode is None:
                    # Process is running
                    platform_health["healthy"] = True
                    platform_health["message"] = "streaming"
                else:
                    # Process has terminated
                    platform_health["message"] = f"terminated_with_code_{returncode}"
                    health["healthy"] = False
            else:
                health["healthy"] = False
            
            health["platforms"].append(platform_health)
        
        return health


# Global instance
multi_streamer = MultiPlatformStreamer()
