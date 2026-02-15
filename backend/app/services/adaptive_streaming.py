"""Adaptive Bitrate Streaming with HLS.

Generates multiple quality levels for adaptive streaming:
- 320kbps (high quality)
- 192kbps (medium quality)
- 128kbps (standard quality)
- 64kbps (low quality - mobile)
"""

import subprocess
import os
from pathlib import Path
from typing import Dict, List
import structlog

logger = structlog.get_logger(__name__)


class AdaptiveBitrateEncoder:
    """Encode audio in multiple bitrates for adaptive streaming."""
    
    def __init__(self, output_dir: str = "/tmp/hls"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality levels
        self.quality_levels = [
            {"name": "high", "bitrate": "320k", "sample_rate": 48000},
            {"name": "medium", "bitrate": "192k", "sample_rate": 44100},
            {"name": "standard", "bitrate": "128k", "sample_rate": 44100},
            {"name": "low", "bitrate": "64k", "sample_rate": 22050},
        ]
        
        logger.info("adaptive_encoder_initialized", output_dir=str(self.output_dir))
    
    def encode_multibitrate(self, audio_path: str, track_id: str) -> Dict[str, str]:
        """Generate HLS playlists with multiple bitrates.
        
        Args:
            audio_path: Path to source audio file
            track_id: Unique track identifier
        
        Returns:
            Dict mapping quality levels to playlist URLs
        """
        track_dir = self.output_dir / track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        
        playlists = {}
        
        # Generate variant playlists for each quality level
        for level in self.quality_levels:
            playlist_path = self._encode_variant(
                audio_path,
                track_dir,
                level["name"],
                level["bitrate"],
                level["sample_rate"],
            )
            playlists[level["name"]] = playlist_path
        
        # Generate master playlist
        master_path = self._generate_master_playlist(track_dir, playlists)
        
        logger.info(
            "multibitrate_encoding_complete",
            track_id=track_id,
            variants=len(playlists),
        )
        
        return {
            "master": master_path,
            "variants": playlists,
        }
    
    def _encode_variant(
        self,
        audio_path: str,
        output_dir: Path,
        quality_name: str,
        bitrate: str,
        sample_rate: int,
    ) -> str:
        """Encode single quality variant.
        
        Args:
            audio_path: Source audio
            output_dir: Output directory
            quality_name: Quality level name
            bitrate: Target bitrate (e.g., "192k")
            sample_rate: Sample rate in Hz
        
        Returns:
            Path to variant playlist
        """
        output_pattern = output_dir / f"{quality_name}_%03d.ts"
        playlist_path = output_dir / f"{quality_name}.m3u8"
        
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-c:a", "aac",
            "-b:a", bitrate,
            "-ar", str(sample_rate),
            "-f", "hls",
            "-hls_time", "10",  # 10-second segments
            "-hls_playlist_type", "vod",
            "-hls_segment_filename", str(output_pattern),
            str(playlist_path),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(
                "variant_encoded",
                quality=quality_name,
                bitrate=bitrate,
                playlist=str(playlist_path),
            )
            return str(playlist_path)
        except subprocess.CalledProcessError as e:
            logger.error(
                "variant_encoding_failed",
                quality=quality_name,
                error=e.stderr,
            )
            raise
    
    def _generate_master_playlist(self, output_dir: Path, variants: Dict[str, str]) -> str:
        """Generate HLS master playlist.
        
        Args:
            output_dir: Output directory
            variants: Dict of quality name to playlist path
        
        Returns:
            Path to master playlist
        """
        master_path = output_dir / "master.m3u8"
        
        # HLS master playlist content
        content = "#EXTM3U\n"
        content += "#EXT-X-VERSION:3\n\n"
        
        # Add each variant
        for level in self.quality_levels:
            quality_name = level["name"]
            if quality_name in variants:
                # Extract bitrate as integer
                bitrate_kbps = int(level["bitrate"].replace("k", ""))
                
                content += f"#EXT-X-STREAM-INF:BANDWIDTH={bitrate_kbps * 1000},NAME=\"{quality_name}\"\n"
                content += f"{quality_name}.m3u8\n\n"
        
        # Write master playlist
        with open(master_path, "w") as f:
            f.write(content)
        
        logger.info("master_playlist_generated", path=str(master_path))
        
        return str(master_path)
    
    def get_playlist_url(self, track_id: str, quality: str = "master") -> str:
        \"\"\"Get URL for playlist.
        
        Args:
            track_id: Track identifier
            quality: Quality level or "master"
        
        Returns:
            Relative URL to playlist
        \"\"\"\n        if quality == \"master\":\n            return f\"/hls/{track_id}/master.m3u8\"\n        else:\n            return f\"/hls/{track_id}/{quality}.m3u8\"\n    \n    def cleanup_old_playlists(self, max_age_hours: int = 24):\n        \"\"\"Clean up old HLS playlists.\n        \n        Args:\n            max_age_hours: Maximum age in hours\n        \"\"\"\n        import time\n        \n        cutoff_time = time.time() - (max_age_hours * 3600)\n        removed_count = 0\n        \n        for track_dir in self.output_dir.iterdir():\n            if track_dir.is_dir():\n                # Check directory age\n                if track_dir.stat().st_mtime < cutoff_time:\n                    import shutil\n                    shutil.rmtree(track_dir)\n                    removed_count += 1\n        \n        logger.info(\"old_playlists_cleaned\", removed=removed_count)\n\n\nclass StreamQualitySelector:\n    \"\"\"Automatically select stream quality based on connection.\"\"\"\n    \n    def __init__(self):\n        self.quality_map = {\n            \"4g\": \"high\",\n            \"wifi\": \"high\",\n            \"3g\": \"standard\",\n            \"2g\": \"low\",\n            \"slow\": \"low\",\n        }\n    \n    def recommend_quality(self, connection_type: str = \"unknown\", bandwidth_kbps: int = 0) -> str:\n        \"\"\"Recommend quality level based on connection.\n        \n        Args:\n            connection_type: Type of connection (4g, wifi, etc.)\n            bandwidth_kbps: Available bandwidth in kbps\n        \n        Returns:\n            Recommended quality level\n        \"\"\"\n        # Use connection type if provided\n        if connection_type in self.quality_map:\n            return self.quality_map[connection_type]\n        \n        # Use bandwidth if available\n        if bandwidth_kbps > 0:\n            if bandwidth_kbps >= 500:\n                return \"high\"\n            elif bandwidth_kbps >= 200:\n                return \"medium\"\n            elif bandwidth_kbps >= 100:\n                return \"standard\"\n            else:\n                return \"low\"\n        \n        # Default to medium\n        return \"medium\"\n\n\n# Global instances\nadaptive_encoder = AdaptiveBitrateEncoder()\nquality_selector = StreamQualitySelector()\n