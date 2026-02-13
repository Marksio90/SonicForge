import uuid

import httpx
import structlog

from ..core.config import GENRE_PROFILES, Genre, get_settings
from ..core.storage import upload_visual
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)

# Visual style definitions per genre
VISUAL_STYLES = {
    "cyberpunk_neon_fractals": {
        "colors": ["#00ff41", "#ff00ff", "#00ffff", "#ff6600"],
        "style": "cyberpunk neon fractals, dark background, glowing lines",
        "animation": "fast-pulsing, bass-reactive, sharp edges",
    },
    "flowing_liquid_colors": {
        "colors": ["#4488ff", "#44ddff", "#8844ff", "#ffffff"],
        "style": "flowing liquid colors, smooth gradients, water-like motion",
        "animation": "smooth-flowing, mid-reactive, organic curves",
    },
    "glitch_art": {
        "colors": ["#ff0000", "#00ff00", "#0000ff", "#ffffff"],
        "style": "glitch art, digital distortion, fragmented visuals",
        "animation": "glitchy, bass-reactive, sharp transitions",
    },
    "warm_geometric_shapes": {
        "colors": ["#ff8844", "#ffaa00", "#ff4466", "#ffddaa"],
        "style": "warm geometric shapes, soft edges, sunset palette",
        "animation": "slow-rotating, mid-reactive, gentle pulsing",
    },
    "evolving_patterns": {
        "colors": ["#6644ff", "#44aaff", "#ff44aa", "#44ffaa"],
        "style": "evolving geometric patterns, kaleidoscope, morphing shapes",
        "animation": "gradually-evolving, all-frequency-reactive",
    },
    "cosmic_landscapes": {
        "colors": ["#0022ff", "#8800ff", "#ff00aa", "#00ffff"],
        "style": "cosmic landscapes, nebulae, star fields, aurora",
        "animation": "sweeping, treble-reactive, expansive motion",
    },
    "psychedelic_fractals": {
        "colors": ["#ff00ff", "#00ff00", "#ffff00", "#ff4400"],
        "style": "psychedelic fractals, mandala patterns, vivid colors",
        "animation": "spiraling, all-frequency-reactive, intense movement",
    },
    "minimal_dark_geometry": {
        "colors": ["#ffffff", "#888888", "#444444", "#00aaff"],
        "style": "minimal dark geometry, clean lines, monochrome with accent",
        "animation": "precise, kick-reactive, mechanical movement",
    },
    "retro_breakbeat_vhs": {
        "colors": ["#ff6600", "#ffff00", "#00ff66", "#ff0066"],
        "style": "retro VHS aesthetic, scan lines, 90s rave visuals",
        "animation": "choppy, beat-synced, nostalgic effects",
    },
    "ethereal_nature_scapes": {
        "colors": ["#228844", "#44aa88", "#88ccaa", "#aaeedd"],
        "style": "ethereal nature scapes, forests, water, mist",
        "animation": "very-slow, ambient-reactive, dreamy movement",
    },
    "dreamy_slow_motion": {
        "colors": ["#886644", "#aa8866", "#ccaa88", "#eeddcc"],
        "style": "dreamy slow motion, lo-fi aesthetic, warm film grain",
        "animation": "slow-drift, low-frequency-reactive, hazy movement",
    },
}


class VisualAgent(BaseAgent):
    """Generates audio-reactive visualizations and graphics for the stream."""

    def __init__(self):
        super().__init__("visual")

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "generate_visual")

        if task_type == "generate_visual":
            return await self.generate_visual(
                track_id=task["track_id"],
                genre=task.get("genre"),
                bpm=task.get("bpm"),
                title=task.get("title"),
            )
        elif task_type == "generate_thumbnail":
            return await self.generate_thumbnail(
                title=task.get("title", "SonicForge Radio"),
                genre=task.get("genre"),
            )
        elif task_type == "generate_overlay":
            return await self.generate_overlay_config(
                track_id=task["track_id"],
                title=task.get("title"),
                bpm=task.get("bpm"),
                key=task.get("key"),
                genre=task.get("genre"),
                score=task.get("score"),
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def generate_visual(
        self,
        track_id: str,
        genre: str | None = None,
        bpm: int | None = None,
        title: str | None = None,
    ) -> dict:
        """Generate audio-reactive visual configuration for a track."""
        visual_theme = "evolving_patterns"
        if genre:
            try:
                genre_enum = Genre(genre)
                visual_theme = GENRE_PROFILES[genre_enum]["visual_theme"]
            except (ValueError, KeyError):
                pass

        style = VISUAL_STYLES.get(visual_theme, VISUAL_STYLES["evolving_patterns"])

        # Generate shader configuration
        shader_config = {
            "theme": visual_theme,
            "colors": style["colors"],
            "style_prompt": style["style"],
            "animation_type": style["animation"],
            "bpm_sync": bpm or 128,
            "audio_reactivity": {
                "bass": {"intensity": 0.8, "smoothing": 0.3},
                "mid": {"intensity": 0.5, "smoothing": 0.5},
                "treble": {"intensity": 0.6, "smoothing": 0.4},
            },
        }

        # Generate overlay configuration
        overlay_config = {
            "track_title": title or "SonicForge Radio",
            "bpm": bpm,
            "genre": genre,
            "position": "bottom-left",
            "font": "Inter",
            "opacity": 0.85,
        }

        self.logger.info(
            "visual_generated",
            track_id=track_id,
            theme=visual_theme,
        )

        return {
            "track_id": track_id,
            "shader_config": shader_config,
            "overlay_config": overlay_config,
            "visual_theme": visual_theme,
        }

    async def generate_thumbnail(self, title: str, genre: str | None = None) -> dict:
        """Generate a YouTube thumbnail for the stream."""
        visual_theme = "evolving_patterns"
        if genre:
            try:
                genre_enum = Genre(genre)
                visual_theme = GENRE_PROFILES[genre_enum]["visual_theme"]
            except (ValueError, KeyError):
                pass

        style = VISUAL_STYLES.get(visual_theme, VISUAL_STYLES["evolving_patterns"])

        # If DALL-E/OpenAI is configured, generate via API
        if settings.openai_api_key:
            return await self._generate_ai_thumbnail(title, style)

        # Return thumbnail specification for manual/template generation
        return {
            "title": title,
            "visual_theme": visual_theme,
            "colors": style["colors"],
            "style_prompt": f"{style['style']}, centered text '{title}', YouTube thumbnail 1280x720",
            "generated": False,
        }

    async def generate_overlay_config(
        self,
        track_id: str,
        title: str | None = None,
        bpm: int | None = None,
        key: str | None = None,
        genre: str | None = None,
        score: float | None = None,
    ) -> dict:
        """Generate the on-stream overlay configuration for a track."""
        overlay = {
            "track_id": track_id,
            "elements": [
                {
                    "type": "text",
                    "content": f"♪ {title or 'SonicForge Radio'}",
                    "position": {"x": 40, "y": 980},
                    "style": {"font_size": 28, "color": "#ffffff", "shadow": True},
                },
                {
                    "type": "badge",
                    "content": f"AI Generated • {genre or 'Electronic'}",
                    "position": {"x": 40, "y": 1020},
                    "style": {"font_size": 16, "color": "#aaaaaa"},
                },
            ],
        }

        if bpm:
            overlay["elements"].append({
                "type": "text",
                "content": f"BPM: {bpm}",
                "position": {"x": 1780, "y": 980},
                "style": {"font_size": 18, "color": "#888888", "align": "right"},
            })

        if key:
            overlay["elements"].append({
                "type": "text",
                "content": f"Key: {key}",
                "position": {"x": 1780, "y": 1010},
                "style": {"font_size": 18, "color": "#888888", "align": "right"},
            })

        if score:
            overlay["elements"].append({
                "type": "badge",
                "content": f"Quality: {score}/10",
                "position": {"x": 1780, "y": 1040},
                "style": {"font_size": 14, "color": "#66ff66", "align": "right"},
            })

        return overlay

    async def _generate_ai_thumbnail(self, title: str, style: dict) -> dict:
        """Generate thumbnail using DALL-E API."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                    json={
                        "model": "dall-e-3",
                        "prompt": (
                            f"Abstract digital art for music streaming thumbnail: {style['style']}. "
                            f"No text in image. Widescreen 16:9 format. Vibrant, professional quality."
                        ),
                        "size": "1792x1024",
                        "quality": "standard",
                        "n": 1,
                    },
                )
                response.raise_for_status()
                data = response.json()
                image_url = data["data"][0]["url"]

                return {
                    "title": title,
                    "image_url": image_url,
                    "generated": True,
                    "style": style["style"],
                }
        except Exception as e:
            self.logger.error("thumbnail_generation_failed", error=str(e))
            return {
                "title": title,
                "generated": False,
                "error": str(e),
            }
