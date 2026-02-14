import json

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import GENRE_PROFILES, Genre, get_settings
from ..core.storage import upload_visual
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)

# Shared httpx client for connection pooling
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=120, write=30, pool=10),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _http_client


# Visual style definitions per genre
VISUAL_STYLES = {
    "cyberpunk_neon_fractals": {
        "colors": ["#00ff41", "#ff00ff", "#00ffff", "#ff6600"],
        "style": "cyberpunk neon fractals, dark background, glowing lines",
        "animation": "fast-pulsing, bass-reactive, sharp edges",
        "sd_prompt": "cyberpunk neon fractal art, dark void background, glowing electric lines, "
        "digital matrix aesthetic, high contrast, octane render, 4k",
        "sd_negative": "text, watermark, blurry, low quality, human faces",
    },
    "flowing_liquid_colors": {
        "colors": ["#4488ff", "#44ddff", "#8844ff", "#ffffff"],
        "style": "flowing liquid colors, smooth gradients, water-like motion",
        "animation": "smooth-flowing, mid-reactive, organic curves",
        "sd_prompt": "flowing liquid abstract art, smooth color gradients, water-like motion, "
        "blue and purple palette, ethereal, dreamlike, 4k",
        "sd_negative": "text, watermark, sharp edges, geometric, human faces",
    },
    "glitch_art": {
        "colors": ["#ff0000", "#00ff00", "#0000ff", "#ffffff"],
        "style": "glitch art, digital distortion, fragmented visuals",
        "animation": "glitchy, bass-reactive, sharp transitions",
        "sd_prompt": "digital glitch art, RGB channel separation, pixel sorting, "
        "data corruption aesthetic, vaporwave, 4k",
        "sd_negative": "text, watermark, blurry, human faces, realistic",
    },
    "warm_geometric_shapes": {
        "colors": ["#ff8844", "#ffaa00", "#ff4466", "#ffddaa"],
        "style": "warm geometric shapes, soft edges, sunset palette",
        "animation": "slow-rotating, mid-reactive, gentle pulsing",
        "sd_prompt": "warm geometric abstract art, sunset colors, soft golden light, "
        "floating shapes, minimal, serene, 4k",
        "sd_negative": "text, watermark, dark, cold colors, human faces",
    },
    "evolving_patterns": {
        "colors": ["#6644ff", "#44aaff", "#ff44aa", "#44ffaa"],
        "style": "evolving geometric patterns, kaleidoscope, morphing shapes",
        "animation": "gradually-evolving, all-frequency-reactive",
        "sd_prompt": "evolving kaleidoscope pattern, morphing geometric shapes, "
        "vibrant multi-color, symmetrical, abstract, 4k",
        "sd_negative": "text, watermark, blurry, human faces",
    },
    "cosmic_landscapes": {
        "colors": ["#0022ff", "#8800ff", "#ff00aa", "#00ffff"],
        "style": "cosmic landscapes, nebulae, star fields, aurora",
        "animation": "sweeping, treble-reactive, expansive motion",
        "sd_prompt": "cosmic nebula landscape, deep space, star fields, aurora borealis, "
        "galaxy, vibrant colors, cinematic, 4k",
        "sd_negative": "text, watermark, human faces, earth, buildings",
    },
    "psychedelic_fractals": {
        "colors": ["#ff00ff", "#00ff00", "#ffff00", "#ff4400"],
        "style": "psychedelic fractals, mandala patterns, vivid colors",
        "animation": "spiraling, all-frequency-reactive, intense movement",
        "sd_prompt": "psychedelic fractal mandala, vivid neon colors, spiraling patterns, "
        "sacred geometry, trippy, 4k",
        "sd_negative": "text, watermark, dull colors, human faces, realistic",
    },
    "minimal_dark_geometry": {
        "colors": ["#ffffff", "#888888", "#444444", "#00aaff"],
        "style": "minimal dark geometry, clean lines, monochrome with accent",
        "animation": "precise, kick-reactive, mechanical movement",
        "sd_prompt": "minimal dark geometric art, clean lines, monochrome with blue accent, "
        "architectural, futuristic, 4k",
        "sd_negative": "text, watermark, colorful, organic, human faces",
    },
    "retro_breakbeat_vhs": {
        "colors": ["#ff6600", "#ffff00", "#00ff66", "#ff0066"],
        "style": "retro VHS aesthetic, scan lines, 90s rave visuals",
        "animation": "choppy, beat-synced, nostalgic effects",
        "sd_prompt": "retro VHS aesthetic, 90s rave visual, scan lines, CRT monitor, "
        "neon colors, nostalgic, lo-fi, 4k",
        "sd_negative": "text, watermark, modern, clean, human faces",
    },
    "ethereal_nature_scapes": {
        "colors": ["#228844", "#44aa88", "#88ccaa", "#aaeedd"],
        "style": "ethereal nature scapes, forests, water, mist",
        "animation": "very-slow, ambient-reactive, dreamy movement",
        "sd_prompt": "ethereal misty forest landscape, morning dew, soft light, "
        "dreamy atmosphere, impressionist, 4k",
        "sd_negative": "text, watermark, human faces, buildings, urban",
    },
    "dreamy_slow_motion": {
        "colors": ["#886644", "#aa8866", "#ccaa88", "#eeddcc"],
        "style": "dreamy slow motion, lo-fi aesthetic, warm film grain",
        "animation": "slow-drift, low-frequency-reactive, hazy movement",
        "sd_prompt": "dreamy lo-fi aesthetic, warm film grain, soft focus, "
        "golden hour, vintage photography, 4k",
        "sd_negative": "text, watermark, sharp, digital, human faces",
    },
}


class VisualAgent(BaseAgent):
    """Generates audio-reactive visualizations using local Stable Diffusion / ComfyUI.

    Supports both static thumbnails and looping AnimateDiff video for live streams.
    Falls back to OpenAI DALL-E if local SD is unavailable.
    """

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
        elif task_type == "generate_video_loop":
            return await self.generate_video_loop(
                track_id=task["track_id"],
                genre=task.get("genre"),
                bpm=task.get("bpm"),
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
        """Generate audio-reactive visual configuration and thumbnail for a track."""
        visual_theme = "evolving_patterns"
        if genre:
            try:
                genre_enum = Genre(genre)
                visual_theme = GENRE_PROFILES[genre_enum]["visual_theme"]
            except (ValueError, KeyError):
                pass

        style = VISUAL_STYLES.get(visual_theme, VISUAL_STYLES["evolving_patterns"])

        # BPM-adaptive reactivity parameters
        bpm_val = bpm or 128
        bass_intensity = min(1.0, 0.6 + (bpm_val - 100) / 200)
        treble_intensity = min(1.0, 0.4 + (bpm_val - 100) / 250)

        shader_config = {
            "theme": visual_theme,
            "colors": style["colors"],
            "style_prompt": style["style"],
            "animation_type": style["animation"],
            "bpm_sync": bpm_val,
            "audio_reactivity": {
                "bass": {"intensity": round(bass_intensity, 2), "smoothing": 0.3},
                "mid": {"intensity": 0.5, "smoothing": 0.5},
                "treble": {"intensity": round(treble_intensity, 2), "smoothing": 0.4},
            },
        }

        # Generate thumbnail via Stable Diffusion
        thumbnail_result = await self._generate_sd_thumbnail(title or "SonicForge", style)

        # Generate video loop if AnimateDiff is enabled
        video_result = None
        if settings.animatediff_enabled:
            video_result = await self._generate_video_loop(style, bpm_val)

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
            has_thumbnail=thumbnail_result.get("generated", False),
            has_video=video_result is not None,
        )

        return {
            "track_id": track_id,
            "shader_config": shader_config,
            "overlay_config": overlay_config,
            "visual_theme": visual_theme,
            "thumbnail": thumbnail_result,
            "video_loop": video_result,
        }

    async def generate_thumbnail(self, title: str, genre: str | None = None) -> dict:
        """Generate a thumbnail using Stable Diffusion (primary) or DALL-E (fallback)."""
        visual_theme = "evolving_patterns"
        if genre:
            try:
                genre_enum = Genre(genre)
                visual_theme = GENRE_PROFILES[genre_enum]["visual_theme"]
            except (ValueError, KeyError):
                pass

        style = VISUAL_STYLES.get(visual_theme, VISUAL_STYLES["evolving_patterns"])

        # Try Stable Diffusion first
        sd_result = await self._generate_sd_thumbnail(title, style)
        if sd_result.get("generated"):
            return sd_result

        # Fallback to OpenAI DALL-E
        if settings.openai_api_key:
            return await self._generate_dalle_thumbnail(title, style)

        return {
            "title": title,
            "visual_theme": visual_theme,
            "colors": style["colors"],
            "style_prompt": f"{style['style']}, centered text '{title}', YouTube thumbnail 1280x720",
            "generated": False,
        }

    async def generate_video_loop(
        self,
        track_id: str,
        genre: str | None = None,
        bpm: int | None = None,
    ) -> dict:
        """Generate a looping video using AnimateDiff synchronized with BPM."""
        visual_theme = "evolving_patterns"
        if genre:
            try:
                genre_enum = Genre(genre)
                visual_theme = GENRE_PROFILES[genre_enum]["visual_theme"]
            except (ValueError, KeyError):
                pass

        style = VISUAL_STYLES.get(visual_theme, VISUAL_STYLES["evolving_patterns"])
        bpm_val = bpm or 128

        result = await self._generate_video_loop(style, bpm_val)
        if result:
            result["track_id"] = track_id
        else:
            result = {"track_id": track_id, "generated": False}

        return result

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _generate_sd_thumbnail(self, title: str, style: dict) -> dict:
        """Generate thumbnail using local Stable Diffusion (A1111/ComfyUI API)."""
        try:
            client = _get_http_client()

            sd_prompt = style.get(
                "sd_prompt", f"abstract digital art, {style['style']}"
            )
            sd_negative = style.get(
                "sd_negative", "text, watermark, human faces, blurry"
            )

            # A1111 WebUI API format (also supported by ComfyUI with a1111 compat)
            response = await client.post(
                f"{settings.stable_diffusion_url}/sdapi/v1/txt2img",
                json={
                    "prompt": f"{sd_prompt}, music streaming thumbnail, widescreen 16:9",
                    "negative_prompt": sd_negative,
                    "width": 1280,
                    "height": 720,
                    "steps": 30,
                    "cfg_scale": 7.5,
                    "sampler_name": "DPM++ 2M Karras",
                    "batch_size": 1,
                },
            )
            response.raise_for_status()
            data = response.json()

            images = data.get("images", [])
            if images:
                # Upload the base64 image to S3
                import base64

                image_data = base64.b64decode(images[0])
                s3_key = upload_visual(f"thumb_{title[:20]}", image_data, "png")

                return {
                    "title": title,
                    "generated": True,
                    "engine": "stable_diffusion",
                    "s3_key": s3_key,
                    "prompt": sd_prompt,
                }

        except Exception as e:
            self.logger.warning("sd_thumbnail_failed", error=str(e))

        return {"title": title, "generated": False, "engine": "stable_diffusion"}

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _generate_video_loop(self, style: dict, bpm: int) -> dict | None:
        """Generate a looping video using AnimateDiff via ComfyUI API.

        Creates an 8-second BPM-synchronized video loop for the live stream.
        """
        if not settings.animatediff_enabled:
            return None

        try:
            client = _get_http_client()

            sd_prompt = style.get(
                "sd_prompt", f"abstract digital art, {style['style']}"
            )
            sd_negative = style.get(
                "sd_negative", "text, watermark, human faces, blurry"
            )

            # Calculate frames for BPM sync
            fps = 16
            loop_duration = settings.visual_loop_duration
            total_frames = fps * loop_duration

            # Beats per loop = BPM / 60 * duration
            beats_per_loop = bpm / 60 * loop_duration

            response = await client.post(
                f"{settings.stable_diffusion_url}/api/prompt",
                json={
                    "prompt": {
                        "3": {
                            "class_type": "AnimateDiffLoader",
                            "inputs": {
                                "model_name": "mm_sd_v15_v2.ckpt",
                                "beta_schedule": "linear",
                            },
                        },
                        "5": {
                            "class_type": "KSampler",
                            "inputs": {
                                "seed": int(bpm * 1000),
                                "steps": 20,
                                "cfg": 7.5,
                                "sampler_name": "euler_ancestral",
                                "scheduler": "normal",
                                "denoise": 1.0,
                            },
                        },
                        "6": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {
                                "text": f"{sd_prompt}, smooth motion, looping animation, "
                                f"synchronized to {bpm} BPM rhythm",
                            },
                        },
                        "7": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": sd_negative},
                        },
                        "10": {
                            "class_type": "EmptyLatentImage",
                            "inputs": {
                                "width": 1920,
                                "height": 1080,
                                "batch_size": total_frames,
                            },
                        },
                    }
                },
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "generated": True,
                    "engine": "animatediff",
                    "fps": fps,
                    "duration": loop_duration,
                    "frames": total_frames,
                    "bpm_sync": bpm,
                    "beats_per_loop": round(beats_per_loop, 1),
                    "prompt_id": result.get("prompt_id"),
                }

        except Exception as e:
            self.logger.warning("animatediff_failed", error=str(e))

        return None

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _generate_dalle_thumbnail(self, title: str, style: dict) -> dict:
        """Generate thumbnail using OpenAI DALL-E 3 (fallback)."""
        try:
            client = _get_http_client()
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
                    "quality": "hd",
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
                "engine": "dall-e-3",
                "style": style["style"],
            }
        except Exception as e:
            self.logger.error("dalle_thumbnail_failed", error=str(e))
            return {
                "title": title,
                "generated": False,
                "engine": "dall-e-3",
                "error": str(e),
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
                    "content": f"{title or 'SonicForge Radio'}",
                    "position": {"x": 40, "y": 980},
                    "style": {"font_size": 28, "color": "#ffffff", "shadow": True},
                },
                {
                    "type": "badge",
                    "content": f"AI Generated | {genre or 'Electronic'}",
                    "position": {"x": 40, "y": 1020},
                    "style": {"font_size": 16, "color": "#aaaaaa"},
                },
            ],
        }

        if bpm:
            overlay["elements"].append(
                {
                    "type": "text",
                    "content": f"BPM: {bpm}",
                    "position": {"x": 1780, "y": 980},
                    "style": {"font_size": 18, "color": "#888888", "align": "right"},
                }
            )

        if key:
            overlay["elements"].append(
                {
                    "type": "text",
                    "content": f"Key: {key}",
                    "position": {"x": 1780, "y": 1010},
                    "style": {"font_size": 18, "color": "#888888", "align": "right"},
                }
            )

        if score:
            overlay["elements"].append(
                {
                    "type": "badge",
                    "content": f"Quality: {score}/10",
                    "position": {"x": 1780, "y": 1040},
                    "style": {
                        "font_size": 14,
                        "color": "#66ff66",
                        "align": "right",
                    },
                }
            )

        return overlay
