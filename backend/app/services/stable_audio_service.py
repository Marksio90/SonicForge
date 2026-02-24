"""Stable Audio Open Small — CPU-Optimized Music Generation Service.

Stable Audio Open Small (341M parameters) by Stability AI + Arm (released 2025):
- Generates 11 seconds of 44.1 kHz stereo audio
- Runs in <8 seconds on mobile-class Arm CPUs
- INT8 quantization: 2.9 GB RAM (down from 5.2 GB), peak 3.6 GB
- Ideal for ambient loops, textures, and short musical phrases
- No vocal support — instrumental/ambient only
- License: Stability AI Community License (commercial use permitted)

Best use case in SonicForge: generating filler ambient loops between
cloud-generated featured tracks on CPU-constrained hardware.

Reference: https://huggingface.co/stabilityai/stable-audio-open-small
"""

import asyncio
import io
import os
import time
import uuid

import structlog

logger = structlog.get_logger(__name__)

# How many 11-second clips to stitch together for longer tracks
DEFAULT_CLIPS_PER_TRACK = int(os.environ.get("STABLE_AUDIO_CLIPS", "8"))  # ~88s
STABLE_AUDIO_MODEL_ID = os.environ.get(
    "STABLE_AUDIO_MODEL_ID", "stabilityai/stable-audio-open-small"
)
STABLE_AUDIO_SAMPLE_RATE = 44100


class StableAudioService:
    """Local Stable Audio Open Small inference with CPU optimization.

    Loads the model lazily and caches it for the process lifetime.
    Uses INT8 dynamic quantization to reduce memory footprint from 5.2 GB
    to ~2.9 GB, making it viable on an 8 GB laptop.
    """

    _model = None
    _processor = None
    _lock: asyncio.Lock | None = None

    def __init__(self):
        self.logger = logger.bind(service="stable_audio")
        if StableAudioService._lock is None:
            StableAudioService._lock = asyncio.Lock()

    async def generate(
        self,
        prompt: str,
        duration: int = 11,
        clips: int | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Generate audio from a text prompt using Stable Audio Open Small.

        For durations longer than 11s, multiple clips are generated and
        concatenated with equal-power crossfade.

        Args:
            prompt: Text description of the desired audio
            duration: Desired duration in seconds (capped at 11s per clip)
            clips: Number of clips to generate and stitch (default from env)
            seed: Random seed (None = random per clip)

        Returns:
            WAV audio bytes at 44.1 kHz stereo
        """
        if clips is None:
            # Determine how many 11s clips needed for the requested duration
            clips = max(1, min(DEFAULT_CLIPS_PER_TRACK, -(-duration // 11)))

        self.logger.info(
            "stable_audio_generate",
            prompt_len=len(prompt),
            clips=clips,
        )

        async with StableAudioService._lock:
            model, processor = await asyncio.to_thread(self._load_model)

        # Generate clips (outside the lock so other requests can proceed)
        all_clips: list[bytes] = []
        for i in range(clips):
            clip_seed = seed if seed is not None else int(uuid.uuid4().int % (2**31))
            clip_bytes = await asyncio.to_thread(
                self._generate_clip, model, processor, prompt, clip_seed
            )
            all_clips.append(clip_bytes)

        if len(all_clips) == 1:
            return all_clips[0]

        # Stitch clips with crossfade
        return await asyncio.to_thread(self._stitch_clips, all_clips)

    def _load_model(self):
        """Lazy-load and cache the Stable Audio Open Small model with INT8 quantization."""
        if StableAudioService._model is not None:
            return StableAudioService._model, StableAudioService._processor

        import os as _os

        # Optimize CPU thread usage before model load
        cpu_count = _os.cpu_count() or 4
        thread_count = max(1, cpu_count // 2)

        try:
            import torch

            torch.set_num_threads(thread_count)
            _os.environ.setdefault("MKL_NUM_THREADS", str(thread_count))
            _os.environ.setdefault("OMP_NUM_THREADS", str(thread_count))
            self.logger.info("cpu_threads_set", threads=thread_count)
        except ImportError:
            pass

        self.logger.info("loading_stable_audio_model", model=STABLE_AUDIO_MODEL_ID)
        start = time.monotonic()

        try:
            from diffusers import StableAudioPipeline
            import torch

            pipe = StableAudioPipeline.from_pretrained(
                STABLE_AUDIO_MODEL_ID,
                torch_dtype=torch.float32,  # float32 for CPU compatibility
            )

            # INT8 dynamic quantization — reduces from 5.2 GB to ~2.9 GB
            try:
                pipe.unet = torch.quantization.quantize_dynamic(
                    pipe.unet,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                self.logger.info("stable_audio_quantized_int8")
            except Exception as e:
                self.logger.warning("stable_audio_quantization_skipped", error=str(e))

            StableAudioService._model = pipe
            StableAudioService._processor = None  # Integrated in pipeline

            elapsed = time.monotonic() - start
            self.logger.info("stable_audio_model_loaded", elapsed_s=round(elapsed, 1))
            return pipe, None

        except ImportError as e:
            raise RuntimeError(
                "diffusers package not installed. "
                "Run: pip install diffusers transformers accelerate"
            ) from e

    def _generate_clip(self, pipe, processor, prompt: str, seed: int) -> bytes:
        """Generate a single 11-second clip synchronously (runs in thread pool)."""
        import torch

        generator = torch.Generator().manual_seed(seed)

        with torch.inference_mode():
            output = pipe(
                prompt,
                negative_prompt="low quality, noisy, distorted",
                num_inference_steps=50,
                audio_length_in_s=11.0,
                num_waveforms_per_prompt=1,
                generator=generator,
            )

        audio_tensor = output.audios[0]  # Shape: (channels, samples)

        # Convert to WAV bytes at 44.1 kHz
        import soundfile as sf

        audio_np = audio_tensor.cpu().numpy().T  # (samples, channels)
        buf = io.BytesIO()
        sf.write(buf, audio_np, STABLE_AUDIO_SAMPLE_RATE, format="WAV")
        return buf.getvalue()

    def _stitch_clips(self, clips: list[bytes]) -> bytes:
        """Concatenate clips with equal-power crossfade (research-recommended approach)."""
        import numpy as np
        import soundfile as sf

        crossfade_s = 1.5  # seconds of overlap between clips
        sr = STABLE_AUDIO_SAMPLE_RATE
        crossfade_samples = int(sr * crossfade_s)

        arrays = []
        for clip_bytes in clips:
            buf = io.BytesIO(clip_bytes)
            data, _ = sf.read(buf, dtype="float32")
            arrays.append(data)

        if not arrays:
            return b""

        result = arrays[0]
        for nxt in arrays[1:]:
            if len(result) < crossfade_samples or len(nxt) < crossfade_samples:
                result = np.concatenate([result, nxt])
                continue

            # Equal-power crossfade curves (sqrt for perceptual linearity)
            fade_out = np.sqrt(np.linspace(1.0, 0.0, crossfade_samples))
            fade_in = np.sqrt(np.linspace(0.0, 1.0, crossfade_samples))

            if result.ndim == 2:
                fade_out = fade_out[:, np.newaxis]
                fade_in = fade_in[:, np.newaxis]

            overlap = (
                result[-crossfade_samples:] * fade_out
                + nxt[:crossfade_samples] * fade_in
            )
            result = np.concatenate([result[:-crossfade_samples], overlap, nxt[crossfade_samples:]])

        buf = io.BytesIO()
        sf.write(buf, result, sr, format="WAV")
        return buf.getvalue()

    async def health(self) -> dict:
        """Return service health including model load status."""
        return {
            "service": "stable_audio_open_small",
            "model": STABLE_AUDIO_MODEL_ID,
            "model_loaded": StableAudioService._model is not None,
            "clips_per_track": DEFAULT_CLIPS_PER_TRACK,
            "sample_rate": STABLE_AUDIO_SAMPLE_RATE,
        }
