"""Audio mastering engine for broadcast-quality output.

Provides LUFS normalization, true peak limiting, multiband compression,
stereo widening, and high-pass filtering to ensure all generated tracks
meet professional radio/streaming standards.
"""

import asyncio
import io
import struct
import math

import numpy as np
import structlog

from ..core.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class MasteringEngine:
    """Professional audio mastering pipeline for AI-generated tracks."""

    def __init__(self):
        self.target_lufs = settings.mastering_target_lufs
        self.true_peak = settings.mastering_true_peak
        self.stereo_width = settings.mastering_stereo_width
        self.logger = logger.bind(component="mastering_engine")

    async def master_audio(self, audio_data: bytes) -> bytes:
        """Apply full mastering chain to raw audio data.

        Pipeline:
        1. High-pass filter (30 Hz) — remove sub-bass rumble
        2. Multiband compression — even out frequency balance
        3. Stereo widening — enhance spatial image
        4. LUFS normalization (-14 LUFS for YouTube standard)
        5. True peak limiter (-1.0 dBTP) — prevent clipping
        """
        if len(audio_data) < 1000:
            return audio_data

        return await asyncio.to_thread(self._master_sync, audio_data)

    def _master_sync(self, audio_data: bytes) -> bytes:
        """Synchronous mastering pipeline (runs in thread pool)."""
        try:
            samples = self._decode_audio(audio_data)
            if samples is None or len(samples) == 0:
                return audio_data

            # Normalize to float [-1, 1]
            samples = samples.astype(np.float64)
            max_val = np.max(np.abs(samples))
            if max_val > 0:
                samples = samples / max_val

            # Step 1: High-pass filter at 30 Hz
            samples = self._apply_highpass(samples, cutoff=30, sample_rate=44100)

            # Step 2: Multiband compression
            samples = self._apply_multiband_compression(samples, sample_rate=44100)

            # Step 3: Stereo widening (if stereo)
            if samples.ndim == 2 and samples.shape[1] == 2:
                samples = self._apply_stereo_widening(samples, width=self.stereo_width)

            # Step 4: LUFS normalization
            samples = self._normalize_lufs(samples, target_lufs=self.target_lufs)

            # Step 5: True peak limiter
            samples = self._apply_limiter(samples, ceiling_db=self.true_peak)

            # Encode back to bytes
            return self._encode_audio(samples)

        except Exception as e:
            self.logger.error("mastering_failed", error=str(e))
            return audio_data

    def _decode_audio(self, audio_data: bytes) -> np.ndarray | None:
        """Decode WAV audio bytes to numpy array."""
        try:
            # Try pyloudnorm/soundfile first
            import soundfile as sf

            audio_io = io.BytesIO(audio_data)
            samples, _ = sf.read(audio_io)
            return samples
        except Exception:
            pass

        # Fallback: try raw PCM interpretation (16-bit stereo 44100 Hz)
        try:
            if len(audio_data) < 44:
                return None
            # Skip WAV header if present
            offset = 44 if audio_data[:4] == b"RIFF" else 0
            pcm = audio_data[offset:]
            num_samples = len(pcm) // 4  # 2 channels * 2 bytes
            if num_samples == 0:
                return None
            samples = np.frombuffer(pcm[: num_samples * 4], dtype=np.int16)
            samples = samples.reshape(-1, 2)
            return samples.astype(np.float64) / 32768.0
        except Exception:
            return None

    def _encode_audio(self, samples: np.ndarray) -> bytes:
        """Encode numpy array back to WAV bytes."""
        try:
            import soundfile as sf

            audio_io = io.BytesIO()
            # Clip to prevent overflow
            samples = np.clip(samples, -1.0, 1.0)
            sf.write(audio_io, samples, 44100, format="WAV", subtype="PCM_16")
            return audio_io.getvalue()
        except Exception:
            # Fallback: raw WAV encoding
            samples = np.clip(samples, -1.0, 1.0)
            int_samples = (samples * 32767).astype(np.int16)
            pcm_data = int_samples.tobytes()

            channels = 2 if samples.ndim == 2 else 1
            sample_rate = 44100
            bits_per_sample = 16
            byte_rate = sample_rate * channels * bits_per_sample // 8
            block_align = channels * bits_per_sample // 8

            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",
                36 + len(pcm_data),
                b"WAVE",
                b"fmt ",
                16,
                1,  # PCM
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
                b"data",
                len(pcm_data),
            )
            return header + pcm_data

    def _apply_highpass(
        self, samples: np.ndarray, cutoff: float, sample_rate: int
    ) -> np.ndarray:
        """Apply a simple first-order high-pass filter."""
        from scipy.signal import butter, sosfilt

        nyquist = sample_rate / 2
        if cutoff >= nyquist:
            return samples
        sos = butter(2, cutoff / nyquist, btype="high", output="sos")
        if samples.ndim == 2:
            for ch in range(samples.shape[1]):
                samples[:, ch] = sosfilt(sos, samples[:, ch])
        else:
            samples = sosfilt(sos, samples)
        return samples

    def _apply_multiband_compression(
        self, samples: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Apply 3-band compression (low/mid/high) for balanced frequency output."""
        from scipy.signal import butter, sosfilt

        nyquist = sample_rate / 2
        low_cut = min(200 / nyquist, 0.99)
        high_cut = min(4000 / nyquist, 0.99)

        # Split into bands
        sos_low = butter(2, low_cut, btype="low", output="sos")
        sos_mid = butter(2, [low_cut, high_cut], btype="band", output="sos")
        sos_high = butter(2, high_cut, btype="high", output="sos")

        def compress_band(band: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
            """Simple soft-knee compressor."""
            result = band.copy()
            mask = np.abs(result) > threshold
            result[mask] = np.sign(result[mask]) * (
                threshold + (np.abs(result[mask]) - threshold) / ratio
            )
            return result

        if samples.ndim == 2:
            output = np.zeros_like(samples)
            for ch in range(samples.shape[1]):
                low = sosfilt(sos_low, samples[:, ch])
                mid = sosfilt(sos_mid, samples[:, ch])
                high = sosfilt(sos_high, samples[:, ch])

                low = compress_band(low, threshold=0.5, ratio=3.0)
                mid = compress_band(mid, threshold=0.4, ratio=2.5)
                high = compress_band(high, threshold=0.3, ratio=2.0)

                output[:, ch] = low + mid + high
        else:
            low = sosfilt(sos_low, samples)
            mid = sosfilt(sos_mid, samples)
            high = sosfilt(sos_high, samples)

            low = compress_band(low, threshold=0.5, ratio=3.0)
            mid = compress_band(mid, threshold=0.4, ratio=2.5)
            high = compress_band(high, threshold=0.3, ratio=2.0)

            output = low + mid + high

        return output

    def _apply_stereo_widening(
        self, samples: np.ndarray, width: float
    ) -> np.ndarray:
        """Widen stereo image using mid/side processing."""
        if samples.ndim != 2 or samples.shape[1] != 2:
            return samples

        mid = (samples[:, 0] + samples[:, 1]) / 2
        side = (samples[:, 0] - samples[:, 1]) / 2

        # Apply width factor to side channel
        side = side * width

        samples[:, 0] = mid + side
        samples[:, 1] = mid - side

        return samples

    def _normalize_lufs(
        self, samples: np.ndarray, target_lufs: float
    ) -> np.ndarray:
        """Normalize audio to target LUFS (integrated loudness)."""
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(44100)
            current_lufs = meter.integrated_loudness(samples)

            if np.isinf(current_lufs) or np.isnan(current_lufs):
                return samples

            gain_db = target_lufs - current_lufs
            gain_linear = 10 ** (gain_db / 20.0)
            return samples * gain_linear
        except Exception:
            # Fallback: simple RMS normalization
            rms = np.sqrt(np.mean(samples**2))
            if rms == 0:
                return samples
            target_rms = 10 ** (target_lufs / 20.0) * 0.1
            return samples * (target_rms / rms)

    def _apply_limiter(
        self, samples: np.ndarray, ceiling_db: float
    ) -> np.ndarray:
        """Apply a brickwall true peak limiter."""
        ceiling_linear = 10 ** (ceiling_db / 20.0)
        peak = np.max(np.abs(samples))

        if peak > ceiling_linear:
            gain = ceiling_linear / peak
            samples = samples * gain

        return np.clip(samples, -ceiling_linear, ceiling_linear)
