"""Professional Audio Mastering Chain with Advanced DSP.

Provides studio-quality mastering with:
- De-esser (harsh frequency removal)
- Multiband compressor (4-band dynamics)
- Harmonic exciter (warmth enhancement)
- Stereo enhancer (width control)
- Limiter (loudness maximization)
- LUFS normalization
"""

import numpy as np
from scipy import signal
import structlog

logger = structlog.get_logger(__name__)


class ProMasteringEngine:
    """Professional mastering with multiband dynamics and harmonic enhancement."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.target_lufs = -14.0  # Streaming standard
        
        # Mastering chain order
        self.chain = [
            self.normalize_input,
            self.de_esser,
            self.multiband_compressor,
            self.harmonic_exciter,
            self.stereo_enhancer,
            self.limiter,
            self.lufs_normalize,
        ]
        
        logger.info("mastering_engine_initialized", sample_rate=sample_rate)
    
    def master(self, audio: np.ndarray) -> np.ndarray:
        """Apply full mastering chain.
        
        Args:
            audio: Audio array (stereo: [channels, samples] or mono: [samples])
        
        Returns:
            Mastered audio
        """
        logger.info("mastering_started", shape=audio.shape)
        
        # Ensure stereo format
        if audio.ndim == 1:
            audio = np.array([audio, audio])
        
        # Apply each processor in chain
        for processor in self.chain:
            audio = processor(audio)
        
        logger.info("mastering_complete")
        return audio
    
    def normalize_input(self, audio: np.ndarray) -> np.ndarray:
        """Normalize input to prevent clipping."""
        peak = np.abs(audio).max()
        if peak > 0.95:
            audio = audio * (0.95 / peak)
        return audio
    
    def de_esser(self, audio: np.ndarray, frequency: float = 6000, reduction_db: float = -3) -> np.ndarray:
        """Remove harsh high frequencies (sibilance).
        
        Args:
            audio: Input audio
            frequency: Center frequency for de-essing (Hz)
            reduction_db: Amount of reduction in dB
        """
        # Design high-shelf filter
        nyquist = self.sample_rate / 2
        normalized_freq = frequency / nyquist
        
        # High-shelf filter coefficients
        b, a = signal.iirfilter(
            4,
            normalized_freq,
            btype='highpass',
            ftype='butter',
            output='ba',
        )
        
        # Apply filter to each channel
        filtered = np.array([signal.filtfilt(b, a, channel) for channel in audio])
        
        # Blend: reduce high frequencies by reduction_db
        blend_factor = 10 ** (reduction_db / 20)
        result = audio * 0.7 + filtered * blend_factor * 0.3
        
        return result
    
    def multiband_compressor(self, audio: np.ndarray) -> np.ndarray:
        """4-band dynamics processing.
        
        Bands:
        - Low: 0-250 Hz (bass)
        - Low-mid: 250-2000 Hz (body)
        - High-mid: 2000-6000 Hz (presence)
        - High: 6000+ Hz (air)
        """
        bands = [
            (0, 250, 3.0, -20),      # Low: ratio 3:1, threshold -20dB
            (250, 2000, 2.5, -18),   # Low-mid: ratio 2.5:1, threshold -18dB
            (2000, 6000, 4.0, -15),  # High-mid: ratio 4:1, threshold -15dB
            (6000, 22000, 2.0, -12), # High: ratio 2:1, threshold -12dB
        ]
        
        result = np.zeros_like(audio)
        
        for low, high, ratio, threshold_db in bands:
            # Create band-pass filter
            band_audio = self._bandpass_filter(audio, low, high)
            
            # Apply compression
            compressed = self._compress(band_audio, ratio, threshold_db)
            
            # Sum bands
            result += compressed
        
        # Normalize to prevent clipping
        result = result / 4.0
        
        return result
    
    def _bandpass_filter(self, audio: np.ndarray, low: float, high: float) -> np.ndarray:
        """Apply band-pass filter."""
        nyquist = self.sample_rate / 2
        
        if low == 0:
            # Low-pass only
            b, a = signal.butter(4, high / nyquist, btype='low')
        elif high >= nyquist:
            # High-pass only
            b, a = signal.butter(4, low / nyquist, btype='high')
        else:
            # Band-pass
            b, a = signal.butter(4, [low / nyquist, high / nyquist], btype='band')
        
        return np.array([signal.filtfilt(b, a, channel) for channel in audio])
    
    def _compress(self, audio: np.ndarray, ratio: float, threshold_db: float) -> np.ndarray:
        """Apply dynamic range compression."""
        threshold_linear = 10 ** (threshold_db / 20)
        
        compressed = np.copy(audio)
        
        for i, channel in enumerate(audio):
            # Calculate envelope
            envelope = np.abs(channel)
            
            # Apply compression where signal exceeds threshold
            mask = envelope > threshold_linear
            excess = envelope[mask] - threshold_linear
            reduction = excess * (1 - 1/ratio)
            
            # Apply gain reduction
            gain = np.ones_like(channel)
            gain[mask] = (threshold_linear + excess - reduction) / envelope[mask]
            
            compressed[i] = channel * gain
        
        return compressed
    
    def harmonic_exciter(self, audio: np.ndarray, amount: float = 0.2) -> np.ndarray:
        """Add harmonic excitement for warmth.
        
        Args:
            audio: Input audio
            amount: Amount of excitation (0-1)
        """
        # Apply soft clipping to generate harmonics
        excited = np.tanh(audio * 2.0) * 0.5
        
        # Blend with original
        return audio * (1 - amount) + excited * amount
    
    def stereo_enhancer(self, audio: np.ndarray, width: float = 1.3) -> np.ndarray:
        """Enhance stereo width.
        
        Args:
            audio: Stereo audio [2, samples]
            width: Width multiplier (1.0 = no change, >1.0 = wider)
        """
        if audio.shape[0] != 2:
            return audio
        
        # Extract mid (mono) and side (difference)
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2
        
        # Enhance side signal
        side = side * width
        
        # Reconstruct stereo
        left = mid + side
        right = mid - side
        
        return np.array([left, right])
    
    def limiter(self, audio: np.ndarray, threshold_db: float = -0.3, release_ms: float = 50) -> np.ndarray:
        """Brick-wall limiter to prevent clipping.
        
        Args:
            audio: Input audio
            threshold_db: Limiter threshold in dB
            release_ms: Release time in milliseconds
        """
        threshold_linear = 10 ** (threshold_db / 20)
        release_samples = int(release_ms * self.sample_rate / 1000)
        
        limited = np.copy(audio)
        
        for i, channel in enumerate(audio):
            envelope = np.abs(channel)
            gain = np.ones_like(channel)
            
            # Calculate gain reduction
            mask = envelope > threshold_linear
            gain[mask] = threshold_linear / envelope[mask]
            
            # Smooth gain reduction (attack/release)
            gain = signal.medfilt(gain, kernel_size=min(release_samples, len(gain)) | 1)
            
            limited[i] = channel * gain
        
        return limited
    
    def lufs_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize to target LUFS.
        
        Uses simplified loudness estimation.
        """
        try:
            import pyloudnorm as pyln
            
            meter = pyln.Meter(self.sample_rate)
            
            # Transpose for pyloudnorm (expects [samples, channels])
            audio_t = audio.T
            
            # Measure loudness
            loudness = meter.integrated_loudness(audio_t)
            
            # Normalize
            normalized = pyln.normalize.loudness(audio_t, loudness, self.target_lufs)
            
            # Transpose back
            return normalized.T
            
        except ImportError:
            logger.warning("pyloudnorm_not_available_using_peak_normalization")
            # Fallback: peak normalization
            peak = np.abs(audio).max()
            if peak > 0:
                target_peak = 10 ** (self.target_lufs / 20)
                return audio * (target_peak / peak)
            return audio
    
    def analyze_audio(self, audio: np.ndarray) -> dict:
        """Analyze audio characteristics.
        
        Returns:
            Dict with peak, RMS, dynamic range, etc.
        """
        peak_db = 20 * np.log10(np.abs(audio).max() + 1e-10)
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        return {
            "peak_db": round(peak_db, 2),
            "rms_db": round(rms_db, 2),
            "dynamic_range_db": round(peak_db - rms_db, 2),
            "sample_rate": self.sample_rate,
            "duration_seconds": audio.shape[-1] / self.sample_rate,
        }
