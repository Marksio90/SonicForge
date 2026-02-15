"""Real-time Audio Analysis and Visualization.

Provides:
- FFT spectrum analysis
- Peak detection
- RMS measurement
- Spectral centroid
- WebSocket streaming for dashboard
"""

import numpy as np
from scipy import signal
import asyncio
import structlog
from typing import Optional

logger = structlog.get_logger(__name__)


class RealtimeAudioAnalyzer:
    """Real-time FFT analysis and visualization data generation."""
    
    def __init__(self, sample_rate: int = 44100, fft_size: int = 2048):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = fft_size // 4
        
        # Frequency bins
        self.frequencies = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
        # Window function for FFT
        self.window = signal.windows.hann(fft_size)
        
        logger.info(
            "realtime_analyzer_initialized",
            sample_rate=sample_rate,
            fft_size=fft_size,
        )
    
    async def analyze_stream(self, audio_stream):
        """Analyze audio stream and yield real-time data.
        
        Args:
            audio_stream: Async iterator of audio chunks
        
        Yields:
            Analysis data dict
        """
        async for chunk in audio_stream:
            if len(chunk) >= self.fft_size:
                analysis = self.analyze_chunk(chunk)
                yield analysis
    
    def analyze_chunk(self, chunk: np.ndarray) -> dict:
        """Analyze a single audio chunk.
        
        Args:
            chunk: Audio samples (mono or stereo)
        
        Returns:
            Analysis data dict
        """
        # Convert to mono if stereo
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=0)
        
        # Apply window
        windowed = chunk[:self.fft_size] * self.window
        
        # Compute FFT
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Calculate metrics
        peak_db = float(np.max(magnitude_db))
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        rms_db = float(20 * np.log10(rms + 1e-10))
        
        # Spectral centroid (brightness)
        spectral_centroid = float(
            np.sum(self.frequencies * magnitude) / (np.sum(magnitude) + 1e-10)
        )
        
        # Find dominant frequencies (peaks)
        peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.3)
        dominant_freqs = self.frequencies[peaks[:5]].tolist()  # Top 5
        
        return {
            "spectrum": magnitude_db[::8].tolist(),  # Downsample for bandwidth
            "frequencies": self.frequencies[::8].tolist(),
            "peak_db": peak_db,
            "rms_db": rms_db,
            "spectral_centroid": spectral_centroid,
            "dominant_frequencies": dominant_freqs,
        }
    
    def analyze_full_track(self, audio: np.ndarray) -> dict:
        """Comprehensive analysis of full track.
        
        Args:
            audio: Full audio array
        
        Returns:
            Detailed analysis dict
        """
        # Convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Compute spectrogram
        frequencies, times, spectrogram = signal.spectrogram(
            audio,
            fs=self.sample_rate,
            nperseg=self.fft_size,
            noverlap=self.fft_size - self.hop_size,
        )
        
        # Convert to dB
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
        
        # Calculate features
        spectral_centroid = np.sum(
            frequencies[:, np.newaxis] * spectrogram, axis=0
        ) / (np.sum(spectrogram, axis=0) + 1e-10)
        
        spectral_rolloff = self._calculate_rolloff(spectrogram, frequencies)
        
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        
        # Peak detection
        peaks, properties = signal.find_peaks(
            audio,
            height=np.max(np.abs(audio)) * 0.5,
            distance=self.sample_rate // 10,  # Min 100ms apart
        )
        
        return {
            "duration_seconds": len(audio) / self.sample_rate,
            "sample_rate": self.sample_rate,
            "peak_amplitude": float(np.max(np.abs(audio))),
            "rms": float(np.sqrt(np.mean(audio ** 2))),
            "dynamic_range_db": float(
                20 * np.log10(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio ** 2)) + 1e-10))
            ),
            "zero_crossing_rate": float(zcr),
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_centroid_std": float(np.std(spectral_centroid)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "num_peaks": len(peaks),
            "spectrogram_shape": spectrogram.shape,
        }
    
    def _calculate_rolloff(self, spectrogram: np.ndarray, frequencies: np.ndarray, threshold: float = 0.85) -> np.ndarray:
        """Calculate spectral rolloff (frequency below which 85% of energy is contained)."""
        cumulative_sum = np.cumsum(spectrogram, axis=0)
        total_energy = cumulative_sum[-1, :]
        
        rolloff_indices = np.argmax(
            cumulative_sum >= threshold * total_energy[np.newaxis, :],
            axis=0,
        )
        
        return frequencies[rolloff_indices]


class AudioVisualizer:
    """Generate visualization data for dashboard."""
    
    def __init__(self):
        self.analyzer = RealtimeAudioAnalyzer()
    
    def generate_waveform(self, audio: np.ndarray, width: int = 800) -> dict:
        """Generate waveform visualization data.
        
        Args:
            audio: Audio array
            width: Number of points in waveform
        
        Returns:
            Waveform data dict
        """
        # Convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Downsample to width
        samples_per_point = len(audio) // width
        
        waveform = []
        for i in range(width):
            start = i * samples_per_point
            end = start + samples_per_point
            chunk = audio[start:end]
            
            if len(chunk) > 0:
                peak_positive = float(np.max(chunk))
                peak_negative = float(np.min(chunk))
                waveform.append([peak_negative, peak_positive])
        
        return {
            "waveform": waveform,
            "duration_seconds": len(audio) / self.analyzer.sample_rate,
            "width": width,
        }
    
    def generate_spectrogram(self, audio: np.ndarray, height: int = 256, width: int = 800) -> dict:
        """Generate spectrogram visualization data.
        
        Args:
            audio: Audio array
            height: Number of frequency bins
            width: Number of time bins
        
        Returns:
            Spectrogram data dict
        """
        # Convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Compute spectrogram
        frequencies, times, spectrogram = signal.spectrogram(
            audio,
            fs=self.analyzer.sample_rate,
            nperseg=512,
            noverlap=256,
        )
        
        # Convert to dB
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
        
        # Resize to target dimensions
        from scipy.ndimage import zoom
        
        zoom_factors = (height / spectrogram_db.shape[0], width / spectrogram_db.shape[1])
        spectrogram_resized = zoom(spectrogram_db, zoom_factors, order=1)
        
        # Normalize to 0-1
        spectrogram_norm = (spectrogram_resized - spectrogram_resized.min()) / (
            spectrogram_resized.max() - spectrogram_resized.min() + 1e-10
        )
        
        return {
            "spectrogram": spectrogram_norm.tolist(),
            "frequencies": frequencies[:height].tolist(),
            "times": times[:width].tolist(),
            "height": height,
            "width": width,
        }


# WebSocket broadcaster for real-time updates
class AudioStreamBroadcaster:
    """Broadcast real-time audio analysis to WebSocket clients."""
    
    def __init__(self):
        self.analyzer = RealtimeAudioAnalyzer()
        self.clients = set()
    
    def register_client(self, websocket):
        """Register a WebSocket client."""
        self.clients.add(websocket)
        logger.info("client_registered", total_clients=len(self.clients))
    
    def unregister_client(self, websocket):
        """Unregister a WebSocket client."""
        self.clients.discard(websocket)
        logger.info("client_unregistered", total_clients=len(self.clients))
    
    async def broadcast_analysis(self, analysis: dict):
        """Broadcast analysis to all connected clients."""
        if not self.clients:
            return
        
        import json
        message = json.dumps({"type": "audio_analysis", "data": analysis})
        
        # Send to all clients
        tasks = [client.send_text(message) for client in self.clients]
        await asyncio.gather(*tasks, return_exceptions=True)


# Global broadcaster instance
broadcaster = AudioStreamBroadcaster()
