import asyncio
import io
import math

import numpy as np
import structlog

from ..core.config import GENRE_PROFILES, Genre, get_settings
from ..core.storage import download_track
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)


class CriticAgent(BaseAgent):
    """Quality control agent with advanced spectral analysis, key detection,
    and mudiness rejection for broadcast-grade audio output."""

    def __init__(self):
        super().__init__("critic")

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "evaluate")

        if task_type == "evaluate":
            return await self.evaluate_track(
                track_id=task["track_id"],
                genre=task.get("genre"),
                audio_data=task.get("audio_data"),
            )
        elif task_type == "evaluate_batch":
            return await self.evaluate_batch(variants=task["variants"])
        elif task_type == "compare":
            return await self.compare_variants(variants=task["variants"])
        elif task_type == "pre_evaluate":
            return await self.pre_evaluate_concept(concept=task["concept"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def pre_evaluate_concept(self, concept: dict) -> dict:
        """Pre-evaluate a concept BEFORE generation to catch obvious issues.

        The Critic reviews the prompt and concept parameters, simulating
        potential problems before expensive audio generation occurs.
        """
        issues = []
        genre = concept.get("genre")
        bpm = concept.get("bpm")
        key = concept.get("key")
        energy = concept.get("energy")

        # Check BPM matches genre expectations
        if genre and bpm:
            try:
                genre_enum = Genre(genre)
                profile = GENRE_PROFILES.get(genre_enum)
                if profile:
                    bpm_min, bpm_max = profile["bpm_range"]
                    if bpm < bpm_min - 10 or bpm > bpm_max + 10:
                        issues.append(
                            f"BPM {bpm} is outside typical range for {genre} "
                            f"({bpm_min}-{bpm_max}). Bass frequencies may clash."
                        )
            except (ValueError, KeyError):
                pass

        # Check key/energy compatibility
        minor_keys = {"Am", "Bm", "Cm", "Dm", "Em", "Fm", "Gm"}
        if key and energy:
            if energy == "high" and key in minor_keys and bpm and bpm < 130:
                issues.append(
                    f"High energy with minor key {key} at low BPM {bpm} "
                    "may produce a muddy, unfocused sound."
                )

        approved = len(issues) == 0
        return {
            "pre_evaluation": True,
            "approved": approved,
            "issues": issues,
            "concept": concept,
        }

    async def evaluate_track(
        self,
        track_id: str,
        genre: str | None = None,
        audio_data: bytes | None = None,
    ) -> dict:
        """Perform comprehensive quality evaluation with parallel analysis steps."""
        self.logger.info("evaluating_track", track_id=track_id, genre=genre)

        if audio_data is None:
            try:
                audio_data = download_track(track_id)
            except Exception as e:
                self.logger.warning("could_not_download_track", error=str(e))
                audio_data = None

        # Load audio as numpy array for advanced analysis
        samples = self._load_audio_samples(audio_data)

        # Run all analysis steps in parallel
        (
            spectral_score,
            structure_score,
            artifact_result,
            dynamics_score,
            genre_score,
            mudiness_result,
            key_result,
        ) = await asyncio.gather(
            self._analyze_spectrum(audio_data, samples, genre),
            self._analyze_structure(audio_data, samples, genre),
            self._detect_artifacts(audio_data, samples),
            self._analyze_dynamics(audio_data, samples),
            self._check_genre_conformity(audio_data, samples, genre),
            self._detect_mudiness(samples),
            self._detect_key(samples, genre),
        )

        weights = {
            "spectral": 0.20,
            "structure": 0.20,
            "artifacts": 0.15,
            "dynamics": 0.15,
            "genre_conformity": 0.10,
            "mudiness": 0.10,
            "key_accuracy": 0.10,
        }

        overall_score = (
            spectral_score * weights["spectral"]
            + structure_score * weights["structure"]
            + (10.0 - artifact_result["severity"]) * weights["artifacts"]
            + dynamics_score * weights["dynamics"]
            + genre_score * weights["genre_conformity"]
            + mudiness_result["score"] * weights["mudiness"]
            + key_result["score"] * weights["key_accuracy"]
        )

        # Rejection conditions
        has_critical_issues = (
            artifact_result["has_artifacts"]
            or mudiness_result["is_muddy"]
            or key_result["is_off_key"]
        )

        approved = (
            overall_score >= settings.quality_threshold and not has_critical_issues
        )

        feedback = self._generate_feedback(
            overall_score,
            spectral_score,
            structure_score,
            artifact_result,
            dynamics_score,
            genre_score,
            mudiness_result,
            key_result,
        )

        result = {
            "track_id": track_id,
            "overall_score": round(overall_score, 2),
            "approved": approved,
            "scores": {
                "spectral": round(spectral_score, 2),
                "structure": round(structure_score, 2),
                "artifacts": round(10.0 - artifact_result["severity"], 2),
                "dynamics": round(dynamics_score, 2),
                "genre_conformity": round(genre_score, 2),
                "mudiness": round(mudiness_result["score"], 2),
                "key_accuracy": round(key_result["score"], 2),
            },
            "has_artifacts": artifact_result["has_artifacts"],
            "is_muddy": mudiness_result["is_muddy"],
            "is_off_key": key_result["is_off_key"],
            "detected_key": key_result.get("detected_key"),
            "expected_key": key_result.get("expected_key"),
            "artifact_details": artifact_result.get("details", []),
            "feedback": feedback,
            "threshold": settings.quality_threshold,
        }

        self.logger.info(
            "evaluation_complete",
            track_id=track_id,
            score=overall_score,
            approved=approved,
            muddy=mudiness_result["is_muddy"],
            off_key=key_result["is_off_key"],
        )

        return result

    async def evaluate_batch(self, variants: list[dict]) -> dict:
        """Evaluate multiple variants in parallel and select the best one."""
        eval_tasks = [
            self.evaluate_track(
                track_id=variant["track_id"],
                genre=variant.get("genre"),
            )
            for variant in variants
        ]

        eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)

        valid_evaluations = []
        for eval_result in eval_results:
            if isinstance(eval_result, Exception):
                self.logger.error("variant_evaluation_failed", error=str(eval_result))
                continue
            valid_evaluations.append(eval_result)

        valid_evaluations.sort(key=lambda x: x["overall_score"], reverse=True)
        approved = [e for e in valid_evaluations if e["approved"]]

        return {
            "evaluations": valid_evaluations,
            "best": valid_evaluations[0] if valid_evaluations else None,
            "approved_count": len(approved),
            "total_count": len(valid_evaluations),
            "best_approved": approved[0] if approved else None,
        }

    async def compare_variants(self, variants: list[dict]) -> dict:
        """Compare variants and return ranked results."""
        batch_result = await self.evaluate_batch(variants)
        ranked = sorted(
            batch_result["evaluations"],
            key=lambda x: x["overall_score"],
            reverse=True,
        )
        return {
            "ranking": [
                {
                    "rank": i + 1,
                    "track_id": e["track_id"],
                    "score": e["overall_score"],
                }
                for i, e in enumerate(ranked)
            ],
            "winner": ranked[0]["track_id"] if ranked else None,
        }

    def _load_audio_samples(self, audio_data: bytes | None) -> np.ndarray | None:
        """Load audio bytes into numpy array for spectral analysis."""
        if audio_data is None or len(audio_data) < 500:
            return None
        try:
            import soundfile as sf

            audio_io = io.BytesIO(audio_data)
            samples, _ = sf.read(audio_io)
            return samples
        except Exception:
            pass

        try:
            # Fallback: raw PCM
            offset = 44 if audio_data[:4] == b"RIFF" else 0
            pcm = audio_data[offset:]
            num_samples = len(pcm) // 4
            if num_samples == 0:
                return None
            samples = np.frombuffer(pcm[: num_samples * 4], dtype=np.int16)
            return samples.reshape(-1, 2).astype(np.float64) / 32768.0
        except Exception:
            return None

    async def _analyze_spectrum(
        self, audio_data: bytes | None, samples: np.ndarray | None, genre: str | None
    ) -> float:
        """Analyze frequency spectrum balance using librosa spectral analysis."""
        if samples is not None and len(samples) > 0:
            try:
                return await asyncio.to_thread(
                    self._spectral_analysis_librosa, samples
                )
            except Exception:
                pass

        # Fallback: byte entropy analysis
        if audio_data is None or len(audio_data) < 500:
            return 7.0

        byte_counts = [0] * 256
        sample_size = min(len(audio_data), 8192)
        for byte in audio_data[:sample_size]:
            byte_counts[byte] += 1

        entropy = 0.0
        for count in byte_counts:
            if count > 0:
                p = count / sample_size
                entropy -= p * math.log2(p)

        normalized = entropy / 8.0
        return min(10.0, max(5.0, 5.0 + normalized * 5.0))

    def _spectral_analysis_librosa(self, samples: np.ndarray) -> float:
        """Advanced spectral analysis using librosa."""
        import librosa

        # Convert to mono if stereo
        if samples.ndim == 2:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples

        mono = mono.astype(np.float32)

        # Spectral centroid — measures brightness
        centroid = librosa.feature.spectral_centroid(y=mono, sr=44100)
        mean_centroid = np.mean(centroid)

        # Spectral bandwidth — measures frequency spread
        bandwidth = librosa.feature.spectral_bandwidth(y=mono, sr=44100)
        mean_bandwidth = np.mean(bandwidth)

        # Spectral rolloff — frequency below which 85% of energy exists
        rolloff = librosa.feature.spectral_rolloff(y=mono, sr=44100)
        mean_rolloff = np.mean(rolloff)

        # Spectral flatness — 0 = tonal, 1 = noise-like
        flatness = librosa.feature.spectral_flatness(y=mono)
        mean_flatness = np.mean(flatness)

        # Score: good music has moderate centroid, wide bandwidth, not too flat
        score = 7.0

        if 500 < mean_centroid < 4000:
            score += 1.0  # Good brightness
        if mean_bandwidth > 1000:
            score += 0.5  # Good frequency spread
        if 2000 < mean_rolloff < 10000:
            score += 0.5  # Good energy distribution
        if 0.01 < mean_flatness < 0.5:
            score += 1.0  # Musical (not pure noise, not pure tone)

        return min(10.0, max(5.0, score))

    async def _analyze_structure(
        self, audio_data: bytes | None, samples: np.ndarray | None, genre: str | None
    ) -> float:
        """Analyze musical structure by detecting energy transitions."""
        if samples is not None and len(samples) > 0:
            try:
                return await asyncio.to_thread(
                    self._structure_analysis_librosa, samples
                )
            except Exception:
                pass

        # Fallback
        if audio_data is None or len(audio_data) < 500:
            return 7.0

        segment_size = max(1, len(audio_data) // 8)
        segment_energies = []
        for i in range(0, len(audio_data), segment_size):
            segment = audio_data[i : i + segment_size]
            if segment:
                energy = sum(b * b for b in segment) / len(segment)
                segment_energies.append(energy)

        if len(segment_energies) < 3:
            return 7.0

        energy_range = max(segment_energies) - min(segment_energies)
        variation_score = min(1.0, energy_range / 5000.0)
        return min(10.0, max(5.0, 6.0 + variation_score * 4.0))

    def _structure_analysis_librosa(self, samples: np.ndarray) -> float:
        """Analyze structure using onset detection and tempo estimation."""
        import librosa

        if samples.ndim == 2:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples
        mono = mono.astype(np.float32)

        # Onset strength envelope
        onset_env = librosa.onset.onset_strength(y=mono, sr=44100)

        # Detect beats
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=44100)

        # Segment the audio by energy
        rms = librosa.feature.rms(y=mono)[0]
        segment_count = 8
        seg_len = max(1, len(rms) // segment_count)
        segment_energies = [
            np.mean(rms[i : i + seg_len])
            for i in range(0, len(rms), seg_len)
        ]

        score = 7.0

        # Good structure has energy variation (intro/build/drop)
        if segment_energies:
            energy_range = max(segment_energies) - min(segment_energies)
            if energy_range > 0.05:
                score += 1.0
            if energy_range > 0.1:
                score += 0.5

        # Having clear beats is good for electronic music
        if len(beats) > 10:
            score += 1.0

        # Reasonable tempo
        tempo_val = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0]) if len(tempo) > 0 else 0
        if 60 < tempo_val < 200:
            score += 0.5

        return min(10.0, max(5.0, score))

    async def _detect_artifacts(
        self, audio_data: bytes | None, samples: np.ndarray | None
    ) -> dict:
        """Detect common AI generation artifacts using statistical analysis."""
        if audio_data is None or len(audio_data) < 500:
            return {"has_artifacts": False, "severity": 0.0, "details": []}

        details = []
        severity = 0.0

        # Check for repeating patterns (common AI artifact)
        chunk_size = 512
        if len(audio_data) > chunk_size * 4:
            chunks = [
                audio_data[i : i + chunk_size]
                for i in range(
                    0, min(len(audio_data), chunk_size * 10), chunk_size
                )
            ]
            for i in range(len(chunks) - 1):
                if chunks[i] == chunks[i + 1]:
                    severity += 2.0
                    details.append("repeated_block_detected")
                    break

        # Check for sudden silence (truncation artifact)
        tail = audio_data[-256:]
        silent_bytes = sum(1 for b in tail if b == 0)
        if silent_bytes > 200:
            severity += 1.5
            details.append("abrupt_silence_ending")

        # Check for spectral anomalies in numpy samples
        if samples is not None and len(samples) > 0:
            try:
                spectral_artifacts = await asyncio.to_thread(
                    self._detect_spectral_artifacts, samples
                )
                severity += spectral_artifacts["severity"]
                details.extend(spectral_artifacts["details"])
            except Exception:
                pass

        has_artifacts = severity >= 2.0
        return {
            "has_artifacts": has_artifacts,
            "severity": min(severity, 10.0),
            "details": details,
        }

    def _detect_spectral_artifacts(self, samples: np.ndarray) -> dict:
        """Detect spectral artifacts like frequency holes or aliasing."""
        details = []
        severity = 0.0

        if samples.ndim == 2:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples

        # Check for DC offset
        dc_offset = abs(np.mean(mono))
        if dc_offset > 0.01:
            severity += 1.0
            details.append(f"dc_offset_detected ({dc_offset:.4f})")

        # Check for clipping
        clipped = np.sum(np.abs(mono) > 0.99) / len(mono)
        if clipped > 0.01:
            severity += 1.5
            details.append(f"clipping_detected ({clipped:.2%})")

        return {"severity": severity, "details": details}

    async def _analyze_dynamics(
        self, audio_data: bytes | None, samples: np.ndarray | None
    ) -> float:
        """Analyze dynamic range using peak-to-average ratio."""
        if samples is not None and len(samples) > 0:
            if samples.ndim == 2:
                mono = np.mean(samples, axis=1)
            else:
                mono = samples

            peak = np.max(np.abs(mono))
            rms = np.sqrt(np.mean(mono**2))
            if rms == 0:
                return 6.0

            crest = peak / rms
            if 3.0 <= crest <= 12.0:
                return min(10.0, 7.0 + (crest - 3.0) / 9.0 * 3.0)
            elif crest < 3.0:
                return max(5.0, 5.0 + crest)
            else:
                return max(5.0, 10.0 - (crest - 12.0) * 0.5)

        # Fallback: byte-level analysis
        if audio_data is None or len(audio_data) < 500:
            return 7.5

        sample = audio_data[: min(len(audio_data), 16384)]
        values = [abs(b - 128) for b in sample]
        if not values:
            return 7.5

        peak = max(values)
        avg = sum(values) / len(values)
        if avg == 0:
            return 6.0

        crest = peak / avg
        if 3.0 <= crest <= 12.0:
            return min(10.0, 7.0 + (crest - 3.0) / 9.0 * 3.0)
        elif crest < 3.0:
            return max(5.0, 5.0 + crest)
        else:
            return max(5.0, 10.0 - (crest - 12.0) * 0.5)

    async def _check_genre_conformity(
        self, audio_data: bytes | None, samples: np.ndarray | None, genre: str | None
    ) -> float:
        """Check if the track conforms to genre expectations using tempo analysis."""
        if genre is None:
            return 7.0

        try:
            genre_enum = Genre(genre)
            profile = GENRE_PROFILES.get(genre_enum)
            if not profile:
                return 7.0
        except (ValueError, KeyError):
            return 7.0

        # If we have numpy samples, do tempo-based conformity check
        if samples is not None and len(samples) > 0:
            try:
                return await asyncio.to_thread(
                    self._genre_conformity_librosa, samples, profile
                )
            except Exception:
                pass

        return 8.0

    def _genre_conformity_librosa(
        self, samples: np.ndarray, profile: dict
    ) -> float:
        """Check genre conformity using librosa tempo detection."""
        import librosa

        if samples.ndim == 2:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples
        mono = mono.astype(np.float32)

        tempo, _ = librosa.beat.beat_track(y=mono, sr=44100)
        tempo_val = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0]) if len(tempo) > 0 else 0

        bpm_min, bpm_max = profile["bpm_range"]

        # Check if detected tempo matches expected range
        if bpm_min <= tempo_val <= bpm_max:
            return 9.5  # Perfect match
        elif bpm_min - 10 <= tempo_val <= bpm_max + 10:
            return 8.0  # Close enough
        elif bpm_min / 2 <= tempo_val <= bpm_max / 2:
            return 7.5  # Half-time detection (common for DnB)
        elif bpm_min * 2 >= tempo_val >= bpm_max * 2:
            return 7.5  # Double-time detection
        else:
            return 5.5  # Doesn't match genre

    async def _detect_mudiness(self, samples: np.ndarray | None) -> dict:
        """Detect mudiness in low frequencies (200-500 Hz buildup).

        Muddy audio has excessive energy in the 200-500 Hz range relative
        to the overall spectrum. This makes bass unreadable on radio.
        """
        if samples is None or len(samples) == 0:
            return {"is_muddy": False, "score": 8.0, "low_mid_ratio": 0.0}

        try:
            return await asyncio.to_thread(self._mudiness_analysis, samples)
        except Exception:
            return {"is_muddy": False, "score": 7.5, "low_mid_ratio": 0.0}

    def _mudiness_analysis(self, samples: np.ndarray) -> dict:
        """Analyze mudiness using spectral energy distribution."""
        if samples.ndim == 2:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples.copy()
        mono = mono.astype(np.float64)

        # FFT analysis
        n = len(mono)
        if n < 2048:
            return {"is_muddy": False, "score": 7.5, "low_mid_ratio": 0.0}

        fft = np.fft.rfft(mono)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(n, d=1.0 / 44100)

        # Energy in frequency bands
        low_mask = (freqs >= 20) & (freqs < 200)
        mud_mask = (freqs >= 200) & (freqs < 500)
        mid_mask = (freqs >= 500) & (freqs < 4000)
        high_mask = (freqs >= 4000) & (freqs < 20000)

        low_energy = np.sum(magnitude[low_mask] ** 2) if np.any(low_mask) else 0
        mud_energy = np.sum(magnitude[mud_mask] ** 2) if np.any(mud_mask) else 0
        mid_energy = np.sum(magnitude[mid_mask] ** 2) if np.any(mid_mask) else 0
        high_energy = np.sum(magnitude[high_mask] ** 2) if np.any(high_mask) else 0

        total_energy = low_energy + mud_energy + mid_energy + high_energy
        if total_energy == 0:
            return {"is_muddy": False, "score": 7.5, "low_mid_ratio": 0.0}

        mud_ratio = mud_energy / total_energy

        # Muddy if > 40% of energy is in 200-500 Hz range
        is_muddy = mud_ratio > 0.40
        score = 10.0 - (mud_ratio * 15.0)  # Higher ratio = lower score
        score = max(3.0, min(10.0, score))

        return {
            "is_muddy": is_muddy,
            "score": round(score, 2),
            "low_mid_ratio": round(mud_ratio, 4),
            "energy_distribution": {
                "low_pct": round(low_energy / total_energy * 100, 1),
                "mud_pct": round(mud_energy / total_energy * 100, 1),
                "mid_pct": round(mid_energy / total_energy * 100, 1),
                "high_pct": round(high_energy / total_energy * 100, 1),
            },
        }

    async def _detect_key(
        self, samples: np.ndarray | None, genre: str | None
    ) -> dict:
        """Detect musical key using chroma features and check against expected key."""
        if samples is None or len(samples) == 0:
            return {"is_off_key": False, "score": 8.0, "detected_key": None, "expected_key": None}

        try:
            return await asyncio.to_thread(self._key_detection, samples, genre)
        except Exception:
            return {"is_off_key": False, "score": 7.5, "detected_key": None, "expected_key": None}

    def _key_detection(self, samples: np.ndarray, genre: str | None) -> dict:
        """Detect key using librosa chroma features."""
        import librosa

        if samples.ndim == 2:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples
        mono = mono.astype(np.float32)

        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=mono, sr=44100)
        chroma_avg = np.mean(chroma, axis=1)

        # Map to pitch classes
        pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        dominant_pitch = int(np.argmax(chroma_avg))
        detected_key = pitch_classes[dominant_pitch]

        # Determine major/minor using correlation with key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_corr = np.corrcoef(np.roll(major_profile, dominant_pitch), chroma_avg)[0, 1]
        minor_corr = np.corrcoef(np.roll(minor_profile, dominant_pitch), chroma_avg)[0, 1]

        if minor_corr > major_corr:
            detected_key += "m"

        # Key detection confidence
        confidence = max(major_corr, minor_corr)

        # Score based on confidence (high confidence = likely clean tonality)
        score = 7.0 + confidence * 3.0
        score = max(5.0, min(10.0, score))

        # Check if off-key (low confidence means messy tonality)
        is_off_key = confidence < 0.3

        return {
            "is_off_key": is_off_key,
            "score": round(score, 2),
            "detected_key": detected_key,
            "expected_key": None,
            "confidence": round(float(confidence), 3),
        }

    def _generate_feedback(
        self,
        overall: float,
        spectral: float,
        structure: float,
        artifacts: dict,
        dynamics: float,
        genre_conf: float,
        mudiness: dict,
        key_result: dict,
    ) -> str:
        """Generate human-readable feedback for the track."""
        parts = []

        if overall >= 9.0:
            parts.append("Excellent quality — ready for broadcast.")
        elif overall >= settings.quality_threshold:
            parts.append("Good quality — approved for broadcast.")
        elif overall >= 7.0:
            parts.append("Decent quality but below threshold. Needs regeneration.")
        else:
            parts.append("Poor quality — rejected. Major issues detected.")

        if spectral < 7.0:
            parts.append(
                "Spectral balance needs improvement — frequency gaps detected."
            )
        if structure < 7.0:
            parts.append(
                "Weak structure — missing clear sections or transitions."
            )
        if artifacts["has_artifacts"]:
            parts.append(
                f"AI artifacts detected: {', '.join(artifacts.get('details', ['general']))}"
            )
        if dynamics < 7.0:
            parts.append(
                "Dynamic range issues — too compressed or too quiet."
            )
        if genre_conf < 7.0:
            parts.append("Doesn't match genre expectations well.")
        if mudiness.get("is_muddy"):
            parts.append(
                f"MUDDY: Excessive low-mid energy ({mudiness.get('low_mid_ratio', 0):.1%} "
                "in 200-500 Hz). Bass will be unreadable on radio."
            )
        if key_result.get("is_off_key"):
            parts.append(
                f"OFF-KEY: Detected key {key_result.get('detected_key', '?')} "
                f"with low confidence ({key_result.get('confidence', 0):.2f}). "
                "Tonality is messy."
            )

        return " ".join(parts)
