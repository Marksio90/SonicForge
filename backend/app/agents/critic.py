import asyncio
import math

import structlog

from ..core.config import GENRE_PROFILES, Genre, get_settings
from ..core.storage import download_track
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)


class CriticAgent(BaseAgent):
    """Quality control agent with parallel evaluation and advanced audio analysis."""

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
        else:
            raise ValueError(f"Unknown task type: {task_type}")

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

        # Run all analysis steps in parallel
        (
            spectral_score,
            structure_score,
            artifact_result,
            dynamics_score,
            genre_score,
        ) = await asyncio.gather(
            self._analyze_spectrum(audio_data, genre),
            self._analyze_structure(audio_data, genre),
            self._detect_artifacts(audio_data),
            self._analyze_dynamics(audio_data),
            self._check_genre_conformity(audio_data, genre),
        )

        weights = {
            "spectral": 0.25,
            "structure": 0.25,
            "artifacts": 0.20,
            "dynamics": 0.15,
            "genre_conformity": 0.15,
        }

        overall_score = (
            spectral_score * weights["spectral"]
            + structure_score * weights["structure"]
            + (10.0 - artifact_result["severity"]) * weights["artifacts"]
            + dynamics_score * weights["dynamics"]
            + genre_score * weights["genre_conformity"]
        )

        approved = overall_score >= settings.quality_threshold and not artifact_result["has_artifacts"]

        feedback = self._generate_feedback(
            overall_score, spectral_score, structure_score,
            artifact_result, dynamics_score, genre_score,
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
            },
            "has_artifacts": artifact_result["has_artifacts"],
            "artifact_details": artifact_result.get("details", []),
            "feedback": feedback,
            "threshold": settings.quality_threshold,
        }

        self.logger.info(
            "evaluation_complete",
            track_id=track_id,
            score=overall_score,
            approved=approved,
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
                {"rank": i + 1, "track_id": e["track_id"], "score": e["overall_score"]}
                for i, e in enumerate(ranked)
            ],
            "winner": ranked[0]["track_id"] if ranked else None,
        }

    async def _analyze_spectrum(self, audio_data: bytes | None, genre: str | None) -> float:
        """Analyze frequency spectrum balance with entropy-based scoring."""
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

    async def _analyze_structure(self, audio_data: bytes | None, genre: str | None) -> float:
        """Analyze musical structure by detecting energy transitions."""
        if audio_data is None or len(audio_data) < 500:
            return 7.0

        segment_size = max(1, len(audio_data) // 8)
        segment_energies = []
        for i in range(0, len(audio_data), segment_size):
            segment = audio_data[i:i + segment_size]
            if segment:
                energy = sum(b * b for b in segment) / len(segment)
                segment_energies.append(energy)

        if len(segment_energies) < 3:
            return 7.0

        energy_range = max(segment_energies) - min(segment_energies)
        variation_score = min(1.0, energy_range / 5000.0)
        return min(10.0, max(5.0, 6.0 + variation_score * 4.0))

    async def _detect_artifacts(self, audio_data: bytes | None) -> dict:
        """Detect common AI generation artifacts using statistical analysis."""
        if audio_data is None or len(audio_data) < 500:
            return {"has_artifacts": False, "severity": 0.0, "details": []}

        details = []
        severity = 0.0

        # Check for repeating patterns (common AI artifact)
        chunk_size = 512
        if len(audio_data) > chunk_size * 4:
            chunks = [
                audio_data[i:i + chunk_size]
                for i in range(0, min(len(audio_data), chunk_size * 10), chunk_size)
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

        has_artifacts = severity >= 2.0
        return {
            "has_artifacts": has_artifacts,
            "severity": min(severity, 10.0),
            "details": details,
        }

    async def _analyze_dynamics(self, audio_data: bytes | None) -> float:
        """Analyze dynamic range using peak-to-average ratio."""
        if audio_data is None or len(audio_data) < 500:
            return 7.5

        sample = audio_data[:min(len(audio_data), 16384)]
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

    async def _check_genre_conformity(self, audio_data: bytes | None, genre: str | None) -> float:
        """Check if the track conforms to genre expectations."""
        if genre is None:
            return 7.0

        try:
            genre_enum = Genre(genre)
            profile = GENRE_PROFILES.get(genre_enum)
            if profile:
                return 8.0
        except (ValueError, KeyError):
            pass

        return 7.0

    def _generate_feedback(
        self, overall: float, spectral: float, structure: float,
        artifacts: dict, dynamics: float, genre_conf: float,
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
            parts.append("Spectral balance needs improvement — frequency gaps detected.")
        if structure < 7.0:
            parts.append("Weak structure — missing clear sections or transitions.")
        if artifacts["has_artifacts"]:
            parts.append(f"AI artifacts detected: {', '.join(artifacts.get('details', ['general']))}")
        if dynamics < 7.0:
            parts.append("Dynamic range issues — too compressed or too quiet.")
        if genre_conf < 7.0:
            parts.append("Doesn't match genre expectations well.")

        return " ".join(parts)
