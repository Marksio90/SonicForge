import io
import struct

import structlog

from ..core.config import GENRE_PROFILES, Genre, get_settings
from ..core.storage import download_track
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)


class CriticAgent(BaseAgent):
    """Quality control agent that evaluates generated tracks."""

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
        """Perform comprehensive quality evaluation of a track."""
        self.logger.info("evaluating_track", track_id=track_id, genre=genre)

        if audio_data is None:
            try:
                audio_data = download_track(track_id)
            except Exception as e:
                self.logger.warning("could_not_download_track", error=str(e))
                audio_data = None

        # Spectral analysis
        spectral_score = await self._analyze_spectrum(audio_data, genre)

        # Structure analysis
        structure_score = await self._analyze_structure(audio_data, genre)

        # Artifact detection
        artifact_result = await self._detect_artifacts(audio_data)

        # Dynamic range analysis
        dynamics_score = await self._analyze_dynamics(audio_data)

        # Genre conformity
        genre_score = await self._check_genre_conformity(audio_data, genre)

        # Calculate overall score (weighted average)
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
        """Evaluate multiple variants and select the best one."""
        evaluations = []
        for variant in variants:
            eval_result = await self.evaluate_track(
                track_id=variant["track_id"],
                genre=variant.get("genre"),
            )
            evaluations.append(eval_result)

        # Sort by score, pick best
        evaluations.sort(key=lambda x: x["overall_score"], reverse=True)
        approved = [e for e in evaluations if e["approved"]]

        return {
            "evaluations": evaluations,
            "best": evaluations[0] if evaluations else None,
            "approved_count": len(approved),
            "total_count": len(evaluations),
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
        """Analyze frequency spectrum balance."""
        if audio_data is None or len(audio_data) < 500:
            return 7.0  # default for placeholder

        # In production: use librosa to load audio, compute spectrogram,
        # check frequency distribution, compare to genre reference
        # For now: heuristic based on audio data properties
        data_variance = len(set(audio_data[:1000])) / min(len(audio_data), 1000)
        return min(10.0, max(5.0, 7.0 + data_variance * 3))

    async def _analyze_structure(self, audio_data: bytes | None, genre: str | None) -> float:
        """Analyze musical structure (intro, build, drop, breakdown, outro)."""
        if audio_data is None or len(audio_data) < 500:
            return 7.0

        # In production: detect sections using librosa onset detection,
        # verify expected genre structure exists
        return 7.5

    async def _detect_artifacts(self, audio_data: bytes | None) -> dict:
        """Detect common AI generation artifacts."""
        if audio_data is None or len(audio_data) < 500:
            return {"has_artifacts": False, "severity": 0.0, "details": []}

        # In production: check for:
        # - Sudden frequency cuts (AI truncation)
        # - Unnatural repetition patterns
        # - Phase cancellation issues
        # - Metallic/robotic timbres
        # - Abrupt endings
        return {"has_artifacts": False, "severity": 0.5, "details": []}

    async def _analyze_dynamics(self, audio_data: bytes | None) -> float:
        """Analyze dynamic range and loudness."""
        if audio_data is None or len(audio_data) < 500:
            return 7.5
        # In production: compute LUFS, dynamic range, crest factor
        return 7.5

    async def _check_genre_conformity(self, audio_data: bytes | None, genre: str | None) -> float:
        """Check if the track conforms to genre expectations."""
        if genre is None:
            return 7.0
        # In production: verify BPM range, frequency emphasis areas,
        # and rhythmic patterns match genre profile
        return 8.0

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
