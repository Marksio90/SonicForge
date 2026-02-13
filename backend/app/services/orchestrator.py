import json
import uuid
from datetime import datetime, timezone

import structlog

from ..agents.analytics import AnalyticsAgent
from ..agents.composer import ComposerAgent
from ..agents.critic import CriticAgent
from ..agents.producer import ProducerAgent
from ..agents.scheduler import SchedulerAgent
from ..agents.stream_master import StreamMasterAgent
from ..agents.visual import VisualAgent
from ..core.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class Orchestrator:
    """Central brain of SonicForge — coordinates all agents through the production pipeline."""

    def __init__(self):
        self.composer = ComposerAgent()
        self.producer = ProducerAgent()
        self.critic = CriticAgent()
        self.scheduler = SchedulerAgent()
        self.stream_master = StreamMasterAgent()
        self.analytics = AnalyticsAgent()
        self.visual = VisualAgent()
        self.logger = logger.bind(component="orchestrator")

    async def run_full_pipeline(self, genre: str | None = None, energy: int | None = None) -> dict:
        """Run the complete production pipeline: concept → generate → evaluate → schedule."""
        pipeline_id = str(uuid.uuid4())
        self.logger.info("pipeline_started", pipeline_id=pipeline_id, genre=genre)

        # Step 1: Create concept
        concept = await self.composer.run({
            "type": "create_concept",
            "genre": genre,
            "energy": energy,
        })
        self.logger.info("concept_created", pipeline_id=pipeline_id, genre=concept["genre"])

        # Step 2: Generate track variants
        generation_result = await self.producer.run({
            "type": "generate",
            "concept": concept,
            "variants": settings.variants_per_concept,
        })
        self.logger.info(
            "variants_generated",
            pipeline_id=pipeline_id,
            count=len(generation_result["variants"]),
        )

        # Step 3: Evaluate all variants
        evaluation = await self.critic.run({
            "type": "evaluate_batch",
            "variants": generation_result["variants"],
        })

        best = evaluation.get("best_approved") or evaluation.get("best")
        if not best:
            self.logger.warning("no_approved_tracks", pipeline_id=pipeline_id)
            # Retry with new concept
            if settings.max_generation_attempts > 1:
                return await self._retry_pipeline(pipeline_id, genre, energy, attempt=2)
            return {"pipeline_id": pipeline_id, "status": "failed", "reason": "no_approved_tracks"}

        self.logger.info(
            "track_approved",
            pipeline_id=pipeline_id,
            track_id=best["track_id"],
            score=best["overall_score"],
        )

        # Step 4: Generate visuals
        visual_config = await self.visual.run({
            "type": "generate_visual",
            "track_id": best["track_id"],
            "genre": concept["genre"],
            "bpm": concept["bpm"],
            "title": self._generate_track_title(concept),
        })

        # Step 5: Generate overlay
        overlay = await self.visual.run({
            "type": "generate_overlay",
            "track_id": best["track_id"],
            "title": self._generate_track_title(concept),
            "bpm": concept["bpm"],
            "key": concept["key"],
            "genre": concept["genre"],
            "score": best["overall_score"],
        })

        # Step 6: Add to stream queue
        track_data = {
            "track_id": best["track_id"],
            "pipeline_id": pipeline_id,
            "title": self._generate_track_title(concept),
            "genre": concept["genre"],
            "bpm": concept["bpm"],
            "key": concept["key"],
            "energy": concept["energy"],
            "score": best["overall_score"],
            "visual_config": visual_config,
            "overlay": overlay,
            "s3_key": best.get("s3_key", generation_result["variants"][0].get("s3_key", "")),
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }

        self.stream_master.redis.rpush("stream:queue", json.dumps(track_data))

        self.logger.info("track_queued", pipeline_id=pipeline_id, track_id=best["track_id"])

        return {
            "pipeline_id": pipeline_id,
            "status": "success",
            "track_id": best["track_id"],
            "title": track_data["title"],
            "genre": concept["genre"],
            "score": best["overall_score"],
            "concept": concept,
            "evaluation": best,
        }

    async def _retry_pipeline(
        self, pipeline_id: str, genre: str | None, energy: int | None, attempt: int
    ) -> dict:
        """Retry the pipeline with a fresh concept."""
        if attempt > settings.max_generation_attempts:
            return {"pipeline_id": pipeline_id, "status": "failed", "reason": "max_attempts_reached"}

        self.logger.info("pipeline_retry", pipeline_id=pipeline_id, attempt=attempt)
        return await self.run_full_pipeline(genre=genre, energy=energy)

    async def ensure_buffer(self) -> dict:
        """Ensure the stream queue has enough tracks."""
        buffer_status = await self.scheduler.run({"type": "check_buffer"})

        if buffer_status["needs_generation"]:
            results = []
            for _ in range(min(buffer_status["deficit"], 5)):
                schedule = await self.scheduler.run({"type": "schedule_next"})
                result = await self.run_full_pipeline(
                    genre=schedule["genre"],
                    energy=schedule["energy"],
                )
                results.append(result)
            return {"generated": len(results), "results": results}

        return {"generated": 0, "buffer_ok": True}

    async def get_all_agent_statuses(self) -> list[dict]:
        """Get status of all agents."""
        agents = [
            self.composer, self.producer, self.critic,
            self.scheduler, self.stream_master, self.analytics, self.visual,
        ]
        statuses = []
        for agent in agents:
            status_data = agent.redis.hgetall(f"agent:status:{agent.name}")
            statuses.append({
                "agent": agent.name,
                "status": status_data.get("status", "idle"),
                "agent_id": status_data.get("agent_id"),
                "timestamp": status_data.get("timestamp"),
            })
        return statuses

    async def get_activity_log(self, limit: int = 50) -> list[dict]:
        """Get recent agent activity log."""
        raw_log = self.composer.redis.lrange("agent:activity_log", 0, limit - 1)
        return [json.loads(entry) for entry in raw_log]

    def _generate_track_title(self, concept: dict) -> str:
        """Generate a track title from concept data."""
        genre_names = {
            "drum_and_bass": "DnB", "liquid_dnb": "Liquid", "dubstep_melodic": "Melodic Dubstep",
            "house_deep": "Deep House", "house_progressive": "Progressive House",
            "trance_uplifting": "Uplifting Trance", "trance_psy": "Psytrance",
            "techno_melodic": "Melodic Techno", "breakbeat": "Breakbeat",
            "ambient": "Ambient", "downtempo": "Downtempo",
        }
        genre_name = genre_names.get(concept["genre"], concept["genre"])
        mood = concept.get("mood", "")
        return f"SonicForge — {genre_name} ({mood.title()} {concept['key']})"
