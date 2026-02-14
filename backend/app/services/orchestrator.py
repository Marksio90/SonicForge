import asyncio
import json
import time
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
    """Central pipeline orchestrator coordinating all agents with parallel execution.

    SonicForge 2.0 pipeline flow:
    1. Compose — create concept (Composer)
    2. Pre-evaluate — check concept before generation (Critic)
    3. Produce — generate variants via MusicGen with auto-mastering (Producer)
    4. Evaluate — parallel quality assessment with key/mudiness detection (Critic)
    5. Visual — generate SD thumbnail + AnimateDiff video loop (Visual)
    6. Queue — add approved track to stream queue (Scheduler)
    """

    def __init__(self):
        self.composer = ComposerAgent()
        self.producer = ProducerAgent()
        self.critic = CriticAgent()
        self.scheduler = SchedulerAgent()
        self.stream_master = StreamMasterAgent()
        self.analytics = AnalyticsAgent()
        self.visual = VisualAgent()
        self.logger = logger.bind(component="orchestrator")

    async def run_full_pipeline(
        self, genre: str | None = None, energy: int | None = None
    ) -> dict:
        """Run the complete production pipeline with pre-evaluation and mastering."""
        pipeline_id = str(uuid.uuid4())
        start_time = time.monotonic()
        self.logger.info("pipeline_started", pipeline_id=pipeline_id, genre=genre)

        try:
            # Step 1: Compose — create concept
            concept = await self.composer.run({
                "type": "create_concept",
                "genre": genre,
                "energy": energy,
            })

            # Step 2: Pre-evaluate — Critic checks concept BEFORE generation
            pre_eval = await self.critic.run({
                "type": "pre_evaluate",
                "concept": concept,
            })

            if not pre_eval.get("approved", True):
                self.logger.warning(
                    "concept_rejected_pre_evaluation",
                    pipeline_id=pipeline_id,
                    issues=pre_eval.get("issues", []),
                )
                # Regenerate concept with feedback
                concept = await self.composer.run({
                    "type": "create_concept",
                    "genre": genre,
                    "energy": energy,
                    "feedback": pre_eval.get("issues", []),
                })

            # Step 3: Produce — generate variants with MusicGen + auto-mastering
            generation = await self.producer.run({
                "type": "generate",
                "concept": concept,
                "variants": settings.variants_per_concept,
            })

            # Step 4: Evaluate — parallel quality assessment (key, mudiness, spectral)
            evaluation = await self.critic.run({
                "type": "evaluate_batch",
                "variants": generation["variants"],
            })

            best_track = evaluation.get("best_approved") or evaluation.get("best")
            if not best_track:
                self.logger.error(
                    "pipeline_failed_no_approved_track", pipeline_id=pipeline_id
                )
                return {
                    "pipeline_id": pipeline_id,
                    "status": "failed",
                    "reason": "no_tracks_approved",
                    "evaluation": evaluation,
                }

            # Step 5: Visual + Overlay + Video loop — run in parallel
            visual_task = self.visual.run({
                "type": "generate_visual",
                "track_id": best_track["track_id"],
                "genre": concept.get("genre"),
                "bpm": concept.get("bpm"),
                "title": concept.get("genre", "").replace("_", " ").title(),
            })

            overlay_task = self.visual.run({
                "type": "generate_overlay",
                "track_id": best_track["track_id"],
                "title": concept.get("genre", "").replace("_", " ").title(),
                "bpm": concept.get("bpm"),
                "key": concept.get("key"),
                "genre": concept.get("genre"),
                "score": best_track.get("overall_score"),
            })

            visual_result, overlay_result = await asyncio.gather(
                visual_task, overlay_task
            )

            # Step 6: Add to stream queue
            queue_item = {
                "track_id": best_track["track_id"],
                "genre": concept.get("genre"),
                "bpm": concept.get("bpm"),
                "key": concept.get("key"),
                "title": concept.get("genre", "").replace("_", " ").title(),
                "score": best_track.get("overall_score"),
                "detected_key": best_track.get("detected_key"),
                "is_muddy": best_track.get("is_muddy", False),
                "engine": best_track.get("engine", "musicgen_local"),
                "queued_at": datetime.now(timezone.utc).isoformat(),
            }
            from ..agents.base import get_shared_redis

            redis = get_shared_redis()
            redis.rpush("stream:queue", json.dumps(queue_item))

            duration_ms = (time.monotonic() - start_time) * 1000

            result = {
                "pipeline_id": pipeline_id,
                "status": "success",
                "concept": concept,
                "pre_evaluation": pre_eval,
                "generation": {
                    "concept_id": generation["concept_id"],
                    "variants_count": len(generation["variants"]),
                    "engine": best_track.get("engine", "unknown"),
                },
                "evaluation": {
                    "best_track_id": best_track["track_id"],
                    "best_score": best_track.get("overall_score"),
                    "approved_count": evaluation.get("approved_count", 0),
                    "total_evaluated": evaluation.get("total_count", 0),
                    "detected_key": best_track.get("detected_key"),
                    "is_muddy": best_track.get("is_muddy", False),
                },
                "visual": visual_result,
                "overlay": overlay_result,
                "duration_ms": round(duration_ms, 1),
            }

            self.logger.info(
                "pipeline_completed",
                pipeline_id=pipeline_id,
                duration_ms=round(duration_ms, 1),
                best_score=best_track.get("overall_score"),
                engine=best_track.get("engine"),
            )

            return result

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.logger.error(
                "pipeline_failed",
                pipeline_id=pipeline_id,
                error=str(e),
                duration_ms=round(duration_ms, 1),
            )
            return {
                "pipeline_id": pipeline_id,
                "status": "error",
                "error": str(e),
                "duration_ms": round(duration_ms, 1),
            }

    async def run_batch_pipeline(
        self, count: int = 5, genre: str | None = None
    ) -> dict:
        """Run multiple pipelines in parallel for batch generation."""
        tasks = [self.run_full_pipeline(genre=genre) for _ in range(count)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        succeeded = []
        failed = []
        for result in results:
            if isinstance(result, Exception):
                failed.append(str(result))
            elif result.get("status") == "success":
                succeeded.append(result)
            else:
                failed.append(result.get("error", "unknown"))

        return {
            "total_requested": count,
            "succeeded": len(succeeded),
            "failed": len(failed),
            "results": succeeded,
            "errors": failed,
        }

    def get_agent_statuses(self) -> list[dict]:
        """Get status of all agents."""
        agents = [
            self.composer,
            self.producer,
            self.critic,
            self.scheduler,
            self.stream_master,
            self.analytics,
            self.visual,
        ]
        return [
            {
                "agent": agent.name,
                "status": agent.status.value,
                "agent_id": agent.agent_id,
                "tasks_completed": agent._task_count,
                "errors": agent._error_count,
            }
            for agent in agents
        ]
