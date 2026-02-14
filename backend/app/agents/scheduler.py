import json
import random
from datetime import datetime, timezone

import structlog
from redis import Redis

from ..core.config import GENRE_PROFILES, TIME_ENERGY_MAP, Genre, get_settings
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)


class SchedulerAgent(BaseAgent):
    """Plans the 24/7 playlist with weighted genre selection and diversity enforcement."""

    def __init__(self):
        super().__init__("scheduler")

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "schedule_next")

        if task_type == "schedule_next":
            return await self.schedule_next_track()
        elif task_type == "check_buffer":
            return await self.check_buffer()
        elif task_type == "process_request":
            return await self.process_listener_request(task["request"])
        elif task_type == "get_queue":
            return await self.get_current_queue()
        elif task_type == "override":
            return await self.override_next(track_id=task["track_id"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def schedule_next_track(self) -> dict:
        """Determine what track should play next based on time, history, and requests."""
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        preferred_genres, energy_level = self._get_time_context(current_hour)
        requests = self._get_pending_requests()
        recent_genres = self._get_recent_genres(count=5)

        # Select genre with weighted diversity algorithm
        selected_genre = self._select_genre(preferred_genres, recent_genres, requests)

        energy_map = {
            "low": 2, "low-medium": 2, "medium": 3,
            "medium-high": 4, "high": 4,
        }
        target_energy = energy_map.get(energy_level, 3)

        schedule_decision = {
            "genre": selected_genre,
            "energy": target_energy,
            "energy_level": energy_level,
            "hour": current_hour,
            "preferred_genres": [g.value for g in preferred_genres],
            "recent_genres": recent_genres,
            "has_listener_request": bool(requests),
            "timestamp": now.isoformat(),
        }

        self.redis.rpush(
            "schedule:decisions",
            json.dumps(schedule_decision),
        )
        self.redis.ltrim("schedule:decisions", -500, -1)

        self.logger.info(
            "scheduled_next",
            genre=selected_genre,
            energy=target_energy,
            hour=current_hour,
        )

        return schedule_decision

    async def check_buffer(self) -> dict:
        """Check if the track buffer is sufficient and trigger generation if needed."""
        queue_length = self.redis.llen("stream:queue")
        min_buffer = settings.buffer_min_tracks

        needs_more = queue_length < min_buffer
        deficit = max(0, min_buffer - queue_length)

        if needs_more:
            priority = "critical" if queue_length < 3 else "high" if queue_length < 5 else "normal"
            self.logger.warning(
                "buffer_low",
                current=queue_length,
                minimum=min_buffer,
                deficit=deficit,
                priority=priority,
            )
            await self.send_message("orchestrator", {
                "type": "generate_tracks",
                "count": deficit,
                "priority": priority,
            })

        return {
            "queue_length": queue_length,
            "minimum_required": min_buffer,
            "needs_generation": needs_more,
            "deficit": deficit,
        }

    async def process_listener_request(self, request: dict) -> dict:
        """Process a listener's request from chat/poll."""
        request_data = {
            "type": request.get("type", "genre"),
            "value": request.get("value"),
            "username": request.get("username"),
            "source": request.get("source", "chat"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.redis.rpush("schedule:requests", json.dumps(request_data))
        self.redis.ltrim("schedule:requests", -100, -1)

        return {"accepted": True, "request": request_data}

    async def get_current_queue(self) -> dict:
        """Get the current playback queue."""
        queue_items = self.redis.lrange("stream:queue", 0, 19)
        return {
            "queue": [json.loads(item) for item in queue_items],
            "total_length": self.redis.llen("stream:queue"),
        }

    async def override_next(self, track_id: str) -> dict:
        """Manual override — force a specific track to play next."""
        self.redis.lpush("stream:queue", json.dumps({
            "track_id": track_id,
            "override": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
        self.logger.info("manual_override", track_id=track_id)
        return {"override": True, "track_id": track_id}

    def _get_time_context(self, hour: int) -> tuple[list[Genre], str]:
        """Get preferred genres and energy level for the current hour."""
        for (start, end), (genres, energy) in TIME_ENERGY_MAP.items():
            if start <= hour < end or (start > end and (hour >= start or hour < end)):
                return genres, energy
        return [Genre.HOUSE_DEEP, Genre.AMBIENT], "low"

    def _get_pending_requests(self) -> list[dict]:
        """Get unfulfilled listener requests from Redis."""
        raw_requests = self.redis.lrange("schedule:requests", 0, -1)
        return [json.loads(r) for r in raw_requests]

    def _get_recent_genres(self, count: int = 5) -> list[str]:
        """Get the genres of recently played tracks."""
        recent = self.redis.lrange("stream:history", 0, count - 1)
        genres = []
        for item in recent:
            try:
                data = json.loads(item)
                genres.append(data.get("genre", ""))
            except (json.JSONDecodeError, KeyError):
                pass
        return genres

    def _select_genre(
        self,
        preferred: list[Genre],
        recent: list[str],
        requests: list[dict],
    ) -> str:
        """Select a genre using weighted random with diversity enforcement."""
        # Priority 1: Recent listener requests for genre
        for req in reversed(requests):
            if req.get("type") == "genre":
                requested = req.get("value", "")
                try:
                    return Genre(requested).value
                except ValueError:
                    pass

        # Priority 2: Weighted selection from preferred genres, penalizing recent repeats
        if preferred:
            weights = []
            for g in preferred:
                base_weight = GENRE_PROFILES.get(g, {}).get("weight", 1.0)
                # Penalize recently played genres for diversity
                if g.value in recent[-2:]:
                    base_weight *= 0.3
                elif g.value in recent:
                    base_weight *= 0.6
                weights.append(base_weight)

            selected = random.choices(preferred, weights=weights, k=1)[0]
            return selected.value

        return random.choice(list(Genre)).value
