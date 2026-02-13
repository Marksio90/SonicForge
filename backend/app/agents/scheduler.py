import json
import random
from datetime import datetime, timezone

import structlog
from redis import Redis

from ..core.config import TIME_ENERGY_MAP, Genre, get_settings
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)


class SchedulerAgent(BaseAgent):
    """Plans the 24/7 playlist based on time-of-day, genre flow, and listener requests."""

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
        """Determine what track should play next based on context."""
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        # Determine preferred genres for current time
        preferred_genres, energy_level = self._get_time_context(current_hour)

        # Check for listener requests
        requests = self._get_pending_requests()

        # Check recently played to avoid repetition
        recent_genres = self._get_recent_genres(count=5)

        # Select genre with variety
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

        # Push decision to Redis queue
        self.redis.rpush(
            "schedule:decisions",
            json.dumps(schedule_decision),
        )

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
            self.logger.warning(
                "buffer_low",
                current=queue_length,
                minimum=min_buffer,
                deficit=deficit,
            )
            # Signal orchestrator to generate more tracks
            await self.send_message("orchestrator", {
                "type": "generate_tracks",
                "count": deficit,
                "priority": "high" if queue_length < 5 else "normal",
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
        self.redis.ltrim("schedule:requests", -100, -1)  # keep last 100

        return {"accepted": True, "request": request_data}

    async def get_current_queue(self) -> dict:
        """Get the current playback queue."""
        queue_items = self.redis.lrange("stream:queue", 0, 19)  # next 20
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
        """Select a genre balancing preferences, variety, and requests."""
        # Priority 1: Recent listener requests for genre
        for req in reversed(requests):
            if req.get("type") == "genre":
                requested = req.get("value", "")
                try:
                    return Genre(requested).value
                except ValueError:
                    pass

        # Priority 2: Preferred genres for this time, avoiding recent repeats
        available = [g for g in preferred if g.value not in recent[-2:]]
        if available:
            return random.choice(available).value

        # Fallback to any preferred
        if preferred:
            return random.choice(preferred).value

        return random.choice(list(Genre)).value
