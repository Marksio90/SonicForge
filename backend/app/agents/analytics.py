import json
from datetime import datetime, timezone

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.config import get_settings
from .base import BaseAgent

settings = get_settings()
logger = structlog.get_logger(__name__)

# Shared httpx client for connection pooling
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=15, write=10, pool=5),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _http_client


class AnalyticsAgent(BaseAgent):
    """Collects and analyzes streaming analytics with connection pooling and caching."""

    def __init__(self):
        super().__init__("analytics")

    async def execute(self, task: dict) -> dict:
        task_type = task.get("type", "snapshot")

        if task_type == "snapshot":
            return await self.take_snapshot()
        elif task_type == "track_performance":
            return await self.get_track_performance(track_id=task["track_id"])
        elif task_type == "genre_analysis":
            return await self.analyze_genre_performance()
        elif task_type == "daily_report":
            return await self.generate_daily_report()
        elif task_type == "ab_test_results":
            return await self.get_ab_test_results()
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def take_snapshot(self) -> dict:
        """Take a snapshot of current streaming analytics."""
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "concurrent_viewers": 0,
            "total_views": 0,
            "chat_messages": 0,
            "likes": 0,
            "new_subscribers": 0,
        }

        if settings.youtube_api_key:
            yt_data = await self._fetch_youtube_analytics()
            snapshot.update(yt_data)

        current_track = self.redis.hgetall("stream:current_track")
        snapshot["current_track"] = current_track
        snapshot["current_genre"] = current_track.get("genre", "unknown")

        self.redis.lpush("analytics:snapshots", json.dumps(snapshot))
        self.redis.ltrim("analytics:snapshots", 0, 8639)

        self.redis.hincrby("analytics:daily", "total_snapshots", 1)

        self.logger.info(
            "snapshot_taken",
            viewers=snapshot.get("concurrent_viewers", 0),
        )

        return snapshot

    async def get_track_performance(self, track_id: str) -> dict:
        """Get performance metrics for a specific track."""
        history = self.redis.lrange("stream:history", 0, -1)
        track_plays = []

        for item in history:
            data = json.loads(item)
            if data.get("track_id") == track_id:
                track_plays.append(data)

        if not track_plays:
            return {"track_id": track_id, "plays": 0, "found": False}

        return {
            "track_id": track_id,
            "plays": len(track_plays),
            "first_played": track_plays[-1].get("played_at"),
            "last_played": track_plays[0].get("played_at"),
            "found": True,
        }

    async def analyze_genre_performance(self) -> dict:
        """Analyze performance metrics by genre with cached aggregation."""
        # Check cache first
        cached = self.redis.get("analytics:genre_perf_cache")
        if cached:
            cache_data = json.loads(cached)
            cache_age = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(cache_data.get("cached_at", "2000-01-01T00:00:00+00:00"))
            ).total_seconds()
            if cache_age < 300:  # 5 minute cache
                return cache_data["data"]

        snapshots = self.redis.lrange("analytics:snapshots", 0, -1)

        genre_data: dict[str, list] = {}
        for raw in snapshots:
            snap = json.loads(raw)
            genre = snap.get("current_genre", "unknown")
            viewers = snap.get("concurrent_viewers", 0)
            if genre not in genre_data:
                genre_data[genre] = []
            genre_data[genre].append(viewers)

        results = {}
        for genre, viewer_counts in genre_data.items():
            if viewer_counts:
                sorted_counts = sorted(viewer_counts)
                results[genre] = {
                    "avg_viewers": sum(viewer_counts) / len(viewer_counts),
                    "peak_viewers": max(viewer_counts),
                    "median_viewers": sorted_counts[len(sorted_counts) // 2],
                    "sample_count": len(viewer_counts),
                }

        result = {"genre_performance": results}

        # Cache the result
        self.redis.setex(
            "analytics:genre_perf_cache",
            300,
            json.dumps({"data": result, "cached_at": datetime.now(timezone.utc).isoformat()}),
        )

        return result

    async def generate_daily_report(self) -> dict:
        """Generate a daily analytics report."""
        snapshots = self.redis.lrange("analytics:snapshots", 0, -1)
        history = self.redis.lrange("stream:history", 0, -1)

        viewer_counts = []
        for raw in snapshots:
            snap = json.loads(raw)
            viewer_counts.append(snap.get("concurrent_viewers", 0))

        tracks_played = len(history)

        report = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "tracks_played": tracks_played,
            "peak_viewers": max(viewer_counts) if viewer_counts else 0,
            "avg_viewers": sum(viewer_counts) / len(viewer_counts) if viewer_counts else 0,
            "total_snapshots": len(snapshots),
            "genre_breakdown": (await self.analyze_genre_performance()).get("genre_performance", {}),
        }

        if settings.telegram_bot_token:
            await self._send_telegram_report(report)

        return report

    async def get_ab_test_results(self) -> dict:
        """Get A/B test results for genre/time combinations."""
        genre_perf = await self.analyze_genre_performance()
        return {
            "test_type": "genre_time_performance",
            "results": genre_perf,
            "recommendation": "Based on collected data",
        }

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def _fetch_youtube_analytics(self) -> dict:
        """Fetch real-time analytics from YouTube Live API with retry."""
        try:
            client = _get_http_client()
            response = await client.get(
                "https://www.googleapis.com/youtube/v3/liveBroadcasts",
                params={
                    "part": "snippet,status,statistics",
                    "broadcastStatus": "active",
                    "key": settings.youtube_api_key,
                },
            )
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                if items:
                    stats = items[0].get("statistics", {})
                    return {
                        "concurrent_viewers": int(stats.get("concurrentViewers", 0)),
                        "total_views": int(stats.get("totalChatMessages", 0)),
                    }
        except Exception as e:
            self.logger.warning("youtube_api_error", error=str(e))

        return {}

    async def _send_telegram_report(self, report: dict) -> None:
        """Send daily report via Telegram bot."""
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return

        message = (
            f"SonicForge Daily Report -- {report['date']}\n\n"
            f"Tracks played: {report['tracks_played']}\n"
            f"Peak viewers: {report['peak_viewers']}\n"
            f"Avg viewers: {report['avg_viewers']:.0f}\n"
        )

        try:
            client = _get_http_client()
            await client.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                json={
                    "chat_id": settings.telegram_chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
            )
        except Exception as e:
            self.logger.warning("telegram_send_failed", error=str(e))
