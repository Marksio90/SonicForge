import asyncio
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


class YouTubeChatBot:
    """AI DJ chat bot that interacts with YouTube Live Chat.

    Responds to viewer commands (!request, !nowplaying, !queue) and
    provides personality-driven responses as the AI DJ.
    """

    COMMANDS = {
        "!request": "handle_request",
        "!req": "handle_request",
        "!nowplaying": "handle_now_playing",
        "!np": "handle_now_playing",
        "!queue": "handle_queue",
        "!q": "handle_queue",
        "!genre": "handle_genre_info",
        "!help": "handle_help",
        "!dj": "handle_dj",
    }

    def __init__(self, redis_client):
        self.redis = redis_client
        self.logger = structlog.get_logger("youtube_chat_bot")
        self._live_chat_id: str | None = None

    async def process_chat_messages(self, messages: list[dict]) -> list[dict]:
        """Process incoming chat messages and generate AI DJ responses."""
        responses = []
        for msg in messages:
            text = msg.get("text", "").strip().lower()
            author = msg.get("author", "Listener")

            # Check if message starts with a command
            for cmd, handler_name in self.COMMANDS.items():
                if text.startswith(cmd):
                    args = text[len(cmd) :].strip()
                    handler = getattr(self, handler_name)
                    response = await handler(author, args)
                    if response:
                        responses.append(response)
                    break

        return responses

    async def handle_request(self, author: str, args: str) -> dict | None:
        """Handle !request <genre> command."""
        if not args:
            return {
                "text": f"@{author} Usage: !request <genre> "
                "(e.g., !request dnb, !request techno)",
                "type": "help",
            }

        # Map common abbreviations
        genre_map = {
            "dnb": "drum_and_bass",
            "drum and bass": "drum_and_bass",
            "liquid": "liquid_dnb",
            "liquid dnb": "liquid_dnb",
            "dubstep": "dubstep_melodic",
            "house": "house_deep",
            "deep house": "house_deep",
            "progressive": "house_progressive",
            "prog house": "house_progressive",
            "trance": "trance_uplifting",
            "uplifting": "trance_uplifting",
            "psy": "trance_psy",
            "psytrance": "trance_psy",
            "techno": "techno_melodic",
            "melodic techno": "techno_melodic",
            "breakbeat": "breakbeat",
            "breaks": "breakbeat",
            "ambient": "ambient",
            "downtempo": "downtempo",
            "chill": "downtempo",
        }

        genre = genre_map.get(args.lower(), args.lower())

        # Add to request queue
        request_data = {
            "genre": genre,
            "requested_by": author,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        self.redis.lpush("schedule:requests", json.dumps(request_data))
        self.redis.ltrim("schedule:requests", 0, 99)

        # Estimate queue position
        queue_len = self.redis.llen("stream:queue")
        est_minutes = max(3, queue_len * 3)

        display_genre = genre.replace("_", " ").title()
        return {
            "text": f"Yo @{author}, wrzucam {display_genre} do kolejki! "
            f"Wejdzie za okolo {est_minutes} minut!",
            "type": "request_confirmed",
            "genre": genre,
        }

    async def handle_now_playing(self, author: str, args: str) -> dict | None:
        """Handle !nowplaying command."""
        current = self.redis.hgetall("stream:current_track")
        if not current:
            return {"text": f"@{author} Nothing playing right now!", "type": "info"}

        title = current.get("title", "Unknown")
        genre = current.get("genre", "Electronic").replace("_", " ").title()
        bpm = current.get("bpm", "?")
        key = current.get("key", "?")
        score = current.get("score", "?")

        return {
            "text": f"@{author} Now Playing: {title} | {genre} | "
            f"BPM: {bpm} | Key: {key} | Quality: {score}/10",
            "type": "now_playing",
        }

    async def handle_queue(self, author: str, args: str) -> dict | None:
        """Handle !queue command — show next 3 tracks."""
        queue_raw = self.redis.lrange("stream:queue", 0, 2)
        if not queue_raw:
            return {
                "text": f"@{author} Queue is empty — AI is cooking up new tracks!",
                "type": "info",
            }

        tracks = []
        for i, raw in enumerate(queue_raw):
            item = json.loads(raw)
            title = item.get("title", "Unknown")
            genre = item.get("genre", "?").replace("_", " ").title()
            tracks.append(f"{i + 1}. {title} ({genre})")

        queue_text = " | ".join(tracks)
        return {
            "text": f"@{author} Coming up: {queue_text}",
            "type": "queue",
        }

    async def handle_genre_info(self, author: str, args: str) -> dict | None:
        """Handle !genre command — show available genres."""
        genres = [
            "DnB", "Liquid DnB", "Dubstep", "Deep House", "Progressive",
            "Trance", "Psytrance", "Techno", "Breakbeat", "Ambient", "Downtempo",
        ]
        return {
            "text": f"@{author} Available genres: {', '.join(genres)}. "
            "Use !request <genre> to add to queue!",
            "type": "info",
        }

    async def handle_help(self, author: str, args: str) -> dict | None:
        """Handle !help command."""
        return {
            "text": f"@{author} Commands: !request <genre> | !nowplaying | "
            "!queue | !genre | !dj",
            "type": "help",
        }

    async def handle_dj(self, author: str, args: str) -> dict | None:
        """Handle !dj command — AI DJ personality response."""
        return {
            "text": f"@{author} Jestem SonicForge AI DJ — generuje muzyke 24/7 "
            "z uzyciem Meta AudioCraft + Stable Diffusion. "
            "Cala muzyka jest tworzona w czasie rzeczywistym przez AI!",
            "type": "personality",
        }

    async def send_response(self, response: dict) -> bool:
        """Send a chat message to YouTube Live Chat."""
        if not self._live_chat_id or not settings.youtube_api_key:
            return False

        try:
            client = _get_http_client()
            await client.post(
                "https://www.googleapis.com/youtube/v3/liveChat/messages",
                params={"part": "snippet", "key": settings.youtube_api_key},
                json={
                    "snippet": {
                        "liveChatId": self._live_chat_id,
                        "type": "textMessageEvent",
                        "textMessageDetails": {"messageText": response["text"]},
                    }
                },
            )
            return True
        except Exception as e:
            logger.warning("chat_send_failed", error=str(e))
            return False

    async def fetch_live_chat_id(self) -> str | None:
        """Get the live chat ID for the current broadcast."""
        if not settings.youtube_api_key:
            return None

        try:
            client = _get_http_client()
            response = await client.get(
                "https://www.googleapis.com/youtube/v3/liveBroadcasts",
                params={
                    "part": "snippet",
                    "broadcastStatus": "active",
                    "key": settings.youtube_api_key,
                },
            )
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                if items:
                    self._live_chat_id = items[0]["snippet"]["liveChatId"]
                    return self._live_chat_id
        except Exception as e:
            logger.warning("fetch_chat_id_failed", error=str(e))

        return None

    async def poll_chat_messages(self) -> list[dict]:
        """Poll for new chat messages from YouTube Live Chat."""
        if not self._live_chat_id or not settings.youtube_api_key:
            return []

        try:
            client = _get_http_client()
            response = await client.get(
                "https://www.googleapis.com/youtube/v3/liveChat/messages",
                params={
                    "liveChatId": self._live_chat_id,
                    "part": "snippet,authorDetails",
                    "key": settings.youtube_api_key,
                },
            )
            if response.status_code == 200:
                data = response.json()
                messages = []
                for item in data.get("items", []):
                    messages.append(
                        {
                            "text": item["snippet"]["textMessageDetails"][
                                "messageText"
                            ],
                            "author": item["authorDetails"]["displayName"],
                            "timestamp": item["snippet"]["publishedAt"],
                        }
                    )
                return messages
        except Exception as e:
            logger.warning("poll_chat_failed", error=str(e))

        return []


class AnalyticsAgent(BaseAgent):
    """Collects and analyzes streaming analytics with YouTube chat bot integration."""

    def __init__(self):
        super().__init__("analytics")
        self._chat_bot = YouTubeChatBot(self.redis)

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
        elif task_type == "process_chat":
            return await self.process_chat()
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    async def take_snapshot(self) -> dict:
        """Take a snapshot of current streaming analytics and process chat."""
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

        # Process chat messages (AI DJ interaction)
        chat_result = await self.process_chat()
        snapshot["chat_responses"] = chat_result.get("responses_sent", 0)

        self.logger.info(
            "snapshot_taken",
            viewers=snapshot.get("concurrent_viewers", 0),
            chat_responses=snapshot.get("chat_responses", 0),
        )

        return snapshot

    async def process_chat(self) -> dict:
        """Process YouTube chat and respond as AI DJ."""
        # Ensure we have the live chat ID
        if not self._chat_bot._live_chat_id:
            await self._chat_bot.fetch_live_chat_id()

        # Poll for new messages
        messages = await self._chat_bot.poll_chat_messages()
        if not messages:
            return {"messages_processed": 0, "responses_sent": 0}

        # Process commands and generate responses
        responses = await self._chat_bot.process_chat_messages(messages)

        # Send responses to chat
        sent_count = 0
        for response in responses:
            success = await self._chat_bot.send_response(response)
            if success:
                sent_count += 1

            # Log chat interactions
            self.redis.lpush(
                "analytics:chat_log",
                json.dumps(
                    {
                        "response": response,
                        "sent": success,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ),
            )
            self.redis.ltrim("analytics:chat_log", 0, 999)

        return {
            "messages_processed": len(messages),
            "responses_sent": sent_count,
            "responses": responses,
        }

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
                - datetime.fromisoformat(
                    cache_data.get("cached_at", "2000-01-01T00:00:00+00:00")
                )
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
            json.dumps(
                {
                    "data": result,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
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
            "avg_viewers": (
                sum(viewer_counts) / len(viewer_counts) if viewer_counts else 0
            ),
            "total_snapshots": len(snapshots),
            "genre_breakdown": (await self.analyze_genre_performance()).get(
                "genre_performance", {}
            ),
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

    @retry(
        stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5)
    )
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
                        "concurrent_viewers": int(
                            stats.get("concurrentViewers", 0)
                        ),
                        "total_views": int(
                            stats.get("totalChatMessages", 0)
                        ),
                    }
        except Exception as e:
            self.logger.warning("youtube_api_error", error=str(e))

        return {}

    async def _send_telegram_report(self, report: dict) -> None:
        """Send daily report via Telegram bot."""
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return

        message = (
            f"SonicForge 2.0 Daily Report -- {report['date']}\n\n"
            f"Tracks played: {report['tracks_played']}\n"
            f"Peak viewers: {report['peak_viewers']}\n"
            f"Avg viewers: {report['avg_viewers']:.0f}\n"
            f"\nPowered by Meta AudioCraft + Stable Diffusion"
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
