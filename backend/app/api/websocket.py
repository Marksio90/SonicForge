import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from redis import Redis

from ..core.config import get_settings

settings = get_settings()

ws_router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time dashboard updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)


manager = ConnectionManager()


@ws_router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await manager.connect(websocket)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)

    try:
        while True:
            # Send current status every 2 seconds
            stream_status = redis.hgetall("stream:status")
            current_track = redis.hgetall("stream:current_track")
            queue_length = redis.llen("stream:queue")

            # Get latest health check
            health_log = redis.lrange("stream:health_log", 0, 0)
            latest_health = json.loads(health_log[0]) if health_log else {}

            await websocket.send_json({
                "type": "status_update",
                "data": {
                    "stream": stream_status,
                    "current_track": current_track,
                    "queue_length": queue_length,
                    "health": latest_health,
                },
            })

            # Check for messages from client
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
                if data.get("type") == "request":
                    # Handle listener requests from dashboard
                    redis.rpush("schedule:requests", json.dumps(data.get("payload", {})))
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
