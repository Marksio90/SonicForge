"""
Real-time WebSocket handlers for live dashboard updates.
"""
import asyncio
import json
import random
from datetime import datetime, timezone
from typing import Dict, List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ws_router = APIRouter()


class LiveDashboardManager:
    """Manages WebSocket connections for real-time dashboard updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task = None
        self._running = False
        
        # Simulated live data state
        self._state = {
            "stream_status": "live",
            "viewers": 127,
            "tracks_generated_today": 42,
            "total_plays": 15847,
            "active_agents": 5,
            "queue_length": 8,
            "cpu_usage": 45.2,
            "memory_usage": 62.8,
            "latency_ms": 23,
            "current_track": {
                "title": "Neon Dreams",
                "genre": "Drum & Bass",
                "bpm": 174,
                "progress": 0,
                "duration": 240,
            },
            "agents": [
                {"id": "composer", "name": "Composer", "status": "idle", "icon": "🎼"},
                {"id": "producer", "name": "Producer", "status": "working", "icon": "🎚️"},
                {"id": "critic", "name": "Critic", "status": "idle", "icon": "🔍"},
                {"id": "scheduler", "name": "Scheduler", "status": "idle", "icon": "📅"},
                {"id": "stream_master", "name": "Stream Master", "status": "working", "icon": "📡"},
                {"id": "analytics", "name": "Analytics", "status": "idle", "icon": "📊"},
                {"id": "visual", "name": "Visual", "status": "idle", "icon": "🎨"},
            ],
            "recent_events": [],
            "genre_distribution": {
                "drum_and_bass": 35,
                "house": 25,
                "techno": 20,
                "trance": 12,
                "ambient": 8,
            },
        }

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Start broadcast loop if not running
        if not self._running and self._broadcast_task is None:
            self._running = True
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        
        # Send initial state
        await websocket.send_json({
            "type": "initial_state",
            "data": self._get_full_state(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        
        # Stop broadcast loop if no connections
        if not self.active_connections and self._running:
            self._running = False

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.active_connections.discard(conn)

    def _get_full_state(self) -> dict:
        """Get complete dashboard state."""
        return {
            "stream": {
                "status": self._state["stream_status"],
                "viewers": self._state["viewers"],
                "current_track": self._state["current_track"],
            },
            "metrics": {
                "tracks_generated_today": self._state["tracks_generated_today"],
                "total_plays": self._state["total_plays"],
                "queue_length": self._state["queue_length"],
                "active_agents": self._state["active_agents"],
            },
            "system": {
                "cpu_usage": self._state["cpu_usage"],
                "memory_usage": self._state["memory_usage"],
                "latency_ms": self._state["latency_ms"],
            },
            "agents": self._state["agents"],
            "recent_events": self._state["recent_events"][-10:],
            "genre_distribution": self._state["genre_distribution"],
        }

    async def _broadcast_loop(self):
        """Background task to broadcast updates every second."""
        while self._running:
            try:
                # Simulate live data changes
                self._update_simulated_data()
                
                # Broadcast incremental update
                await self.broadcast({
                    "type": "live_update",
                    "data": {
                        "viewers": self._state["viewers"],
                        "current_track": self._state["current_track"],
                        "system": {
                            "cpu_usage": self._state["cpu_usage"],
                            "memory_usage": self._state["memory_usage"],
                            "latency_ms": self._state["latency_ms"],
                        },
                        "agents": self._state["agents"],
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Broadcast error: {e}")
                await asyncio.sleep(1)

    def _update_simulated_data(self):
        """Simulate live data changes for demo purposes."""
        # Fluctuate viewers
        self._state["viewers"] = max(10, self._state["viewers"] + random.randint(-5, 7))
        
        # Update system metrics
        self._state["cpu_usage"] = round(max(10, min(95, self._state["cpu_usage"] + random.uniform(-3, 3))), 1)
        self._state["memory_usage"] = round(max(30, min(90, self._state["memory_usage"] + random.uniform(-2, 2))), 1)
        self._state["latency_ms"] = max(5, min(100, self._state["latency_ms"] + random.randint(-5, 5)))
        
        # Progress current track
        if self._state["current_track"]["progress"] < self._state["current_track"]["duration"]:
            self._state["current_track"]["progress"] += 1
        else:
            # New track
            self._state["current_track"] = self._generate_new_track()
            self._state["tracks_generated_today"] += 1
            self._state["total_plays"] += 1
            self._add_event("track_generated", f"New track: {self._state['current_track']['title']}")
        
        # Random agent status changes
        if random.random() < 0.1:
            agent_idx = random.randint(0, len(self._state["agents"]) - 1)
            new_status = random.choice(["idle", "working", "idle", "idle"])
            old_status = self._state["agents"][agent_idx]["status"]
            if old_status != new_status:
                self._state["agents"][agent_idx]["status"] = new_status
                agent_name = self._state["agents"][agent_idx]["name"]
                self._add_event("agent_status", f"{agent_name} → {new_status}")
        
        # Count active agents
        self._state["active_agents"] = sum(
            1 for a in self._state["agents"] if a["status"] == "working"
        )
        
        # Random queue changes
        if random.random() < 0.05:
            self._state["queue_length"] = max(0, self._state["queue_length"] + random.randint(-1, 2))

    def _generate_new_track(self) -> dict:
        """Generate a new random track."""
        titles = [
            "Neon Dreams", "Midnight Pulse", "Digital Storm", "Cosmic Wave",
            "Electric Soul", "Binary Sunset", "Quantum Flow", "Neural Path",
            "Cyber Dawn", "Plasma Rush", "Void Walker", "Signal Fire",
        ]
        genres = ["Drum & Bass", "House", "Techno", "Trance", "Ambient"]
        bpms = {"Drum & Bass": 174, "House": 128, "Techno": 135, "Trance": 140, "Ambient": 90}
        
        genre = random.choice(genres)
        return {
            "title": f"{random.choice(titles)} {random.randint(1, 99):02d}",
            "genre": genre,
            "bpm": bpms[genre] + random.randint(-5, 5),
            "progress": 0,
            "duration": random.randint(180, 300),
        }

    def _add_event(self, event_type: str, message: str):
        """Add event to recent events list."""
        event = {
            "type": event_type,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._state["recent_events"].append(event)
        
        # Keep only last 50 events
        if len(self._state["recent_events"]) > 50:
            self._state["recent_events"] = self._state["recent_events"][-50:]


# Global manager instance
live_manager = LiveDashboardManager()


@ws_router.websocket("/ws/live")
async def live_dashboard_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live dashboard updates.
    
    Sends updates every second with:
    - Stream status and viewer count
    - Current track with progress
    - System metrics (CPU, Memory, Latency)
    - Agent statuses
    - Recent events
    """
    await live_manager.connect(websocket)
    
    try:
        while True:
            # Wait for client messages (commands)
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                # Handle client commands
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "request_full_state":
                    await websocket.send_json({
                        "type": "full_state",
                        "data": live_manager._get_full_state(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        live_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        live_manager.disconnect(websocket)
