"""
Recommendation Engine (Phase 5 - User Experience)

Implements AI-powered music recommendations:
- Content-based filtering (genre, BPM, mood)
- Collaborative filtering (user preferences)
- Trending tracks
- Personalized playlists
"""

import random
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)

# Simulated track database
_tracks: dict[str, dict] = {}
_user_history: dict[str, list[str]] = {}  # user_id -> [track_ids]
_user_preferences: dict[str, dict] = {}  # user_id -> {genre: weight, ...}


class TrackMetadata(BaseModel):
    """Track metadata for recommendations."""
    track_id: str
    title: str
    genre: str
    bpm: int
    energy: float = Field(..., ge=0, le=1)
    mood: str
    key: str
    duration: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecommendationItem(BaseModel):
    """Single recommendation item."""
    track_id: str
    title: str
    genre: str
    score: float = Field(..., ge=0, le=100)
    reason: str


class RecommendationResponse(BaseModel):
    """Recommendation response."""
    user_id: Optional[str] = None
    recommendations: list[RecommendationItem]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecommendationEngine:
    """AI-powered recommendation engine."""
    
    def __init__(self):
        self._genres = [
            "drum_and_bass", "liquid_dnb", "dubstep_melodic",
            "house_deep", "house_progressive", "trance_uplifting",
            "trance_psy", "techno_melodic", "breakbeat", "ambient"
        ]
        self._moods = ["energetic", "chill", "dark", "euphoric", "melancholic", "uplifting"]
    
    def add_track(self, track: TrackMetadata) -> None:
        """Add track to recommendation database."""
        _tracks[track.track_id] = track.model_dump()
        logger.info("track_added_to_recommendations", track_id=track.track_id)
    
    def record_listen(self, user_id: str, track_id: str) -> None:
        """Record that a user listened to a track."""
        if user_id not in _user_history:
            _user_history[user_id] = []
        
        # Keep last 100 tracks
        _user_history[user_id].append(track_id)
        if len(_user_history[user_id]) > 100:
            _user_history[user_id] = _user_history[user_id][-100:]
        
        # Update preferences
        if track_id in _tracks:
            track = _tracks[track_id]
            if user_id not in _user_preferences:
                _user_preferences[user_id] = {}
            
            genre = track.get("genre", "unknown")
            _user_preferences[user_id][genre] = _user_preferences[user_id].get(genre, 0) + 1
    
    async def get_personalized_recommendations(
        self,
        user_id: str,
        limit: int = 10,
    ) -> RecommendationResponse:
        """Get personalized recommendations for a user."""
        recommendations = []
        
        # Get user preferences
        prefs = _user_preferences.get(user_id, {})
        history = set(_user_history.get(user_id, []))
        
        # Score each track
        for track_id, track in _tracks.items():
            if track_id in history:
                continue  # Skip already listened tracks
            
            score = 50.0  # Base score
            reason = "Recommended for you"
            
            # Genre preference boost
            genre = track.get("genre", "unknown")
            if genre in prefs:
                genre_score = min(prefs[genre] * 5, 30)
                score += genre_score
                reason = f"Based on your love for {genre.replace('_', ' ')}"
            
            # Add some randomness
            score += random.uniform(-10, 10)
            score = max(0, min(100, score))
            
            recommendations.append(RecommendationItem(
                track_id=track_id,
                title=track.get("title", f"Track {track_id[:8]}"),
                genre=genre,
                score=round(score, 2),
                reason=reason,
            ))
        
        # Sort by score and limit
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(
            "personalized_recommendations_generated",
            user_id=user_id,
            count=len(recommendations[:limit]),
        )
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations[:limit],
        )
    
    async def get_similar_tracks(
        self,
        track_id: str,
        limit: int = 5,
    ) -> RecommendationResponse:
        """Get tracks similar to a given track."""
        if track_id not in _tracks:
            return RecommendationResponse(recommendations=[])
        
        source_track = _tracks[track_id]
        recommendations = []
        
        for tid, track in _tracks.items():
            if tid == track_id:
                continue
            
            score = 0.0
            reasons = []
            
            # Same genre
            if track.get("genre") == source_track.get("genre"):
                score += 40
                reasons.append("Same genre")
            
            # Similar BPM (within 10%)
            source_bpm = source_track.get("bpm", 0)
            track_bpm = track.get("bpm", 0)
            if source_bpm and track_bpm:
                bpm_diff = abs(source_bpm - track_bpm) / source_bpm
                if bpm_diff < 0.1:
                    score += 30
                    reasons.append("Similar tempo")
            
            # Same mood
            if track.get("mood") == source_track.get("mood"):
                score += 20
                reasons.append("Same vibe")
            
            # Same key
            if track.get("key") == source_track.get("key"):
                score += 10
                reasons.append("Harmonic match")
            
            if score > 0:
                recommendations.append(RecommendationItem(
                    track_id=tid,
                    title=track.get("title", f"Track {tid[:8]}"),
                    genre=track.get("genre", "unknown"),
                    score=round(score, 2),
                    reason=" • ".join(reasons) if reasons else "Similar vibes",
                ))
        
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return RecommendationResponse(
            recommendations=recommendations[:limit],
        )
    
    async def get_trending_tracks(self, limit: int = 10) -> RecommendationResponse:
        """Get currently trending tracks."""
        # In production, this would use real-time data
        recommendations = []
        
        for track_id, track in list(_tracks.items())[:limit]:
            recommendations.append(RecommendationItem(
                track_id=track_id,
                title=track.get("title", f"Track {track_id[:8]}"),
                genre=track.get("genre", "unknown"),
                score=round(random.uniform(70, 100), 2),
                reason="Trending now",
            ))
        
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return RecommendationResponse(
            recommendations=recommendations,
        )
    
    async def get_genre_recommendations(
        self,
        genre: str,
        limit: int = 10,
    ) -> RecommendationResponse:
        """Get recommendations for a specific genre."""
        recommendations = []
        
        for track_id, track in _tracks.items():
            if track.get("genre") == genre:
                recommendations.append(RecommendationItem(
                    track_id=track_id,
                    title=track.get("title", f"Track {track_id[:8]}"),
                    genre=genre,
                    score=round(random.uniform(60, 100), 2),
                    reason=f"Top {genre.replace('_', ' ')} track",
                ))
        
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return RecommendationResponse(
            recommendations=recommendations[:limit],
        )


# Global instance
recommendation_engine = RecommendationEngine()
