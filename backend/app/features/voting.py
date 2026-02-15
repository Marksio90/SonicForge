"""
Voting System (Phase 5 - User Experience)

Implements track voting functionality:
- Star ratings (1-5)
- Upvotes/downvotes
- Real-time vote aggregation
- Popularity scoring
"""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)

# In-memory storage (use Redis/MongoDB in production)
_votes: dict[str, dict[str, int]] = {}  # track_id -> {user_id: vote}
_vote_counts: dict[str, dict] = {}  # track_id -> {total, count, avg}


class VoteRequest(BaseModel):
    """Vote request schema."""
    track_id: str
    vote: int = Field(..., ge=1, le=5, description="Rating 1-5 stars")


class VoteResponse(BaseModel):
    """Vote response schema."""
    track_id: str
    vote: int
    total_votes: int
    average_rating: float
    message: str


class TrackPopularity(BaseModel):
    """Track popularity data."""
    track_id: str
    total_votes: int
    average_rating: float
    upvotes: int
    downvotes: int
    popularity_score: float


class VotingService:
    """Service for managing track votes."""
    
    def __init__(self):
        self._redis = None
    
    async def initialize(self, redis_client=None):
        """Initialize with Redis client."""
        self._redis = redis_client
    
    async def submit_vote(
        self,
        track_id: str,
        user_id: str,
        vote: int,
    ) -> VoteResponse:
        """Submit a vote for a track."""
        if vote < 1 or vote > 5:
            raise ValueError("Vote must be between 1 and 5")
        
        # Initialize track if needed
        if track_id not in _votes:
            _votes[track_id] = {}
            _vote_counts[track_id] = {"total": 0, "count": 0, "upvotes": 0, "downvotes": 0}
        
        # Check if user already voted
        old_vote = _votes[track_id].get(user_id)
        
        if old_vote:
            # Update existing vote
            _vote_counts[track_id]["total"] -= old_vote
            if old_vote >= 4:
                _vote_counts[track_id]["upvotes"] -= 1
            elif old_vote <= 2:
                _vote_counts[track_id]["downvotes"] -= 1
        else:
            # New vote
            _vote_counts[track_id]["count"] += 1
        
        # Record new vote
        _votes[track_id][user_id] = vote
        _vote_counts[track_id]["total"] += vote
        
        if vote >= 4:
            _vote_counts[track_id]["upvotes"] += 1
        elif vote <= 2:
            _vote_counts[track_id]["downvotes"] += 1
        
        # Calculate average
        count = _vote_counts[track_id]["count"]
        total = _vote_counts[track_id]["total"]
        avg = total / count if count > 0 else 0
        
        logger.info(
            "vote_submitted",
            track_id=track_id,
            user_id=user_id,
            vote=vote,
            avg_rating=avg,
        )
        
        return VoteResponse(
            track_id=track_id,
            vote=vote,
            total_votes=count,
            average_rating=round(avg, 2),
            message="Vote submitted successfully" if not old_vote else "Vote updated successfully",
        )
    
    async def get_track_votes(self, track_id: str) -> TrackPopularity:
        """Get voting statistics for a track."""
        if track_id not in _vote_counts:
            return TrackPopularity(
                track_id=track_id,
                total_votes=0,
                average_rating=0.0,
                upvotes=0,
                downvotes=0,
                popularity_score=0.0,
            )
        
        stats = _vote_counts[track_id]
        count = stats["count"]
        avg = stats["total"] / count if count > 0 else 0
        
        # Calculate popularity score (weighted)
        # Higher weight for more votes + higher ratings
        popularity = (avg * 0.6 + min(count / 100, 1) * 0.4) * 100
        
        return TrackPopularity(
            track_id=track_id,
            total_votes=count,
            average_rating=round(avg, 2),
            upvotes=stats["upvotes"],
            downvotes=stats["downvotes"],
            popularity_score=round(popularity, 2),
        )
    
    async def get_user_vote(self, track_id: str, user_id: str) -> Optional[int]:
        """Get a user's vote for a track."""
        if track_id not in _votes:
            return None
        return _votes[track_id].get(user_id)
    
    async def get_top_tracks(self, limit: int = 10) -> list[TrackPopularity]:
        """Get top rated tracks."""
        tracks = []
        for track_id in _vote_counts:
            popularity = await self.get_track_votes(track_id)
            tracks.append(popularity)
        
        # Sort by popularity score
        tracks.sort(key=lambda x: x.popularity_score, reverse=True)
        return tracks[:limit]
    
    async def remove_vote(self, track_id: str, user_id: str) -> bool:
        """Remove a user's vote."""
        if track_id not in _votes or user_id not in _votes[track_id]:
            return False
        
        old_vote = _votes[track_id].pop(user_id)
        _vote_counts[track_id]["total"] -= old_vote
        _vote_counts[track_id]["count"] -= 1
        
        if old_vote >= 4:
            _vote_counts[track_id]["upvotes"] -= 1
        elif old_vote <= 2:
            _vote_counts[track_id]["downvotes"] -= 1
        
        logger.info("vote_removed", track_id=track_id, user_id=user_id)
        return True


# Global instance
voting_service = VotingService()
