"""
Phase 5 API Routes: User Experience & Monetization

Implements endpoints for:
- Track voting
- Recommendations
- Social sharing
- Payments & subscriptions
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from ..features.voting import (
    voting_service,
    VoteRequest,
    VoteResponse,
    TrackPopularity,
)
from ..features.recommendations import (
    recommendation_engine,
    RecommendationResponse,
    TrackMetadata,
)
from ..features.social_sharing import (
    social_sharing,
    ShareLink,
    SocialShareUrls,
    ShareAnalytics,
)
from ..features.payments import (
    payment_service,
    SubscriptionPlan,
    UserSubscription,
    CheckoutResponse,
    SUBSCRIPTION_PLANS,
    CREDIT_PACKAGES,
)
from ..security.auth import get_current_user, get_current_active_user, TokenPayload

router = APIRouter(tags=["phase5"])


# ==================== VOTING ====================

@router.post("/vote", response_model=VoteResponse)
async def submit_vote(
    request: VoteRequest,
    user: TokenPayload = Depends(get_current_active_user),
):
    """Submit a vote for a track (1-5 stars)."""
    return await voting_service.submit_vote(
        track_id=request.track_id,
        user_id=user.sub,
        vote=request.vote,
    )


@router.get("/vote/{track_id}", response_model=TrackPopularity)
async def get_track_votes(track_id: str):
    """Get voting statistics for a track."""
    return await voting_service.get_track_votes(track_id)


@router.get("/vote/{track_id}/user")
async def get_user_vote(
    track_id: str,
    user: TokenPayload = Depends(get_current_active_user),
):
    """Get current user's vote for a track."""
    vote = await voting_service.get_user_vote(track_id, user.sub)
    return {"track_id": track_id, "vote": vote}


@router.delete("/vote/{track_id}")
async def remove_vote(
    track_id: str,
    user: TokenPayload = Depends(get_current_active_user),
):
    """Remove user's vote for a track."""
    removed = await voting_service.remove_vote(track_id, user.sub)
    if not removed:
        raise HTTPException(status_code=404, detail="Vote not found")
    return {"message": "Vote removed"}


@router.get("/top-tracks", response_model=list[TrackPopularity])
async def get_top_tracks(limit: int = 10):
    """Get top rated tracks."""
    return await voting_service.get_top_tracks(limit)


# ==================== RECOMMENDATIONS ====================

@router.get("/recommendations/personalized", response_model=RecommendationResponse)
async def get_personalized_recommendations(
    limit: int = 10,
    user: TokenPayload = Depends(get_current_active_user),
):
    """Get personalized track recommendations."""
    return await recommendation_engine.get_personalized_recommendations(
        user_id=user.sub,
        limit=limit,
    )


@router.get("/recommendations/similar/{track_id}", response_model=RecommendationResponse)
async def get_similar_tracks(track_id: str, limit: int = 5):
    """Get tracks similar to a given track."""
    return await recommendation_engine.get_similar_tracks(track_id, limit)


@router.get("/recommendations/trending", response_model=RecommendationResponse)
async def get_trending():
    """Get currently trending tracks."""
    return await recommendation_engine.get_trending_tracks()


@router.get("/recommendations/genre/{genre}", response_model=RecommendationResponse)
async def get_genre_recommendations(genre: str, limit: int = 10):
    """Get recommendations for a specific genre."""
    return await recommendation_engine.get_genre_recommendations(genre, limit)


@router.post("/recommendations/listen/{track_id}")
async def record_listen(
    track_id: str,
    user: Optional[TokenPayload] = Depends(get_current_user),
):
    """Record that a user listened to a track."""
    if user:
        recommendation_engine.record_listen(user.sub, track_id)
    return {"recorded": True}


# ==================== SOCIAL SHARING ====================

class CreateShareRequest(BaseModel):
    track_id: str
    title: str
    genre: Optional[str] = None


@router.post("/share", response_model=ShareLink)
async def create_share_link(
    request: CreateShareRequest,
    user: Optional[TokenPayload] = Depends(get_current_user),
):
    """Create a shareable link for a track."""
    return await social_sharing.create_share_link(
        track_id=request.track_id,
        title=request.title,
        genre=request.genre,
        user_id=user.sub if user else None,
    )


@router.get("/share/{share_code}", response_model=SocialShareUrls)
async def get_share_urls(share_code: str, request: Request):
    """Get social media share URLs for a share code."""
    base_url = str(request.base_url).rstrip("/")
    urls = await social_sharing.get_social_urls(share_code, base_url)
    if not urls:
        raise HTTPException(status_code=404, detail="Share link not found")
    return urls


@router.get("/share/{share_code}/track")
async def get_shared_track(share_code: str):
    """Get track info from a share code."""
    track = await social_sharing.get_track_from_share(share_code)
    if not track:
        raise HTTPException(status_code=404, detail="Share link not found")
    
    # Record click
    await social_sharing.record_click(share_code)
    return track


@router.get("/share/{share_code}/analytics", response_model=ShareAnalytics)
async def get_share_analytics(
    share_code: str,
    user: TokenPayload = Depends(get_current_active_user),
):
    """Get analytics for a share link."""
    analytics = await social_sharing.get_analytics(share_code)
    if not analytics:
        raise HTTPException(status_code=404, detail="Share link not found")
    return analytics


# ==================== PAYMENTS & SUBSCRIPTIONS ====================

@router.get("/plans", response_model=list[SubscriptionPlan])
async def get_subscription_plans():
    """Get all available subscription plans."""
    return payment_service.get_plans()


@router.get("/plans/credits")
async def get_credit_packages():
    """Get all available credit packages."""
    return payment_service.get_credit_packages()


@router.get("/subscription", response_model=UserSubscription)
async def get_subscription_status(
    user: TokenPayload = Depends(get_current_active_user),
):
    """Get current user's subscription status."""
    return payment_service.get_user_subscription(user.sub)


class CreateCheckoutRequest(BaseModel):
    plan_id: Optional[str] = None
    credit_package: Optional[str] = None
    origin_url: str


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(
    request: CreateCheckoutRequest,
    user: TokenPayload = Depends(get_current_active_user),
):
    """Create a Stripe checkout session."""
    if not request.plan_id and not request.credit_package:
        raise HTTPException(
            status_code=400,
            detail="Must specify plan_id or credit_package",
        )
    
    result = await payment_service.create_checkout_session(
        user_id=user.sub,
        plan_id=request.plan_id,
        credit_package=request.credit_package,
        origin_url=request.origin_url,
    )
    
    if not result:
        raise HTTPException(
            status_code=500,
            detail="Failed to create checkout session. Stripe may not be configured.",
        )
    
    return result


@router.get("/checkout/status/{session_id}")
async def get_checkout_status(session_id: str):
    """Get payment status for a checkout session."""
    status = await payment_service.check_payment_status(session_id)
    if not status:
        raise HTTPException(
            status_code=500,
            detail="Failed to check payment status",
        )
    return status


@router.get("/can-generate")
async def check_can_generate(
    user: TokenPayload = Depends(get_current_active_user),
):
    """Check if user can generate a track."""
    can_gen, reason = payment_service.can_generate(user.sub)
    return {
        "can_generate": can_gen,
        "reason": reason,
    }
