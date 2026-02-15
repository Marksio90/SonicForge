# Features module for Phase 5: User Experience & Monetization

from .voting import VotingService, voting_service, VoteRequest, VoteResponse, TrackPopularity
from .recommendations import (
    RecommendationEngine,
    recommendation_engine,
    TrackMetadata,
    RecommendationItem,
    RecommendationResponse,
)
from .social_sharing import (
    SocialSharingService,
    social_sharing,
    ShareRequest,
    ShareLink,
    SocialShareUrls,
    ShareAnalytics,
)
from .payments import (
    PaymentService,
    payment_service,
    SubscriptionPlan,
    UserSubscription,
    CheckoutRequest,
    CheckoutResponse,
    PaymentTransaction,
    SUBSCRIPTION_PLANS,
    CREDIT_PACKAGES,
)

__all__ = [
    # Voting
    "VotingService",
    "voting_service",
    "VoteRequest",
    "VoteResponse",
    "TrackPopularity",
    # Recommendations
    "RecommendationEngine",
    "recommendation_engine",
    "TrackMetadata",
    "RecommendationItem",
    "RecommendationResponse",
    # Social Sharing
    "SocialSharingService",
    "social_sharing",
    "ShareRequest",
    "ShareLink",
    "SocialShareUrls",
    "ShareAnalytics",
    # Payments
    "PaymentService",
    "payment_service",
    "SubscriptionPlan",
    "UserSubscription",
    "CheckoutRequest",
    "CheckoutResponse",
    "PaymentTransaction",
    "SUBSCRIPTION_PLANS",
    "CREDIT_PACKAGES",
]
