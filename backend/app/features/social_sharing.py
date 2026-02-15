"""
Social Sharing (Phase 5 - User Experience)

Implements social sharing functionality:
- Share tracks to social media
- Generate shareable links
- Track sharing analytics
- Embeddable player widgets
"""

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)

# In-memory storage
_share_links: dict[str, dict] = {}  # share_code -> track info
_share_analytics: dict[str, dict] = {}  # share_code -> analytics


class ShareRequest(BaseModel):
    """Share request schema."""
    track_id: str
    title: str
    genre: Optional[str] = None
    artist: str = "SonicForge AI"
    platform: Optional[str] = None  # twitter, facebook, etc.


class ShareLink(BaseModel):
    """Share link response."""
    share_code: str
    share_url: str
    track_id: str
    title: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


class SocialShareUrls(BaseModel):
    """Social media share URLs."""
    share_code: str
    direct_link: str
    twitter: str
    facebook: str
    linkedin: str
    whatsapp: str
    telegram: str
    email: str
    embed_code: str


class ShareAnalytics(BaseModel):
    """Share analytics data."""
    share_code: str
    track_id: str
    total_clicks: int
    clicks_by_platform: dict[str, int]
    clicks_by_country: dict[str, int]
    created_at: datetime
    last_clicked_at: Optional[datetime] = None


class SocialSharingService:
    """Service for social sharing functionality."""
    
    def __init__(self, base_url: str = "https://sonicforge.ai"):
        self.base_url = base_url
    
    def _generate_share_code(self, track_id: str) -> str:
        """Generate a unique share code."""
        unique = f"{track_id}:{secrets.token_hex(4)}"
        return hashlib.sha256(unique.encode()).hexdigest()[:12]
    
    async def create_share_link(
        self,
        track_id: str,
        title: str,
        genre: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ShareLink:
        """Create a shareable link for a track."""
        share_code = self._generate_share_code(track_id)
        
        _share_links[share_code] = {
            "track_id": track_id,
            "title": title,
            "genre": genre,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
        }
        
        _share_analytics[share_code] = {
            "track_id": track_id,
            "total_clicks": 0,
            "clicks_by_platform": {},
            "clicks_by_country": {},
            "created_at": datetime.now(timezone.utc),
            "last_clicked_at": None,
        }
        
        share_url = f"{self.base_url}/share/{share_code}"
        
        logger.info(
            "share_link_created",
            share_code=share_code,
            track_id=track_id,
        )
        
        return ShareLink(
            share_code=share_code,
            share_url=share_url,
            track_id=track_id,
            title=title,
        )
    
    async def get_social_urls(
        self,
        share_code: str,
        base_url: Optional[str] = None,
    ) -> Optional[SocialShareUrls]:
        """Get social media share URLs for a share code."""
        if share_code not in _share_links:
            return None
        
        link_data = _share_links[share_code]
        title = link_data["title"]
        genre = link_data.get("genre", "AI Generated")
        
        url = base_url or self.base_url
        direct_link = f"{url}/share/{share_code}"
        
        # Encode for URLs
        encoded_title = quote_plus(f"🎵 {title}")
        encoded_text = quote_plus(f"Check out this AI-generated {genre} track on SonicForge!")
        encoded_url = quote_plus(direct_link)
        
        # Generate platform-specific URLs
        twitter_url = (
            f"https://twitter.com/intent/tweet?"
            f"text={encoded_text}&url={encoded_url}"
        )
        
        facebook_url = (
            f"https://www.facebook.com/sharer/sharer.php?"
            f"u={encoded_url}&quote={encoded_text}"
        )
        
        linkedin_url = (
            f"https://www.linkedin.com/sharing/share-offsite/?"
            f"url={encoded_url}"
        )
        
        whatsapp_url = (
            f"https://api.whatsapp.com/send?"
            f"text={encoded_text}%20{encoded_url}"
        )
        
        telegram_url = (
            f"https://t.me/share/url?"
            f"url={encoded_url}&text={encoded_text}"
        )
        
        email_url = (
            f"mailto:?subject={encoded_title}&"
            f"body={encoded_text}%0A%0A{encoded_url}"
        )
        
        # Embed code
        embed_code = (
            f'<iframe src="{direct_link}/embed" '
            f'width="400" height="200" frameborder="0" '
            f'allow="autoplay; encrypted-media" allowfullscreen></iframe>'
        )
        
        return SocialShareUrls(
            share_code=share_code,
            direct_link=direct_link,
            twitter=twitter_url,
            facebook=facebook_url,
            linkedin=linkedin_url,
            whatsapp=whatsapp_url,
            telegram=telegram_url,
            email=email_url,
            embed_code=embed_code,
        )
    
    async def record_click(
        self,
        share_code: str,
        platform: Optional[str] = None,
        country: Optional[str] = None,
    ) -> bool:
        """Record a share link click."""
        if share_code not in _share_analytics:
            return False
        
        analytics = _share_analytics[share_code]
        analytics["total_clicks"] += 1
        analytics["last_clicked_at"] = datetime.now(timezone.utc)
        
        if platform:
            analytics["clicks_by_platform"][platform] = (
                analytics["clicks_by_platform"].get(platform, 0) + 1
            )
        
        if country:
            analytics["clicks_by_country"][country] = (
                analytics["clicks_by_country"].get(country, 0) + 1
            )
        
        logger.info(
            "share_click_recorded",
            share_code=share_code,
            platform=platform,
            total_clicks=analytics["total_clicks"],
        )
        
        return True
    
    async def get_analytics(self, share_code: str) -> Optional[ShareAnalytics]:
        """Get analytics for a share link."""
        if share_code not in _share_analytics:
            return None
        
        data = _share_analytics[share_code]
        
        return ShareAnalytics(
            share_code=share_code,
            track_id=data["track_id"],
            total_clicks=data["total_clicks"],
            clicks_by_platform=data["clicks_by_platform"],
            clicks_by_country=data["clicks_by_country"],
            created_at=data["created_at"],
            last_clicked_at=data.get("last_clicked_at"),
        )
    
    async def get_track_from_share(self, share_code: str) -> Optional[dict]:
        """Get track info from share code."""
        return _share_links.get(share_code)


# Global instance
social_sharing = SocialSharingService()
