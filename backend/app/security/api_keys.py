"""
API Key Management (Phase 4 - Item 7.5)

Implements API key authentication and rotation:
- API key generation and validation
- Key rotation without downtime
- Key scoping and rate limiting
- Key revocation
"""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from ..core.config import get_settings

settings = get_settings()


class KeyScope(str, Enum):
    """API key scopes."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    PIPELINE = "pipeline"
    ANALYTICS = "analytics"
    STREAM = "stream"


@dataclass
class APIKey:
    """API key model."""
    key_id: str
    key_hash: str
    name: str
    scopes: list[KeyScope]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit: str = "100/minute"
    metadata: dict = None
    
    def is_valid(self) -> bool:
        """Check if key is valid (active and not expired)."""
        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < datetime.now(timezone.utc):
            return False
        return True
    
    def has_scope(self, scope: KeyScope) -> bool:
        """Check if key has a specific scope."""
        return scope in self.scopes or KeyScope.ADMIN in self.scopes


# In-memory key storage (in production, use database)
_api_keys: dict[str, APIKey] = {}

# Security header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key and its hash."""
    # Generate a secure random key
    key = f"sf_{secrets.token_urlsafe(32)}"
    
    # Hash the key for storage
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    
    return key, key_hash


def hash_api_key(key: str) -> str:
    """Hash an API key."""
    return hashlib.sha256(key.encode()).hexdigest()


def create_api_key(
    name: str,
    scopes: list[KeyScope],
    expires_in_days: Optional[int] = None,
    rate_limit: str = "100/minute",
    metadata: dict = None,
) -> tuple[str, APIKey]:
    """Create a new API key."""
    key, key_hash = generate_api_key()
    key_id = secrets.token_urlsafe(8)
    
    expires_at = None
    if expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
    
    api_key = APIKey(
        key_id=key_id,
        key_hash=key_hash,
        name=name,
        scopes=scopes,
        created_at=datetime.now(timezone.utc),
        expires_at=expires_at,
        rate_limit=rate_limit,
        metadata=metadata or {},
    )
    
    # Store the key
    _api_keys[key_hash] = api_key
    
    return key, api_key


def validate_api_key(key: str) -> Optional[APIKey]:
    """Validate an API key and return its data."""
    key_hash = hash_api_key(key)
    
    api_key = _api_keys.get(key_hash)
    if not api_key:
        return None
    
    if not api_key.is_valid():
        return None
    
    # Update last used time
    api_key.last_used_at = datetime.now(timezone.utc)
    
    return api_key


def revoke_api_key(key_id: str) -> bool:
    """Revoke an API key by its ID."""
    for key_hash, api_key in _api_keys.items():
        if api_key.key_id == key_id:
            api_key.is_active = False
            return True
    return False


def rotate_api_key(key_id: str) -> Optional[tuple[str, APIKey]]:
    """Rotate an API key (create new key, mark old as deprecated)."""
    # Find the existing key
    old_key = None
    for key_hash, api_key in _api_keys.items():
        if api_key.key_id == key_id:
            old_key = api_key
            break
    
    if not old_key:
        return None
    
    # Create new key with same configuration
    new_key, new_api_key = create_api_key(
        name=f"{old_key.name} (rotated)",
        scopes=old_key.scopes,
        rate_limit=old_key.rate_limit,
        metadata=old_key.metadata,
    )
    
    # Mark old key for deprecation (expires in 24 hours for graceful transition)
    old_key.expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    
    return new_key, new_api_key


def list_api_keys() -> list[dict]:
    """List all API keys (without sensitive data)."""
    return [
        {
            "key_id": key.key_id,
            "name": key.name,
            "scopes": [s.value for s in key.scopes],
            "created_at": key.created_at.isoformat(),
            "expires_at": key.expires_at.isoformat() if key.expires_at else None,
            "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
            "is_active": key.is_active,
            "rate_limit": key.rate_limit,
        }
        for key in _api_keys.values()
    ]


class APIKeyValidator:
    """Dependency for validating API keys with scope checking."""
    
    def __init__(self, required_scopes: list[KeyScope] = None, require_all: bool = True):
        self.required_scopes = required_scopes or []
        self.require_all = require_all
    
    async def __call__(
        self,
        api_key: Optional[str] = Security(api_key_header),
    ) -> APIKey:
        """Validate API key and check scopes."""
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        key_data = validate_api_key(api_key)
        if not key_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        # Check scopes
        if self.required_scopes:
            if self.require_all:
                has_scopes = all(key_data.has_scope(s) for s in self.required_scopes)
            else:
                has_scopes = any(key_data.has_scope(s) for s in self.required_scopes)
            
            if not has_scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "insufficient_scope",
                        "required": [s.value for s in self.required_scopes],
                    },
                )
        
        return key_data


# Convenience dependencies
RequireAPIKey = Depends(APIKeyValidator())
RequireReadKey = Depends(APIKeyValidator([KeyScope.READ]))
RequireWriteKey = Depends(APIKeyValidator([KeyScope.WRITE]))
RequireAdminKey = Depends(APIKeyValidator([KeyScope.ADMIN]))
RequirePipelineKey = Depends(APIKeyValidator([KeyScope.PIPELINE]))


# Combined auth (JWT or API Key)
async def get_auth(
    api_key: Optional[str] = Security(api_key_header),
    jwt_credentials=None,  # Would be JWT token
) -> dict:
    """Get authentication from either API key or JWT."""
    if api_key:
        key_data = validate_api_key(api_key)
        if key_data:
            return {"type": "api_key", "data": key_data}
    
    if jwt_credentials:
        # JWT validation would go here
        pass
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required (API key or JWT token)",
    )
