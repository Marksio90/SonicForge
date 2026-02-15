"""
Authentication API Routes (Phase 4)

Implements authentication endpoints:
- Login
- Register
- Token refresh
- Logout
- API key management
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, field_validator

from ..security.auth import (
    TokenResponse,
    User,
    create_tokens,
    get_password_hash,
    refresh_access_token,
    verify_password,
    blacklist_token,
    get_current_active_user,
    TokenPayload,
)
from ..security.api_keys import (
    APIKey,
    KeyScope,
    create_api_key,
    list_api_keys,
    revoke_api_key,
    rotate_api_key,
)
from ..security.rate_limiter import limiter, RateLimits
from ..security.input_validation import validate_email, validate_password
from ..security.rbac import Role, RoleChecker

router = APIRouter(prefix="/auth", tags=["authentication"])


# --- Request/Response Models ---

class LoginRequest(BaseModel):
    """Login request schema."""
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    """Registration request schema."""
    email: EmailStr
    username: str
    password: str
    
    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        return validate_password(v)
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(v) > 50:
            raise ValueError("Username must be at most 50 characters")
        return v.strip()


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    refresh_token: str


class CreateAPIKeyRequest(BaseModel):
    """Create API key request schema."""
    name: str
    scopes: list[str]
    expires_in_days: Optional[int] = None
    rate_limit: str = "100/minute"


class APIKeyResponse(BaseModel):
    """API key response schema."""
    key: str
    key_id: str
    name: str
    scopes: list[str]
    expires_at: Optional[str] = None


# --- In-memory user storage (in production, use database) ---

_users: dict[str, User] = {}


def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email."""
    for user in _users.values():
        if user.email == email:
            return user
    return None


def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID."""
    return _users.get(user_id)


# --- Authentication Endpoints ---

@router.post("/register", response_model=TokenResponse)
@limiter.limit(RateLimits.REGISTER)
async def register(request: Request, data: RegisterRequest):
    """Register a new user."""
    # Check if email already exists
    if get_user_by_email(data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Create user
    import uuid
    user_id = str(uuid.uuid4())
    
    user = User(
        id=user_id,
        email=data.email.lower(),
        username=data.username,
        hashed_password=get_password_hash(data.password),
        roles=["user"],
    )
    
    _users[user_id] = user
    
    # Generate tokens
    return create_tokens(user)


@router.post("/login", response_model=TokenResponse)
@limiter.limit(RateLimits.LOGIN)
async def login(request: Request, data: LoginRequest):
    """Authenticate user and return tokens."""
    user = get_user_by_email(data.email.lower())
    
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    
    return create_tokens(user)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit(RateLimits.TOKEN_REFRESH)
async def refresh_token(request: Request, data: RefreshTokenRequest):
    """Refresh access token using refresh token."""
    return await refresh_access_token(data.refresh_token)


@router.post("/logout")
async def logout(
    request: Request,
    user: TokenPayload = Depends(get_current_active_user),
):
    """Logout user (blacklist current token)."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        blacklist_token(token)
    
    return {"message": "Successfully logged out"}


@router.get("/me")
async def get_current_user_info(
    user: TokenPayload = Depends(get_current_active_user),
):
    """Get current user information."""
    user_data = get_user_by_id(user.sub)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return {
        "id": user_data.id,
        "email": user_data.email,
        "username": user_data.username,
        "roles": user_data.roles,
        "is_active": user_data.is_active,
    }


# --- API Key Management Endpoints ---

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_new_api_key(
    data: CreateAPIKeyRequest,
    user: TokenPayload = Depends(RoleChecker(Role.ADMIN)),
):
    """Create a new API key (admin only)."""
    # Parse scopes
    scopes = []
    for scope_str in data.scopes:
        try:
            scopes.append(KeyScope(scope_str))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid scope: {scope_str}",
            )
    
    key, api_key = create_api_key(
        name=data.name,
        scopes=scopes,
        expires_in_days=data.expires_in_days,
        rate_limit=data.rate_limit,
    )
    
    return APIKeyResponse(
        key=key,
        key_id=api_key.key_id,
        name=api_key.name,
        scopes=[s.value for s in api_key.scopes],
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
    )


@router.get("/api-keys")
async def get_api_keys(
    user: TokenPayload = Depends(RoleChecker(Role.ADMIN)),
):
    """List all API keys (admin only)."""
    return list_api_keys()


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: str,
    user: TokenPayload = Depends(RoleChecker(Role.ADMIN)),
):
    """Revoke an API key (admin only)."""
    if not revoke_api_key(key_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    return {"message": "API key revoked successfully"}


@router.post("/api-keys/{key_id}/rotate")
async def rotate_existing_api_key(
    key_id: str,
    user: TokenPayload = Depends(RoleChecker(Role.ADMIN)),
):
    """Rotate an API key (admin only)."""
    result = rotate_api_key(key_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    key, api_key = result
    return APIKeyResponse(
        key=key,
        key_id=api_key.key_id,
        name=api_key.name,
        scopes=[s.value for s in api_key.scopes],
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
    )
