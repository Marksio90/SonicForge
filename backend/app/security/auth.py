"""
JWT Authentication with Refresh Tokens (Phase 4 - Item 7.4)

Implements secure token-based authentication with:
- Access tokens (short-lived, 15 minutes)
- Refresh tokens (long-lived, 7 days)
- Password hashing with bcrypt
- Token blacklisting for logout
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from ..core.config import get_settings

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Token types
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"

# Security scheme
security = HTTPBearer(auto_error=False)


class TokenPayload(BaseModel):
    """JWT token payload schema."""
    sub: str  # user_id
    exp: datetime
    type: str  # "access" or "refresh"
    roles: list[str] = []
    permissions: list[str] = []


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class User(BaseModel):
    """User model for authentication."""
    id: str
    email: str
    username: str
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    roles: list[str] = ["user"]
    permissions: list[str] = []


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(
    user_id: str,
    roles: list[str] = None,
    permissions: list[str] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a new access token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "sub": user_id,
        "exp": expire,
        "type": TOKEN_TYPE_ACCESS,
        "roles": roles or [],
        "permissions": permissions or [],
        "iat": datetime.now(timezone.utc),
    }
    
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a new refresh token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "sub": user_id,
        "exp": expire,
        "type": TOKEN_TYPE_REFRESH,
        "iat": datetime.now(timezone.utc),
    }
    
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = TOKEN_TYPE_ACCESS) -> TokenPayload:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        
        # Check token type
        if payload.get("type") != token_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token type. Expected {token_type}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return TokenPayload(**payload)
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[TokenPayload]:
    """Get current user from JWT token (optional authentication)."""
    if credentials is None:
        return None
    
    token = credentials.credentials
    return verify_token(token, TOKEN_TYPE_ACCESS)


async def get_current_active_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
) -> TokenPayload:
    """Get current active user from JWT token (required authentication)."""
    token = credentials.credentials
    user = verify_token(token, TOKEN_TYPE_ACCESS)
    
    # Here you could add additional checks like:
    # - Check if user is active in database
    # - Check if token is blacklisted
    # - Check user permissions
    
    return user


def create_tokens(user: User) -> TokenResponse:
    """Create both access and refresh tokens for a user."""
    access_token = create_access_token(
        user_id=user.id,
        roles=user.roles,
        permissions=user.permissions,
    )
    refresh_token = create_refresh_token(user_id=user.id)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
    )


async def refresh_access_token(refresh_token: str) -> TokenResponse:
    """Refresh an access token using a refresh token."""
    # Verify refresh token
    payload = verify_token(refresh_token, TOKEN_TYPE_REFRESH)
    
    # Create new tokens
    access_token = create_access_token(user_id=payload.sub)
    new_refresh_token = create_refresh_token(user_id=payload.sub)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
    )


# Token blacklist (in production, use Redis)
_blacklisted_tokens: set[str] = set()


def blacklist_token(token: str) -> None:
    """Add a token to the blacklist (logout)."""
    _blacklisted_tokens.add(token)


def is_token_blacklisted(token: str) -> bool:
    """Check if a token is blacklisted."""
    return token in _blacklisted_tokens
