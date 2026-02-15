# Security module for SonicForge
# Phase 4: Security & Reliability

from .auth import (
    create_access_token,
    create_refresh_token,
    verify_token,
    get_current_user,
    get_current_active_user,
    get_password_hash,
    verify_password,
)
from .rate_limiter import limiter, rate_limit_exceeded_handler
from .input_validation import sanitize_input, validate_input
from .rbac import Permission, Role, require_permission, require_role

__all__ = [
    # Auth
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "get_current_user",
    "get_current_active_user",
    "get_password_hash",
    "verify_password",
    # Rate Limiting
    "limiter",
    "rate_limit_exceeded_handler",
    # Input Validation
    "sanitize_input",
    "validate_input",
    # RBAC
    "Permission",
    "Role",
    "require_permission",
    "require_role",
]
