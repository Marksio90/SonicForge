"""
Role-Based Access Control (RBAC) (Phase 4 - Item 7.4)

Implements comprehensive RBAC with:
- Predefined roles (admin, moderator, user, viewer)
- Granular permissions
- Role hierarchy
- Permission decorators for endpoints
"""

from enum import Enum
from functools import wraps
from typing import Callable, Optional

from fastapi import Depends, HTTPException, status

from .auth import TokenPayload, get_current_active_user


class Permission(str, Enum):
    """Available permissions in the system."""
    
    # Track permissions
    TRACK_READ = "track:read"
    TRACK_CREATE = "track:create"
    TRACK_UPDATE = "track:update"
    TRACK_DELETE = "track:delete"
    TRACK_EVALUATE = "track:evaluate"
    
    # Stream permissions
    STREAM_VIEW = "stream:view"
    STREAM_CONTROL = "stream:control"
    STREAM_MANAGE = "stream:manage"
    
    # Queue permissions
    QUEUE_VIEW = "queue:view"
    QUEUE_MODIFY = "queue:modify"
    QUEUE_OVERRIDE = "queue:override"
    
    # Analytics permissions
    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_EXPORT = "analytics:export"
    
    # User permissions
    USER_VIEW = "user:view"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Admin permissions
    ADMIN_ACCESS = "admin:access"
    ADMIN_MANAGE = "admin:manage"
    SYSTEM_CONFIG = "system:config"
    
    # Pipeline permissions
    PIPELINE_RUN = "pipeline:run"
    PIPELINE_BATCH = "pipeline:batch"
    PIPELINE_BENCHMARK = "pipeline:benchmark"
    
    # Visual permissions
    VISUAL_GENERATE = "visual:generate"
    VISUAL_MANAGE = "visual:manage"


class Role(str, Enum):
    """Available roles in the system."""
    
    SUPERADMIN = "superadmin"
    ADMIN = "admin"
    MODERATOR = "moderator"
    DJ = "dj"
    USER = "user"
    VIEWER = "viewer"
    API_CLIENT = "api_client"


# Role hierarchy (higher roles inherit lower role permissions)
ROLE_HIERARCHY = {
    Role.SUPERADMIN: 100,
    Role.ADMIN: 80,
    Role.MODERATOR: 60,
    Role.DJ: 50,
    Role.USER: 30,
    Role.API_CLIENT: 20,
    Role.VIEWER: 10,
}

# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, list[Permission]] = {
    Role.VIEWER: [
        Permission.TRACK_READ,
        Permission.STREAM_VIEW,
        Permission.QUEUE_VIEW,
        Permission.ANALYTICS_VIEW,
    ],
    
    Role.USER: [
        # Inherits VIEWER permissions
        Permission.TRACK_READ,
        Permission.STREAM_VIEW,
        Permission.QUEUE_VIEW,
        Permission.ANALYTICS_VIEW,
        # Plus:
        Permission.VISUAL_GENERATE,
    ],
    
    Role.API_CLIENT: [
        Permission.TRACK_READ,
        Permission.STREAM_VIEW,
        Permission.QUEUE_VIEW,
        Permission.ANALYTICS_VIEW,
        Permission.PIPELINE_RUN,
    ],
    
    Role.DJ: [
        # Inherits USER permissions
        Permission.TRACK_READ,
        Permission.STREAM_VIEW,
        Permission.QUEUE_VIEW,
        Permission.ANALYTICS_VIEW,
        Permission.VISUAL_GENERATE,
        # Plus:
        Permission.TRACK_CREATE,
        Permission.TRACK_EVALUATE,
        Permission.QUEUE_MODIFY,
        Permission.STREAM_CONTROL,
        Permission.PIPELINE_RUN,
    ],
    
    Role.MODERATOR: [
        # Inherits DJ permissions
        Permission.TRACK_READ,
        Permission.TRACK_CREATE,
        Permission.TRACK_UPDATE,
        Permission.TRACK_EVALUATE,
        Permission.STREAM_VIEW,
        Permission.STREAM_CONTROL,
        Permission.QUEUE_VIEW,
        Permission.QUEUE_MODIFY,
        Permission.QUEUE_OVERRIDE,
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT,
        Permission.VISUAL_GENERATE,
        Permission.VISUAL_MANAGE,
        Permission.PIPELINE_RUN,
        Permission.USER_VIEW,
    ],
    
    Role.ADMIN: [
        # Inherits MODERATOR permissions
        Permission.TRACK_READ,
        Permission.TRACK_CREATE,
        Permission.TRACK_UPDATE,
        Permission.TRACK_DELETE,
        Permission.TRACK_EVALUATE,
        Permission.STREAM_VIEW,
        Permission.STREAM_CONTROL,
        Permission.STREAM_MANAGE,
        Permission.QUEUE_VIEW,
        Permission.QUEUE_MODIFY,
        Permission.QUEUE_OVERRIDE,
        Permission.ANALYTICS_VIEW,
        Permission.ANALYTICS_EXPORT,
        Permission.USER_VIEW,
        Permission.USER_CREATE,
        Permission.USER_UPDATE,
        Permission.VISUAL_GENERATE,
        Permission.VISUAL_MANAGE,
        Permission.PIPELINE_RUN,
        Permission.PIPELINE_BATCH,
        Permission.PIPELINE_BENCHMARK,
        Permission.ADMIN_ACCESS,
    ],
    
    Role.SUPERADMIN: [
        # All permissions
        permission for permission in Permission
    ],
}


def get_role_permissions(role: Role) -> list[Permission]:
    """Get all permissions for a role."""
    return ROLE_PERMISSIONS.get(role, [])


def get_user_permissions(roles: list[str]) -> set[Permission]:
    """Get combined permissions for all user roles."""
    permissions = set()
    for role_str in roles:
        try:
            role = Role(role_str)
            permissions.update(get_role_permissions(role))
        except ValueError:
            continue
    return permissions


def has_permission(user_roles: list[str], required_permission: Permission) -> bool:
    """Check if user has a specific permission."""
    user_permissions = get_user_permissions(user_roles)
    return required_permission in user_permissions


def has_any_permission(user_roles: list[str], permissions: list[Permission]) -> bool:
    """Check if user has any of the specified permissions."""
    user_permissions = get_user_permissions(user_roles)
    return any(p in user_permissions for p in permissions)


def has_all_permissions(user_roles: list[str], permissions: list[Permission]) -> bool:
    """Check if user has all specified permissions."""
    user_permissions = get_user_permissions(user_roles)
    return all(p in user_permissions for p in permissions)


def has_role(user_roles: list[str], required_role: Role) -> bool:
    """Check if user has a specific role or higher."""
    required_level = ROLE_HIERARCHY.get(required_role, 0)
    
    for role_str in user_roles:
        try:
            role = Role(role_str)
            user_level = ROLE_HIERARCHY.get(role, 0)
            if user_level >= required_level:
                return True
        except ValueError:
            continue
    
    return False


class PermissionChecker:
    """Dependency for checking permissions."""
    
    def __init__(self, required_permissions: list[Permission], require_all: bool = True):
        self.required_permissions = required_permissions
        self.require_all = require_all
    
    async def __call__(
        self,
        user: TokenPayload = Depends(get_current_active_user),
    ) -> TokenPayload:
        """Check if user has required permissions."""
        if self.require_all:
            has_perms = has_all_permissions(user.roles, self.required_permissions)
        else:
            has_perms = has_any_permission(user.roles, self.required_permissions)
        
        if not has_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "insufficient_permissions",
                    "required": [p.value for p in self.required_permissions],
                    "require_all": self.require_all,
                },
            )
        
        return user


class RoleChecker:
    """Dependency for checking roles."""
    
    def __init__(self, required_role: Role):
        self.required_role = required_role
    
    async def __call__(
        self,
        user: TokenPayload = Depends(get_current_active_user),
    ) -> TokenPayload:
        """Check if user has required role."""
        if not has_role(user.roles, self.required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "insufficient_role",
                    "required_role": self.required_role.value,
                },
            )
        
        return user


def require_permission(*permissions: Permission, require_all: bool = True):
    """Decorator to require specific permissions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        # Add dependency
        wrapper.__annotations__["_permission_check"] = PermissionChecker(
            list(permissions), require_all
        )
        return wrapper
    return decorator


def require_role(role: Role):
    """Decorator to require a specific role."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        # Add dependency
        wrapper.__annotations__["_role_check"] = RoleChecker(role)
        return wrapper
    return decorator


# Convenience dependencies
RequireAdmin = Depends(RoleChecker(Role.ADMIN))
RequireModerator = Depends(RoleChecker(Role.MODERATOR))
RequireDJ = Depends(RoleChecker(Role.DJ))
RequireUser = Depends(RoleChecker(Role.USER))

RequireTrackRead = Depends(PermissionChecker([Permission.TRACK_READ]))
RequireTrackCreate = Depends(PermissionChecker([Permission.TRACK_CREATE]))
RequireStreamControl = Depends(PermissionChecker([Permission.STREAM_CONTROL]))
RequireQueueModify = Depends(PermissionChecker([Permission.QUEUE_MODIFY]))
RequireAnalytics = Depends(PermissionChecker([Permission.ANALYTICS_VIEW]))
RequirePipelineRun = Depends(PermissionChecker([Permission.PIPELINE_RUN]))
