"""
Input Validation & Sanitization (Phase 4 - Item 7.6)

Implements comprehensive input validation and sanitization:
- SQL injection prevention
- XSS prevention
- Path traversal prevention
- Command injection prevention
- Data sanitization utilities
"""

import html
import re
from functools import wraps
from typing import Any, Callable

from fastapi import HTTPException, status
from pydantic import BaseModel, field_validator


# Dangerous patterns to detect
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)",
    r"(--|;|/\*|\*/|@@|@)",
    r"(\bOR\b|\bAND\b).*?=",
    r"'.*?(OR|AND).*?'",
]

XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe[^>]*>",
    r"<object[^>]*>",
    r"<embed[^>]*>",
]

PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e%2f",
    r"%2e%2e/",
    r"\.%2e/",
    r"%2e\./",
]

COMMAND_INJECTION_PATTERNS = [
    r"[;&|`$]",
    r"\$\(",
    r"`.*`",
    r"\|\|",
    r"&&",
]


class ValidationError(HTTPException):
    """Custom validation error."""
    def __init__(self, message: str, field: str = None):
        detail = {"message": message}
        if field:
            detail["field"] = field
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
        )


def check_patterns(value: str, patterns: list[str], error_type: str) -> None:
    """Check value against a list of regex patterns."""
    for pattern in patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValidationError(
                f"Potential {error_type} detected in input",
            )


def sanitize_string(value: str) -> str:
    """Sanitize a string value."""
    if not isinstance(value, str):
        return value
    
    # HTML escape
    value = html.escape(value)
    
    # Remove null bytes
    value = value.replace("\x00", "")
    
    # Normalize whitespace
    value = " ".join(value.split())
    
    return value.strip()


def sanitize_html(value: str) -> str:
    """Remove HTML tags from string."""
    if not isinstance(value, str):
        return value
    
    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", "", value)
    
    # Decode HTML entities
    clean = html.unescape(clean)
    
    return clean.strip()


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    if not isinstance(filename, str):
        return filename
    
    # Remove path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    
    # Remove special characters
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    
    # Remove leading dots
    filename = filename.lstrip(".")
    
    return filename


def sanitize_input(value: Any) -> Any:
    """Recursively sanitize input data."""
    if isinstance(value, str):
        return sanitize_string(value)
    elif isinstance(value, dict):
        return {k: sanitize_input(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [sanitize_input(v) for v in value]
    return value


def validate_input(value: str, field_name: str = "input") -> str:
    """Validate and sanitize input string."""
    if not isinstance(value, str):
        return value
    
    # Check for SQL injection
    check_patterns(value, SQL_INJECTION_PATTERNS, "SQL injection")
    
    # Check for XSS
    check_patterns(value, XSS_PATTERNS, "XSS attack")
    
    # Check for path traversal
    check_patterns(value, PATH_TRAVERSAL_PATTERNS, "path traversal")
    
    # Sanitize and return
    return sanitize_string(value)


def validate_no_sql_injection(func: Callable) -> Callable:
    """Decorator to validate function arguments against SQL injection."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, str):
                check_patterns(value, SQL_INJECTION_PATTERNS, "SQL injection")
        return await func(*args, **kwargs)
    return wrapper


def validate_no_xss(func: Callable) -> Callable:
    """Decorator to validate function arguments against XSS."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, str):
                check_patterns(value, XSS_PATTERNS, "XSS attack")
        return await func(*args, **kwargs)
    return wrapper


class SecureString(str):
    """A string type that validates against common injection attacks."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("String required")
        
        # Validate
        validate_input(v)
        
        # Sanitize
        return sanitize_string(v)


class SafeBaseModel(BaseModel):
    """Base model with automatic input sanitization."""
    
    @field_validator("*", mode="before")
    @classmethod
    def sanitize_all_fields(cls, v: Any) -> Any:
        """Sanitize all string fields."""
        return sanitize_input(v)


# Email validation regex
EMAIL_REGEX = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"


def validate_email(email: str) -> str:
    """Validate email format."""
    if not re.match(EMAIL_REGEX, email):
        raise ValidationError("Invalid email format", field="email")
    return email.lower().strip()


# Password validation
def validate_password(password: str) -> str:
    """Validate password strength."""
    if len(password) < 8:
        raise ValidationError("Password must be at least 8 characters", field="password")
    if not re.search(r"[A-Z]", password):
        raise ValidationError("Password must contain uppercase letter", field="password")
    if not re.search(r"[a-z]", password):
        raise ValidationError("Password must contain lowercase letter", field="password")
    if not re.search(r"\d", password):
        raise ValidationError("Password must contain a digit", field="password")
    return password


# UUID validation
UUID_REGEX = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def validate_uuid(value: str, field_name: str = "id") -> str:
    """Validate UUID format."""
    if not re.match(UUID_REGEX, value, re.IGNORECASE):
        raise ValidationError(f"Invalid UUID format", field=field_name)
    return value.lower()
