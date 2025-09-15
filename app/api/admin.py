"""
Admin endpoints for RAG Patient API operational management.

Provides runtime configuration updates for rate limiting, feature flags,
and other operational parameters.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.settings import settings

admin = APIRouter(prefix="/admin", tags=["admin"])


class RateLimitUpdate(BaseModel):
    session_per_min: int | None = Field(None, ge=0, le=10000)
    ip_per_min: int | None = Field(None, ge=0, le=100000)
    enabled: bool | None = None
    fail_open: bool | None = None


@admin.post("/rate_limit")
def update_rate_limit(payload: RateLimitUpdate):
    """
    Update rate limiting configuration at runtime.

    Args:
        payload: Rate limit configuration updates

    Returns:
        dict: Current rate limit configuration

    Example:
        curl -X POST localhost:8000/admin/rate_limit \
          -H 'content-type: application/json' \
          -d '{"session_per_min":5,"ip_per_min":1000,"enabled":true}'
    """
    if payload.session_per_min is not None:
        settings.RATE_LIMIT_SESSION_PER_MIN = payload.session_per_min
    if payload.ip_per_min is not None:
        settings.RATE_LIMIT_IP_PER_MIN = payload.ip_per_min
    if payload.enabled is not None:
        settings.RATE_LIMIT_ENABLED = payload.enabled
    if payload.fail_open is not None:
        settings.RATE_LIMIT_FAIL_OPEN = payload.fail_open

    return {
        "enabled": settings.RATE_LIMIT_ENABLED,
        "session_per_min": settings.RATE_LIMIT_SESSION_PER_MIN,
        "ip_per_min": settings.RATE_LIMIT_IP_PER_MIN,
        "fail_open": settings.RATE_LIMIT_FAIL_OPEN,
    }


@admin.get("/rate_limit")
def get_rate_limit():
    """
    Get current rate limiting configuration.

    Returns:
        dict: Current rate limit configuration
    """
    return {
        "enabled": settings.RATE_LIMIT_ENABLED,
        "session_per_min": settings.RATE_LIMIT_SESSION_PER_MIN,
        "ip_per_min": settings.RATE_LIMIT_IP_PER_MIN,
        "fail_open": settings.RATE_LIMIT_FAIL_OPEN,
    }