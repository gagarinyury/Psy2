"""
Tests for admin endpoints functionality.

Tests runtime configuration updates for rate limiting and operational parameters.
"""

import pytest
from httpx import AsyncClient

from app.core.settings import settings


@pytest.mark.anyio
async def test_get_rate_limit_config(client: AsyncClient):
    """Test getting current rate limit configuration."""
    response = await client.get("/admin/rate_limit")
    assert response.status_code == 200

    data = response.json()
    assert "enabled" in data
    assert "session_per_min" in data
    assert "ip_per_min" in data
    assert "fail_open" in data

    # Check types
    assert isinstance(data["enabled"], bool)
    assert isinstance(data["session_per_min"], int)
    assert isinstance(data["ip_per_min"], int)
    assert isinstance(data["fail_open"], bool)


@pytest.mark.anyio
async def test_update_rate_limit_config(client: AsyncClient):
    """Test updating rate limit configuration."""
    # Get initial config
    initial_response = await client.get("/admin/rate_limit")
    initial_config = initial_response.json()

    # Update some values
    update_payload = {
        "session_per_min": 10,
        "ip_per_min": 500,
        "enabled": True,
        "fail_open": False
    }

    response = await client.post("/admin/rate_limit", json=update_payload)
    assert response.status_code == 200

    data = response.json()
    assert data["session_per_min"] == 10
    assert data["ip_per_min"] == 500
    assert data["enabled"] is True
    assert data["fail_open"] is False

    # Verify settings were actually updated
    assert settings.RATE_LIMIT_SESSION_PER_MIN == 10
    assert settings.RATE_LIMIT_IP_PER_MIN == 500
    assert settings.RATE_LIMIT_ENABLED is True
    assert settings.RATE_LIMIT_FAIL_OPEN is False


@pytest.mark.anyio
async def test_partial_rate_limit_update(client: AsyncClient):
    """Test partial updates to rate limit configuration."""
    # Update only session limit
    response = await client.post(
        "/admin/rate_limit",
        json={"session_per_min": 15}
    )
    assert response.status_code == 200

    data = response.json()
    assert data["session_per_min"] == 15
    # Other values should remain unchanged from settings
    assert "ip_per_min" in data
    assert "enabled" in data
    assert "fail_open" in data


@pytest.mark.anyio
async def test_rate_limit_update_validation(client: AsyncClient):
    """Test validation of rate limit update values."""
    # Test invalid session_per_min (negative)
    response = await client.post(
        "/admin/rate_limit",
        json={"session_per_min": -1}
    )
    assert response.status_code == 422  # Validation error

    # Test invalid session_per_min (too large)
    response = await client.post(
        "/admin/rate_limit",
        json={"session_per_min": 20000}
    )
    assert response.status_code == 422  # Validation error

    # Test invalid ip_per_min (negative)
    response = await client.post(
        "/admin/rate_limit",
        json={"ip_per_min": -1}
    )
    assert response.status_code == 422  # Validation error