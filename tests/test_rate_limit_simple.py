"""
Простые интеграционные тесты rate limiting согласно требованиям задания 9.1
"""

import uuid
from typing import AsyncGenerator
from unittest.mock import patch

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.settings import settings
from app.infra.rate_limit import RateLimitMiddleware, TokenBucketLimiter, per_min_to_refill
from app.main import create_app


class MockTokenBucketLimiter:
    """Mock version of TokenBucketLimiter for testing without Lua scripts"""

    def __init__(self, redis_client, capacity: int, refill_per_sec: float, key_prefix: str = "rl"):
        self.redis = redis_client
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.key_prefix = key_prefix

    async def allow(self, identifier: str) -> bool:
        """Check if request should be allowed for given identifier"""
        import time

        key = f"{self.key_prefix}:{identifier}"
        current_time = time.time()

        # Get current bucket state
        bucket_data = await self.redis.hmget(key, "tokens", "ts")
        # Handle bytes returned by fakeredis
        tokens = float(bucket_data[0].decode()) if bucket_data[0] is not None else self.capacity
        last_refill = float(bucket_data[1].decode()) if bucket_data[1] is not None else current_time

        # Calculate tokens to add
        elapsed = current_time - last_refill
        tokens_to_add = elapsed * self.refill_per_sec
        tokens = min(self.capacity, tokens + tokens_to_add)

        if tokens < 1:
            await self.redis.expire(key, 120)
            return False

        # Consume token
        tokens -= 1
        await self.redis.hset(key, mapping={"tokens": tokens, "ts": current_time})
        await self.redis.expire(key, 120)
        return True


@pytest.fixture(scope="function")
async def fake_redis():
    """Fixture providing a fake Redis client for testing"""
    redis_client = fakeredis.aioredis.FakeRedis()
    try:
        yield redis_client
    finally:
        await redis_client.aclose()


@pytest.fixture(scope="function")
async def app_with_rate_limit():
    """Fixture providing FastAPI app with rate limiting enabled"""
    app = create_app()
    # Clear middleware and add only rate limiting
    app.middleware_stack = None
    app.add_middleware(RateLimitMiddleware)
    app.build_middleware_stack()
    return app


@pytest.fixture(scope="function")
async def client(app_with_rate_limit: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Fixture providing AsyncClient for testing"""
    transport = ASGITransport(app=app_with_rate_limit)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def valid_turn_payload():
    """Generate a minimal valid /turn request payload"""
    return {
        "therapist_utterance": "test utterance",
        "session_state": {
            "affect": "neutral",
            "trust": 0.5,
            "fatigue": 0.0,
            "access_level": 1,
            "risk_status": "low",
            "last_turn_summary": "test",
        },
        "case_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
    }


# Test 1: IP-limit - 6 запросов → 5 OK, 6-й = 429
@pytest.mark.anyio
async def test_ip_rate_limit_simple(client: AsyncClient, fake_redis):
    """Test IP rate limiting: 5 requests OK, 6th returns 429"""

    with (
        patch("app.infra.rate_limit.TokenBucketLimiter", MockTokenBucketLimiter),
        patch("app.infra.rate_limit.redis.from_url", return_value=fake_redis),
        patch.object(settings, "RATE_LIMIT_ENABLED", True),
        patch.object(settings, "RATE_LIMIT_IP_PER_MIN", 5),
    ):  # Малый лимит для теста
        payload = valid_turn_payload()
        # Remove session_id to test IP limiting only
        payload.pop("session_id")

        success_count = 0
        rate_limited_count = 0

        # Make 6 requests
        for i in range(6):
            response = await client.post("/turn", json=payload)
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited_count += 1
                data = response.json()
                assert data["detail"] == "rate_limited"
                assert data["scope"] == "ip"

        # Assertions
        assert success_count == 5, f"Expected 5 successful requests, got {success_count}"
        assert rate_limited_count == 1, f"Expected 1 rate limited request, got {rate_limited_count}"


# Test 2: Session-limit с X-Session-ID → 3-й = 429
@pytest.mark.anyio
async def test_session_rate_limit_with_header(client: AsyncClient, fake_redis):
    """Test session rate limiting with X-Session-ID header: 2 OK, 3rd returns 429"""

    with (
        patch("app.infra.rate_limit.TokenBucketLimiter", MockTokenBucketLimiter),
        patch("app.infra.rate_limit.redis.from_url", return_value=fake_redis),
        patch.object(settings, "RATE_LIMIT_ENABLED", True),
        patch.object(settings, "RATE_LIMIT_SESSION_PER_MIN", 2),
    ):  # Малый лимит
        session_id_1 = str(uuid.uuid4())
        session_id_2 = str(uuid.uuid4())
        payload = valid_turn_payload()
        payload.pop("session_id")  # Remove from body to test header

        success_count_1 = 0
        rate_limited_count_1 = 0

        # Test session 1 - should allow 2, block 3rd
        for i in range(3):
            response = await client.post(
                "/turn", json=payload, headers={"X-Session-ID": session_id_1}
            )
            if response.status_code == 200:
                success_count_1 += 1
            elif response.status_code == 429:
                rate_limited_count_1 += 1
                data = response.json()
                assert data["detail"] == "rate_limited"
                assert data["scope"] == "session"

        assert success_count_1 == 2
        assert rate_limited_count_1 == 1

        # Test session 2 - should not be blocked (different session)
        response = await client.post("/turn", json=payload, headers={"X-Session-ID": session_id_2})
        assert response.status_code == 200


# Test 3: Refill тест с монкингом времени
@pytest.mark.anyio
async def test_token_bucket_refill_with_time_mock(fake_redis):
    """Test that tokens are refilled after time passes using time mocking"""

    # Mock the script registration to avoid evalsha issues
    capacity = 3
    refill_rate = per_min_to_refill(capacity)  # 3/60 = 0.05 tokens/sec

    # Override the limiter initialization to not use Lua scripts
    class MockLimiter(TokenBucketLimiter):
        def __init__(self, redis_client, capacity, refill_per_sec, key_prefix):
            self.redis = redis_client
            self.capacity = capacity
            self.refill_per_sec = refill_per_sec
            self.key_prefix = key_prefix
            # Skip script registration

        async def allow(self, identifier: str) -> bool:
            """Simple non-atomic version for testing"""
            import time

            key = f"{self.key_prefix}:{identifier}"
            current_time = time.time()

            # Get current bucket state
            bucket_data = await self.redis.hmget(key, "tokens", "ts")
            tokens = float(bucket_data[0].decode()) if bucket_data[0] is not None else self.capacity
            last_refill = (
                float(bucket_data[1].decode()) if bucket_data[1] is not None else current_time
            )

            # Calculate tokens to add
            elapsed = current_time - last_refill
            tokens_to_add = elapsed * self.refill_per_sec
            tokens = min(self.capacity, tokens + tokens_to_add)

            if tokens < 1:
                await self.redis.expire(key, 120)
                return False

            # Consume token
            tokens -= 1
            await self.redis.hset(key, mapping={"tokens": tokens, "ts": current_time})
            await self.redis.expire(key, 120)
            return True

    limiter = MockLimiter(fake_redis, capacity, refill_rate, "test")
    identifier = "test_user"
    base_time = 1000.0

    # Use all tokens at base time
    with patch("time.time", return_value=base_time):
        for i in range(3):
            result = await limiter.allow(identifier)
            assert result is True, f"Request {i + 1} should be allowed"

        # 4th request should be denied
        result = await limiter.allow(identifier)
        assert result is False, "4th request should be denied"

    # Advance time by 60 seconds
    with patch("time.time", return_value=base_time + 60.0):
        # After 60 seconds, bucket should be refilled
        # 60 seconds * (3/60) tokens/sec = 3 tokens should be available
        success_count = 0
        for i in range(3):
            result = await limiter.allow(identifier)
            if result:
                success_count += 1

        assert success_count == 3, f"After refill, should allow 3 requests, got {success_count}"


# Test 4: Disabled flag → все запросы проходят
@pytest.mark.anyio
async def test_rate_limiting_disabled(client: AsyncClient, fake_redis):
    """Test that when rate limiting is disabled, all requests pass through"""

    with (
        patch("app.infra.rate_limit.TokenBucketLimiter", MockTokenBucketLimiter),
        patch("app.infra.rate_limit.redis.from_url", return_value=fake_redis),
        patch.object(settings, "RATE_LIMIT_ENABLED", False),
    ):
        payload = valid_turn_payload()
        success_count = 0

        # Make 10 requests - all should succeed
        for i in range(10):
            response = await client.post("/turn", json=payload)
            if response.status_code == 200:
                success_count += 1
            else:
                print(f"Request {i}: Status {response.status_code}")
                # Stop on first error for debugging
                if i == 0:
                    break

        assert success_count == 10, (
            f"All requests should succeed when rate limiting disabled, got {success_count}"
        )


# Test 5: Session-ID извлечение из JSON body
@pytest.mark.anyio
async def test_session_id_from_json_body(client: AsyncClient, fake_redis):
    """Test session_id extraction from JSON body (legacy behavior)"""

    with (
        patch("app.infra.rate_limit.TokenBucketLimiter", MockTokenBucketLimiter),
        patch("app.infra.rate_limit.redis.from_url", return_value=fake_redis),
        patch.object(settings, "RATE_LIMIT_ENABLED", True),
        patch.object(settings, "RATE_LIMIT_SESSION_PER_MIN", 2),
    ):
        payload = valid_turn_payload()
        success_count = 0

        # Test with session_id in JSON body (no header) - 2 success, 3rd blocked
        for i in range(3):
            response = await client.post("/turn", json=payload)
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                data = response.json()
                assert data["scope"] == "session"
                break

        assert success_count == 2
