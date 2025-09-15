import uuid
from unittest.mock import patch

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.responses import JSONResponse

from app.core.settings import settings
from app.infra.rate_limit import RateLimitMiddleware, per_min_to_refill


@pytest.fixture(scope="function")
async def fake_redis():
    """Fixture providing a fake Redis client for testing"""
    redis_client = fakeredis.aioredis.FakeRedis()
    try:
        yield redis_client
    finally:
        # Clear all data after each test
        await redis_client.flushall()
        await redis_client.close()


def create_test_app(fake_redis):
    """Create test app with mocked /turn endpoint and rate limiting"""
    app = FastAPI()

    # Add mocked /turn endpoint
    @app.post("/turn")
    async def mock_turn():
        return JSONResponse({"status": "success", "response": "mocked"})

    # Add test middleware with pre-configured Redis client
    app.add_middleware(RateLimitMiddleware, redis_client=fake_redis)

    return app




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
            "last_turn_summary": "test"
        },
        "case_id": "00000000-0000-0000-0000-000000000001",  # Fixed case ID for testing
        "session_id": str(uuid.uuid4())
    }


# Test 1: IP-limit - 121 запрос → 120 OK, 121-й = 429
@pytest.mark.anyio
async def test_ip_rate_limit_121_requests(fake_redis):
    """Test that IP rate limiting works: 120 requests OK, 121st returns 429"""

    # Patch settings only
    with patch.object(settings, 'RATE_LIMIT_ENABLED', True), \
         patch.object(settings, 'RATE_LIMIT_IP_PER_MIN', 120):

        # Create test app
        app = create_test_app(fake_redis)

        # Create test client
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = valid_turn_payload()
            # Remove session_id to test IP limiting only
            payload.pop('session_id')

            success_count = 0
            rate_limited_count = 0

            # Make 121 requests
            for i in range(121):
                response = await client.post("/turn", json=payload)
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    rate_limited_count += 1
                    # Check the response format
                    data = response.json()
                    assert data["detail"] == "rate_limited"
                    assert data["scope"] == "ip"

            # Assertions
            assert success_count == 120, f"Expected 120 successful requests, got {success_count}"
            assert rate_limited_count == 1, f"Expected 1 rate limited request, got {rate_limited_count}"


# Test 2: Session-limit с X-Session-ID → 21-й = 429
@pytest.mark.anyio
async def test_session_rate_limit_with_header(fake_redis):
    """Test session rate limiting with X-Session-ID header: 20 OK, 21st returns 429"""

    with patch.object(settings, 'RATE_LIMIT_ENABLED', True), \
         patch.object(settings, 'RATE_LIMIT_SESSION_PER_MIN', 20), \
         patch.object(settings, 'RATE_LIMIT_IP_PER_MIN', 10000):

        # Create test app
        app = create_test_app(fake_redis)

        # Create test client
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            session_id_1 = str(uuid.uuid4())
            session_id_2 = str(uuid.uuid4())
            payload = valid_turn_payload()
            # Remove session_id from body to test header-only extraction
            payload.pop('session_id')

            success_count_1 = 0
            rate_limited_count_1 = 0

            # Test session 1 - should allow 20, block 21st
            for i in range(21):
                response = await client.post(
                    "/turn",
                    json=payload,
                    headers={"X-Session-ID": session_id_1}
                )
                if response.status_code == 200:
                    success_count_1 += 1
                elif response.status_code == 429:
                    rate_limited_count_1 += 1
                    data = response.json()
                    assert data["detail"] == "rate_limited"
                    assert data["scope"] == "session"

            assert success_count_1 == 20
            assert rate_limited_count_1 == 1

            # Test session 2 - should not be blocked (different session)
            response = await client.post(
                "/turn",
                json=payload,
                headers={"X-Session-ID": session_id_2}
            )
            assert response.status_code == 200


# Test 3: Refill тест с монкингом времени +60s
@pytest.mark.anyio
async def test_token_bucket_refill(fake_redis):
    """Test that tokens are refilled after time passes"""

    # Create limiter with 20 tokens capacity, refill 20 per minute
    from app.infra.rate_limit import TokenBucketLimiter
    capacity = 20
    refill_rate = per_min_to_refill(capacity)  # 20/60 = 0.333 tokens/sec
    limiter = TokenBucketLimiter(fake_redis, capacity, refill_rate, "test")

    identifier = "test_user"
    base_time = 1000.0

    # Use all tokens (make 20 requests) at base time
    with patch('app.infra.rate_limit.now', return_value=base_time):
        for i in range(20):
            result = await limiter.allow(identifier)
            assert result is True, f"Request {i+1} should be allowed"

        # 21st request should be denied
        result = await limiter.allow(identifier)
        assert result is False, "21st request should be denied"

    # Now advance time by 60 seconds
    with patch('app.infra.rate_limit.now', return_value=base_time + 60.0):
        # After 60 seconds, bucket should be refilled
        # 60 seconds * (20/60) tokens/sec = 20 tokens should be added
        success_count = 0
        for i in range(20):
            result = await limiter.allow(identifier)
            if result:
                success_count += 1

        assert success_count == 20, f"After refill, should allow 20 requests, got {success_count}"


# Test 4: Disabled flag → 200 запросов проходят
@pytest.mark.anyio
async def test_rate_limiting_disabled(fake_redis):
    """Test that when rate limiting is disabled, all requests pass through"""

    with patch.object(settings, 'RATE_LIMIT_ENABLED', False):

        # Create test app
        app = create_test_app(fake_redis)

        # Create test client
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = valid_turn_payload()
            success_count = 0

            # Make 200 requests - all should succeed
            for i in range(200):
                response = await client.post("/turn", json=payload)
                if response.status_code == 200:
                    success_count += 1
                else:
                    print(f"Request {i}: Status {response.status_code}, Response: {response.text[:200]}")
                    if i == 0:  # Show first error
                        break

            assert success_count == 200, f"All 200 requests should succeed when rate limiting disabled, got {success_count}"


# Additional test: Verify session_id extraction from JSON body still works
@pytest.mark.anyio
async def test_session_id_from_json_body(fake_redis):
    """Test session_id extraction from JSON body (legacy behavior)"""

    with patch.object(settings, 'RATE_LIMIT_ENABLED', True), \
         patch.object(settings, 'RATE_LIMIT_SESSION_PER_MIN', 20), \
         patch.object(settings, 'RATE_LIMIT_IP_PER_MIN', 10000):

        # Create test app
        app = create_test_app(fake_redis)

        # Create test client
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = valid_turn_payload()
            success_count = 0

            # Test with session_id in JSON body (no header)
            for i in range(21):
                response = await client.post("/turn", json=payload)
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    data = response.json()
                    assert data["scope"] == "session"
                    break

            assert success_count == 20


# Test basic token bucket logic
@pytest.mark.anyio
async def test_simple_token_bucket_logic(fake_redis):
    """Test basic token bucket behavior"""

    from app.infra.rate_limit import TokenBucketLimiter
    capacity = 5
    refill_rate = per_min_to_refill(capacity)
    limiter = TokenBucketLimiter(fake_redis, capacity, refill_rate, "test")

    identifier = "test_user"

    # Should allow 5 requests initially
    for i in range(5):
        result = await limiter.allow(identifier)
        assert result is True

    # 6th request should be denied
    result = await limiter.allow(identifier)
    assert result is False