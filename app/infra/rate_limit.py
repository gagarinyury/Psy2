import json
import time
from typing import Optional

import redis.asyncio as redis
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.settings import settings
from app.infra.logging import get_logger

logger = get_logger()


def now() -> float:
    """Get current time - mockable for tests"""
    return time.time()


# Lua script for atomic token bucket operations
LUA_TOKEN_BUCKET = r"""
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])   -- tokens per second
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])

if tokens == nil or ts == nil then
  tokens = capacity
  ts = now
else
  local delta = math.max(0, now - ts)
  tokens = math.min(capacity, tokens + delta * refill)
  ts = now
end

local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end

redis.call('HSET', key, 'tokens', tokens, 'ts', ts)
redis.call('EXPIRE', key, ttl)
return allowed
"""


def per_min_to_refill(capacity: int) -> float:
    """Convert per-minute capacity to refill-per-second rate"""
    return capacity / 60.0


async def allow(redis_client: redis.Redis, key: str, capacity: int) -> bool:
    """Simple allow function using token bucket algorithm"""
    refill_rate = per_min_to_refill(capacity)
    current_time = int(time.time())
    ttl = 120
    try:
        result = await redis_client.eval(
            LUA_TOKEN_BUCKET, 1, key, capacity, refill_rate, current_time, ttl
        )
        return result == 1 or result == "1"
    except Exception as e:
        # FakeRedis doesn't support eval, use pure Python fallback
        if "unknown command" in str(e) or "eval" in str(e).lower():
            logger.debug(f"Using Python fallback for rate limiting {key}: {e}")
            return await _python_fallback(
                redis_client, key, capacity, refill_rate, float(current_time), ttl
            )
        raise


async def _python_fallback(
    redis_client: redis.Redis,
    key: str,
    capacity: int,
    refill_rate: float,
    current_time: float,
    ttl: int,
) -> bool:
    """Python implementation of token bucket logic for testing with FakeRedis"""
    # Get current bucket state
    bucket_data = await redis_client.hmget(key, "tokens", "ts")

    # Parse current state
    if bucket_data[0] is None:
        # Initialize bucket
        tokens = float(capacity)
        last_refill = current_time
    else:
        tokens = float(bucket_data[0])
        last_refill = float(bucket_data[1]) if bucket_data[1] is not None else current_time

    # Calculate tokens to add based on elapsed time
    elapsed = current_time - last_refill
    tokens_to_add = elapsed * refill_rate
    tokens = min(capacity, tokens + tokens_to_add)

    # Check if request can be allowed
    if tokens < 1:
        # Set TTL and deny request
        await redis_client.expire(key, ttl)
        return False

    # Allow request - consume 1 token
    tokens -= 1

    # Update bucket state
    await redis_client.hset(key, mapping={"tokens": tokens, "ts": current_time})
    await redis_client.expire(key, ttl)

    return True


def get_client_ip(request: Request) -> str:
    """Extract client IP from request headers with X-Forwarded-For support"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fallback to direct client IP
    client_host = request.client.host if request.client else "unknown"
    return client_host


def extract_session_id(body: bytes, headers) -> Optional[str]:
    """Extract session_id from X-Session-ID header or JSON body"""
    # First try header
    session_id = headers.get("X-Session-ID")
    if session_id:
        return session_id

    # Then try JSON body if content-type is application/json
    if headers.get("content-type", "").startswith("application/json") and body:
        try:
            request_data = json.loads(body)
            return request_data.get("session_id")
        except (json.JSONDecodeError, KeyError):
            pass

    return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting /turn endpoint"""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Only apply rate limiting to /turn endpoint
        if request.url.path != "/turn" or request.method != "POST":
            return await call_next(request)

        # Check if rate limiting is enabled
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        try:
            # Get Redis client from app state
            redis_client = request.app.state.redis

            # Extract client IP
            client_ip = get_client_ip(request)

            # Read body for session extraction and downstream processing
            body = await request.body()

            # Extract session_id from header or request body
            session_id = extract_session_id(body, request.headers)

            # Check limits based on session_id presence
            if session_id:
                # If session_id exists, check only session limit
                session_allowed = await allow(
                    redis_client, f"rl:session:{session_id}", settings.RATE_LIMIT_SESSION_PER_MIN
                )
                if not session_allowed:
                    logger.warning(f"Rate limit exceeded for session: {session_id}")
                    return JSONResponse(
                        status_code=429, content={"detail": "rate_limited", "scope": "session"}
                    )
                # Session OK - do NOT check IP
            else:
                # No session_id, check IP limit only
                ip_allowed = await allow(
                    redis_client, f"rl:ip:{client_ip}", settings.RATE_LIMIT_IP_PER_MIN
                )
                if not ip_allowed:
                    logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                    return JSONResponse(
                        status_code=429, content={"detail": "rate_limited", "scope": "ip"}
                    )

            # Restore request body for downstream processing
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request._receive = receive

            # Continue with request processing
            return await call_next(request)

        except Exception:
            logger.exception("Rate limiting failed")
            if settings.RATE_LIMIT_FAIL_OPEN:
                return await call_next(request)
            return JSONResponse(status_code=503, content={"detail": "rate_limiter_unavailable"})
