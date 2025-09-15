import asyncio
import time

import fakeredis.aioredis


class MockTokenBucketLimiter:
    """Mock version of TokenBucketLimiter for testing without Lua scripts"""

    def __init__(self, redis_client, capacity: int, refill_per_sec: float, key_prefix: str = "rl"):
        self.redis = redis_client
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.key_prefix = key_prefix

    async def allow(self, identifier: str) -> bool:
        """Check if request should be allowed for given identifier"""
        key = f"{self.key_prefix}:{identifier}"
        current_time = time.time()

        print(f"DEBUG: key={key}, capacity={self.capacity}, refill_per_sec={self.refill_per_sec}")

        # Get current bucket state
        bucket_data = await self.redis.hmget(key, "tokens", "ts")
        print(f"DEBUG: bucket_data={bucket_data}")

        # Handle bytes returned by fakeredis
        tokens = float(bucket_data[0].decode()) if bucket_data[0] is not None else self.capacity
        last_refill = float(bucket_data[1].decode()) if bucket_data[1] is not None else current_time

        print(f"DEBUG: tokens={tokens}, last_refill={last_refill}, current_time={current_time}")

        # Calculate tokens to add
        elapsed = current_time - last_refill
        tokens_to_add = elapsed * self.refill_per_sec
        tokens = min(self.capacity, tokens + tokens_to_add)

        print(f"DEBUG: elapsed={elapsed}, tokens_to_add={tokens_to_add}, final_tokens={tokens}")

        if tokens < 1:
            await self.redis.expire(key, 120)
            print("DEBUG: DENIED - not enough tokens")
            return False

        # Consume token
        tokens -= 1
        await self.redis.hset(key, mapping={"tokens": tokens, "ts": current_time})
        await self.redis.expire(key, 120)
        print(f"DEBUG: ALLOWED - tokens after consumption: {tokens}")
        return True


async def test_mock_limiter():
    redis_client = fakeredis.aioredis.FakeRedis()

    capacity = 5
    refill_rate = 0.1  # 0.1 tokens per second
    limiter = MockTokenBucketLimiter(redis_client, capacity, refill_rate, "test")

    identifier = "test_user"

    print("Testing mock limiter...")

    # Test 6 requests
    for i in range(6):
        print(f"\n--- Request {i + 1} ---")
        result = await limiter.allow(identifier)
        print(f"Result: {result}\n")

    await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(test_mock_limiter())
