import asyncio

import fakeredis.aioredis

from tests.test_rate_limit import SimpleTokenBucketLimiter, per_min_to_refill


async def debug_token_bucket():
    redis_client = fakeredis.aioredis.FakeRedis()

    capacity = 5
    refill_rate = per_min_to_refill(capacity)
    limiter = SimpleTokenBucketLimiter(redis_client, capacity, refill_rate, "test")

    identifier = "test_user"

    print(f"Capacity: {capacity}")
    print(f"Refill rate: {refill_rate}")

    # Test 10 requests
    for i in range(10):
        result = await limiter.allow(identifier)
        print(f"Request {i + 1}: {result}")

        # Check bucket state
        bucket_data = await redis_client.hmget(f"test:{identifier}", "tokens", "ts")
        tokens = float(bucket_data[0]) if bucket_data[0] is not None else "None"
        ts = float(bucket_data[1]) if bucket_data[1] is not None else "None"
        print(f"  Bucket state: tokens={tokens}, ts={ts}")

    await redis_client.close()


if __name__ == "__main__":
    asyncio.run(debug_token_bucket())
