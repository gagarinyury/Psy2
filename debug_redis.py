import asyncio

import fakeredis.aioredis


async def debug_redis():
    redis_client = fakeredis.aioredis.FakeRedis()

    key = "test:user"
    print("Testing fakeredis behavior...")

    # Test HMGET on non-existent key
    result = await redis_client.hmget(key, 'tokens', 'ts')
    print(f"HMGET on non-existent key: {result}")
    print(f"Type: {type(result)}")
    print(f"Length: {len(result)}")
    print(f"result[0]: {result[0]} (type: {type(result[0])})")
    print(f"result[1]: {result[1]} (type: {type(result[1])})")

    # Check boolean values
    print(f"result[0] is None: {result[0] is None}")
    print(f"result[1] is None: {result[1] is None}")

    # Set some data and try again
    await redis_client.hmset(key, {'tokens': 5.0, 'ts': 1000.0})
    result2 = await redis_client.hmget(key, 'tokens', 'ts')
    print(f"HMGET after setting data: {result2}")

    await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(debug_redis())