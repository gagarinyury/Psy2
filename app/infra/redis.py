from redis import asyncio as aioredis

from app.core.settings import settings

_redis = None


async def get_redis():
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
    return _redis