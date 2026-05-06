import hashlib
import json
import os
from typing import Optional


try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    _redis = redis.from_url(REDIS_URL, decode_responses=True)
    _redis.ping()
    USE_REDIS = True
    print("Redis connected")
except Exception:
    USE_REDIS = False
    _local_cache: dict = {}
    print("Redis unavailable — using in-memory cache")

CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))


def _cache_key(text: str, top_k: int) -> str:
    h = hashlib.sha256(f"{text}:{top_k}".encode()).hexdigest()
    return f"sentiment:{h}"


def get(text: str, top_k: int) -> Optional[list]:
    key = _cache_key(text, top_k)
    if USE_REDIS:
        val = _redis.get(key)
        return json.loads(val) if val else None
    return _local_cache.get(key)


def set(text: str, top_k: int, result: list) -> None:
    key = _cache_key(text, top_k)
    if USE_REDIS:
        _redis.setex(key, CACHE_TTL, json.dumps(result))
    else:
        _local_cache[key] = result