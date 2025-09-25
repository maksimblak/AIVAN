from __future__ import annotations

import time
from collections import deque

try:
    from redis import asyncio as redis_async  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis_async = None  # type: ignore


class RateLimiter:
    """Per-user fixed window limiter with optional Redis backend.

    Fallback to in-memory deque timestamps when Redis is not configured.
    """

    def __init__(self, *, redis_url: str | None, max_requests: int, window_seconds: int) -> None:
        self.redis_url = redis_url
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._redis: redis_async.Redis | None = None
        self._local: dict[int, deque[float]] = {}

    async def init(self) -> None:
        if self.redis_url and redis_async is not None:
            try:
                self._redis = redis_async.Redis.from_url(
                    self.redis_url, encoding="utf-8", decode_responses=True
                )
                await self._redis.ping()
            except Exception:
                self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._redis = None

    async def allow(self, user_id: int) -> bool:
        now = time.time()
        if self._redis is not None:
            key = f"rl:{user_id}"
            try:
                async with self._redis.pipeline(transaction=True) as pipe:  # type: ignore[union-attr]
                    pipe.incr(key)
                    pipe.expire(key, self.window_seconds)
                    res = await pipe.execute()
                count = int(res[0]) if res else 0
                return count <= self.max_requests
            except Exception:
                # Fallback to local if Redis temporarily fails
                pass

        dq = self._local.get(user_id)
        if dq is None:
            dq = deque()
            self._local[user_id] = dq
        # Drop old entries
        threshold = now - self.window_seconds
        while dq and dq[0] < threshold:
            dq.popleft()
        if len(dq) >= self.max_requests:
            return False
        dq.append(now)
        return True
