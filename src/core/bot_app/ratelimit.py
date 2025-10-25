from __future__ import annotations

import time
from collections import deque
from typing import Any

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

    async def get_stats(self) -> dict[str, Any]:
        """Return non-destructive snapshot of limiter state for health checks."""
        snapshot: dict[str, Any] = {
            "backend": "redis" if self._redis else "memory",
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
        }

        if self._redis is not None:
            try:
                pong = await self._redis.ping()  # type: ignore[union-attr]
                snapshot["redis_ok"] = True
                snapshot["redis_response"] = pong
            except Exception as exc:  # pragma: no cover - diagnostics only
                snapshot["redis_ok"] = False
                snapshot["redis_error"] = str(exc)
        else:
            now = time.time()
            active_counts = [
                sum(1 for ts in dq if ts >= now - self.window_seconds)
                for dq in self._local.values()
            ]
            snapshot["tracked_users"] = len(self._local)
            snapshot["max_active_requests"] = max(active_counts, default=0)
            snapshot["limiter_saturated"] = (
                self.max_requests > 0 and snapshot["max_active_requests"] >= self.max_requests
            )

        return snapshot
