"""Простейший лимитер запросов."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Deque, Dict


class RateLimiter:
    """Ограничивает количество запросов за указанный период."""

    def __init__(self, max_calls: int, period_seconds: int) -> None:
        self._max_calls = max_calls
        self._period = period_seconds
        self._calls: Dict[int, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: int) -> bool:
        """Проверяет, может ли пользователь выполнить ещё один запрос."""

        async with self._lock:
            now = time.monotonic()
            calls = self._calls[key]

            while calls and now - calls[0] > self._period:
                calls.popleft()

            if len(calls) >= self._max_calls:
                return False

            calls.append(now)
            return True

    async def remaining(self, key: int) -> int:
        """Возвращает оставшееся количество запросов."""

        async with self._lock:
            now = time.monotonic()
            calls = self._calls[key]
            while calls and now - calls[0] > self._period:
                calls.popleft()
            return max(self._max_calls - len(calls), 0)
