from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Deque, Dict


class RateLimiter:
    """
    Простой лимитер: не более `max_calls` за `period_seconds` на ключ (обычно user_id).
    Асинхронно-безопасный (общий lock для скорости; при очень больших нагрузках можно
    заменить на шардирование по ключам).
    """

    def __init__(self, max_calls: int, period_seconds: int) -> None:
        self.max_calls = max(1, int(max_calls))
        self.period = max(1, int(period_seconds))
        self._hits: Dict[int, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: int) -> bool:
        """
        Возвращает True, если вызов можно пропустить (не превышен лимит).
        """
        now = time.time()
        cutoff = now - self.period
        async with self._lock:
            q = self._hits[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) < self.max_calls:
                q.append(now)
                return True
            return False

    async def remaining(self, key: int) -> int:
        """
        Сколько ещё вызовов доступно в текущем окне.
        """
        now = time.time()
        cutoff = now - self.period
        async with self._lock:
            q = self._hits[key]
            while q and q[0] < cutoff:
                q.popleft()
            return max(0, self.max_calls - len(q))
