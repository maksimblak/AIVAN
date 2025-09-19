import time
import asyncio
from collections import defaultdict, deque
from typing import Deque, Dict, Hashable

class RateLimiter:
    """
    Простой асинхронный лимитер по ключу (user/chat):
    - max_requests: сколько событий разрешено
    - period: окно (сек)
    - max_tracked_keys: максимум одновременно отслеживаемых ключей
    """
    def __init__(self, max_requests: int, period: float, max_tracked_keys: int = 50_000) -> None:
        self.max_requests = int(max_requests)
        self.period = float(period)
        self.max_tracked_keys = int(max_tracked_keys)
        self._hits: Dict[Hashable, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: Hashable) -> bool:
        now = time.time()
        cutoff = now - self.period
        async with self._lock:
            q = self._hits[key]

            # убрать старые
            while q and q[0] <= cutoff:
                q.popleft()

            # лимит достигнут?
            if len(q) >= self.max_requests:
                return False

            # записать хит
            q.append(now)

            # мягкая уборка LRU при переполнении словаря ключей
            if len(self._hits) > self.max_tracked_keys:
                overflow = len(self._hits) - self.max_tracked_keys
                if overflow > 0:
                    # сортируем по последнему обращению (старые — первыми)
                    keys_by_lru = sorted(self._hits.items(), key=lambda kv: kv[1][-1] if kv[1] else 0.0)
                    removed = 0
                    for k, _ in keys_by_lru:
                        if k == key:
                            continue  # не трогаем текущий активный ключ
                        del self._hits[k]
                        removed += 1
                        if removed >= overflow:
                            break
            return True

    async def remaining(self, key: Hashable) -> int:
        async with self._lock:
            q = self._hits.get(key)
            if not q:
                return self.max_requests
            return max(0, self.max_requests - len(q))

    async def retry_after(self, key: Hashable) -> float:
        """
        Сколько секунд ждать до освобождения первого слота (0.0 — если уже можно).
        """
        now = time.time()
        cutoff = now - self.period
        async with self._lock:
            q = self._hits.get(key)
            if not q:
                return 0.0
            # убрать старые
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) < self.max_requests:
                return 0.0
            # время истечения самого «старого» события в окне
            return max(0.0, q[0] + self.period - now)
