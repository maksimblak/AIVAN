from __future__ import annotations

import asyncio
import time
import random
from collections import defaultdict, deque
from typing import Deque, Dict, Set


class RateLimiter:
    """
    Простой лимитер с защитой от утечек памяти: не более `max_calls` за `period_seconds` на ключ.
    
    Особенности:
    - Асинхронно-безопасный (общий lock для скорости)
    - Автоматическая очистка неактивных пользователей
    - Ограничение на максимальное количество отслеживаемых ключей
    - Периодическая очистка для предотвращения утечек памяти
    """

    def __init__(self, max_calls: int, period_seconds: int, max_tracked_keys: int = 50000) -> None:
        self.max_calls = max(1, int(max_calls))
        self.period = max(1, int(period_seconds))
        self.max_tracked_keys = max(100, int(max_tracked_keys))  # Минимум 100 ключей
        
        self._hits: Dict[int, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        
        # Для оптимизации очистки
        self._last_cleanup = time.time()
        self._cleanup_interval = max(300, self.period)  # Очистка не чаще раз в 5 минут
        
    async def _cleanup_inactive_keys(self) -> None:
        """Удаляет неактивные ключи для предотвращения утечек памяти."""
        now = time.time()
        cutoff = now - self.period * 2  # Удаляем ключи, неактивные в 2 раза дольше периода
        
        # Находим неактивные ключи
        inactive_keys: Set[int] = set()
        for key, queue in self._hits.items():
            if not queue or (queue and queue[-1] < cutoff):
                inactive_keys.add(key)
        
        # Удаляем неактивные ключи
        for key in inactive_keys:
            del self._hits[key]
        
        # Если все еще слишком много ключей, удаляем самые старые
        if len(self._hits) > self.max_tracked_keys:
            # Сортируем по времени последнего обращения
            sorted_keys = sorted(
                self._hits.items(),
                key=lambda x: x[1][-1] if x[1] else 0
            )
            
            # Удаляем половину самых старых ключей
            keys_to_remove = len(self._hits) - self.max_tracked_keys // 2
            for key, _ in sorted_keys[:keys_to_remove]:
                del self._hits[key]
        
        self._last_cleanup = now

    async def check(self, key: int) -> bool:
        """
        Возвращает True, если вызов можно пропустить (не превышен лимит).
        Включает защиту от timing атак.
        """
        # Добавляем небольшую случайную задержку для защиты от timing атак
        await asyncio.sleep(random.uniform(0.001, 0.005))
        
        now = time.time()
        cutoff = now - self.period
        
        async with self._lock:
            # Периодическая очистка памяти
            if now - self._last_cleanup > self._cleanup_interval:
                await self._cleanup_inactive_keys()
            
            q = self._hits[key]
            # Очищаем устаревшие записи для текущего ключа
            while q and q[0] < cutoff:
                q.popleft()
            
            # Проверяем лимит
            if len(q) < self.max_calls:
                q.append(now + random.uniform(-0.001, 0.001))  # Небольшой джиттер
                # Дополнительная задержка для всех запросов (не выдаем результат сразу)
                await asyncio.sleep(random.uniform(0.001, 0.003))
                return True
            
            # Одинаковая задержка для отклоненных запросов
            await asyncio.sleep(random.uniform(0.001, 0.003))
            return False

    async def remaining(self, key: int) -> int:
        """
        Сколько ещё вызовов доступно в текущем окне.
        """
        now = time.time()
        cutoff = now - self.period
        
        async with self._lock:
            q = self._hits[key]
            # Очищаем устаревшие записи
            while q and q[0] < cutoff:
                q.popleft()
            return max(0, self.max_calls - len(q))
    
    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику для мониторинга."""
        return {
            "tracked_keys": len(self._hits),
            "max_tracked_keys": self.max_tracked_keys,
            "cleanup_interval": self._cleanup_interval,
        }
