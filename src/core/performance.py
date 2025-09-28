"""
Оптимизации производительности для AIVAN
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import weakref
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class PerformanceMetrics:
    """Сбор метрик производительности"""

    def __init__(self):
        self._metrics: Dict[str, list] = {}
        self._counters: Dict[str, int] = {}

    def record_timing(self, name: str, duration: float):
        """Запись времени выполнения"""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(duration)

        # Ограничиваем размер истории
        if len(self._metrics[name]) > 1000:
            self._metrics[name] = self._metrics[name][-500:]

    def increment_counter(self, name: str, value: int = 1):
        """Увеличение счетчика"""
        self._counters[name] = self._counters.get(name, 0) + value

    def get_average_timing(self, name: str) -> Optional[float]:
        """Получение среднего времени выполнения"""
        if name not in self._metrics or not self._metrics[name]:
            return None
        return sum(self._metrics[name]) / len(self._metrics[name])

    def get_counter(self, name: str) -> int:
        """Получение значения счетчика"""
        return self._counters.get(name, 0)

    def get_summary(self) -> Dict[str, Any]:
        """Получение сводки метрик"""
        summary = {}
        for name, timings in self._metrics.items():
            if timings:
                summary[f"{name}_avg"] = sum(timings) / len(timings)
                summary[f"{name}_min"] = min(timings)
                summary[f"{name}_max"] = max(timings)
                summary[f"{name}_count"] = len(timings)

        for name, count in self._counters.items():
            summary[f"{name}_total"] = count

        return summary


# Глобальный экземпляр метрик
metrics = PerformanceMetrics()


def timing(name: Optional[str] = None):
    """Декоратор для измерения времени выполнения функций"""

    def decorator(func: F) -> F:
        metric_name = name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    metrics.record_timing(metric_name, duration)
                    if duration > 1.0:  # Логируем долгие операции
                        logger.warning(f"Slow operation {metric_name}: {duration:.3f}s")

            return async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    metrics.record_timing(metric_name, duration)
                    if duration > 1.0:
                        logger.warning(f"Slow operation {metric_name}: {duration:.3f}s")

            return sync_wrapper

    return decorator


class LRUCache:
    """LRU кеш с TTL поддержкой"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        """Получение значения из кеша"""
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                # Проверяем TTL
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    # Перемещаем в конец (LRU)
                    self._cache.move_to_end(key)
                    return value
                else:
                    # Удаляем устаревшее значение
                    del self._cache[key]
        return None

    async def set(self, key: str, value: Any):
        """Установка значения в кеш"""
        async with self._lock:
            # Удаляем старое значение если есть
            if key in self._cache:
                del self._cache[key]

            # Добавляем новое значение
            self._cache[key] = (value, datetime.now())

            # Проверяем размер кеша
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # Удаляем самый старый

    async def clear(self):
        """Очистка кеша"""
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self):
        """Очистка устаревших записей"""
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if now - timestamp >= timedelta(seconds=self.ttl_seconds)
            ]
            for key in expired_keys:
                del self._cache[key]

    def size(self) -> int:
        """Текущий размер кеша"""
        return len(self._cache)


def cached(cache_instance: LRUCache, key_func: Optional[Callable] = None):
    """Декоратор для кеширования результатов функций"""

    def decorator(func: F) -> F:

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Генерируем ключ кеша
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"

            # Пытаемся получить из кеша
            cached_result = await cache_instance.get(cache_key)
            if cached_result is not None:
                metrics.increment_counter(f"cache_hit_{func.__name__}")
                return cached_result

            # Выполняем функцию
            metrics.increment_counter(f"cache_miss_{func.__name__}")
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Сохраняем в кеш
            await cache_instance.set(cache_key, result)
            return result

        return wrapper

    return decorator


class BatchProcessor:
    """Пакетная обработка операций для повышения производительности"""

    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._batches: Dict[str, list] = {}
        self._processors: Dict[str, Callable] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_flush: Dict[str, float] = {}

    def register_processor(self, batch_type: str, processor: Callable[[list], Awaitable[None]]):
        """Регистрация обработчика пакетов"""
        self._processors[batch_type] = processor
        self._batches[batch_type] = []
        self._locks[batch_type] = asyncio.Lock()
        self._last_flush[batch_type] = time.time()

    async def add_item(self, batch_type: str, item: Any):
        """Добавление элемента в пакет"""
        if batch_type not in self._processors:
            raise ValueError(f"Unknown batch type: {batch_type}")

        async with self._locks[batch_type]:
            self._batches[batch_type].append(item)

            # Проверяем нужно ли обработать пакет
            should_flush = (
                len(self._batches[batch_type]) >= self.batch_size
                or time.time() - self._last_flush[batch_type] > self.flush_interval
            )

            if should_flush:
                await self._flush_batch(batch_type)

    async def _flush_batch(self, batch_type: str):
        """Обработка пакета"""
        if not self._batches[batch_type]:
            return

        batch = self._batches[batch_type].copy()
        self._batches[batch_type].clear()
        self._last_flush[batch_type] = time.time()

        try:
            await self._processors[batch_type](batch)
            metrics.increment_counter(f"batch_processed_{batch_type}")
            metrics.record_timing(f"batch_size_{batch_type}", len(batch))
        except Exception as e:
            logger.error(f"Error processing batch {batch_type}: {e}")
            # Возвращаем элементы обратно в пакет при ошибке
            self._batches[batch_type].extend(batch)

    async def flush_all(self):
        """Принудительная обработка всех пакетов"""
        for batch_type in self._batches:
            async with self._locks[batch_type]:
                await self._flush_batch(batch_type)


class AsyncResourcePool:
    """Пул ресурсов с ограничением количества одновременных операций"""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._semaphore = asyncio.Semaphore(max_size)
        self._active_count = 0

    async def acquire(self):
        """Получение ресурса из пула"""
        await self._semaphore.acquire()
        self._active_count += 1

    def release(self):
        """Освобождение ресурса"""
        self._semaphore.release()
        self._active_count = max(0, self._active_count - 1)

    @property
    def active_count(self) -> int:
        """Количество активных ресурсов"""
        return self._active_count

    @property
    def available_count(self) -> int:
        """Количество доступных ресурсов"""
        return self.max_size - self._active_count


def rate_limit(calls_per_second: float):
    """Декоратор для ограничения частоты вызовов"""
    min_interval = 1.0 / calls_per_second
    last_call_time = 0.0
    lock = asyncio.Lock()

    def decorator(func: F) -> F:

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal last_call_time

            async with lock:
                now = time.time()
                time_since_last = now - last_call_time

                if time_since_last < min_interval:
                    sleep_time = min_interval - time_since_last
                    await asyncio.sleep(sleep_time)

                last_call_time = time.time()

            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


class MemoryOptimizer:
    """Оптимизатор использования памяти"""

    @staticmethod
    def create_weak_callback(callback: Callable):
        """Создание weak reference для callback'ов"""
        if hasattr(callback, '__self__'):
            # Это метод объекта
            obj_ref = weakref.ref(callback.__self__)
            func_name = callback.__name__

            def weak_callback(*args, **kwargs):
                obj = obj_ref()
                if obj is not None:
                    return getattr(obj, func_name)(*args, **kwargs)
                else:
                    logger.debug(f"Weak callback {func_name} called on deleted object")

            return weak_callback
        else:
            return callback

    @staticmethod
    def optimize_string_storage(text: str) -> str:
        """Оптимизация хранения строк через интернирование"""
        if len(text) < 100:  # Интернируем только короткие строки
            return text.__intern__() if hasattr(text, '__intern__') else text
        return text


# Глобальные экземпляры для использования в приложении
response_cache = LRUCache(max_size=1000, ttl_seconds=1800)  # 30 минут
batch_processor = BatchProcessor(batch_size=50, flush_interval=2.0)
resource_pool = AsyncResourcePool(max_size=20)


async def initialize_performance_components():
    """Инициализация компонентов производительности"""
    logger.info("Initializing performance components")

    # Регистрируем обработчик для пакетного логирования
    async def log_batch_processor(items: list):
        # Группируем логи по уровням
        log_levels = {}
        for item in items:
            level = item.get('level', 'INFO')
            if level not in log_levels:
                log_levels[level] = []
            log_levels[level].append(item['message'])

        # Логируем пакетами
        for level, messages in log_levels.items():
            if messages:
                logger.log(getattr(logging, level), f"Batch log ({len(messages)} items): {messages[:5]}...")

    batch_processor.register_processor("logs", log_batch_processor)

    # Запускаем фоновую задачу очистки кеша
    asyncio.create_task(_cache_cleanup_task())


async def _cache_cleanup_task():
    """Фоновая задача очистки кеша"""
    while True:
        try:
            await asyncio.sleep(300)  # 5 минут
            await response_cache.cleanup_expired()
            logger.debug(f"Cache cleanup completed. Size: {response_cache.size()}")
        except Exception as e:
            logger.error(f"Error in cache cleanup task: {e}")


def get_performance_summary() -> Dict[str, Any]:
    """Получение сводки производительности"""
    return {
        "metrics": metrics.get_summary(),
        "cache": {
            "size": response_cache.size(),
            "max_size": response_cache.max_size,
        },
        "resource_pool": {
            "active": resource_pool.active_count,
            "available": resource_pool.available_count,
            "max_size": resource_pool.max_size,
        },
    }