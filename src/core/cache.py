"""
Система кеширования для ответов OpenAI с поддержкой Redis и in-memory fallback
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Запись в кеше"""

    key: str
    value: Any
    created_at: float
    ttl_seconds: int
    access_count: int = 0
    last_accessed: float = 0.0

    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at

    @property
    def is_expired(self) -> bool:
        """Проверка истечения TTL"""
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Возраст записи в секундах"""
        return time.time() - self.created_at

    def to_dict(self) -> dict[str, Any]:
        """Сериализация для хранения в Redis"""
        return {
            "value": self.value,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, key: str, data: dict[str, Any]) -> CacheEntry:
        """Десериализация из Redis"""
        return cls(
            key=key,
            value=data["value"],
            created_at=data["created_at"],
            ttl_seconds=data["ttl_seconds"],
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", data["created_at"]),
        )


class CacheBackend(ABC):
    """Абстрактный интерфейс для cache backend"""

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """Получить значение из кеша"""
        pass

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> None:
        """Сохранить значение в кеше"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Удалить значение из кеша"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Очистить весь кеш"""
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Получить статистику кеша"""
        pass


class InMemoryCacheBackend(CacheBackend):
    """In-memory кеш с LRU eviction"""

    def __init__(self, max_size: int = 1000, cleanup_interval: float = 300):
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # LRU порядок
        self._lock = asyncio.Lock()

        # Статистика
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Фоновая очистка
        self._cleanup_task: asyncio.Task | None = None
        if cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def get(self, key: str) -> CacheEntry | None:
        async with self._lock:
            entry = self._cache.get(key)
            if not entry:
                self.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self.misses += 1
                return None

            # Обновляем статистику доступа
            entry.access_count += 1
            entry.last_accessed = time.time()

            # Обновляем LRU порядок
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            self.hits += 1
            return entry

    async def set(self, key: str, entry: CacheEntry) -> None:
        async with self._lock:
            # Проверяем лимит размера
            if key not in self._cache and len(self._cache) >= self.max_size:
                await self._evict_lru()

            self._cache[key] = entry

            # Обновляем LRU порядок
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()

    async def _evict_lru(self) -> None:
        """Удаление наименее используемого элемента"""
        if not self._access_order:
            return

        lru_key = self._access_order.pop(0)
        if lru_key in self._cache:
            del self._cache[lru_key]
            self.evictions += 1

    async def _cleanup_expired(self) -> None:
        """Очистка истекших записей"""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]

        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _cleanup_loop(self) -> None:
        """Фоновая очистка истекших записей"""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval)
                async with self._lock:
                    await self._cleanup_expired()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Cache cleanup loop error: {e}")

    async def get_stats(self) -> dict[str, Any]:
        async with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests) if total_requests > 0 else 0

            return {
                "backend": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "oldest_entry_age": min(
                    [entry.age_seconds for entry in self._cache.values()], default=0
                ),
            }

    async def close(self) -> None:
        """Закрытие кеша"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class RedisCacheBackend(CacheBackend):
    """Redis-based кеш"""

    def __init__(self, redis_url: str, key_prefix: str = "aivan:cache:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis: redis.Redis | None = None

        # Статистика (приблизительная для Redis)
        self.hits = 0
        self.misses = 0

    async def connect(self) -> None:
        """Подключение к Redis"""
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis is not available")

        self._redis = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

        # Проверяем соединение
        await self._redis.ping()
        logger.info("Connected to Redis cache backend")

    def _make_key(self, key: str) -> str:
        """Создание полного ключа с префиксом"""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> CacheEntry | None:
        if not self._redis:
            return None

        try:
            redis_key = self._make_key(key)
            data = await self._redis.get(redis_key)

            if not data:
                self.misses += 1
                return None

            # Десериализуем
            entry_data = json.loads(data)
            entry = CacheEntry.from_dict(key, entry_data)

            if entry.is_expired:
                await self._redis.delete(redis_key)
                self.misses += 1
                return None

            # Обновляем статистику доступа
            entry.access_count += 1
            entry.last_accessed = time.time()

            # Обновляем запись в Redis
            await self.set(key, entry)

            self.hits += 1
            return entry

        except Exception as e:
            logger.warning(f"Redis cache get error: {e}")
            self.misses += 1
            return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        if not self._redis:
            return

        try:
            redis_key = self._make_key(key)
            data = json.dumps(entry.to_dict())

            # Устанавливаем с TTL
            await self._redis.setex(redis_key, entry.ttl_seconds, data)

        except Exception as e:
            logger.warning(f"Redis cache set error: {e}")

    async def delete(self, key: str) -> bool:
        if not self._redis:
            return False

        try:
            redis_key = self._make_key(key)
            result = await self._redis.delete(redis_key)
            return result > 0
        except Exception as e:
            logger.warning(f"Redis cache delete error: {e}")
            return False

    async def clear(self) -> None:
        if not self._redis:
            return

        try:
            # Удаляем все ключи с нашим префиксом
            pattern = f"{self.key_prefix}*"
            keys = await self._redis.keys(pattern)
            if keys:
                await self._redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis cache clear error: {e}")

    async def get_stats(self) -> dict[str, Any]:
        stats = {
            "backend": "redis",
            "hits": self.hits,
            "misses": self.misses,
            "connected": self._redis is not None,
        }

        if self._redis:
            try:
                # Примерная статистика из Redis
                info = await self._redis.info()
                stats.update(
                    {
                        "redis_memory": info.get("used_memory_human", "unknown"),
                        "redis_connections": info.get("connected_clients", 0),
                    }
                )

                # Подсчет ключей с нашим префиксом
                pattern = f"{self.key_prefix}*"
                keys = await self._redis.keys(pattern)
                stats["size"] = len(keys)

            except Exception as e:
                logger.warning(f"Redis stats error: {e}")
                stats["redis_error"] = str(e)

        return stats

    async def close(self) -> None:
        """Закрытие соединения с Redis"""
        if self._redis:
            await self._redis.close()
            self._redis = None


class ResponseCache:
    """Кеш для ответов OpenAI с интеллектуальным хешированием"""

    def __init__(
        self,
        backend: CacheBackend,
        default_ttl: int = 3600,  # 1 час
        enable_compression: bool = True,
    ):
        self.backend = backend
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression

        # Статистика
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_cache_key(
        self, system_prompt: str, user_text: str, model_params: dict[str, Any] | None = None
    ) -> str:
        """Генерация ключа кеша на основе входных данных"""
        # Создаем детерминированный hash от всех параметров
        content = {
            "system_prompt": system_prompt.strip(),
            "user_text": user_text.strip().lower(),  # Нормализуем регистр для лучшего кеширования
            "model_params": model_params or {},
        }

        # Сериализуем в стабильный JSON
        content_json = json.dumps(content, sort_keys=True, ensure_ascii=False)

        # Создаем SHA-256 hash
        return hashlib.sha256(content_json.encode("utf-8")).hexdigest()

    def _compress_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Сжатие ответа для экономии места"""
        if not self.enable_compression:
            return response

        # Удаляем избыточную информацию
        compressed = response.copy()

        # Удаляем debug информацию если есть
        if "debug" in compressed:
            del compressed["debug"]

        # Сжимаем usage информацию
        if "usage" in compressed and isinstance(compressed["usage"], dict):
            usage = compressed["usage"]
            compressed["usage"] = {
                "total_tokens": usage.get("total_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }

        return compressed

    async def get_cached_response(
        self, system_prompt: str, user_text: str, model_params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Получение кешированного ответа"""
        self.total_requests += 1

        try:
            cache_key = self._generate_cache_key(system_prompt, user_text, model_params)
            entry = await self.backend.get(cache_key)

            if entry:
                self.cache_hits += 1
                logger.debug(f"Cache hit for key {cache_key[:16]}...")
                return entry.value
            else:
                self.cache_misses += 1
                return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_misses += 1
            return None

    async def cache_response(
        self,
        system_prompt: str,
        user_text: str,
        response: dict[str, Any],
        model_params: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> None:
        """Кеширование ответа"""
        try:
            cache_key = self._generate_cache_key(system_prompt, user_text, model_params)
            compressed_response = self._compress_response(response)

            entry = CacheEntry(
                key=cache_key,
                value=compressed_response,
                created_at=time.time(),
                ttl_seconds=ttl or self.default_ttl,
            )

            await self.backend.set(cache_key, entry)
            logger.debug(f"Cached response for key {cache_key[:16]}...")

        except Exception as e:
            logger.error(f"Cache set error: {e}")

    async def clear_cache(self) -> None:
        """Очистка всего кеша"""
        await self.backend.clear()
        logger.info("Response cache cleared")

    async def get_cache_stats(self) -> dict[str, Any]:
        """Статистика кеша"""
        backend_stats = await self.backend.get_stats()

        hit_rate = (self.cache_hits / self.total_requests) if self.total_requests > 0 else 0

        return {
            **backend_stats,
            "response_cache": {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "compression_enabled": self.enable_compression,
            },
        }

    async def close(self) -> None:
        """Закрытие кеша"""
        if hasattr(self.backend, "close"):
            await self.backend.close()


async def create_cache_backend(
    redis_url: str | None = None, fallback_to_memory: bool = True, memory_max_size: int = 1000
) -> CacheBackend:
    """Фабрика для создания cache backend с автоматическим fallback"""

    # Пытаемся создать Redis backend
    if redis_url and REDIS_AVAILABLE:
        try:
            redis_backend = RedisCacheBackend(redis_url)
            await redis_backend.connect()
            logger.info("Using Redis cache backend")
            return redis_backend
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            if not fallback_to_memory:
                raise

    # Fallback на memory backend
    if fallback_to_memory:
        memory_backend = InMemoryCacheBackend(max_size=memory_max_size)
        logger.info(f"Using in-memory cache backend (max_size={memory_max_size})")
        return memory_backend
    else:
        raise RuntimeError("No cache backend available")
