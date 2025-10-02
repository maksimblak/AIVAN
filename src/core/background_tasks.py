"""
Система фоновых задач для автоматического обслуживания приложения
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .exceptions import ErrorContext, ErrorHandler
from src.documents.base import DocumentStorage

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Статус фоновой задачи"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Результат выполнения задачи"""

    status: TaskStatus
    started_at: float
    completed_at: float | None = None
    duration: float | None = None
    result: Any | None = None
    error: str | None = None
    retry_count: int = 0

    def __post_init__(self):
        if self.completed_at and self.duration is None:
            self.duration = self.completed_at - self.started_at


class BackgroundTask(ABC):
    """Базовый класс для фоновых задач"""

    def __init__(
        self,
        name: str,
        interval_seconds: float = 300,  # 5 минут по умолчанию
        max_retries: int = 3,
        retry_delay: float = 60,
        enabled: bool = True,
    ):
        self.name = name
        self.interval_seconds = interval_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enabled = enabled

        # Состояние
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_run: float | None = None
        self._run_count = 0
        self._error_count = 0
        self._last_result: TaskResult | None = None

        # Error handler
        self._error_handler: ErrorHandler | None = None

    def set_error_handler(self, error_handler: ErrorHandler) -> None:
        """Установка обработчика ошибок"""
        self._error_handler = error_handler

    @abstractmethod
    async def execute(self) -> Any:
        """Выполнение задачи (должно быть переопределено в наследниках)"""
        pass

    async def _safe_execute(self) -> TaskResult:
        """Безопасное выполнение с обработкой ошибок"""
        start_time = time.time()
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                result = await self.execute()

                task_result = TaskResult(
                    status=TaskStatus.COMPLETED,
                    started_at=start_time,
                    completed_at=time.time(),
                    result=result,
                    retry_count=retry_count,
                )

                logger.debug(f"Background task '{self.name}' completed successfully")
                return task_result

            except Exception as e:
                retry_count += 1
                self._error_count += 1

                # Обрабатываем через error handler если есть
                if self._error_handler:
                    try:
                        context = ErrorContext(
                            function_name=f"background_task_{self.name}",
                            additional_data={"retry_count": retry_count},
                        )
                        await self._error_handler.handle_exception(e, context)
                    except:
                        pass

                if retry_count > self.max_retries:
                    logger.error(
                        f"Background task '{self.name}' failed after {self.max_retries} retries: {e}"
                    )
                    return TaskResult(
                        status=TaskStatus.FAILED,
                        started_at=start_time,
                        completed_at=time.time(),
                        error=str(e),
                        retry_count=retry_count - 1,
                    )
                else:
                    logger.warning(
                        f"Background task '{self.name}' failed (retry {retry_count}/{self.max_retries}): {e}"
                    )
                    await asyncio.sleep(self.retry_delay)

        return TaskResult(
            status=TaskStatus.FAILED,
            started_at=start_time,
            completed_at=time.time(),
            error="Max retries exceeded",
            retry_count=retry_count,
        )

    async def _run_loop(self) -> None:
        """Основной цикл выполнения задачи"""
        logger.info(
            f"Starting background task '{self.name}' with interval {self.interval_seconds}s"
        )

        try:
            while self._running and self.enabled:
                self._run_count += 1
                self._last_run = time.time()

                # Выполняем задачу
                self._last_result = await self._safe_execute()

                # Ждем до следующего выполнения
                if self._running:
                    await asyncio.sleep(self.interval_seconds)

        except asyncio.CancelledError:
            logger.info(f"Background task '{self.name}' was cancelled")
            self._last_result = TaskResult(
                status=TaskStatus.CANCELLED, started_at=time.time(), completed_at=time.time()
            )
        except Exception as e:
            logger.error(f"Background task '{self.name}' loop failed: {e}")
            self._last_result = TaskResult(
                status=TaskStatus.FAILED,
                started_at=time.time(),
                completed_at=time.time(),
                error=str(e),
            )

    async def start(self) -> None:
        """Запуск фоновой задачи"""
        if self._running:
            logger.warning(f"Background task '{self.name}' is already running")
            return

        if not self.enabled:
            logger.info(f"Background task '{self.name}' is disabled")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Background task '{self.name}' started")

    async def stop(self) -> None:
        """Остановка фоновой задачи"""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(f"Background task '{self.name}' stopped")

    def get_stats(self) -> dict[str, Any]:
        """Получение статистики задачи"""
        stats = {
            "name": self.name,
            "enabled": self.enabled,
            "running": self._running,
            "interval_seconds": self.interval_seconds,
            "run_count": self._run_count,
            "error_count": self._error_count,
            "last_run": self._last_run,
            "uptime": time.time() - self._last_run if self._last_run else None,
        }

        if self._last_result:
            stats["last_result"] = {
                "status": self._last_result.status.value,
                "duration": self._last_result.duration,
                "retry_count": self._last_result.retry_count,
                "error": self._last_result.error,
            }

        return stats


# Конкретные реализации фоновых задач


class DatabaseCleanupTask(BackgroundTask):
    """Задача очистки базы данных"""

    def __init__(
        self,
        database,
        max_old_transactions_days: int = 90,
        cleanup_expired_users: bool = True,
        **kwargs,
    ):
        super().__init__(name="database_cleanup", **kwargs)
        self.database = database
        self.max_old_transactions_days = max_old_transactions_days
        self.cleanup_expired_users = cleanup_expired_users

    async def execute(self) -> dict[str, int]:
        """Выполнение очистки базы данных"""
        results = {"transactions_deleted": 0, "expired_sessions_cleaned": 0}

        # Очистка старых транзакций
        if hasattr(self.database, "pool"):
            async with self.database.pool.acquire() as conn:
                cutoff_timestamp = int(time.time() - (self.max_old_transactions_days * 86400))

                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM transactions WHERE created_at < ? AND status IN ('success', 'failed')",
                    (cutoff_timestamp,),
                )
                old_count = (await cursor.fetchone())[0]
                await cursor.close()

                if old_count > 0:
                    await conn.execute(
                        "DELETE FROM transactions WHERE created_at < ? AND status IN ('success', 'failed')",
                        (cutoff_timestamp,),
                    )
                    results["transactions_deleted"] = old_count
                    logger.info(f"Deleted {old_count} old transactions")

        return results


class CacheCleanupTask(BackgroundTask):
    """Задача очистки кеша"""

    def __init__(self, cache_services: list[Any], **kwargs):
        super().__init__(name="cache_cleanup", **kwargs)
        self.cache_services = cache_services

    async def execute(self) -> dict[str, Any]:
        """Выполнение очистки кеша"""
        results = {"services_cleaned": 0, "total_stats": {}}

        for service in self.cache_services:
            try:
                if hasattr(service, "cache") and service.cache:
                    # Получаем статистику до очистки
                    stats_before = await service.cache.get_cache_stats()

                    # Выполняем cleanup если есть такой метод
                    if hasattr(service.cache.backend, "_cleanup_expired"):
                        await service.cache.backend._cleanup_expired()

                    stats_after = await service.cache.get_cache_stats()

                    results["services_cleaned"] += 1
                    results["total_stats"][service.__class__.__name__] = {
                        "before": stats_before,
                        "after": stats_after,
                    }

            except Exception as e:
                logger.warning(f"Failed to cleanup cache for {service.__class__.__name__}: {e}")

        return results


class SessionCleanupTask(BackgroundTask):
    """Задача очистки сессий"""

    def __init__(self, session_store, **kwargs):
        super().__init__(name="session_cleanup", **kwargs)
        self.session_store = session_store

    async def execute(self) -> dict[str, int]:
        """Выполнение очистки сессий"""
        if hasattr(self.session_store, "cleanup"):
            sessions_before = len(getattr(self.session_store, "_sessions", {}))
            self.session_store.cleanup()
            sessions_after = len(getattr(self.session_store, "_sessions", {}))

            cleaned = sessions_before - sessions_after
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired user sessions")

            return {"sessions_cleaned": cleaned}

        return {"sessions_cleaned": 0}


class DocumentStorageCleanupTask(BackgroundTask):
    """Periodically remove stale files from local document storage."""

    def __init__(
        self,
        storage: DocumentStorage,
        *,
        max_age_hours: int,
        interval_seconds: float = 3600.0,
    ) -> None:
        super().__init__(name="document_storage_cleanup", interval_seconds=interval_seconds)
        self.storage = storage
        self.max_age_hours = max_age_hours

    async def execute(self) -> dict[str, Any]:
        removed = await self.storage.cleanup_all_users(self.max_age_hours)
        return {"removed": removed, "max_age_hours": self.max_age_hours}



class HealthCheckTask(BackgroundTask):
    """Задача проверки состояния системы"""

    def __init__(self, components: dict[str, Any], **kwargs):
        super().__init__(name="health_check", **kwargs)
        self.components = components

    async def execute(self) -> dict[str, Any]:
        """Проверка состояния всех компонентов"""
        results = {"healthy_components": 0, "unhealthy_components": 0, "checks": {}}

        for name, component in self.components.items():
            try:
                # Специальная обработка для health_checker
                if hasattr(component, "get_all_stats"):
                    # Это HealthChecker
                    stats = component.get_all_stats()  # Синхронный метод
                    results["checks"][name] = {"status": "healthy", "stats": stats}
                    results["healthy_components"] += 1
                # Пытаемся получить статистику/здоровье компонента
                elif hasattr(component, "get_stats"):
                    # Проверяем, это async метод или нет
                    stats_method = component.get_stats
                    if inspect.iscoroutinefunction(stats_method):
                        stats = await stats_method()
                    else:
                        stats = stats_method()
                    results["checks"][name] = {"status": "healthy", "stats": stats}
                    results["healthy_components"] += 1
                elif hasattr(component, "ping"):
                    ping_method = component.ping
                    if inspect.iscoroutinefunction(ping_method):
                        await ping_method()
                    else:
                        ping_method()
                    results["checks"][name] = {"status": "healthy"}
                    results["healthy_components"] += 1
                # Базовая проверка - компонент существует и не None
                elif component is not None:
                    results["checks"][name] = {"status": "healthy", "note": "basic_check"}
                    results["healthy_components"] += 1
                else:
                    results["checks"][name] = {"status": "unhealthy", "error": "component_is_none"}
                    results["unhealthy_components"] += 1

            except Exception as e:
                results["checks"][name] = {"status": "unhealthy", "error": str(e)}
                results["unhealthy_components"] += 1
                logger.warning(f"Health check failed for {name}: {e}")

        # Логгируем общее состояние
        total = results["healthy_components"] + results["unhealthy_components"]
        logger.info(f"Health check: {results['healthy_components']}/{total} components healthy")

        return results


class MetricsCollectionTask(BackgroundTask):
    """Задача сбора метрик"""

    def __init__(self, components: dict[str, Any], **kwargs):
        super().__init__(name="metrics_collection", **kwargs)
        self.components = components
        self.metrics_history: list[dict[str, Any]] = []
        self.max_history = 100  # Храним последние 100 записей

    async def execute(self) -> dict[str, Any]:
        """Сбор метрик со всех компонентов"""
        timestamp = time.time()
        metrics = {"timestamp": timestamp, "components": {}}

        for name, component in self.components.items():
            try:
                if hasattr(component, "get_stats"):
                    stats = await component.get_stats()
                    metrics["components"][name] = stats

            except Exception as e:
                logger.warning(f"Failed to collect metrics from {name}: {e}")
                metrics["components"][name] = {"error": str(e)}

        # Добавляем в историю
        self.metrics_history.append(metrics)

        # Ограничиваем размер истории
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history :]

        return {
            "components_collected": len(metrics["components"]),
            "history_size": len(self.metrics_history),
        }

    def get_metrics_history(self) -> list[dict[str, Any]]:
        """Получение истории метрик"""
        return self.metrics_history.copy()


class BackgroundTaskManager:
    """Менеджер фоновых задач"""

    def __init__(self, error_handler: ErrorHandler | None = None):
        self.tasks: dict[str, BackgroundTask] = {}
        self.error_handler = error_handler

    def register_task(self, task: BackgroundTask) -> None:
        """Регистрация фоновой задачи"""
        if self.error_handler:
            task.set_error_handler(self.error_handler)

        self.tasks[task.name] = task
        logger.info(f"Registered background task '{task.name}'")

    async def start_all(self) -> None:
        """Запуск всех зарегистрированных задач"""
        for task in self.tasks.values():
            await task.start()

        logger.info(f"Started {len(self.tasks)} background tasks")

    async def stop_all(self) -> None:
        """Остановка всех задач"""
        for task in self.tasks.values():
            await task.stop()

        logger.info("All background tasks stopped")

    async def restart_task(self, task_name: str) -> bool:
        """Перезапуск конкретной задачи"""
        if task_name not in self.tasks:
            return False

        task = self.tasks[task_name]
        await task.stop()
        await task.start()
        logger.info(f"Restarted background task '{task_name}'")
        return True

    def get_all_stats(self) -> dict[str, Any]:
        """Получение статистики всех задач"""
        return {
            "total_tasks": len(self.tasks),
            "running_tasks": sum(1 for task in self.tasks.values() if task._running),
            "enabled_tasks": sum(1 for task in self.tasks.values() if task.enabled),
            "tasks": {name: task.get_stats() for name, task in self.tasks.items()},
        }
