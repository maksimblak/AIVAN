"""
Система health checks для мониторинга состояния всех компонентов системы
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Статус здоровья компонента"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Результат проверки здоровья"""

    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = None
    response_time: float | None = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.details is None:
            self.details = {}

    def to_dict(self) -> dict[str, Any]:
        """Конвертация в словарь для сериализации"""
        return {
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "response_time": self.response_time,
            "timestamp": self.timestamp,
        }


class HealthCheck(ABC):
    """Абстрактный базовый класс для health check'ов"""

    def __init__(
        self, name: str, timeout: float = 5.0, critical: bool = True, tags: list[str] | None = None
    ):
        self.name = name
        self.timeout = timeout
        self.critical = critical  # Влияет ли на общий статус системы
        self.tags = tags or []

        # Статистика
        self.total_checks = 0
        self.failed_checks = 0
        self.degraded_checks = 0
        self.unknown_checks = 0
        self.last_result: HealthCheckResult | None = None

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Выполнение проверки здоровья"""
        pass

    async def execute(self) -> HealthCheckResult:
        """Выполнение проверки с обработкой таймаутов"""
        start_time = time.time()
        self.total_checks += 1

        try:
            result = await asyncio.wait_for(self.check(), timeout=self.timeout)
            result.response_time = time.time() - start_time
        except (asyncio.TimeoutError, TimeoutError):
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timeout after {self.timeout}s",
                response_time=time.time() - start_time,
            )
        except Exception as e:
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time=time.time() - start_time,
            )

        self._update_counters(result.status)
        self.last_result = result
        return result


    def _update_counters(self, status: HealthStatus) -> None:
        """Обновляет статистику по категориям статусов."""
        if status == HealthStatus.HEALTHY:
            return
        if status == HealthStatus.DEGRADED:
            self.degraded_checks += 1
        elif status == HealthStatus.UNHEALTHY:
            self.failed_checks += 1
        elif status == HealthStatus.UNKNOWN:
            self.unknown_checks += 1

    def get_stats(self) -> dict[str, Any]:
        """Статистика health check'а"""
        healthy_checks = max(
            self.total_checks - self.failed_checks - self.degraded_checks - self.unknown_checks,
            0,
        )
        success_rate = healthy_checks / self.total_checks if self.total_checks else 0.0

        return {
            "name": self.name,
            "critical": self.critical,
            "tags": self.tags,
            "total_checks": self.total_checks,
            "healthy_checks": healthy_checks,
            "failed_checks": self.failed_checks,
            "degraded_checks": self.degraded_checks,
            "unknown_checks": self.unknown_checks,
            "success_rate": success_rate,
            "last_check": self.last_result.to_dict() if self.last_result else None,
        }

# Конкретные реализации health check'ов


class DatabaseHealthCheck(HealthCheck):
    """Проверка здоровья базы данных"""

    def __init__(self, database, **kwargs):
        super().__init__("database", **kwargs)
        self.database = database

    async def check(self) -> HealthCheckResult:
        try:
            # Проверяем подключение к БД
            if hasattr(self.database, "pool") and self.database.pool:
                async with self.database.pool.acquire() as conn:
                    # Выполняем простой запрос
                    cursor = await conn.execute("SELECT 1")
                    result = await cursor.fetchone()
                    await cursor.close()

                    if result and result[0] == 1:
                        # Получаем статистику пула
                        pool_stats = self.database.pool.get_stats()

                        status = HealthStatus.HEALTHY
                        message = "Database connection healthy"

                        # Degraded только если проблемы с пулом соединений
                        if pool_stats["is_closed"]:
                            status = HealthStatus.UNHEALTHY
                            message = "Database pool is closed"
                        elif pool_stats["total_connections"] == 0:
                            status = HealthStatus.DEGRADED
                            message = "No database connections available"

                        return HealthCheckResult(status=status, message=message, details=pool_stats)
            else:
                # Fallback для старой версии БД
                await self.database.get_user(1)  # Тестовый запрос
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    message="Database connection healthy (legacy mode)",
                    details={"connection_type": "legacy"},
                )

            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, message="Database connection failed"
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, message=f"Database error: {str(e)}"
            )


class OpenAIHealthCheck(HealthCheck):
    """Проверка здоровья OpenAI сервиса"""

    def __init__(self, openai_service, **kwargs):
        super().__init__("openai", **kwargs)
        self.openai_service = openai_service

    async def check(self) -> HealthCheckResult:
        try:
            # Получаем статистику сервиса
            stats = await self.openai_service.get_stats()

            # Проверяем кеш если доступен
            cache_status = "not_available"
            if hasattr(self.openai_service, "cache") and self.openai_service.cache:
                cache_stats = await self.openai_service.cache.get_cache_stats()
                cache_status = "available"
                stats["cache"] = cache_stats

            # Определяем статус на основе статистики
            status = HealthStatus.HEALTHY
            message = "OpenAI service healthy"

            # Проверяем процент неудачных запросов
            if stats["total_requests"] > 0:
                error_rate = stats["failed_requests"] / stats["total_requests"]
                if error_rate > 0.5:  # Более 50% ошибок
                    status = HealthStatus.UNHEALTHY
                    message = f"Critical error rate: {error_rate:.2%}"
                elif error_rate > 0.1:  # Более 10% ошибок
                    status = HealthStatus.DEGRADED
                    message = f"High error rate: {error_rate:.2%}"

            return HealthCheckResult(
                status=status, message=message, details={**stats, "cache_status": cache_status}
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, message=f"OpenAI service error: {str(e)}"
            )

class SessionStoreHealthCheck(HealthCheck):
    """Проверка здоровья session store"""

    def __init__(self, session_store, **kwargs):
        super().__init__("session_store", critical=False, **kwargs)
        self.session_store = session_store

    async def check(self) -> HealthCheckResult:
        try:
            if not self.session_store:
                return HealthCheckResult(
                    status=HealthStatus.UNKNOWN, message="Session store not initialized"
                )

            # Получаем статистику сессий
            sessions_count = len(getattr(self.session_store, "_sessions", {}))
            max_size = getattr(self.session_store, "_max_size", 0)

            # Определяем статус
            status = HealthStatus.HEALTHY
            message = f"Session store healthy ({sessions_count} sessions)"

            if max_size > 0:
                usage_percent = sessions_count / max_size
                if usage_percent >= 1.0:  # Session store overflow
                    status = HealthStatus.UNHEALTHY
                    message = "Session store overflow"
                elif usage_percent > 0.8:  # Session store nearly full
                    status = HealthStatus.DEGRADED
                    message = f"Session store nearly full ({usage_percent:.1%})"

            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "sessions_count": sessions_count,
                    "max_size": max_size,
                    "usage_percent": usage_percent if max_size > 0 else 0,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, message=f"Session store error: {str(e)}"
            )


class RateLimiterHealthCheck(HealthCheck):
    """Проверка здоровья rate limiter"""

    def __init__(self, rate_limiter, **kwargs):
        super().__init__("rate_limiter", **kwargs)
        self.rate_limiter = rate_limiter

    async def check(self) -> HealthCheckResult:
        try:
            if not self.rate_limiter:
                return HealthCheckResult(
                    status=HealthStatus.UNKNOWN, message="Rate limiter not initialized"
                )

            stats: dict[str, Any] | None = None
            if hasattr(self.rate_limiter, "get_stats"):
                stats_method = getattr(self.rate_limiter, "get_stats")
                if callable(stats_method):
                    maybe_stats = stats_method()
                    if inspect.isawaitable(maybe_stats):
                        stats = await maybe_stats
                    else:
                        stats = maybe_stats

            if stats is None:
                stats = {
                    "backend": "unknown",
                    "max_requests": getattr(self.rate_limiter, "max_requests", None),
                    "window_seconds": getattr(self.rate_limiter, "window_seconds", None),
                }

            status = HealthStatus.HEALTHY
            message = "Rate limiter healthy"

            if stats.get("backend") == "redis" and stats.get("redis_ok") is False:
                status = HealthStatus.UNHEALTHY
                message = "Redis backend not reachable"
            elif stats.get("limiter_saturated"):
                status = HealthStatus.DEGRADED
                message = "Limiter saturated for at least one user"

            return HealthCheckResult(
                status=status,
                message=message,
                details=stats,
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Rate limiter error: {str(e)}",
            )


class SystemResourcesHealthCheck(HealthCheck):
    """Проверка системных ресурсов"""

    def __init__(self, *, history_size: int = 10, **kwargs):
        super().__init__("system_resources", **kwargs)
        self._history_size = history_size
        self._metrics_history: deque[dict[str, float]] = deque(maxlen=history_size)
        try:  # pragma: no cover - prime psutil for non-blocking calls
            import psutil

            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    async def check(self) -> HealthCheckResult:
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            status = HealthStatus.HEALTHY
            issues = []

            if cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent}%")
                status = HealthStatus.DEGRADED

            if memory.percent > 85:
                issues.append(f"High memory usage: {memory.percent}%")
                status = HealthStatus.DEGRADED

            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent}%")
                status = HealthStatus.DEGRADED

            if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
                status = HealthStatus.UNHEALTHY

            message = "System resources healthy"
            if issues:
                message = f"Resource issues: {', '.join(issues)}"

            snapshot = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
            }

            self._metrics_history.append({"timestamp": time.time(), **snapshot})
            history_samples = list(self._metrics_history)
            history_summary: dict[str, dict[str, float]] = {}
            if history_samples:
                for metric in (
                    "cpu_percent",
                    "memory_percent",
                    "memory_available_gb",
                    "disk_percent",
                    "disk_free_gb",
                ):
                    values = [entry[metric] for entry in history_samples]
                    history_summary[metric] = {
                        "min": round(min(values), 2),
                        "max": round(max(values), 2),
                        "avg": round(sum(values) / len(values), 2),
                    }

            details = {
                **snapshot,
                "history": {
                    "window_size": self._history_size,
                    "samples": history_samples,
                    "summary": history_summary,
                },
            }

            return HealthCheckResult(
                status=status,
                message=message,
                details=details,
            )

        except ImportError:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN, message="psutil not available for system monitoring"
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, message=f"System resources error: {str(e)}"
            )


class HealthChecker:
    """Менеджер health check'ов"""

    def __init__(self, check_interval: float = 30.0):
        self.checks: dict[str, HealthCheck] = {}
        self.check_interval = check_interval
        self.last_full_check: float | None = None
        self._background_task: asyncio.Task | None = None
        self._running = False

    def register_check(self, health_check: HealthCheck) -> None:
        """Регистрация health check'а"""
        self.checks[health_check.name] = health_check
        logger.info(f"Registered health check '{health_check.name}'")

    async def check_all(self) -> dict[str, HealthCheckResult]:
        """Проверка всех компонентов"""
        results = {}

        # Выполняем все проверки параллельно
        tasks = {name: check.execute() for name, check in self.checks.items()}

        completed = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for (name, _), outcome in zip(tasks.items(), completed, strict=False):
            if isinstance(outcome, BaseException):
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                if isinstance(outcome, (KeyboardInterrupt, SystemExit)):
                    raise outcome
                logger.error("Health check '%s' failed during execution: %s", name, outcome)
                fallback = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check execution failed: {str(outcome)}",
                )
                check = self.checks.get(name)
                if check is not None:
                    check._update_counters(fallback.status)
                    check.last_result = fallback
                results[name] = fallback
            else:
                results[name] = outcome
        self.last_full_check = time.time()
        return results

    async def start_background_checks(self) -> None:
        """Запуск фоновых проверок"""
        if self._running:
            logger.warning("Background health checks already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._background_check_loop())
        logger.info(f"Started background health checks (interval: {self.check_interval}s)")

    async def stop_background_checks(self) -> None:
        """Остановка фоновых проверок"""
        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped background health checks")

    async def _background_check_loop(self) -> None:
        """Цикл фоновых проверок"""
        try:
            while self._running:
                try:
                    results = await self.check_all()

                    # Логгируем только проблемы
                    for name, result in results.items():
                        if result.status != HealthStatus.HEALTHY:
                            level = (
                                logging.WARNING
                                if result.status == HealthStatus.DEGRADED
                                else logging.ERROR
                            )
                            logger.log(
                                level,
                                f"Health check '{name}': {result.status.value} - {result.message}",
                            )

                except Exception as e:
                    logger.error(f"Background health check failed: {e}")

                if self._running:
                    await asyncio.sleep(self.check_interval)

        except asyncio.CancelledError:
            pass

    def get_stats(self) -> dict[str, Any]:
        """Статистика health checker'а"""
        return {
            "registered_checks": len(self.checks),
            "background_running": self._running,
            "check_interval": self.check_interval,
            "last_full_check": self.last_full_check,
            "checks": {name: check.get_stats() for name, check in self.checks.items()},
        }

    def get_all_stats(self) -> dict[str, Any]:
        """Alias для get_stats для совместимости с BackgroundTask"""
        return self.get_stats()
