"""
Система health checks для мониторинга состояния всех компонентов системы
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
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

            if result.status != HealthStatus.HEALTHY:
                self.failed_checks += 1

            self.last_result = result
            return result

        except TimeoutError:
            self.failed_checks += 1
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timeout after {self.timeout}s",
                response_time=time.time() - start_time,
            )
            self.last_result = result
            return result

        except Exception as e:
            self.failed_checks += 1
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time=time.time() - start_time,
            )
            self.last_result = result
            return result

    def get_stats(self) -> dict[str, Any]:
        """Статистика health check'а"""
        success_rate = 0.0
        if self.total_checks > 0:
            success_rate = (self.total_checks - self.failed_checks) / self.total_checks

        return {
            "name": self.name,
            "critical": self.critical,
            "tags": self.tags,
            "total_checks": self.total_checks,
            "failed_checks": self.failed_checks,
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
                user = await self.database.get_user(1)  # Тестовый запрос
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
                if error_rate > 0.1:  # Более 10% ошибок
                    status = HealthStatus.DEGRADED
                    message = f"High error rate: {error_rate:.2%}"
                elif error_rate > 0.5:  # Более 50% ошибок
                    status = HealthStatus.UNHEALTHY
                    message = f"Critical error rate: {error_rate:.2%}"

            return HealthCheckResult(
                status=status, message=message, details={**stats, "cache_status": cache_status}
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, message=f"OpenAI service error: {str(e)}"
            )


class RedisHealthCheck(HealthCheck):
    """Проверка здоровья Redis"""

    def __init__(self, redis_client, **kwargs):
        super().__init__("redis", critical=False, **kwargs)  # Redis не критичен
        self.redis_client = redis_client

    async def check(self) -> HealthCheckResult:
        try:
            if not self.redis_client:
                return HealthCheckResult(
                    status=HealthStatus.UNKNOWN, message="Redis client not initialized"
                )

            # Ping Redis
            await self.redis_client.ping()

            # Получаем базовую информацию
            info = await self.redis_client.info()

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Redis connection healthy",
                details={
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY, message=f"Redis error: {str(e)}"
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
                if usage_percent > 0.8:  # Более 80% заполненности
                    status = HealthStatus.DEGRADED
                    message = f"Session store nearly full ({usage_percent:.1%})"
                elif usage_percent >= 1.0:  # Переполнение
                    status = HealthStatus.UNHEALTHY
                    message = "Session store overflow"

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
                    stats = await stats_method()

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

    def __init__(self, **kwargs):
        super().__init__("system_resources", **kwargs)

    async def check(self) -> HealthCheckResult:
        try:
            import psutil

            # Получаем информацию о системе
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Определяем статус
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

            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "disk_percent": disk.percent,
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                },
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

    async def check_component(self, component_name: str) -> HealthCheckResult | None:
        """Проверка конкретного компонента"""
        if component_name not in self.checks:
            return None

        return await self.checks[component_name].execute()

    async def check_all(self) -> dict[str, HealthCheckResult]:
        """Проверка всех компонентов"""
        results = {}

        # Выполняем все проверки параллельно
        tasks = {name: check.execute() for name, check in self.checks.items()}

        completed = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for (name, _), result in zip(tasks.items(), completed, strict=False):
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check execution failed: {str(result)}",
                )
            else:
                results[name] = result

        self.last_full_check = time.time()
        return results

    async def get_system_status(self) -> dict[str, Any]:
        """Получение общего статуса системы"""
        results = await self.check_all()

        # Определяем общий статус
        critical_checks = {
            name: result for name, result in results.items() if self.checks[name].critical
        }

        overall_status = HealthStatus.HEALTHY
        unhealthy_critical = []
        degraded_critical = []

        for name, result in critical_checks.items():
            if result.status == HealthStatus.UNHEALTHY:
                unhealthy_critical.append(name)
            elif result.status == HealthStatus.DEGRADED:
                degraded_critical.append(name)

        if unhealthy_critical:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_critical:
            overall_status = HealthStatus.DEGRADED

        # Статистика
        total_checks = len(results)
        healthy_count = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)

        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "summary": {
                "total_checks": total_checks,
                "healthy": healthy_count,
                "degraded": len(degraded_critical)
                + sum(
                    1
                    for r in results.values()
                    if r.status == HealthStatus.DEGRADED and r not in critical_checks.values()
                ),
                "unhealthy": len(unhealthy_critical)
                + sum(
                    1
                    for r in results.values()
                    if r.status == HealthStatus.UNHEALTHY and r not in critical_checks.values()
                ),
                "unknown": sum(1 for r in results.values() if r.status == HealthStatus.UNKNOWN),
            },
            "critical_issues": unhealthy_critical,
            "degraded_services": degraded_critical,
            "checks": {name: result.to_dict() for name, result in results.items()},
        }

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

async def check_health(*, raise_on_degraded: bool = True) -> dict[str, Any]:
    """Run a single-pass health check suitable for Docker health probes."""
    from src.core.app_context import get_settings, set_settings

    from src.core.bootstrap import build_runtime
    from src.core.settings import AppSettings
    from src.core.db_advanced import DatabaseAdvanced
    from src.core.openai_service import OpenAIService
    from src.core.session_store import SessionStore
    from src.telegram_legal_bot.ratelimit import RateLimiter

    health_logger = logging.getLogger("ai-ivan.healthcheck")

    settings = get_settings()
    set_settings(settings)
    runtime, container = build_runtime(settings, logger=health_logger)

    db = None
    rate_limiter = None
    session_store = None
    openai_service = None

    try:
        db = runtime.db or container.get(DatabaseAdvanced)
        runtime.db = db
        rate_limiter = runtime.rate_limiter or container.get(RateLimiter)
        runtime.rate_limiter = rate_limiter
        session_store = runtime.session_store or container.get(SessionStore)
        runtime.session_store = session_store
        openai_service = runtime.openai_service or container.get(OpenAIService)
        runtime.openai_service = openai_service

        await container.init_async_services()

        checker = HealthChecker(check_interval=settings.health_check_interval)
        checker.register_check(DatabaseHealthCheck(db))
        checker.register_check(OpenAIHealthCheck(openai_service))
        checker.register_check(SessionStoreHealthCheck(session_store))
        checker.register_check(RateLimiterHealthCheck(rate_limiter))
        if settings.enable_system_monitoring:
            checker.register_check(SystemResourcesHealthCheck())

        results = await checker.check_all()

        status = HealthStatus.HEALTHY
        degraded: list[str] = []
        unhealthy: list[str] = []

        for name, result in results.items():
            if result.status == HealthStatus.UNHEALTHY:
                unhealthy.append(f"{name}: {result.message}")
                status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED:
                degraded.append(f"{name}: {result.message}")
                if status != HealthStatus.UNHEALTHY:
                    status = HealthStatus.DEGRADED

        if unhealthy:
            health_logger.error("Health check failed: %s", "; ".join(unhealthy))
        if degraded and status == HealthStatus.DEGRADED:
            health_logger.warning("Health check degraded: %s", "; ".join(degraded))

        if status == HealthStatus.UNHEALTHY or (raise_on_degraded and status == HealthStatus.DEGRADED):
            details = [
                f"{name}: {result.message}"
                for name, result in results.items()
                if result.status != HealthStatus.HEALTHY
            ]
            raise RuntimeError("Health check failed: " + "; ".join(details))

        return {"status": status.value, "results": {name: result.to_dict() for name, result in results.items()}}

    finally:
        if container is not None:
            with suppress(Exception):
                await container.cleanup()
