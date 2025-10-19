"""
Система метрик и мониторинга с поддержкой Prometheus
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Enum,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Info = Enum = None
    CollectorRegistry = generate_latest = CONTENT_TYPE_LATEST = None

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Значение метрики с временной меткой"""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Сборщик метрик с fallback на простое логирование"""

    def __init__(self, enable_prometheus: bool = True, prometheus_port: int | None = None):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port

        # Prometheus registry
        self.registry: CollectorRegistry | None = None
        self._prometheus_server = None

        # Fallback storage для метрик
        self._fallback_metrics: dict[str, list[MetricValue]] = {}
        self._fallback_max_values = 1000  # Максимум значений на метрику

        # Prometheus метрики
        self._prometheus_metrics: dict[str, Any] = {}

        # Инициализация
        if self.enable_prometheus:
            self._init_prometheus()
        else:
            logger.warning("Prometheus не доступен, используем fallback метрики")

    def _init_prometheus(self) -> None:
        """Инициализация Prometheus"""
        try:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()

            if self.prometheus_port:
                self._start_prometheus_server()

            logger.info("Prometheus metrics initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus: {e}")
            self.enable_prometheus = False

    def _setup_prometheus_metrics(self) -> None:
        """Настройка стандартных метрик Prometheus"""
        if not self.enable_prometheus or not self.registry:
            return

        # Метрики Telegram бота
        self._prometheus_metrics.update(
            {
                # Счетчики запросов
                "telegram_messages_total": Counter(
                    "telegram_messages_total",
                    "Total number of Telegram messages processed",
                    ["user_id", "message_type", "status"],
                    registry=self.registry,
                ),
                "openai_requests_total": Counter(
                    "openai_requests_total",
                    "Total number of OpenAI API requests",
                    ["model", "status"],
                    registry=self.registry,
                ),
                "cache_operations_total": Counter(
                    "cache_operations_total",
                    "Total number of cache operations",
                    ["operation", "backend", "status"],
                    registry=self.registry,
                ),
                "database_operations_total": Counter(
                    "database_operations_total",
                    "Total number of database operations",
                    ["operation", "table", "status"],
                    registry=self.registry,
                ),
                "payment_transactions_total": Counter(
                    "payment_transactions_total",
                    "Total number of payment transactions",
                    ["provider", "currency", "status"],
                    registry=self.registry,
                ),
                # Security metrics
                "security_violations_total": Counter(
                    "security_violations_total",
                    "Total number of security violations detected",
                    ["violation_type", "severity", "source"],
                    registry=self.registry,
                ),
                "sql_injection_attempts_total": Counter(
                    "sql_injection_attempts_total",
                    "Total number of SQL injection attempts detected",
                    ["pattern_type", "source"],
                    registry=self.registry,
                ),
                "xss_attempts_total": Counter(
                    "xss_attempts_total",
                    "Total number of XSS attempts detected",
                    ["pattern_type", "source"],
                    registry=self.registry,
                ),
                # Гистограммы времени отклика
                "openai_request_duration_seconds": Histogram(
                    "openai_request_duration_seconds",
                    "OpenAI request duration in seconds",
                    ["model"],
                    registry=self.registry,
                    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
                ),
                "telegram_response_duration_seconds": Histogram(
                    "telegram_response_duration_seconds",
                    "Telegram response duration in seconds",
                    ["message_type"],
                    registry=self.registry,
                    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                ),
                "database_query_duration_seconds": Histogram(
                    "database_query_duration_seconds",
                    "Database query duration in seconds",
                    ["operation", "table"],
                    registry=self.registry,
                ),
                # Gauges для текущих состояний
                "active_user_sessions": Gauge(
                    "active_user_sessions", "Number of active user sessions", registry=self.registry
                ),
                "database_connections_active": Gauge(
                    "database_connections_active",
                    "Number of active database connections",
                    registry=self.registry,
                ),
                "cache_size_bytes": Gauge(
                    "cache_size_bytes",
                    "Current cache size in bytes",
                    ["backend"],
                    registry=self.registry,
                ),
                "openai_tokens_used_total": Counter(
                    "openai_tokens_used_total",
                    "Total OpenAI tokens used",
                    ["model", "token_type"],
                    registry=self.registry,
                ),
                # Информационные метрики
                "bot_info": Info(
                    "bot_info", "Bot version and configuration info", registry=self.registry
                ),
                # Состояние системы
                "system_status": Enum(
                    "system_status",
                    "Current system status",
                    states=["starting", "running", "degraded", "maintenance", "stopping"],
                    registry=self.registry,
                ),
            }
        )

    def _start_prometheus_server(self) -> None:
        """Запуск HTTP сервера для Prometheus метрик"""
        try:
            from prometheus_client import start_http_server

            start_http_server(self.prometheus_port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

    # Методы для записи метрик

    def inc_counter(
        self, metric_name: str, labels: dict[str, str] | None = None, value: float = 1.0
    ) -> None:
        """Увеличение счетчика"""
        if self.enable_prometheus and metric_name in self._prometheus_metrics:
            metric = self._prometheus_metrics[metric_name]
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
        else:
            self._record_fallback_metric(metric_name, value, labels)

    def observe_histogram(
        self, metric_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Запись значения в гистограмму"""
        if self.enable_prometheus and metric_name in self._prometheus_metrics:
            metric = self._prometheus_metrics[metric_name]
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
        else:
            self._record_fallback_metric(metric_name, value, labels)

    def set_gauge(
        self, metric_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Установка значения gauge"""
        if self.enable_prometheus and metric_name in self._prometheus_metrics:
            metric = self._prometheus_metrics[metric_name]
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        else:
            self._record_fallback_metric(metric_name, value, labels)

    def set_info(self, metric_name: str, info_dict: dict[str, str]) -> None:
        """Установка информационной метрики"""
        if self.enable_prometheus and metric_name in self._prometheus_metrics:
            metric = self._prometheus_metrics[metric_name]
            metric.info(info_dict)
        else:
            self._record_fallback_metric(metric_name, 1.0, info_dict)

    def set_enum(self, metric_name: str, state: str) -> None:
        """Установка enum метрики"""
        if self.enable_prometheus and metric_name in self._prometheus_metrics:
            metric = self._prometheus_metrics[metric_name]
            metric.state(state)
        else:
            self._record_fallback_metric(metric_name, 1.0, {"state": state})

    def _record_fallback_metric(
        self, metric_name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Запись метрики в fallback хранилище"""
        if metric_name not in self._fallback_metrics:
            self._fallback_metrics[metric_name] = []

        metric_value = MetricValue(value=value, labels=labels or {})

        self._fallback_metrics[metric_name].append(metric_value)

        # Ограничиваем размер
        if len(self._fallback_metrics[metric_name]) > self._fallback_max_values:
            self._fallback_metrics[metric_name] = self._fallback_metrics[metric_name][
                -self._fallback_max_values :
            ]

    # Контекстные менеджеры для измерения времени

    @asynccontextmanager
    async def time_openai_request(self, model: str = "unknown"):
        """Контекстный менеджер для измерения времени OpenAI запросов"""
        start_time = time.time()
        success = False
        try:
            yield
            success = True
        except Exception:
            raise
        finally:
            duration = time.time() - start_time
            status = "success" if success else "error"

            self.observe_histogram("openai_request_duration_seconds", duration, {"model": model})
            self.inc_counter("openai_requests_total", {"model": model, "status": status})

    @asynccontextmanager
    async def time_telegram_response(self, message_type: str = "text"):
        """Контекстный менеджер для измерения времени Telegram ответов"""
        start_time = time.time()
        success = False
        try:
            yield
            success = True
        except Exception:
            raise
        finally:
            duration = time.time() - start_time
            status = "success" if success else "error"

            self.observe_histogram(
                "telegram_response_duration_seconds", duration, {"message_type": message_type}
            )
            self.inc_counter(
                "telegram_responses_total",
                {"message_type": message_type, "status": status},
            )

    @asynccontextmanager
    async def time_database_query(self, operation: str, table: str = "unknown"):
        """Контекстный менеджер для измерения времени БД запросов"""
        start_time = time.time()
        success = False
        try:
            yield
            success = True
        except Exception:
            raise
        finally:
            duration = time.time() - start_time
            status = "success" if success else "error"

            self.observe_histogram(
                "database_query_duration_seconds",
                duration,
                {"operation": operation, "table": table},
            )
            self.inc_counter(
                "database_operations_total",
                {"operation": operation, "table": table, "status": status},
            )

    # Специфичные методы для различных компонентов

    def record_user_message(self, user_id: str, message_type: str, status: str) -> None:
        """Запись пользовательского сообщения"""
        self.inc_counter(
            "telegram_messages_total",
            {"user_id": user_id, "message_type": message_type, "status": status},
        )

    def record_cache_operation(self, operation: str, backend: str, status: str) -> None:
        """Запись операции кеша"""
        self.inc_counter(
            "cache_operations_total", {"operation": operation, "backend": backend, "status": status}
        )

    def record_payment(self, provider: str, currency: str, status: str) -> None:
        """Запись платежной транзакции"""
        self.inc_counter(
            "payment_transactions_total",
            {"provider": provider, "currency": currency, "status": status},
        )

    def record_security_violation(
        self, violation_type: str, severity: str, source: str = "unknown"
    ) -> None:
        """Запись нарушения безопасности"""
        self.inc_counter(
            "security_violations_total",
            {"violation_type": violation_type, "severity": severity, "source": source},
        )

    def record_sql_injection_attempt(self, pattern_type: str, source: str = "user_input") -> None:
        """Запись попытки SQL injection"""
        self.inc_counter(
            "sql_injection_attempts_total", {"pattern_type": pattern_type, "source": source}
        )

    def record_xss_attempt(self, pattern_type: str, source: str = "user_input") -> None:
        """Запись попытки XSS атаки"""
        self.inc_counter("xss_attempts_total", {"pattern_type": pattern_type, "source": source})

    def record_openai_tokens(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Запись использованных токенов OpenAI"""
        self.inc_counter(
            "openai_tokens_used_total", {"model": model, "token_type": "prompt"}, prompt_tokens
        )
        self.inc_counter(
            "openai_tokens_used_total",
            {"model": model, "token_type": "completion"},
            completion_tokens,
        )

    def update_system_gauges(self, **gauges: float) -> None:
        """Обновление системных gauge метрик"""
        for gauge_name, value in gauges.items():
            if gauge_name in self._prometheus_metrics:
                self.set_gauge(gauge_name, value)

    # Получение данных

    def get_prometheus_data(self) -> str | None:
        """Получение данных в формате Prometheus"""
        if self.enable_prometheus and self.registry:
            try:
                return generate_latest(self.registry)
            except Exception as e:
                logger.error(f"Failed to generate Prometheus data: {e}")
        return None

    def get_fallback_metrics(self) -> dict[str, Any]:
        """Получение fallback метрик"""
        result = {}
        for metric_name, values in self._fallback_metrics.items():
            if not values:
                continue

            # Базовая статистика
            numeric_values = [v.value for v in values]
            result[metric_name] = {
                "count": len(values),
                "latest_value": values[-1].value,
                "latest_timestamp": values[-1].timestamp,
                "total": sum(numeric_values),
                "average": sum(numeric_values) / len(numeric_values) if numeric_values else 0,
                "min": min(numeric_values) if numeric_values else 0,
                "max": max(numeric_values) if numeric_values else 0,
            }

        return result

    def get_stats(self) -> dict[str, Any]:
        """Общая статистика системы метрик"""
        stats = {
            "prometheus_enabled": self.enable_prometheus,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "prometheus_port": self.prometheus_port,
            "fallback_metrics_count": len(self._fallback_metrics),
        }

        if not self.enable_prometheus:
            fallback_metrics = self.get_fallback_metrics()
            if fallback_metrics is not None:
                stats["fallback_metrics"] = fallback_metrics

        return stats


# Глобальный экземпляр collector'а
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector | None:
    """Получение глобального collector'а метрик"""
    return _metrics_collector


def init_metrics(
    enable_prometheus: bool = True, prometheus_port: int | None = 8000
) -> MetricsCollector:
    """Инициализация системы метрик"""
    global _metrics_collector

    if _metrics_collector is not None:
        logger.warning("Metrics collector already initialized")
        return _metrics_collector

    _metrics_collector = MetricsCollector(
        enable_prometheus=enable_prometheus, prometheus_port=prometheus_port
    )

    # Устанавливаем базовую информацию о боте
    if _metrics_collector.enable_prometheus:
        _metrics_collector.set_info(
            "bot_info",
            {"version": "1.0.0", "name": "ai-ivan", "description": "Legal AI Assistant Bot"},
        )
        _metrics_collector.set_enum("system_status", "starting")

    logger.info("Metrics system initialized")
    return _metrics_collector


def set_system_status(status: str) -> None:
    """Установка статуса системы"""
    if _metrics_collector:
        _metrics_collector.set_enum("system_status", status)


# Декораторы для автоматического измерения метрик


def track_telegram_message(message_type: str = "text"):
    """Декоратор для отслеживания Telegram сообщений"""

    def decorator(func):
        async def wrapper(message, *args, **kwargs):
            user_id = str(getattr(message.from_user, "id", "unknown"))
            success = False

            try:
                if _metrics_collector:
                    async with _metrics_collector.time_telegram_response(message_type):
                        result = await func(message, *args, **kwargs)
                        success = True
                        return result
                else:
                    return await func(message, *args, **kwargs)
            except Exception:
                raise
            finally:
                if _metrics_collector:
                    status = "success" if success else "error"
                    _metrics_collector.record_user_message(user_id, message_type, status)

        return wrapper

    return decorator


def track_openai_request(model: str = "unknown"):
    """Декоратор для отслеживания OpenAI запросов"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            if _metrics_collector:
                async with _metrics_collector.time_openai_request(model):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

        return wrapper

    return decorator
