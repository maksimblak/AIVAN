"""
Централизованная обработка исключений с детальной классификацией ошибок
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# Типы ошибок
class ErrorType(Enum):
    """Типы ошибок в системе"""

    VALIDATION = "validation"
    DATABASE = "database"
    OPENAI_API = "openai_api"
    TELEGRAM_API = "telegram_api"
    NETWORK = "network"
    PAYMENT = "payment"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Уровни критичности ошибок"""

    LOW = "low"  # Логгируем, продолжаем работу
    MEDIUM = "medium"  # Логгируем, возможно уведомляем админа
    HIGH = "high"  # Логгируем, уведомляем админа, пытаемся восстановить
    CRITICAL = "critical"  # Логгируем, уведомляем админа, возможно останавливаем сервис


@dataclass
class ErrorContext:
    """Контекст ошибки для детального анализа"""

    user_id: int | None = None
    chat_id: int | None = None
    message_id: int | None = None
    command: str | None = None
    function_name: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseCustomException(Exception):
    """Базовое исключение с расширенным функционалом"""

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: ErrorContext | None = None,
        recoverable: bool = True,
        user_message: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.user_message = user_message or "Произошла ошибка. Попробуйте позже."
        self.timestamp = datetime.now()


# Специализированные исключения


class ValidationException(BaseCustomException):
    """Ошибки валидации входных данных"""

    def __init__(self, message: str, context: ErrorContext | None = None):
        super().__init__(
            message,
            ErrorType.VALIDATION,
            ErrorSeverity.LOW,
            context,
            recoverable=True,
            user_message="Проверьте правильность введенных данных",
        )


class DatabaseException(BaseCustomException):
    """Ошибки базы данных"""

    def __init__(
        self, message: str, context: ErrorContext | None = None, is_connection_error: bool = False
    ):
        severity = ErrorSeverity.HIGH if is_connection_error else ErrorSeverity.MEDIUM
        super().__init__(
            message,
            ErrorType.DATABASE,
            severity,
            context,
            recoverable=not is_connection_error,
            user_message="Временные проблемы с базой данных. Попробуйте позже.",
        )


class OpenAIException(BaseCustomException):
    """Ошибки OpenAI API"""

    def __init__(
        self, message: str, context: ErrorContext | None = None, is_quota_error: bool = False
    ):
        severity = ErrorSeverity.HIGH if is_quota_error else ErrorSeverity.MEDIUM
        user_msg = (
            "Превышен лимит запросов к ИИ. Попробуйте позже."
            if is_quota_error
            else "Проблемы с ИИ-сервисом. Попробуйте позже."
        )
        super().__init__(
            message,
            ErrorType.OPENAI_API,
            severity,
            context,
            recoverable=True,
            user_message=user_msg,
        )


class TelegramException(BaseCustomException):
    """Ошибки Telegram API"""

    def __init__(self, message: str, context: ErrorContext | None = None):
        super().__init__(
            message,
            ErrorType.TELEGRAM_API,
            ErrorSeverity.MEDIUM,
            context,
            recoverable=True,
            user_message="Проблемы с Telegram API. Сообщение может быть доставлено с задержкой.",
        )


class NetworkException(BaseCustomException):
    """Сетевые ошибки"""

    def __init__(self, message: str, context: ErrorContext | None = None):
        super().__init__(
            message,
            ErrorType.NETWORK,
            ErrorSeverity.MEDIUM,
            context,
            recoverable=True,
            user_message="Проблемы с сетевым соединением. Попробуйте позже.",
        )


class PaymentException(BaseCustomException):
    """Ошибки платежей"""

    def __init__(
        self, message: str, context: ErrorContext | None = None, is_user_error: bool = False
    ):
        severity = ErrorSeverity.LOW if is_user_error else ErrorSeverity.HIGH
        super().__init__(
            message,
            ErrorType.PAYMENT,
            severity,
            context,
            recoverable=True,
            user_message="Проблемы с обработкой платежа. Обратитесь в поддержку.",
        )


class AuthException(BaseCustomException):
    """Ошибки авторизации"""

    def __init__(self, message: str, context: ErrorContext | None = None):
        super().__init__(
            message,
            ErrorType.AUTH,
            ErrorSeverity.MEDIUM,
            context,
            recoverable=True,
            user_message="Проблемы с доступом. Проверьте подписку или обратитесь к администратору.",
        )


class RateLimitException(BaseCustomException):
    """Превышение лимитов запросов"""

    def __init__(self, message: str, context: ErrorContext | None = None):
        super().__init__(
            message,
            ErrorType.RATE_LIMIT,
            ErrorSeverity.LOW,
            context,
            recoverable=True,
            user_message="Слишком много запросов. Попробуйте через несколько минут.",
        )


class SystemException(BaseCustomException):
    """Системные ошибки"""

    def __init__(self, message: str, context: ErrorContext | None = None):
        super().__init__(
            message,
            ErrorType.SYSTEM,
            ErrorSeverity.CRITICAL,
            context,
            recoverable=False,
            user_message="Системная ошибка. Администраторы уведомлены.",
        )


class ErrorHandler:
    """Централизованный обработчик ошибок"""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: dict[ErrorType, int] = {}
        self.recovery_handlers: dict[ErrorType, Callable] = {}

    def register_recovery_handler(self, error_type: ErrorType, handler: Callable):
        """Регистрация обработчика восстановления для типа ошибки"""
        self.recovery_handlers[error_type] = handler

    async def handle_exception(
        self, exception: Exception, context: ErrorContext | None = None
    ) -> BaseCustomException:
        """Главный обработчик исключений"""

        # Если это уже наше кастомное исключение
        if isinstance(exception, BaseCustomException):
            custom_exc = exception
        else:
            # Преобразуем стандартное исключение в кастомное
            custom_exc = self._classify_exception(exception, context)

        # Обновляем статистику
        self.error_counts[custom_exc.error_type] = (
            self.error_counts.get(custom_exc.error_type, 0) + 1
        )

        # Логгируем ошибку
        await self._log_error(custom_exc, exception)

        # Пытаемся восстановиться, если возможно
        if custom_exc.recoverable and custom_exc.error_type in self.recovery_handlers:
            try:
                await self.recovery_handlers[custom_exc.error_type](custom_exc)
            except Exception as recovery_error:
                self.logger.error(
                    f"Failed to recover from {custom_exc.error_type}: {recovery_error}"
                )

        return custom_exc

    def _classify_exception(
        self, exception: Exception, context: ErrorContext | None = None
    ) -> BaseCustomException:
        """Классификация стандартных исключений"""

        exc_type = type(exception).__name__
        exc_message = str(exception)

        # Классификация по типу исключения
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return NetworkException(f"Network error: {exc_message}", context)

        elif isinstance(exception, asyncio.TimeoutError):
            return NetworkException(f"Timeout error: {exc_message}", context)

        elif "rate limit" in exc_message.lower() or "too many requests" in exc_message.lower():
            return RateLimitException(f"Rate limit exceeded: {exc_message}", context)

        elif "database" in exc_message.lower() or "sqlite" in exc_message.lower():
            is_conn_error = (
                "database is locked" in exc_message.lower() or "connection" in exc_message.lower()
            )
            return DatabaseException(f"Database error: {exc_message}", context, is_conn_error)

        elif "openai" in exc_message.lower() or "api" in exc_message.lower():
            is_quota = "quota" in exc_message.lower() or "billing" in exc_message.lower()
            return OpenAIException(f"OpenAI error: {exc_message}", context, is_quota)

        elif "telegram" in exc_message.lower() or "bot" in exc_message.lower():
            return TelegramException(f"Telegram error: {exc_message}", context)

        elif "payment" in exc_message.lower() or "invoice" in exc_message.lower():
            return PaymentException(f"Payment error: {exc_message}", context)

        elif isinstance(exception, PermissionError):
            return AuthException(f"Permission denied: {exc_message}", context)

        elif isinstance(exception, MemoryError):
            return SystemException(f"Memory error: {exc_message}", context)

        else:
            return BaseCustomException(
                f"Unclassified error ({exc_type}): {exc_message}",
                ErrorType.UNKNOWN,
                ErrorSeverity.MEDIUM,
                context,
            )

    async def _log_error(self, custom_exc: BaseCustomException, original_exc: Exception):
        """Детальное логгирование ошибки"""

        log_data = {
            "error_type": custom_exc.error_type.value,
            "severity": custom_exc.severity.value,
            "error_message": custom_exc.message,
            "user_message": custom_exc.user_message,
            "recoverable": custom_exc.recoverable,
            "timestamp": custom_exc.timestamp.isoformat(),
        }

        # Добавляем контекст если есть
        if custom_exc.context:
            log_data.update(
                {
                    "user_id": custom_exc.context.user_id,
                    "chat_id": custom_exc.context.chat_id,
                    "function": custom_exc.context.function_name,
                    "additional_data": custom_exc.context.additional_data,
                }
            )

        # Добавляем стек трейс для критических ошибок
        if custom_exc.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            log_data["traceback"] = traceback.format_exception(
                type(original_exc), original_exc, original_exc.__traceback__
            )

        # Выбираем уровень логгирования
        if custom_exc.severity == ErrorSeverity.LOW:
            self.logger.info("Error handled", extra=log_data)
        elif custom_exc.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Error handled", extra=log_data)
        elif custom_exc.severity == ErrorSeverity.HIGH:
            self.logger.error("Error handled", extra=log_data)
        else:  # CRITICAL
            self.logger.critical("Critical error handled", extra=log_data)

    def get_error_stats(self) -> dict[str, int]:
        """Получить статистику ошибок"""
        return {error_type.value: count for error_type, count in self.error_counts.items()}


# Декораторы для обработки ошибок


def handle_exceptions(error_handler: ErrorHandler, context_func: Callable | None = None):
    """Декоратор для автоматической обработки исключений"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = None
                if context_func:
                    try:
                        context = context_func(*args, **kwargs)
                    except:
                        pass

                custom_exc = await error_handler.handle_exception(e, context)
                raise custom_exc

        return wrapper

    return decorator


def safe_execute(error_handler: ErrorHandler, default_return=None):
    """Декоратор для безопасного выполнения с возвратом значения по умолчанию"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await error_handler.handle_exception(e)
                return default_return

        return wrapper

    return decorator
