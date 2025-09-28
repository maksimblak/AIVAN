"""
Простой Dependency Injection контейнер для AIVAN
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable, Dict, Optional, Type, TypeVar, get_type_hints

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DIContainer:
    """Простой DI контейнер с автоматическим разрешением зависимостей"""

    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._config: Dict[str, Any] = {}

    def register_singleton(self, interface: Type[T], implementation: Type[T] | T) -> None:
        """Регистрация singleton сервиса"""
        if isinstance(implementation, type):
            self._factories[interface] = implementation
        else:
            self._singletons[interface] = implementation
        logger.debug(f"Registered singleton: {interface.__name__}")

    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Регистрация transient сервиса (новый экземпляр каждый раз)"""
        self._services[interface] = implementation
        logger.debug(f"Registered transient: {interface.__name__}")

    def register_factory(self, interface: Type[T], factory: Callable[..., T]) -> None:
        """Регистрация factory функции"""
        self._factories[interface] = factory
        logger.debug(f"Registered factory: {interface.__name__}")

    def register_config(self, key: str, value: Any) -> None:
        """Регистрация конфигурационного значения"""
        self._config[key] = value
        logger.debug(f"Registered config: {key}")

    def get(self, interface: Type[T]) -> T:
        """Получение экземпляра сервиса"""
        # Проверяем сначала singleton кеш
        if interface in self._singletons:
            return self._singletons[interface]

        # Проверяем factory или singleton фабрику
        if interface in self._factories:
            factory = self._factories[interface]
            instance = self._create_instance(factory)
            # Если это была фабрика для singleton, кешируем
            if interface not in self._services:
                self._singletons[interface] = instance
            return instance

        # Проверяем transient сервисы
        if interface in self._services:
            implementation = self._services[interface]
            return self._create_instance(implementation)

        # Пытаемся создать автоматически
        return self._create_instance(interface)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Получение конфигурационного значения"""
        return self._config.get(key, default)

    def _create_instance(self, cls_or_factory: Type[T] | Callable) -> T:
        """Создание экземпляра с автоматическим разрешением зависимостей"""
        if not inspect.isclass(cls_or_factory) and callable(cls_or_factory):
            # Это factory функция
            return self._call_with_dependencies(cls_or_factory)

        # Это класс
        constructor = cls_or_factory.__init__
        return self._call_with_dependencies(cls_or_factory)

    def _call_with_dependencies(self, func: Callable) -> Any:
        """Вызов функции/конструктора с автоматическим разрешением зависимостей"""
        if inspect.isclass(func):
            # Это конструктор класса
            signature = inspect.signature(func.__init__)
            type_hints = get_type_hints(func.__init__)
            args = []
            kwargs = {}

            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue

                param_type = type_hints.get(param_name, param.annotation)

                if param_type == inspect.Parameter.empty:
                    # Нет аннотации типа, пропускаем
                    continue

                if hasattr(param_type, "__origin__"):
                    # Это generic тип, пропускаем пока
                    continue

                try:
                    dependency = self.get(param_type)
                    kwargs[param_name] = dependency
                except Exception as e:
                    # Если не можем разрешить зависимость и параметр обязательный
                    if param.default == inspect.Parameter.empty:
                        logger.warning(f"Cannot resolve dependency {param_type} for {func}: {e}")
                        # Пытаемся найти в конфиге
                        config_value = self.get_config(param_name)
                        if config_value is not None:
                            kwargs[param_name] = config_value
                    else:
                        # Есть значение по умолчанию, используем его
                        pass

            return func(**kwargs)
        else:
            # Это обычная функция
            signature = inspect.signature(func)
            type_hints = get_type_hints(func)
            kwargs = {}

            for param_name, param in signature.parameters.items():
                param_type = type_hints.get(param_name, param.annotation)

                if param_type == inspect.Parameter.empty:
                    continue

                if hasattr(param_type, "__origin__"):
                    continue

                try:
                    dependency = self.get(param_type)
                    kwargs[param_name] = dependency
                except Exception as e:
                    if param.default == inspect.Parameter.empty:
                        logger.warning(f"Cannot resolve dependency {param_type} for {func}: {e}")
                        config_value = self.get_config(param_name)
                        if config_value is not None:
                            kwargs[param_name] = config_value

            return func(**kwargs)

    async def init_async_services(self) -> None:
        """Инициализация асинхронных сервисов"""
        for service in self._singletons.values():
            if hasattr(service, "init") and callable(service.init):
                if inspect.iscoroutinefunction(service.init):
                    await service.init()
                else:
                    service.init()

    async def cleanup(self) -> None:
        """Очистка ресурсов"""
        for service in self._singletons.values():
            if hasattr(service, "close") and callable(service.close):
                if inspect.iscoroutinefunction(service.close):
                    await service.close()
                else:
                    service.close()
            elif hasattr(service, "cleanup") and callable(service.cleanup):
                if inspect.iscoroutinefunction(service.cleanup):
                    await service.cleanup()
                else:
                    service.cleanup()


def create_container() -> DIContainer:
    """Создание и настройка DI контейнера для AIVAN"""
    container = DIContainer()

    # Регистрация конфигурации из переменных окружения
    container.register_config("db_path", os.getenv("DB_PATH", "data/bot.sqlite3"))
    container.register_config("db_max_connections", int(os.getenv("DB_MAX_CONNECTIONS", "5")))
    container.register_config("openai_api_key", os.getenv("OPENAI_API_KEY"))
    container.register_config("telegram_bot_token", os.getenv("TELEGRAM_BOT_TOKEN"))
    container.register_config("trial_requests", int(os.getenv("TRIAL_REQUESTS", "10")))
    container.register_config("admin_ids", set(map(int, os.getenv("ADMIN_IDS", "").split(","))) if os.getenv("ADMIN_IDS") else set())

    # Регистрация основных сервисов
    from .db_advanced import DatabaseAdvanced
    from .access import AccessService
    from .openai_service import OpenAIService
    from .audio_service import AudioService
    from ..telegram_legal_bot.ratelimit import RateLimiter

    # Database как singleton
    container.register_factory(
        DatabaseAdvanced,
        lambda: DatabaseAdvanced(
            container.get_config("db_path"),
            max_connections=container.get_config("db_max_connections"),
            enable_metrics=True
        )
    )

    # AccessService как singleton
    container.register_factory(
        AccessService,
        lambda db: AccessService(
            db=db,
            trial_limit=container.get_config("trial_requests"),
            admin_ids=container.get_config("admin_ids")
        )
    )

    # OpenAIService как singleton
    container.register_factory(
        OpenAIService,
        lambda: OpenAIService(api_key=container.get_config("openai_api_key"))
    )

    # AudioService как singleton
    container.register_singleton(AudioService, AudioService)

    # RateLimiter как singleton
    container.register_singleton(RateLimiter, RateLimiter)

    logger.info("DI Container configured successfully")
    return container


# Глобальный экземпляр контейнера
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Получение глобального экземпляра DI контейнера"""
    global _container
    if _container is None:
        _container = create_container()
    return _container


def reset_container() -> None:
    """Сброс глобального контейнера (для тестов)"""
    global _container
    _container = None