from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar, get_type_hints

from src.core.settings import AppSettings
from src.core.app_context import set_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DIContainer:
    """Minimalistic dependency injection container."""

    def __init__(self) -> None:
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[..., Any]] = {}
        self._singletons: Dict[Type, Any] = {}
        self._config: Dict[str, Any] = {}

    def register_singleton(self, interface: Type[T], implementation: Type[T] | T) -> None:
        if isinstance(implementation, type):
            self._factories[interface] = implementation
        else:
            self._singletons[interface] = implementation
        logger.debug("Registered singleton: %s", interface.__name__)

    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        self._services[interface] = implementation
        logger.debug("Registered transient: %s", interface.__name__)

    def register_factory(self, interface: Type[T], factory: Callable[..., T]) -> None:
        self._factories[interface] = factory
        logger.debug("Registered factory: %s", interface.__name__)

    def register_config(self, key: str, value: Any) -> None:
        self._config[key] = value
        logger.debug("Registered config value: %s", key)

    def get(self, interface: Type[T]) -> T:
        if interface in self._singletons:
            return self._singletons[interface]

        if interface in self._factories:
            factory = self._factories[interface]
            instance = self._create_instance(factory)
            if interface not in self._services:
                self._singletons[interface] = instance
            return instance

        if interface in self._services:
            implementation = self._services[interface]
            return self._create_instance(implementation)

        return self._create_instance(interface)

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def _create_instance(self, cls_or_factory: Type[T] | Callable[..., T]) -> T:
        if not inspect.isclass(cls_or_factory) and callable(cls_or_factory):
            return self._call_with_dependencies(cls_or_factory)
        return self._call_with_dependencies(cls_or_factory)

    def _call_with_dependencies(self, func: Callable[..., T]) -> T:
        if inspect.isclass(func):
            signature = inspect.signature(func.__init__)
            type_hints = get_type_hints(func.__init__)
        else:
            signature = inspect.signature(func)
            type_hints = get_type_hints(func)

        kwargs: Dict[str, Any] = {}
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue

            hinted_type = type_hints.get(name, parameter.annotation)
            if hinted_type == inspect.Parameter.empty or hasattr(hinted_type, "__origin__"):
                continue

            try:
                dependency = self.get(hinted_type)
                kwargs[name] = dependency
            except Exception as exc:
                if parameter.default is inspect.Parameter.empty:
                    logger.warning("Cannot resolve dependency %s for %s: %s", hinted_type, func, exc)
                    config_value = self.get_config(name)
                    if config_value is not None:
                        kwargs[name] = config_value

        return func(**kwargs)  # type: ignore[arg-type]

    async def init_async_services(self) -> None:
        for service in self._singletons.values():
            init_method = getattr(service, "init", None)
            if callable(init_method):
                if inspect.iscoroutinefunction(init_method):
                    await init_method()
                else:
                    init_method()

    async def cleanup(self) -> None:
        for service in self._singletons.values():
            close_method = getattr(service, "close", None)
            cleanup_method = getattr(service, "cleanup", None)
            target = close_method or cleanup_method
            if callable(target):
                if inspect.iscoroutinefunction(target):
                    await target()
                else:
                    target()


def create_container(settings: AppSettings) -> DIContainer:
    container = DIContainer()
    set_settings(settings)
    container.register_singleton(AppSettings, settings)

    from src.core.db_advanced import DatabaseAdvanced
    from src.core.access import AccessService
    from src.core.openai_service import OpenAIService
    from src.core.audio_service import AudioService
    from src.core.session_store import SessionStore
    from src.core.payments import CryptoPayProvider
    from src.core.rag.judicial_rag import JudicialPracticeRAG
    from src.bot.ratelimit import RateLimiter

    container.register_factory(
        DatabaseAdvanced,
        lambda: DatabaseAdvanced(
            settings.db_path,
            max_connections=settings.db_max_connections,
            enable_metrics=True,
        ),
    )

    container.register_factory(
        AccessService,
        lambda: AccessService(
            db=container.get(DatabaseAdvanced),
            trial_limit=settings.trial_requests,
            admin_ids=set(settings.admin_ids),
        ),
    )

    container.register_factory(OpenAIService, lambda: OpenAIService())

    container.register_factory(
        AudioService,
        lambda: AudioService(
            stt_model=settings.voice_stt_model,
            tts_model=settings.voice_tts_model,
            tts_voice=settings.voice_tts_voice,
            tts_format=settings.voice_tts_format,
            max_duration_seconds=settings.voice_max_duration_seconds,
            tts_voice_male=settings.voice_tts_voice_male,
            tts_chunk_char_limit=settings.voice_tts_chunk_char_limit,
            tts_speed=settings.voice_tts_speed,
            tts_style=settings.voice_tts_style,
            tts_sample_rate=settings.voice_tts_sample_rate,
            tts_backend=settings.voice_tts_backend,
        ),
    )

    container.register_factory(
        RateLimiter,
        lambda: RateLimiter(
            redis_url=settings.redis_url,
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        ),
    )

    container.register_factory(
        SessionStore,
        lambda: SessionStore(
            max_size=settings.user_sessions_max,
            ttl_seconds=settings.user_session_ttl_seconds,
        ),
    )

    container.register_factory(
        CryptoPayProvider,
        lambda: CryptoPayProvider(asset=settings.crypto_asset, settings=settings),
    )

    container.register_factory(
        JudicialPracticeRAG,
        lambda: JudicialPracticeRAG(settings=settings),
    )

    container.register_config("subscription_price_rub", settings.subscription_price_rub)
    container.register_config("subscription_price_xtr", settings.subscription_price_xtr)

    logger.info("DI container configured successfully")
    return container


_container: Optional[tuple[AppSettings, DIContainer]] = None


def get_container(settings: AppSettings | None = None) -> DIContainer:
    """Return cached container instance creating it when required."""
    global _container
    if _container is None:
        if settings is None:
            raise RuntimeError("AppSettings required for initial container build")
        _container = (settings, create_container(settings))
        return _container[1]

    cached_settings, cached_container = _container
    if settings is not None and settings is not cached_settings:
        _container = (settings, create_container(settings))
        return _container[1]

    return cached_container


def reset_container() -> None:
    """Reset cached container (used by tests)."""
    global _container
    _container = None




