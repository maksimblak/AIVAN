"""
Простая версия Telegram бота ИИ-Иван
Только /start и обработка вопросов, никаких кнопок и лишних команд
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Optional

from src.documents.document_manager import DocumentManager

if TYPE_CHECKING:
    from src.core.db_advanced import DatabaseAdvanced, TransactionStatus

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    BotCommand,
    BotCommandScopeChat,
    ErrorEvent,
    User,
)

from src.bot.retention_notifier import RetentionNotifier
from src.bot.ui_components import Emoji
from src.core.audio_service import AudioService
from src.core.access import AccessService
from src.core.db_advanced import DatabaseAdvanced, TransactionStatus
from src.core.exceptions import (
    ErrorContext,
    ErrorHandler,
    ErrorType,
)
from src.core.middlewares.error_middleware import ErrorHandlingMiddleware
from src.core.openai_service import OpenAIService
from src.core.payments import CryptoPayProvider
from src.core.admin_modules.admin_commands import setup_admin_commands
from src.core.session_store import SessionStore
from src.core.runtime import SubscriptionPlanPricing, WelcomeMedia
from src.bot.ratelimit import RateLimiter

from src.core.simple_bot.menus import register_menu_handlers, cmd_start
from src.core.simple_bot.documents import register_document_handlers
from src.core.simple_bot.feedback import register_feedback_handlers
from src.core.simple_bot.admin import register_admin_handlers
from src.core.simple_bot.retention import register_retention_handlers
from src.core.simple_bot import context as simple_context
from src.core.simple_bot.formatting import (
    _format_currency,
    _format_datetime,
    _format_hour_label,
    _format_number,
    _format_progress_extras,
    _format_risk_count,
    _format_response_time,
    _format_stat_row,
    _format_trend_value,
    _split_plain_text,
)
from src.core.simple_bot.questions import process_question, register_question_handlers
from src.core.simple_bot.stats import (
    FEATURE_LABELS,
    DAY_NAMES,
    build_stats_keyboard,
    describe_primary_summary,
    describe_secondary_summary,
    generate_user_stats_response,
    normalize_stats_period,
    peak_summary,
    progress_line,
    translate_payment_status,
    translate_plan_name,
)
from src.core.simple_bot.payments import get_plan_pricing, register_payment_handlers
from src.core.simple_bot.voice import register_voice_handlers

SECTION_DIVIDER = "<code>────────────────────</code>"


retention_notifier = None


# ============ КОНФИГУРАЦИЯ ============

logger = logging.getLogger("ai-ivan.simple")

set_runtime = simple_context.set_runtime
get_runtime = simple_context.get_runtime
settings = simple_context.settings
derived = simple_context.derived

WELCOME_MEDIA: WelcomeMedia | None = None
BOT_TOKEN = ""
BOT_USERNAME = ""
USE_ANIMATION = True
USE_STREAMING = True
SAFE_LIMIT = 3900
MAX_MESSAGE_LENGTH = 4000
DB_PATH = ""
TRIAL_REQUESTS = 0
SUB_DURATION_DAYS = 0
RUB_PROVIDER_TOKEN = ""
SUB_PRICE_RUB = 0
SUB_PRICE_RUB_KOPEKS = 0
STARS_PROVIDER_TOKEN = ""
SUB_PRICE_XTR = 0
DYNAMIC_PRICE_XTR = 0
SUBSCRIPTION_PLANS: tuple[SubscriptionPlanPricing, ...] = ()
SUBSCRIPTION_PLAN_MAP: dict[str, SubscriptionPlanPricing] = {}
DEFAULT_SUBSCRIPTION_PLAN: SubscriptionPlanPricing | None = None
ADMIN_IDS: set[int] = set()
USER_SESSIONS_MAX = 0
USER_SESSION_TTL_SECONDS = 0

db: DatabaseAdvanced | None = None
rate_limiter: RateLimiter | None = None
access_service: AccessService | None = None
openai_service: OpenAIService | None = None
audio_service: AudioService | None = None
session_store: SessionStore | None = None
crypto_provider: CryptoPayProvider | None = None
robokassa_provider: Any | None = None
yookassa_provider: Any | None = None
error_handler: ErrorHandler | None = None
document_manager: DocumentManager | None = None
response_cache: Any | None = None
metrics_collector: Any | None = None
task_manager: Any | None = None
health_checker: Any | None = None
scaling_components: dict[str, Any] | None = None
judicial_rag: Any | None = None

_SYNCED_ATTRS = (
    "WELCOME_MEDIA",
    "BOT_TOKEN",
    "BOT_USERNAME",
    "USE_ANIMATION",
    "USE_STREAMING",
    "SAFE_LIMIT",
    "MAX_MESSAGE_LENGTH",
    "DB_PATH",
    "TRIAL_REQUESTS",
    "SUB_DURATION_DAYS",
    "RUB_PROVIDER_TOKEN",
    "SUB_PRICE_RUB",
    "SUB_PRICE_RUB_KOPEKS",
    "STARS_PROVIDER_TOKEN",
    "SUB_PRICE_XTR",
    "DYNAMIC_PRICE_XTR",
    "SUBSCRIPTION_PLANS",
    "SUBSCRIPTION_PLAN_MAP",
    "DEFAULT_SUBSCRIPTION_PLAN",
    "ADMIN_IDS",
    "USER_SESSIONS_MAX",
    "USER_SESSION_TTL_SECONDS",
    "db",
    "rate_limiter",
    "access_service",
    "openai_service",
    "audio_service",
    "session_store",
    "crypto_provider",
    "robokassa_provider",
    "yookassa_provider",
    "error_handler",
    "document_manager",
    "response_cache",
    "metrics_collector",
    "task_manager",
    "health_checker",
    "scaling_components",
    "judicial_rag",
)


def _sync_local_globals() -> None:
    for attr in _SYNCED_ATTRS:
        globals()[attr] = getattr(simple_context, attr, None)


_sync_local_globals()


def refresh_runtime_globals() -> None:
    simple_context.refresh_runtime_globals()
    _sync_local_globals()


def __getattr__(name: str) -> Any:
    return getattr(simple_context, name)
# ============ СИСТЕМА РЕЙТИНГА ============



# ============ ОБРАБОТКА ОШИБОК ============


async def log_only_aiogram_error(event: ErrorEvent):
    """Глобальный обработчик ошибок"""
    logger.exception("Critical error in bot: %s", event.exception)


# ============ ГЛАВНАЯ ФУНКЦИЯ ============


async def _maybe_call(coro_or_func):
    """Вспомогательный вызов: поддерживает sync/async методы init()/close()."""
    if coro_or_func is None:
        return
    try:
        res = coro_or_func()
    except TypeError:
        # Если передали уже корутину
        res = coro_or_func
    if asyncio.iscoroutine(res):
        return await res
    return res


async def run_bot() -> None:
    """Main coroutine launching the bot."""
    global BOT_USERNAME
    global metrics_collector, db, response_cache, rate_limiter, access_service, openai_service
    global audio_service, session_store, crypto_provider, error_handler, document_manager
    global scaling_components, health_checker, task_manager
    ctx = get_runtime()
    cfg = ctx.settings
    container = ctx.get_dependency('container')
    if container is None:
        raise RuntimeError('DI container is not available')

    refresh_runtime_globals()

    if not cfg.telegram_bot_token:
        raise RuntimeError('TELEGRAM_BOT_TOKEN is required')

    session = None
    proxy_url = (cfg.telegram_proxy_url or '').strip()
    if proxy_url:
        logger.info('Using proxy: %s', proxy_url.split('@')[-1])
        proxy_user = (cfg.telegram_proxy_user or '').strip()
        proxy_pass = (cfg.telegram_proxy_pass or '').strip()
        if proxy_user and proxy_pass:
            from urllib.parse import quote, urlparse, urlunparse

            if '://' not in proxy_url:
                proxy_url = 'http://' + proxy_url
            u = urlparse(proxy_url)
            userinfo = f"{quote(proxy_user, safe='')}:{quote(proxy_pass, safe='')}"
            netloc = f"{userinfo}@{u.hostname}{':' + str(u.port) if u.port else ''}"
            proxy_url = urlunparse((u.scheme, netloc, u.path or '', u.params, u.query, u.fragment))
        session = AiohttpSession(proxy=proxy_url)

    bot = Bot(cfg.telegram_bot_token, session=session)
    try:
        bot_info = await bot.get_me()
        simple_context.BOT_USERNAME = (bot_info.username or '').strip()
        BOT_USERNAME = simple_context.BOT_USERNAME
    except Exception as exc:
        logger.warning('Could not fetch bot username: %s', exc)
    dp = Dispatcher()
    register_progressbar(dp)

    # Инициализация системы метрик/кэша/т.п.
    from src.core.background_tasks import (
        BackgroundTaskManager,
        CacheCleanupTask,
        DatabaseCleanupTask,
        DocumentStorageCleanupTask,
        HealthCheckTask,
        MetricsCollectionTask,
        SessionCleanupTask,
    )
    from src.core.cache import ResponseCache, create_cache_backend
    from src.core.health import (
        DatabaseHealthCheck,
        HealthChecker,
        OpenAIHealthCheck,
        RateLimiterHealthCheck,
        SessionStoreHealthCheck,
        SystemResourcesHealthCheck,
    )
    from src.core.metrics import init_metrics, set_system_status
    from src.core.scaling import LoadBalancer, ScalingManager, ServiceRegistry, SessionAffinity

    prometheus_port = cfg.prometheus_port
    metrics_collector = init_metrics(
        enable_prometheus=cfg.enable_prometheus,
        prometheus_port=prometheus_port,
    )
    ctx.metrics_collector = metrics_collector
    simple_context.metrics_collector = metrics_collector
    set_system_status("starting")

    logger.info("🚀 Starting AI-Ivan (simple)")

    # Используем продвинутую базу данных с connection pooling
    logger.info("Using advanced database with connection pooling")
    db = ctx.db or container.get(DatabaseAdvanced)
    ctx.db = db
    simple_context.db = db
    await db.init()

    setup_admin_commands(dp, db, ADMIN_IDS)

    cache_backend = await create_cache_backend(
        redis_url=cfg.redis_url,
        fallback_to_memory=True,
        memory_max_size=cfg.cache_max_size,
    )

    response_cache = ResponseCache(
        backend=cache_backend,
        default_ttl=cfg.cache_ttl,
        enable_compression=cfg.cache_compression,
    )
    ctx.response_cache = response_cache
    simple_context.response_cache = response_cache

    rate_limiter = ctx.rate_limiter or container.get(RateLimiter)
    ctx.rate_limiter = rate_limiter
    simple_context.rate_limiter = rate_limiter
    await rate_limiter.init()

    access_service = ctx.access_service or container.get(AccessService)
    ctx.access_service = access_service
    simple_context.access_service = access_service

    openai_service = ctx.openai_service or container.get(OpenAIService)
    openai_service.cache = response_cache
    ctx.openai_service = openai_service
    simple_context.openai_service = openai_service

    if cfg.voice_mode_enabled:
        audio_service = AudioService(
            stt_model=cfg.voice_stt_model,
            tts_model=cfg.voice_tts_model,
            tts_voice=cfg.voice_tts_voice,
            tts_format=cfg.voice_tts_format,
            max_duration_seconds=cfg.voice_max_duration_seconds,
            tts_voice_male=cfg.voice_tts_voice_male,
            tts_chunk_char_limit=cfg.voice_tts_chunk_char_limit,
            tts_speed=cfg.voice_tts_speed,
            tts_style=cfg.voice_tts_style,
            tts_sample_rate=cfg.voice_tts_sample_rate,
            tts_backend=cfg.voice_tts_backend,
        )
        ctx.audio_service = audio_service
        simple_context.audio_service = audio_service
        logger.info(
            "Voice mode enabled (stt=%s, tts=%s, voice=%s, male_voice=%s, format=%s, chunk_limit=%s)",
            cfg.voice_stt_model,
            cfg.voice_tts_model,
            cfg.voice_tts_voice,
            cfg.voice_tts_voice_male,
            cfg.voice_tts_format,
            cfg.voice_tts_chunk_char_limit,
        )
    else:
        audio_service = None
        ctx.audio_service = None
        simple_context.audio_service = None
        logger.info("Voice mode disabled")

    session_store = ctx.session_store or container.get(SessionStore)
    ctx.session_store = session_store
    simple_context.session_store = session_store
    crypto_provider = ctx.crypto_provider or container.get(CryptoPayProvider)
    ctx.crypto_provider = crypto_provider
    simple_context.crypto_provider = crypto_provider

    error_handler = ErrorHandler(logger=logger)
    ctx.error_handler = error_handler
    simple_context.error_handler = error_handler

    dp.update.middleware(ErrorHandlingMiddleware(error_handler, logger=logger))

    document_manager = DocumentManager(openai_service=openai_service, settings=cfg)
    ctx.document_manager = document_manager
    simple_context.document_manager = document_manager
    logger.info("Document processing system initialized")

    refresh_runtime_globals()

    # Регистрируем recovery handler для БД
    async def database_recovery_handler(exc):
        if db is not None and hasattr(db, "init"):
            try:
                await _maybe_call(db.init)
                logger.info("Database recovery completed")
            except Exception as recovery_error:
                logger.error(f"Database recovery failed: {recovery_error}")

    try:
        error_handler.register_recovery_handler(ErrorType.DATABASE, database_recovery_handler)
    except Exception:
        # Если ErrorType/handler не поддерживает регистрацию — просто логируем
        logger.debug("Recovery handler registration skipped")

    # Инициализация компонентов масштабирования (опционально)
    scaling_components = None
    ctx.scaling_components = None
    simple_context.scaling_components = None
    if cfg.enable_scaling:
        try:
            service_registry = ServiceRegistry(
                redis_url=cfg.redis_url,
                heartbeat_interval=cfg.heartbeat_interval,
            )
            await service_registry.initialize()
            await service_registry.start_background_tasks()

            load_balancer = LoadBalancer(service_registry)
            session_affinity = SessionAffinity(
                redis_client=getattr(cache_backend, "_redis", None),
                ttl=cfg.session_affinity_ttl,
            )
            scaling_manager = ScalingManager(
                service_registry=service_registry,
                load_balancer=load_balancer,
                session_affinity=session_affinity,
            )

            scaling_components = {
                "service_registry": service_registry,
                "load_balancer": load_balancer,
                "session_affinity": session_affinity,
                "scaling_manager": scaling_manager,
            }
            ctx.scaling_components = scaling_components
            simple_context.scaling_components = scaling_components
            logger.info("🔄 Scaling components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize scaling components: {e}")

    # Health checks
    health_checker = HealthChecker(check_interval=cfg.health_check_interval)
    ctx.health_checker = health_checker
    simple_context.health_checker = health_checker
    health_checker.register_check(DatabaseHealthCheck(db))
    health_checker.register_check(OpenAIHealthCheck(openai_service))
    health_checker.register_check(SessionStoreHealthCheck(session_store))
    health_checker.register_check(RateLimiterHealthCheck(rate_limiter))
    if cfg.enable_system_monitoring:
        health_checker.register_check(SystemResourcesHealthCheck())
    await health_checker.start_background_checks()

    # Фоновые задачи
    task_manager = BackgroundTaskManager(error_handler)
    ctx.task_manager = task_manager
    simple_context.task_manager = task_manager
    task_manager.register_task(
        DatabaseCleanupTask(
            db,
            interval_seconds=cfg.db_cleanup_interval,
            max_old_transactions_days=cfg.db_cleanup_days,
        )
    )
    task_manager.register_task(
        CacheCleanupTask(
            [openai_service], interval_seconds=cfg.cache_cleanup_interval
        )
    )
    task_manager.register_task(
        SessionCleanupTask(
            session_store, interval_seconds=cfg.session_cleanup_interval
        )
    )
    task_manager.register_task(
        DocumentStorageCleanupTask(
            document_manager.storage,
            max_age_hours=document_manager.storage.cleanup_max_age_hours,
            interval_seconds=document_manager.storage.cleanup_interval_seconds,
        )
    )

    all_components = {
        "database": db,
        "openai_service": openai_service,
        "rate_limiter": rate_limiter,
        "session_store": session_store,
        "error_handler": error_handler,
        "health_checker": health_checker,
    }
    if scaling_components:
        all_components.update(scaling_components)

    task_manager.register_task(
        HealthCheckTask(
            all_components, interval_seconds=cfg.health_check_task_interval
        )
    )
    if getattr(metrics_collector, "enable_prometheus", False):
        task_manager.register_task(
            MetricsCollectionTask(
                all_components,
                interval_seconds=cfg.metrics_collection_interval,
            )
        )
    await task_manager.start_all()
    logger.info("Started %s background tasks", len(task_manager.tasks))

    # Запускаем retention notifier
    global retention_notifier
    retention_notifier = RetentionNotifier(bot, db)
    await retention_notifier.start()
    logger.info("✉️ Retention notifier started")

    refresh_runtime_globals()

    # Команды
    base_commands = [
        BotCommand(command="start", description=f"{Emoji.ROBOT} Начать работу"),
        BotCommand(command="buy", description=f"{Emoji.MAGIC} Оформить подписку"),
        BotCommand(command="status", description=f"{Emoji.STATS} Статус подписки"),
        BotCommand(command="mystats", description="📊 Моя статистика"),

    ]
    await bot.set_my_commands(base_commands)

    if ADMIN_IDS:
        admin_commands = base_commands + [
            BotCommand(command="ratings", description="📈 Статистика рейтингов (админ)"),
            BotCommand(command="errors", description="🚨 Статистика ошибок (админ)"),
        ]
        for admin_id in ADMIN_IDS:
            try:
                await bot.set_my_commands(
                    admin_commands,
                    scope=BotCommandScopeChat(chat_id=admin_id),
                )
            except TelegramBadRequest as exc:
                logger.warning(
                    "Failed to set admin command list for %s: %s",
                    admin_id,
                    exc,
                )

    # Роутинг
    register_payment_handlers(dp)

    register_menu_handlers(dp)
    register_document_handlers(dp)
    register_retention_handlers(dp)
    register_feedback_handlers(dp)
    register_admin_handlers(dp)
    register_question_handlers(dp)

    if settings().voice_mode_enabled:
        register_voice_handlers(dp, process_question)

    # Глобальный обработчик ошибок aiogram (с интеграцией ErrorHandler при наличии)
    async def telegram_error_handler(event: ErrorEvent):
        if error_handler:
            try:
                context = ErrorContext(
                    function_name="telegram_error_handler",
                    additional_data={
                        "update": str(event.update) if event.update else None,
                        "exception_type": type(event.exception).__name__,
                    },
                )
                await error_handler.handle_exception(event.exception, context)
            except Exception as handler_error:
                logger.error(f"Error handler failed: {handler_error}")
        logger.exception("Critical error in bot: %s", event.exception)

    dp.error.register(telegram_error_handler)

    # Лог старта
    set_system_status("running")
    startup_info = [
        "🤖 AI-Ivan (simple) successfully started!",
        f"🎞 Animation: {'enabled' if USE_ANIMATION else 'disabled'}",
        f"🗄️ Database: advanced",
        f"🔄 Cache: {cache_backend.__class__.__name__}",
        f"📈 Metrics: {'enabled' if getattr(metrics_collector, 'enable_prometheus', False) else 'disabled'}",
        f"🏥 Health checks: {len(health_checker.checks)} registered",
        f"⚙️ Background tasks: {len(task_manager.tasks)} running",
        f"🔄 Scaling: {'enabled' if scaling_components else 'disabled'}",
    ]
    for info in startup_info:
        logger.info(info)
    if prometheus_port:
        logger.info(
            f"📊 Prometheus metrics available at http://localhost:{prometheus_port}/metrics"
        )

    try:
        logger.info("🚀 Starting bot polling...")
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("🛑 AI-Ivan stopped by user")
        set_system_status("stopping")
    except Exception as e:
        logger.exception("💥 Fatal error in main loop: %s", e)
        set_system_status("stopping")
        raise
    finally:
        logger.info("🔧 Shutting down services...")
        set_system_status("stopping")

        # Останавливаем retention notifier
        if retention_notifier:
            try:
                await retention_notifier.stop()
            except Exception as e:
                logger.error(f"Error stopping retention notifier: {e}")

        # Останавливаем фоновые задачи
        try:
            await task_manager.stop_all()
        except Exception as e:
            logger.error(f"Error stopping background tasks: {e}")

        # Останавливаем health checks
        try:
            await health_checker.stop_background_checks()
        except Exception as e:
            logger.error(f"Error stopping health checks: {e}")

        # Останавливаем компоненты масштабирования
        if scaling_components:
            try:
                await scaling_components["service_registry"].stop_background_tasks()
            except Exception as e:
                logger.error(f"Error stopping scaling components: {e}")

        # Закрываем основные сервисы (поддержка sync/async close)
        services_to_close = [
            ("Bot session", lambda: bot.session.close()),
            ("Database", lambda: getattr(db, "close", None) and db.close()),
            ("Rate limiter", lambda: getattr(rate_limiter, "close", None) and rate_limiter.close()),
            (
                "OpenAI service",
                lambda: getattr(openai_service, "close", None) and openai_service.close(),
            ),
            (
                "Audio service",
                lambda: getattr(audio_service, "aclose", None) and audio_service.aclose(),
            ),
            (
                "Response cache",
                lambda: getattr(response_cache, "close", None) and response_cache.close(),
            ),
        ]
        for service_name, close_func in services_to_close:
            try:
                await _maybe_call(close_func)
                logger.debug(f"✅ {service_name} closed")
            except Exception as e:
                logger.error(f"❌ Error closing {service_name}: {e}")

        logger.info("👋 AI-Ivan shutdown complete")
