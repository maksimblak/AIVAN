"""
Простая версия Telegram бота ИИ-Иван
Только /start и обработка вопросов, никаких кнопок и лишних команд
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

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

from src.bot.ui_components import Emoji
from src.core.audio_service import AudioService
from src.core.access import AccessService
from src.core.db_advanced import DatabaseAdvanced, TransactionStatus
from src.core.exceptions import (
    ErrorContext,
    ErrorHandler,
)
from src.core.openai_service import OpenAIService
from src.core.payments import CryptoPayProvider
from src.core.session_store import SessionStore
from src.core.runtime import SubscriptionPlanPricing, WelcomeMedia
from src.core.metrics import set_system_status
from src.bot.ratelimit import RateLimiter
from src.bot.status_manager import register_progressbar

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
from src.core.simple_bot.startup import maybe_call, setup_bot_runtime
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


# ============ КОНФИГУРАЦИЯ ============

logger = logging.getLogger("ai-ivan.simple")

set_runtime = simple_context.set_runtime
get_runtime = simple_context.get_runtime
settings = simple_context.settings
derived = simple_context.derived

def refresh_runtime_globals() -> None:
    simple_context.refresh_runtime_globals()


def _build_base_commands() -> list[BotCommand]:
    return [
        BotCommand(command="start", description=f"{Emoji.ROBOT} Начать работу"),
        BotCommand(command="buy", description=f"{Emoji.MAGIC} Оформить подписку"),
        BotCommand(command="status", description=f"{Emoji.STATS} Статус подписки"),
        BotCommand(command="mystats", description="📊 Моя статистика"),
    ]


def _build_admin_commands(base_commands: Sequence[BotCommand]) -> list[BotCommand]:
    admin_specific = [
        BotCommand(command="ratings", description="📈 Статистика рейтингов (админ)"),
        BotCommand(command="errors", description="🚨 Статистика ошибок (админ)"),
    ]
    return list(base_commands) + admin_specific


def _create_telegram_error_handler(
    error_handler: ErrorHandler | None,
) -> Callable[[ErrorEvent], Awaitable[None]]:
    async def telegram_error_handler(event: ErrorEvent) -> None:
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
            except Exception as handler_error:  # noqa: BLE001
                logger.error("Error handler failed: %s", handler_error)
        logger.exception("Critical error in bot: %s", event.exception)

    return telegram_error_handler


def __getattr__(name: str) -> Any:
    return getattr(simple_context, name)
# ============ СИСТЕМА РЕЙТИНГА ============



# ============ ОБРАБОТКА ОШИБОК ============


async def run_bot() -> None:
    """Main coroutine launching the bot."""
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
    except Exception as exc:
        logger.warning('Could not fetch bot username: %s', exc)
    dp = Dispatcher()
    register_progressbar(dp)

    prometheus_port = cfg.prometheus_port
    runtime = await setup_bot_runtime(
        dispatcher=dp,
        bot=bot,
        ctx=ctx,
        cfg=cfg,
        container=container,
        logger=logger,
        admin_ids=simple_context.ADMIN_IDS,
    )

    metrics_collector = runtime.metrics_collector
    cache_backend = runtime.cache_backend
    response_cache = runtime.response_cache
    db = runtime.db
    rate_limiter = runtime.rate_limiter
    access_service = runtime.access_service
    openai_service = runtime.openai_service
    audio_service = runtime.audio_service
    session_store = runtime.session_store
    crypto_provider = runtime.crypto_provider
    error_handler = runtime.error_handler
    document_manager = runtime.document_manager
    scaling_components = runtime.scaling_components
    health_checker = runtime.health_checker
    task_manager = runtime.task_manager
    retention_notifier = runtime.retention_notifier

    refresh_runtime_globals()
    admin_ids = simple_context.ADMIN_IDS

    # Команды
    base_commands = _build_base_commands()
    await bot.set_my_commands(base_commands)

    if admin_ids:
        admin_commands = _build_admin_commands(base_commands)
        for admin_id in admin_ids:
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

    dp.error.register(_create_telegram_error_handler(error_handler))

    # Лог старта
    set_system_status("running")
    startup_info = [
        "🤖 AI-Ivan (simple) successfully started!",
        f"🎞 Animation: {'enabled' if cfg.use_status_animation else 'disabled'}",
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
                await maybe_call(close_func)
                logger.debug(f"✅ {service_name} closed")
            except Exception as e:
                logger.error(f"❌ Error closing {service_name}: {e}")

        logger.info("👋 AI-Ivan shutdown complete")
