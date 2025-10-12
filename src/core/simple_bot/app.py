from __future__ import annotations

import logging
from typing import Awaitable, Callable, Sequence

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import BotCommand, BotCommandScopeChat, ErrorEvent

from src.bot.status_manager import register_progressbar
from src.bot.ui_components import Emoji
from src.core.exceptions import ErrorContext, ErrorHandler
from src.core.metrics import set_system_status
from src.core.simple_bot import context as simple_context
from src.core.simple_bot.admin import register_admin_handlers
from src.core.simple_bot.documents import register_document_handlers
from src.core.simple_bot.feedback import register_feedback_handlers
from src.core.simple_bot.menus import register_menu_handlers
from src.core.simple_bot.payments import register_payment_handlers
from src.core.simple_bot.questions import process_question, register_question_handlers
from src.core.simple_bot.retention import register_retention_handlers
from src.core.simple_bot.startup import RuntimeBundle, maybe_call, setup_bot_runtime
from src.core.simple_bot.voice import register_voice_handlers

logger = logging.getLogger("ai-ivan.simple")

set_runtime = simple_context.set_runtime
get_runtime = simple_context.get_runtime
settings = simple_context.settings
derived = simple_context.derived

__all__ = [
    "set_runtime",
    "get_runtime",
    "settings",
    "derived",
    "refresh_runtime_globals",
    "run_bot",
]


def refresh_runtime_globals() -> None:
    simple_context.refresh_runtime_globals()


def _build_base_commands() -> list[BotCommand]:
    return [
        BotCommand(command="start", description=f"{Emoji.ROBOT} Start the bot"),
        BotCommand(command="buy", description=f"{Emoji.MAGIC} Purchase subscription"),
        BotCommand(command="status", description=f"{Emoji.STATS} Subscription status"),
        BotCommand(command="mystats", description="📊 Personal statistics"),
    ]


def _build_admin_commands(base_commands: Sequence[BotCommand]) -> list[BotCommand]:
    admin_specific = [
        BotCommand(command="ratings", description="📈 Ratings dashboard (admin)"),
        BotCommand(command="errors", description="🚨 Error dashboard (admin)"),
    ]
    return [*base_commands, *admin_specific]


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


def _register_core_handlers(dp: Dispatcher) -> None:
    register_payment_handlers(dp)
    register_menu_handlers(dp)
    register_document_handlers(dp)
    register_retention_handlers(dp)
    register_feedback_handlers(dp)
    register_admin_handlers(dp)
    register_question_handlers(dp)


async def _register_commands(bot: Bot, admin_ids: Sequence[int]) -> None:
    base_commands = _build_base_commands()
    await bot.set_my_commands(base_commands)

    if not admin_ids:
        return

    admin_commands = _build_admin_commands(base_commands)
    for admin_id in admin_ids:
        try:
            await bot.set_my_commands(
                admin_commands,
                scope=BotCommandScopeChat(chat_id=admin_id),
            )
        except TelegramBadRequest as exc:
            logger.warning("Failed to set admin command list for %s: %s", admin_id, exc)


def _log_startup_banner(runtime: RuntimeBundle, cfg) -> None:
    cache_backend = runtime.cache_backend
    metrics_collector = runtime.metrics_collector
    health_checker = runtime.health_checker
    task_manager = runtime.task_manager
    scaling_components = runtime.scaling_components

    startup_info = [
        "🤖 AI-Ivan (simple) successfully started!",
        f"🎞 Animation: {'enabled' if cfg.use_status_animation else 'disabled'}",
        "🗄️ Database: advanced",
        f"🔄 Cache: {cache_backend.__class__.__name__}",
        f"📈 Metrics: {'enabled' if getattr(metrics_collector, 'enable_prometheus', False) else 'disabled'}",
        f"🏥 Health checks: {len(health_checker.checks)} registered",
        f"⚙️ Background tasks: {len(task_manager.tasks)} running",
        f"🔄 Scaling: {'enabled' if scaling_components else 'disabled'}",
    ]

    for line in startup_info:
        logger.info(line)


async def _graceful_shutdown(bot: Bot, runtime: RuntimeBundle) -> None:
    retention_notifier = runtime.retention_notifier
    task_manager = runtime.task_manager
    health_checker = runtime.health_checker
    scaling_components = runtime.scaling_components
    rate_limiter = runtime.rate_limiter
    openai_service = runtime.openai_service
    audio_service = runtime.audio_service
    response_cache = runtime.response_cache
    db = runtime.db

    if retention_notifier:
        try:
            await retention_notifier.stop()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error stopping retention notifier: %s", exc)

    try:
        await task_manager.stop_all()
    except Exception as exc:  # noqa: BLE001
        logger.error("Error stopping background tasks: %s", exc)

    try:
        await health_checker.stop_background_checks()
    except Exception as exc:  # noqa: BLE001
        logger.error("Error stopping health checks: %s", exc)

    if scaling_components:
        try:
            await scaling_components["service_registry"].stop_background_tasks()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error stopping scaling components: %s", exc)

    services_to_close = [
        ("Bot session", lambda: bot.session.close()),
        ("Database", lambda: getattr(db, "close", None) and db.close()),
        ("Rate limiter", lambda: getattr(rate_limiter, "close", None) and rate_limiter.close()),
        ("OpenAI service", lambda: getattr(openai_service, "close", None) and openai_service.close()),
        ("Audio service", lambda: getattr(audio_service, "aclose", None) and audio_service.aclose()),
        ("Response cache", lambda: getattr(response_cache, "close", None) and response_cache.close()),
    ]

    for service_name, close_func in services_to_close:
        try:
            await maybe_call(close_func)
            logger.debug("✅ %s closed", service_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ Error closing %s: %s", service_name, exc)


async def run_bot() -> None:
    ctx = get_runtime()
    cfg = ctx.settings
    container = ctx.get_dependency("container")
    if container is None:
        raise RuntimeError("DI container is not available")

    refresh_runtime_globals()

    if not cfg.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    session: AiohttpSession | None = None
    proxy_url = (cfg.telegram_proxy_url or "").strip()
    if proxy_url:
        logger.info("Using proxy: %s", proxy_url.split("@")[-1])
        proxy_user = (cfg.telegram_proxy_user or "").strip()
        proxy_pass = (cfg.telegram_proxy_pass or "").strip()
        if proxy_user and proxy_pass:
            from urllib.parse import quote, urlparse, urlunparse

            if "://" not in proxy_url:
                proxy_url = "http://" + proxy_url
            parsed = urlparse(proxy_url)
            userinfo = f"{quote(proxy_user, safe='')}:{quote(proxy_pass, safe='')}"
            netloc = f"{userinfo}@{parsed.hostname}{':' + str(parsed.port) if parsed.port else ''}"
            proxy_url = urlunparse(
                (parsed.scheme, netloc, parsed.path or "", parsed.params, parsed.query, parsed.fragment),
            )
        session = AiohttpSession(proxy=proxy_url)

    bot = Bot(cfg.telegram_bot_token, session=session)
    try:
        bot_info = await bot.get_me()
        simple_context.BOT_USERNAME = (bot_info.username or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch bot username: %s", exc)

    dp = Dispatcher()
    register_progressbar(dp)

    runtime = await setup_bot_runtime(
        dispatcher=dp,
        bot=bot,
        ctx=ctx,
        cfg=cfg,
        container=container,
        logger=logger,
        admin_ids=simple_context.ADMIN_IDS,
    )

    refresh_runtime_globals()
    admin_ids = simple_context.ADMIN_IDS

    await _register_commands(bot, admin_ids)
    _register_core_handlers(dp)

    if settings().voice_mode_enabled:
        register_voice_handlers(dp, process_question)

    dp.error.register(_create_telegram_error_handler(runtime.error_handler))

    set_system_status("running")
    _log_startup_banner(runtime, cfg)
    if cfg.prometheus_port:
        logger.info("📊 Prometheus metrics available at http://localhost:%s/metrics", cfg.prometheus_port)

    try:
        logger.info("🚀 Starting bot polling...")
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("🛑 AI-Ivan stopped by user")
        set_system_status("stopping")
    except Exception as exc:  # noqa: BLE001
        logger.exception("💥 Fatal error in main loop: %s", exc)
        set_system_status("stopping")
        raise
    finally:
        logger.info("🔧 Shutting down services...")
        set_system_status("stopping")
        await _graceful_shutdown(bot, runtime)
        logger.info("👋 AI-Ivan shutdown complete")
