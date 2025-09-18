"""Точка входа в приложение Telegram Legal Bot."""
from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from telegram_legal_bot.config import Settings, load_settings
from telegram_legal_bot.handlers.legal_query import LegalQueryHandler
from telegram_legal_bot.handlers.start import help_command, start

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _error_handler(update: object, context: Any) -> None:
    """Глобальный обработчик ошибок бота."""

    logger.exception("Произошла ошибка при обработке update=%s", update)


async def _run_application(settings: Settings) -> None:
    """Запускает приложение и обрабатывает graceful shutdown."""

    application = (
        Application.builder()
        .token(settings.bot.token)
        .parse_mode("MarkdownV2")
        .post_init(lambda app: logger.info("Бот инициализирован"))
        .build()
    )

    legal_query_handler = LegalQueryHandler(settings)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, legal_query_handler.handle))
    application.add_error_handler(_error_handler)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:  # pragma: no cover - Windows compatibility
            signal.signal(sig, lambda *_: stop_event.set())

    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("LegalBot запущен и ожидает сообщения")

    await stop_event.wait()

    logger.info("Остановка бота...")
    await application.updater.stop()
    await application.stop()
    await application.shutdown()
    logger.info("Бот остановлен корректно")


def main() -> None:
    """Инициализирует настройки и запускает бота."""

    try:
        settings = load_settings()
    except ValueError as exc:
        logger.error("Не удалось загрузить настройки: %s", exc)
        raise

    asyncio.run(_run_application(settings))


if __name__ == "__main__":
    main()
