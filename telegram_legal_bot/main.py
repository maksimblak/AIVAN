from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from typing import Sequence

from telegram import BotCommand, Update
from telegram.constants import ParseMode
from telegram.ext import (
    AIORateLimiter,
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from config import load_settings
from handlers.legal_query import build_legal_message_handler
from handlers.start import cmd_help, cmd_start
from services.openai_service import OpenAIService


# ----------- Логирование (поддержка JSON-логов по флажку) -------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        data = {
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def _setup_logging(level: str, json_logs: bool) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level, logging.INFO))
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    root.addHandler(handler)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.exception("Unhandled error: %s", context.error)
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "⚠️ Случилась непредвиденная ошибка. Попробуйте позже."
            )
    except Exception:
        pass


async def _set_bot_commands(app: Application) -> None:
    commands: Sequence[BotCommand] = [
        BotCommand("start", "Начать работу"),
        BotCommand("help", "Помощь"),
    ]
    await app.bot.set_my_commands(commands)


async def main_async() -> None:
    settings = load_settings()
    _setup_logging(settings.log_level, settings.json_logs)
    logging.info("Запуск telegram_legal_bot…")

    ai_service = OpenAIService(settings)

    application: Application = (
        ApplicationBuilder()
        .token(settings.telegram_token)
        .rate_limiter(AIORateLimiter())
        .build()
    )

    # Команды
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))

    # Юридические вопросы
    application.add_handler(build_legal_message_handler(settings, ai_service))

    # Глобальный обработчик ошибок
    application.add_error_handler(on_error)

    # Установим команды
    await _set_bot_commands(application)

    # Грациозная остановка по сигналу
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop(*_: object) -> None:
        logging.info("Получен сигнал остановки — завершаем…")
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _stop)
        loop.add_signal_handler(signal.SIGTERM, _stop)
    except NotImplementedError:
        # Windows: fallback
        signal.signal(signal.SIGINT, lambda *_: _stop())
        signal.signal(signal.SIGTERM, lambda *_: _stop())

    # run_polling сам стартует/останавливает updater и сетку тасков
    await application.initialize()
    await application.start()
    await application.updater.start_polling(
        allowed_updates=("message",), drop_pending_updates=True
    )

    try:
        await stop_event.wait()
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        logging.info("Остановлено. Пока 👋")


if __name__ == "__main__":
    # никаких get_event_loop() — создаём и управляем лупом правильно
    asyncio.run(main_async())
