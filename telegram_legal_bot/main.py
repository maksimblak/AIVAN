from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from typing import Sequence

from telegram import BotCommand, Update
from telegram.ext import (
    AIORateLimiter,
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

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
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    root.addHandler(handler)


# -------------------------- Error handler --------------------------------
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.exception("Unhandled error: %s", context.error)
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "⚠️ Случилась непредвиденная ошибка. Попробуйте позже."
            )
    except Exception:
        pass


# -------------------------- post_init: команды ---------------------------
async def _set_bot_commands(app: Application) -> None:
    commands: Sequence[BotCommand] = [
        BotCommand("start", "Начать работу"),
        BotCommand("help", "Помощь"),
    ]
    await app.bot.set_my_commands(commands)


# ------------------------------- main -----------------------------------
def main() -> None:
    settings = load_settings()
    _setup_logging(settings.log_level, settings.json_logs)
    logging.info("Запуск telegram_legal_bot…")

    ai_service = OpenAIService(settings)

    application: Application = (
        ApplicationBuilder()
        .token(settings.telegram_token)
        .rate_limiter(AIORateLimiter())  # не про наш лимитер — это телега-флуди
        .post_init(_set_bot_commands)
        .build()
    )

    # Команды
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))

    # Юридические вопросы: все текстовые сообщения
    application.add_handler(build_legal_message_handler(settings, ai_service))

    # Глобальный обработчик ошибок
    application.add_error_handler(on_error)

    # Грациозное завершение (SIGINT/SIGTERM)
    loop = asyncio.get_event_loop()

    stop_event = asyncio.Event()

    def _graceful_shutdown(*_: object) -> None:
        logging.info("Получен сигнал остановки — завершаем…")
        stop_event.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _graceful_shutdown)
        except NotImplementedError:
            # Windows
            signal.signal(s, lambda *_: _graceful_shutdown())

    async def runner() -> None:
        async with application:
            await application.start()
            await application.updater.start_polling(
                allowed_updates=("message",), drop_pending_updates=True
            )
            await stop_event.wait()
            await application.updater.stop()
            await application.stop()

    loop.run_until_complete(runner())
    logging.info("Остановлено. Пока 👋")


if __name__ == "__main__":
    main()
