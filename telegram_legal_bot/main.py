from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from contextlib import suppress
from typing import Sequence
from urllib.parse import urlparse, urlunparse

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.types import BotCommand

# ── надёжные импорты (работает и как модуль, и как файл) ───────────────
try:
    from .config import load_settings
    from .handlers.start import router as start_router
    from .handlers.legal_query import router as legal_router, setup_context
    from .services.openai_service import OpenAIService
except ImportError:
    from telegram_legal_bot.config import load_settings
    from telegram_legal_bot.handlers.start import router as start_router
    from telegram_legal_bot.handlers.legal_query import router as legal_router, setup_context
    from telegram_legal_bot.services.openai_service import OpenAIService


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
    handler.setFormatter(
        JsonFormatter()
        if json_logs
        else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)


async def _set_bot_commands(bot: Bot) -> None:
    commands: Sequence[BotCommand] = [
        BotCommand(command="start", description="Начать работу"),
        BotCommand(command="help", description="Помощь"),
    ]
    await bot.set_my_commands(commands)


def _embed_basic_auth(url: str | None, user: str | None, pwd: str | None) -> str | None:
    if not url:
        return None
    if not user:
        return url
    p = urlparse(url)
    # если уже есть креды — оставляем
    if "@" in (p.netloc or ""):
        return url
    host = p.hostname or ""
    netloc = f"{user}:{pwd or ''}@{host}"
    if p.port:
        netloc += f":{p.port}"
    return urlunparse((p.scheme, netloc, p.path or "", p.params or "", p.query or "", p.fragment or ""))


async def main_async() -> None:
    settings = load_settings()
    _setup_logging(settings.log_level, settings.json_logs)
    logging.info("Запуск telegram_legal_bot (aiogram)…")

    # ── Telegram proxy (вшиваем креды прямо в URL; никаких proxy_auth) ──
    proxy_url = _embed_basic_auth(
        getattr(settings, "telegram_proxy_url", None),
        getattr(settings, "telegram_proxy_user", None),
        getattr(settings, "telegram_proxy_pass", None),
    )
    session = AiohttpSession(proxy=proxy_url) if proxy_url else None

    bot = Bot(
        token=settings.telegram_token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2),
        session=session,  # может быть None — тогда без прокси
    )
    dp = Dispatcher()

    # DI: прокидываем конфиг и OpenAI-сервис в хэндлеры
    ai = OpenAIService(settings)
    setup_context(settings, ai)

    # Подключаем роутеры
    dp.include_router(start_router)
    dp.include_router(legal_router)

    # Команды
    await _set_bot_commands(bot)

    # Грациозная остановка
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop(*_: object) -> None:
        logging.info("Получен сигнал — останавливаемся…")
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _stop)
        loop.add_signal_handler(signal.SIGTERM, _stop)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda *_: _stop())
        signal.signal(signal.SIGTERM, lambda *_: _stop())

    # Стартуем polling в отдельной таске, ждём сигнал, затем отменяем таску
    polling_task = asyncio.create_task(
        dp.start_polling(
            bot,
            skip_updates=True,
            allowed_updates=dp.resolve_used_update_types(),
        )
    )
    try:
        await stop_event.wait()
    finally:
        polling_task.cancel()
        with suppress(asyncio.CancelledError):
            await polling_task
        await bot.session.close()
        logging.info("Остановлено. Пока 👋")


def run() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
