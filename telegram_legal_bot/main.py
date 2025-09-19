from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import suppress
from typing import Optional
from urllib.parse import quote

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer

# ── Импорты пакета (и фоллбек для плоской структуры) ─────────────────────────
try:
    from telegram_legal_bot.config import Settings, load_settings
    from telegram_legal_bot.services import OpenAIService
    from telegram_legal_bot.handlers.start import router as start_router
    from telegram_legal_bot.handlers.legal_query import (
        router as legal_router,
        setup_context as setup_legal_context,
    )
except ImportError:
    from config import Settings, load_settings
    try:
        from services import OpenAIService
    except ImportError:
        from services.openai_service import OpenAIService
    from handlers.start import router as start_router
    from handlers.legal_query import router as legal_router, setup_context as setup_legal_context


# ── Логирование ───────────────────────────────────────────────────────────────
def _setup_logging(json_mode: bool) -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if json_mode:
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                import json, time
                payload = {
                    "t": int(time.time()),
                    "lvl": record.levelname,
                    "msg": record.getMessage(),
                    "name": record.name,
                }
                if record.exc_info:
                    payload["exc"] = self.formatException(record.exc_info)
                return json.dumps(payload, ensure_ascii=False)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
    else:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # 🔉 уровни библиотек
    logging.getLogger("aiogram").setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # подними до INFO при дебаге
    # наши
    logging.getLogger("openai_service").setLevel(level)
    logging.getLogger("legal_query").setLevel(level)


# ── Утилиты ───────────────────────────────────────────────────────────────────
def _build_proxy_url(url: str | None, user: Optional[str], pwd: Optional[str]) -> Optional[str]:
    """http(s)://user:pass@host:port — с экранированием логина/пароля."""
    if not url:
        return None
    if user and pwd and "@" not in url and "://" in url:
        scheme, rest = url.split("://", 1)
        u = quote(user, safe="")
        p = quote(pwd, safe="")
        return f"{scheme}://{u}:{p}@{rest}"
    return url


def _get_api_server(base: Optional[str]) -> TelegramAPIServer:
    """Всегда возвращаем валидный TelegramAPIServer (официальный по умолчанию)."""
    official = getattr(TelegramAPIServer, "official", None)
    if base:
        return TelegramAPIServer.from_base(base)
    if callable(official):
        return official()
    return TelegramAPIServer.from_base("https://api.telegram.org")


# ── entrypoint ────────────────────────────────────────────────────────────────
async def main_async() -> None:
    settings = load_settings()
    _setup_logging(settings.log_json)
    log = logging.getLogger("main")

    # Telegram: корректно задаём API-сервер (official или self-hosted) и HTTP-прокси
    api = _get_api_server(getattr(settings, "telegram_api_base", None))
    tg_proxy = _build_proxy_url(settings.telegram_proxy_url, settings.telegram_proxy_user, settings.telegram_proxy_pass)

    # ВАЖНО: proxy передаём через параметр proxy= (а не в api=)
    session = AiohttpSession(api=api, proxy=tg_proxy)

    bot = Bot(
        token=settings.telegram_token,
        session=session,
        default=DefaultBotProperties(parse_mode=settings.parse_mode),
    )
    dp = Dispatcher()

    # OpenAI service + контекст хэндлеров
    ai = OpenAIService(settings)
    setup_legal_context(settings, ai)

    # Роутеры
    dp.include_router(start_router)
    dp.include_router(legal_router)

    # Запуск
    log.info("Запуск бота…")
    try:
        await dp.start_polling(bot)
    finally:
        with suppress(Exception):
            await ai.aclose()
        with suppress(Exception):
            await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except (KeyboardInterrupt, SystemExit):
        pass
