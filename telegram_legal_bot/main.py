# telegram_legal_bot/main.py
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

from telegram_legal_bot.utils.pro_logging import setup_pro_logging
from telegram_legal_bot.middlewares.request_context import RequestContextMiddleware

# ── Импорты пакета (и фоллбек для плоской структуры) ─────────────────────────
try:
    from telegram_legal_bot.config import load_settings
    from telegram_legal_bot.services.openai_service import OpenAIService
    from telegram_legal_bot.handlers.start import router as start_router
    from telegram_legal_bot.handlers.legal_query import (
        router as legal_router,
        setup_context as setup_legal_context,
    )
except ImportError:
    from config import load_settings  # type: ignore
    try:
        from services.openai_service import OpenAIService  # type: ignore
    except ImportError:
        from services import OpenAIService  # type: ignore
    from handlers.start import router as start_router  # type: ignore
    from handlers.legal_query import (  # type: ignore
        router as legal_router,
        setup_context as setup_legal_context,
    )


# ── Утилиты ───────────────────────────────────────────────────────────────────
def _build_proxy_url(url: str | None, user: Optional[str], pwd: Optional[str]) -> Optional[str]:
    """
    Делает валидный прокси-URL для AiohttpSession:
      - добавляет схему http:// при её отсутствии;
      - при наличии user/password и отсутствии userinfo — внедряет user:pass@;
      - логин/пароль экранируются.
    Примеры: "localhost:8080", "http://host:8080", "socks5://1.2.3.4:1080".
    """
    if not url:
        return None

    u = url.strip()
    if "://" not in u:
        u = "http://" + u  # по умолчанию http://

    if user and pwd and "@" not in u:
        scheme, rest = u.split("://", 1)
        u_enc = quote(user, safe="")
        p_enc = quote(pwd, safe="")
        u = f"{scheme}://{u_enc}:{p_enc}@{rest}"

    return u


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
    # (!) Настройки берём из готового загрузчика
    settings = load_settings()  # читает ENV/.env и задаёт openai_model и токены

    # Профи-логирование: JSON + контекст (cid/uid/chat); уровень можно через ENV LOG_LEVEL
    setup_pro_logging(json_mode=getattr(settings, "log_json", True),
                      level=os.getenv("LOG_LEVEL", "INFO"))
    log = logging.getLogger("main")

    # Telegram: корректно задаём API-сервер (official или self-hosted) и HTTP-прокси
    api = _get_api_server(getattr(settings, "telegram_api_base", None))
    tg_proxy = _build_proxy_url(
        getattr(settings, "telegram_proxy_url", None),
        getattr(settings, "telegram_proxy_user", None),
        getattr(settings, "telegram_proxy_pass", None),
    )

    # Прокси передаём в session, не в api
    session = AiohttpSession(api=api, proxy=tg_proxy, timeout=70)

    default_props = (
        DefaultBotProperties(parse_mode=settings.parse_mode)
        if getattr(settings, "parse_mode", None) is not None
        else DefaultBotProperties()
    )

    bot = Bot(
        token=settings.telegram_token,   # поле так и называется в Settings
        session=session,
        default=default_props,
    )
    dp = Dispatcher()

    # ВКЛЮЧАЕМ контекстное middleware → все логи будут с cid/uid/chat
    dp.update.outer_middleware(RequestContextMiddleware())

    # OpenAI service + контекст хэндлеров
    ai = OpenAIService(settings)
    setup_legal_context(settings, ai)

    # Роутеры
    dp.include_router(start_router)
    dp.include_router(legal_router)

    # Перехват необработанных ошибок, чтобы не ронять процесс
    @dp.errors()
    async def _on_error(event, exception):
        logging.getLogger("aiogram.errors").exception("Unhandled error: %r", exception)

    # Запуск с авто-перезапуском polling при падениях
    log.info("Запуск бота…")
    try:
        while True:
            try:
                await dp.start_polling(bot)
            except Exception as e:
                log.exception("Polling crashed, restarting in 3s: %r", e)
                await asyncio.sleep(3.0)
                continue
            else:
                break
    finally:
        with suppress(Exception):
            await ai.aclose()
        with suppress(Exception):
            await bot.session.close()
def main() -> None:
    """CLI entrypoint for Poetry script."""
    try:
        asyncio.run(main_async())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
