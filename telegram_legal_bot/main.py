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

# ── Импорты пакета (и фоллбек для плоской структуры) ─────────────────────────
try:
    from telegram_legal_bot.config import Settings, load_settings
    from telegram_legal_bot.services.openai_service import OpenAIService
    from telegram_legal_bot.handlers.start import router as start_router
    from telegram_legal_bot.handlers.legal_query import (
        router as legal_router,
        setup_context as setup_legal_context,
    )

except ImportError:
    from config import Settings, load_settings  # type: ignore
    try:
        from services.openai_service import OpenAIService  # type: ignore
    except ImportError:
        from services import OpenAIService  # type: ignore
    from handlers.start import router as start_router  # type: ignore
    from handlers.legal_query import (  # type: ignore
        router as legal_router,
        setup_context as setup_legal_context,
    )
    try:
        from handlers.ui_demo import router as ui_demo_router  # type: ignore
    except Exception:
        ui_demo_router = None  # type: ignore[assignment]


# ── Логирование ───────────────────────────────────────────────────────────────
def _setup_logging(json_mode: bool) -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if json_mode:
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                import json as _json
                import time as _time
                payload = {
                    "t": int(_time.time()),
                    "lvl": record.levelname,
                    "msg": record.getMessage(),
                    "name": record.name,
                }
                if record.exc_info:
                    payload["exc"] = self.formatException(record.exc_info)
                return _json.dumps(payload, ensure_ascii=False)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    # 🔉 уровни библиотек
    logging.getLogger("aiogram").setLevel(level)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("openai_service").setLevel(logging.DEBUG)
    logging.getLogger("legal_query").setLevel(level)


# ── Утилиты ───────────────────────────────────────────────────────────────────
def _build_proxy_url(url: str | None, user: Optional[str], pwd: Optional[str]) -> Optional[str]:
    """
    Делает валидный прокси-URL для AiohttpSession:
      - добавляет схему http:// при её отсутствии;
      - при наличии user/password и отсутствии userinfo — внедряет user:pass@;
      - логин/пароль экранируются.
    Примеры входа: "localhost:8080", "http://host:8080", "socks5://1.2.3.4:1080".
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
    settings = load_settings()
    _setup_logging(settings.log_json)
    log = logging.getLogger("main")

    # Telegram: корректно задаём API-сервер (official или self-hosted) и HTTP-прокси
    api = _get_api_server(getattr(settings, "telegram_api_base", None))
    tg_proxy = _build_proxy_url(
        getattr(settings, "telegram_proxy_url", None),
        getattr(settings, "telegram_proxy_user", None),
        getattr(settings, "telegram_proxy_pass", None),
    )

    # ВАЖНО: proxy передаём через параметр proxy= (а не в api=)
    session = AiohttpSession(api=api, proxy=tg_proxy, timeout=70)

    default_props = (
        DefaultBotProperties(parse_mode=settings.parse_mode)
        if getattr(settings, "parse_mode", None) is not None
        else DefaultBotProperties()
    )

    bot = Bot(
        token=settings.telegram_token,
        session=session,
        default=default_props,
    )
    dp = Dispatcher()

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


if __name__ == "__main__":
    try:
        # Не используем uvloop принудительно: совместимость шире
        asyncio.run(main_async())
    except (KeyboardInterrupt, SystemExit):
        pass
