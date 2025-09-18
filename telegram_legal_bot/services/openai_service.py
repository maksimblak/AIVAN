from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Sequence

import httpx
from openai import AsyncOpenAI
from openai._exceptions import OpenAIError, PermissionDeniedError
from urllib.parse import urlparse, urlunparse

from telegram_legal_bot.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class LegalAdvice:
    """Нормализованный ответ от модели."""
    summary: str
    details: str
    laws: list[str]


class OpenAIService:
    """
    Обёртка над OpenAI Responses API (GPT-5):
      - developer + user сообщения
      - строгий JSON-ответ
      - таймаут + ретраи с экспоненциальным бэкоффом
      - управление стилем: text.verbosity, reasoning.effort
      - опциональный системный прокси (НЕ для обхода гео-блоков)
    """

    def __init__(self, settings: Settings):
        self._s = settings

        # --- httpx клиент с прокси (если указан в .env) ---
        http_client: httpx.AsyncClient | None = None
        if getattr(settings, "openai_proxy_url", None):
            proxy_url = self._inject_basic_auth(
                settings.openai_proxy_url or "",
                settings.openai_proxy_user,
                settings.openai_proxy_pass,
            )
            http_client = httpx.AsyncClient(
                proxies=proxy_url,
                timeout=httpx.Timeout(45.0),
            )

        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            http_client=http_client,  # None = без прокси
        )

    # --------------------------- Публичное API ---------------------------

    async def generate_legal_advice(
        self,
        user_question: str,
        short_history: Sequence[str] | None = None,
    ) -> LegalAdvice:
        """Формирует юридический ответ в нормализованной структуре."""
        hist = ""
        if short_history:
            hist_json = json.dumps(list(short_history), ensure_ascii=False)
            hist = f"\nИстория последних ответов (кратко): {hist_json}\n"

        user_msg = (
            "Вот юридический вопрос пользователя.\n"
            "Сформируй ответ строго в формате JSON:\n"
            '{ "summary": "Краткий ответ (2-3 предложения)", '
            '"details": "Развернутое объяснение", '
            '"laws": ["Норма/ссылка 1", "Норма/ссылка 2"] }\n'
            "Никакого текста вне JSON не добавляй.\n"
            f"{hist}\n"
            f"Вопрос: {user_question}"
        )

        raw = await self._ask_with_retries(user_msg)

        # Жёсткий парс JSON (с отрезанием мусора вокруг)
        try:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start : end + 1]
            data = json.loads(raw)
        except Exception as e:
            logger.warning("JSON-парсер споткнулся. Фрагмент: %r", raw[:800])
            raise ValueError("Модель вернула невалидный JSON") from e

        summary = (data.get("summary") or "").strip() or "Краткий ответ не сформирован."
        details = (data.get("details") or "").strip() or "Подробное объяснение не сформировано."
        laws = [str(x).strip() for x in (data.get("laws") or []) if str(x).strip()]
        return LegalAdvice(summary=summary, details=details, laws=laws)

    # --------------------------- Внутренности ---------------------------

    async def _ask_with_retries(self, user_msg: str) -> str:
        delay = 1.5
        for attempt in range(1, 4):
            try:
                return await self._ask_once(user_msg)
            except asyncio.TimeoutError as e:
                if attempt == 3:
                    raise
                logger.warning("Timeout (попытка %s) — ретраим через %.1fs", attempt, delay)
            except PermissionDeniedError as e:
                # Специально ловим 403 с гео-блоком, чтобы не молотить ретраи
                if self._is_unsupported_region(e):
                    logger.error("OpenAI: регион/территория не поддерживается (403). Останавливаем ретраи.")
                    raise RuntimeError(
                        "OpenAI отклонил запрос: регион/территория не поддерживается (403)."
                    ) from e
                if attempt == 3:
                    raise
                logger.warning("PermissionDenied (попытка %s) — ретраим через %.1fs", attempt, delay)
            except OpenAIError as e:
                if attempt == 3:
                    raise
                logger.warning("Попытка %s: %s — ретраим через %.1fs", attempt, e, delay)
            await asyncio.sleep(delay)
            delay *= 2
        # Теоретически не достигнем
        raise RuntimeError("Не удалось получить ответ от OpenAI")

    async def _ask_once(self, user_msg: str) -> str:
        """Один запрос к Responses API с управлением стилем и таймаутом."""
        try:
            resp = await asyncio.wait_for(
                self._client.responses.create(
                    model=self._s.openai_model,
                    input=[
                        {"role": "developer", "content": self._s.system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    text={"verbosity": self._s.openai_verbosity},
                    reasoning={"effort": self._s.openai_reasoning_effort},
                    max_output_tokens=self._s.openai_max_tokens,
                    temperature=self._s.openai_temperature,
                ),
                timeout=45.0,
            )
        except asyncio.TimeoutError as e:
            raise asyncio.TimeoutError("Timeout при обращении к OpenAI") from e

        # Нормально достаём текст
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        # Фоллбек: собрать текст из output.*.content[].text
        parts: list[str] = []
        for block in getattr(resp, "output", []) or []:
            for item in getattr(block, "content", []) or []:
                t = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if t:
                    parts.append(str(t))
        return "".join(parts).strip()

    # --------------------------- Utils ----------------------------------

    @staticmethod
    def _inject_basic_auth(url: str, user: str | None, password: str | None) -> str:
        """Если логин/пароль заданы и не вписаны в URL — вписываем их."""
        if not url or not user:
            return url
        p = urlparse(url)
        if "@" in (p.netloc or ""):
            return url  # уже есть креды в URL
        netloc = f"{user}:{password or ''}@{p.hostname or ''}"
        if p.port:
            netloc += f":{p.port}"
        return urlunparse((p.scheme, netloc, p.path or "", p.params or "", p.query or "", p.fragment or ""))

    @staticmethod
    def _is_unsupported_region(err: OpenAIError) -> bool:
        """
        Пытаемся аккуратно распознать 403 'unsupported_country_region_territory'
        из тела ответа, чтобы не крутить бессмысленные ретраи.
        """
        try:
            resp = getattr(err, "response", None)
            if resp is None:
                return False
            data = resp.json()
            code = (data or {}).get("error", {}).get("code")
            return str(code) == "unsupported_country_region_territory"
        except Exception:
            return False
