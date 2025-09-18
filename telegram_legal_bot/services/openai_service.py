from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from openai import AsyncOpenAI
from openai._exceptions import OpenAIError

from telegram_legal_bot.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class LegalAdvice:
    """
    Нормализованный ответ от модели.
    """
    summary: str
    details: str
    laws: list[str]


class OpenAIService:
    """
    Обёртка над OpenAI Chat Completions с:
      - системным промптом
      - жёстким ограничением на формат JSON
      - таймаутом и ретраями (экспоненциальный бэкофф)
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def _ask_chat(self, messages: list[dict[str, str]]) -> str:
        """
        Делает один вызов к Chat Completions с таймаутом.
        Возвращает content первого ответа.
        """
        # Жёсткая гарантия JSON вывода через инструкцию и проверку при парсе
        tools_hint = (
            "Ответь строго в формате JSON со следующими полями:\n"
            "{\n"
            '  "summary": "Краткий ответ (2-3 предложения)",\n'
            '  "details": "Развернутое объяснение",\n'
            '  "laws": ["Норма/ссылка 1", "Норма/ссылка 2"]\n'
            "}\n"
            "Никакого текста вне JSON не добавляй."
        )

        # Добавляем системный промпт + техническую инструкцию о JSON
        msgs = [
            {"role": "system", "content": self._settings.system_prompt},
            {"role": "system", "content": tools_hint},
            *messages,
        ]

        # Таймаут + аккуратный размер ответа
        try:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._settings.openai_model,
                    temperature=self._settings.openai_temperature,
                    max_tokens=self._settings.openai_max_tokens,
                    messages=msgs,
                    # Можно включить "response_format={'type': 'json_object'}",
                    # но оставим чистую промпт-инструкцию для совместимости.
                ),
                timeout=45.0,
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError("Timeout при обращении к OpenAI") from e

        except OpenAIError:
            logger.exception("OpenAI API error")
            raise

        choice = (resp.choices or [None])[0]
        content = (choice and choice.message and choice.message.content) or ""
        return content.strip()

    async def _ask_with_retries(self, messages: list[dict[str, str]]) -> str:
        """
        Ретраи с экспоненциальным бэкоффом: 3 попытки.
        """
        delay = 1.5
        for attempt in range(1, 4):
            try:
                return await self._ask_chat(messages)
            except (TimeoutError, OpenAIError) as e:
                if attempt == 3:
                    raise
                logger.warning("Попытка %s не удалась: %s — ретраим через %.1fs", attempt, e, delay)
                await asyncio.sleep(delay)
                delay *= 2
        # теоретически недостижимо
        raise RuntimeError("Не удалось получить ответ от OpenAI")

    async def generate_legal_advice(
        self,
        user_question: str,
        short_history: Sequence[str] | None = None,
    ) -> LegalAdvice:
        """
        Запрашивает у модели юридическое разъяснение.

        :param user_question: вопрос пользователя
        :param short_history: список нескольких прошлых кратких ответов
                              (для лёгкого контекста; опционально)
        """
        history_block = ""
        if short_history:
            # Короткий контекст — помогает связности диалога, но не переутяжеляет запрос
            history_json = json.dumps(list(short_history), ensure_ascii=False)
            history_block = f"\nИстория последних ответов (кратко): {history_json}\n"

        user_msg = (
            "Вот юридический вопрос пользователя.\n"
            "Сформируй понятный и проверяемый ответ по установленному JSON-шаблону.\n"
            f"{history_block}\n"
            f"Вопрос: {user_question}"
        )

        raw = await self._ask_with_retries([{"role": "user", "content": user_msg}])

        # Строгий парс JSON (обрезаем случайный префикс/постфикс, если модель упрямится)
        try:
            # На всякий пожарный — находим первую { и последнюю }
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start : end + 1]
            data = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            logger.warning("Парсер JSON споткнулся, отладочная копия ответа: %r", raw[:500])
            raise ValueError("Модель вернула невалидный JSON") from e

        summary = str(data.get("summary") or "").strip() or "Краткий ответ не сформирован."
        details = str(data.get("details") or "").strip() or "Подробное объяснение не сформировано."
        laws = data.get("laws") or []
        laws = [str(x).strip() for x in laws if str(x).strip()]

        return LegalAdvice(summary=summary, details=details, laws=laws)
