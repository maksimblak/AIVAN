from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from openai import AsyncOpenAI
from openai._exceptions import OpenAIError

from config import Settings

logger = logging.getLogger(__name__)


@dataclass
class LegalAdvice:
    """Нормализованный ответ от модели."""
    summary: str
    details: str
    laws: list[str]


class OpenAIService:
    """
    Обёртка над OpenAI Responses API для GPT-5:
      - системный промпт как developer message
      - JSON-ответ (жёсткое требование в инструкциях)
      - timeout + ретраи с экспоненциальным бэкоффом
      - управление стилем через text.verbosity и reasoning.effort
    """

    def __init__(self, settings: Settings):
        self._s = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    # ------------------------- публичный метод -------------------------
    async def generate_legal_advice(
        self,
        user_question: str,
        short_history: Sequence[str] | None = None,
    ) -> LegalAdvice:
        history_block = ""
        if short_history:
            history_json = json.dumps(list(short_history), ensure_ascii=False)
            history_block = f"\nИстория последних ответов (кратко): {history_json}\n"

        user_msg = (
            "Вот юридический вопрос пользователя.\n"
            "Сформируй ответ **строго в формате JSON** со следующими полями:\n"
            '{ "summary": "Краткий ответ (2-3 предложения)", '
            '"details": "Развернутое объяснение", '
            '"laws": ["Норма/ссылка 1", "Норма/ссылка 2"] }\n'
            "Никакого текста вне JSON не добавляй.\n"
            f"{history_block}\n"
            f"Вопрос: {user_question}"
        )

        raw = await self._ask_with_retries(user_msg)

        # Жёсткий парс JSON (на случай, если вокруг прилетит мусор)
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start : end + 1]
            data = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            logger.warning("JSON-парсер споткнулся. Фрагмент ответа: %r", raw[:600])
            raise ValueError("Модель вернула невалидный JSON") from e

        summary = (data.get("summary") or "").strip() or "Краткий ответ не сформирован."
        details = (data.get("details") or "").strip() or "Подробное объяснение не сформировано."
        laws = [str(x).strip() for x in (data.get("laws") or []) if str(x).strip()]
        return LegalAdvice(summary=summary, details=details, laws=laws)

    # ------------------------- внутренности ---------------------------
    async def _ask_with_retries(self, user_msg: str) -> str:
        delay = 1.5
        for attempt in range(1, 4):
            try:
                return await self._ask_once(user_msg)
            except (asyncio.TimeoutError, OpenAIError) as e:
                if attempt == 3:
                    raise
                logger.warning("Попытка %s не удалась: %s — ретраим через %.1fs", attempt, e, delay)
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError("Не удалось получить ответ от OpenAI")

    async def _ask_once(self, user_msg: str) -> str:
        """
        Один запрос к Responses API (GPT-5) с таймаутом.
        Управляем лаконичностью/скоростью через text.verbosity (low/med/high)
        и глубиной размышления через reasoning.effort (minimal/medium/high).
        См. гайд по GPT-5. :contentReference[oaicite:2]{index=2}
        """
        try:
            resp = await asyncio.wait_for(
                self._client.responses.create(
                    model=self._s.openai_model,  # "gpt-5"
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

        # Достаём текст надёжно
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        # Фоллбек: собрать всё текстовое из output[*].content[*].text
        parts: list[str] = []
        for block in getattr(resp, "output", []) or []:
            for item in getattr(block, "content", []) or []:
                t = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if t:
                    parts.append(str(t))
        return "".join(parts).strip()
