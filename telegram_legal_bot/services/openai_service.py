"""Сервис взаимодействия с OpenAI API."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

try:  # pragma: no cover - импорт зависит от версии SDK
    from openai import AsyncOpenAI, OpenAIError
except ImportError:  # pragma: no cover - совместимость с будущими версиями
    from openai import AsyncOpenAI  # type: ignore[assignment]
    from openai import APIError as OpenAIError  # type: ignore

from telegram_legal_bot.config import OpenAISettings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LegalAdvice:
    """Структурированный ответ модели."""

    summary: str
    details: str
    laws: List[str]


class OpenAIService:
    """Инкапсулирует логику работы с OpenAI GPT."""

    def __init__(self, settings: OpenAISettings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.api_key)

    async def generate_legal_advice(self, query: str) -> LegalAdvice:
        """Возвращает структурированный ответ на юридический вопрос."""

        logger.debug("Отправка запроса в OpenAI: %s", query)
        try:
            response = await self._client.responses.create(
                model=self._settings.model,
                temperature=self._settings.temperature,
                max_output_tokens=self._settings.max_tokens,
                response_format={"type": "json_object"},
                input=[
                    {"role": "system", "content": self._settings.system_prompt},
                    {"role": "user", "content": query},
                ],
            )
        except OpenAIError as exc:
            logger.exception("Ошибка OpenAI API")
            raise exc

        content = self._extract_text(response)
        logger.debug("Ответ OpenAI: %s", content)

        data = self._parse_response(content)
        return LegalAdvice(
            summary=data.get("summary", "Не удалось сформировать краткий ответ."),
            details=data.get("details", "Попробуйте задать вопрос иначе."),
            laws=data.get("laws", []),
        )

    @staticmethod
    def _parse_response(content: str) -> Dict[str, Any]:
        """Преобразует ответ модели в словарь."""

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Не удалось распарсить JSON. Возвращаем ответ в текстовом виде.")
            return {
                "summary": content.strip(),
                "details": content.strip(),
                "laws": [],
            }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Извлекает текст из ответа Responses API."""

        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        parts: List[str] = []
        output_blocks = getattr(response, "output", None)
        if not output_blocks:
            return ""

        for block in output_blocks:
            block_content = getattr(block, "content", None) or []
            for item in block_content:
                text_piece: str | None = None
                if isinstance(item, dict):
                    text_piece = item.get("text")
                else:
                    text_piece = getattr(item, "text", None)

                if text_piece:
                    parts.append(text_piece)

        return "".join(parts)
