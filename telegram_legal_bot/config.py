"""Модуль конфигурации бота."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenAISettings:
    """Настройки доступа к OpenAI API."""

    api_key: str
    model: str
    temperature: float = 0.3
    max_tokens: int = 1500
    system_prompt: str = (
        "Ты - квалифицированный юрист-консультант. Отвечай на юридические вопросы "
        "четко и структурированно. Всегда указывай применимые нормы права. "
        "Предупреждай, что консультация носит информационный характер и не заменяет "
        "профессиональную юридическую помощь. Отвечай в формате JSON с ключами: "
        "summary (краткий ответ), details (подробное разъяснение), laws (список ссылок на законы)."
    )


@dataclass(slots=True)
class BotSettings:
    """Настройки телеграм-бота."""

    token: str
    max_requests_per_hour: int = 10
    min_question_length: int = 20


@dataclass(slots=True)
class Settings:
    """Объединённая конфигурация приложения."""

    bot: BotSettings
    openai: OpenAISettings


def load_settings(env_file: Optional[str] = None) -> Settings:
    """Загружает настройки из переменных окружения."""

    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    max_requests_raw = os.getenv("MAX_REQUESTS_PER_HOUR", "10")

    if not token:
        logger.error("Переменная TELEGRAM_BOT_TOKEN не найдена")
        raise ValueError("Необходимо указать TELEGRAM_BOT_TOKEN в переменных окружения")

    if not api_key:
        logger.error("Переменная OPENAI_API_KEY не найдена")
        raise ValueError("Необходимо указать OPENAI_API_KEY в переменных окружения")

    try:
        max_requests = int(max_requests_raw)
    except ValueError as exc:  # pragma: no cover - защитный код
        logger.warning(
            "Некорректное значение MAX_REQUESTS_PER_HOUR=%s, используется значение по умолчанию", max_requests_raw
        )
        raise ValueError("MAX_REQUESTS_PER_HOUR должно быть целым числом") from exc

    bot_settings = BotSettings(token=token, max_requests_per_hour=max_requests)
    openai_settings = OpenAISettings(api_key=api_key, model=model)

    logger.debug("Настройки успешно загружены")
    return Settings(bot=bot_settings, openai=openai_settings)
