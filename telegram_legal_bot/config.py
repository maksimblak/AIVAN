from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """
    Настройки приложения, читаемые из .env.

    Все значения безопасно парсятся и имеют дефолты,
    чтобы бот мог стартовать даже при частично заполненной конфигурации.
    """

    telegram_token: str
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 1500

    max_requests_per_hour: int = 10
    min_question_length: int = 20

    log_level: str = "INFO"
    json_logs: bool = False

    system_prompt: str = (
        "Ты — квалифицированный юрист-консультант. Отвечай на юридические вопросы "
        "четко и структурированно. Всегда указывай применимые нормы права "
        "(если они уместны и известны). Предупреждай, что консультация носит "
        "информационный характер и не заменяет профессиональную юридическую помощь. "
        "Форматируй ответ по шаблону: краткий ответ, подробности, нормы, дисклеймер."
    )


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings() -> Settings:
    """Безопасно собирает конфиг из .env с адекватными фоллбэками."""
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not telegram_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY не задан в .env")

    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))

    # Лимитер
    max_req_raw = os.getenv("MAX_REQUESTS_PER_HOUR", "10")
    try:
        max_requests_per_hour = int(max_req_raw)
    except ValueError:
        logging.warning(
            "Некорректное MAX_REQUESTS_PER_HOUR=%s — используем 10 по умолчанию", max_req_raw
        )
        max_requests_per_hour = 10

    min_question_length = int(os.getenv("MIN_QUESTION_LENGTH", "20"))

    # Логи
    log_level = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    json_logs = _get_bool("JSON_LOGS", False)

    system_prompt = os.getenv("SYSTEM_PROMPT") or Settings.system_prompt

    return Settings(
        telegram_token=telegram_token,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_temperature=openai_temperature,
        openai_max_tokens=openai_max_tokens,
        max_requests_per_hour=max_requests_per_hour,
        min_question_length=min_question_length,
        log_level=log_level,
        json_logs=json_logs,
        system_prompt=system_prompt,
    )
