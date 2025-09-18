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
    Настройки приложения (из .env), с дефолтами.
    """
    telegram_token: str
    openai_api_key: str

    # GPT-5 + Responses API
    openai_model: str = "gpt-5"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 1500
    openai_verbosity: str = "medium"           # low|medium|high
    openai_reasoning_effort: str = "medium" # minimal|medium|high

    # Бот
    max_requests_per_hour: int = 10
    min_question_length: int = 20

    # Логи
    log_level: str = "INFO"
    json_logs: bool = False

    system_prompt: str = (
        "Ты — квалифицированный юрист-консультант. Отвечай на юридические вопросы "
        "четко и структурированно. Всегда указывай применимые нормы права "
        "(если уместно и известно). Предупреждай, что консультация носит "
        "информационный характер и не заменяет профессиональную юридическую помощь. "
        "Формат: краткий ответ, подробности, нормы, дисклеймер."
    )


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_settings() -> Settings:
    """Собирает конфиг из .env с адекватными фоллбэками."""
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not telegram_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY не задан в .env")

    # GPT-5 и генерация
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5").strip()
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))
    openai_verbosity = os.getenv("OPENAI_VERBOSITY", "low").strip().lower()
    openai_reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "medium").strip().lower()

    # Лимитер и валидация
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
        openai_verbosity=openai_verbosity,
        openai_reasoning_effort=openai_reasoning_effort,
        max_requests_per_hour=max_requests_per_hour,
        min_question_length=min_question_length,
        log_level=log_level,
        json_logs=json_logs,
        system_prompt=system_prompt,
    )
