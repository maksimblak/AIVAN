from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    openai_api_key: str

    # GPT-5 + Responses API
    openai_model: str = "gpt-5"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 1500
    openai_verbosity: str = "low"           # low|medium|high
    openai_reasoning_effort: str = "medium" # minimal|medium|high

    # Бот
    max_requests_per_hour: int = 10
    min_question_length: int = 20

    telegram_proxy_url: str | None = None
    telegram_proxy_user: str | None = None
    telegram_proxy_pass: str | None = None

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
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    if not key:
        raise RuntimeError("OPENAI_API_KEY не задан в .env")

    model = os.getenv("OPENAI_MODEL", "gpt-5").strip()
    temp = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))
    verbosity = os.getenv("OPENAI_VERBOSITY", "low").strip().lower()
    effort = os.getenv("OPENAI_REASONING_EFFORT", "medium").strip().lower()

    try:
        max_per_hour = int(os.getenv("MAX_REQUESTS_PER_HOUR", "10"))
    except ValueError:
        logging.warning("Некорректное MAX_REQUESTS_PER_HOUR — используем 10")
        max_per_hour = 10

    min_len = int(os.getenv("MIN_QUESTION_LENGTH", "20"))
    log_level = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    json_logs = _get_bool("JSON_LOGS", False)
    system_prompt = os.getenv("SYSTEM_PROMPT") or Settings.system_prompt
    telegram_proxy_url = os.getenv("TELEGRAM_PROXY_URL") or None
    telegram_proxy_user = os.getenv("TELEGRAM_PROXY_USER") or None
    telegram_proxy_pass = os.getenv("TELEGRAM_PROXY_PASS") or None

    return Settings(
        telegram_token=token,
        openai_api_key=key,
        openai_model=model,
        openai_temperature=temp,
        openai_max_tokens=max_tokens,
        openai_verbosity=verbosity,
        openai_reasoning_effort=effort,
        max_requests_per_hour=max_per_hour,
        min_question_length=min_len,
        log_level=log_level,
        json_logs=json_logs,
        system_prompt=system_prompt,
        telegram_proxy_url=telegram_proxy_url,
        telegram_proxy_user=telegram_proxy_user,
        telegram_proxy_pass=telegram_proxy_pass,
    )
