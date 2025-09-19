from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ── утилиты чтения ENV ────────────────────────────────────────────────────────

def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on", "y"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _maybe_load_dotenv() -> None:
    """
    Лёгкий .env-лоадер без зависимостей.
    Ищем .env в (по порядку):
      • текущей рабочей директории
      • каталоге этого файла (…/telegram_legal_bot)
      • его родителе (корень проекта)
    Подставляем только те ключи, которых нет в os.environ.
    Формат строк: KEY=VALUE, строки с # — комментарии.
    """
    candidates: Iterable[Path] = {
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    }
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        except Exception:
            # best-effort: тихо игнорим
            pass


def _env_first(*names: str, default: str = "") -> str:
    """Возвращает первое непустое значение из списка имён ENV."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v.strip()
    return default


# ── модель настроек ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """
    Конфигурация бота и LLM. Значения берутся из ENV (+ .env, если найден).

    Обязательные:
      - TELEGRAM_BOT_TOKEN (или TELEGRAM_TOKEN / BOT_TOKEN / TG_BOT_TOKEN / TELEGRAM_BOT_API_TOKEN)
      - OPENAI_API_KEY     (или OPENAI_KEY / OPENAI_TOKEN)
    """

    # обязательные (в начале!)
    telegram_token: str
    openai_api_key: str

    # Telegram proxy (опционально)
    telegram_proxy_url: str | None = None
    telegram_proxy_user: str | None = None
    telegram_proxy_pass: str | None = None

    # Кастомный Bot API сервер (self-hosted), напр. http://127.0.0.1:8081
    telegram_api_base: str | None = None

    # OpenAI базовые настройки
    openai_model: str = "gpt-5"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 1500
    openai_verbosity: str = "low"            # low|medium|high
    openai_reasoning_effort: str = "medium"  # low|medium|high

    # OpenAI proxy (опционально)
    openai_proxy_url: str | None = None
    openai_proxy_user: str | None = None
    openai_proxy_pass: str | None = None

    # App
    parse_mode: str = "MarkdownV2"
    log_json: bool = True
    min_question_length: int = 20
    max_requests_per_hour: int = 10
    history_size: int = 5


def load_settings() -> Settings:
    """Собирает настройки. Пытается подхватить .env, если переменных нет в окружении."""
    _maybe_load_dotenv()

    # --- Telegram token: поддерживаем альтернативные имена
    tg_token = _env_first(
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_TOKEN",
        "BOT_TOKEN",
        "TG_BOT_TOKEN",
        "TELEGRAM_BOT_API_TOKEN",
    )
    # --- OpenAI key: поддерживаем альтернативные имена
    oaikey = _env_first(
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "OPENAI_TOKEN",
    )

    if not tg_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is required "
            "(также поддерж.: TELEGRAM_TOKEN / BOT_TOKEN / TG_BOT_TOKEN / TELEGRAM_BOT_API_TOKEN). "
            "Проверь .env или переменные окружения процесса."
        )
    if not oaikey:
        raise RuntimeError(
            "OPENAI_API_KEY is required "
            "(также поддерж.: OPENAI_KEY / OPENAI_TOKEN). "
            "Проверь .env или переменные окружения процесса."
        )

    # Telegram proxy
    tg_proxy_url = os.getenv("TELEGRAM_PROXY_URL") or None
    tg_proxy_user = os.getenv("TELEGRAM_PROXY_USER") or None
    tg_proxy_pass = os.getenv("TELEGRAM_PROXY_PASS") or None

    # Кастомный Bot API сервер (если используешь self-hosted)
    tg_api_base = os.getenv("TELEGRAM_API_BASE") or os.getenv("BOT_API_BASE") or None

    # OpenAI base
    model = os.getenv("OPENAI_MODEL", "gpt-5")
    temp = _get_float("OPENAI_TEMPERATURE", 0.3)
    max_tokens = _get_int("OPENAI_MAX_TOKENS", 1500)
    verbosity = (os.getenv("OPENAI_VERBOSITY", "low") or "low").lower()
    effort = (os.getenv("OPENAI_REASONING_EFFORT", "medium") or "medium").lower()

    # OpenAI proxy
    oai_proxy_url = os.getenv("OPENAI_PROXY_URL") or None
    oai_proxy_user = os.getenv("OPENAI_PROXY_USER") or None
    oai_proxy_pass = os.getenv("OPENAI_PROXY_PASS") or None

    # App
    parse_mode = os.getenv("PARSE_MODE", "MarkdownV2")
    log_json = _get_bool("LOG_JSON", True)
    min_len = _get_int("MIN_QUESTION_LENGTH", 20)
    max_per_hour = _get_int("MAX_REQUESTS_PER_HOUR", 10)
    history_size = _get_int("HISTORY_SIZE", 5)

    # Лёгкие валидации
    if not (0.0 <= temp <= 2.0):
        temp = 0.3
    if max_tokens < 1:
        max_tokens = 1500
    if verbosity not in {"low", "medium", "high"}:
        verbosity = "low"
    if effort not in {"low", "medium", "high"}:
        effort = "medium"
    if parse_mode not in {"MarkdownV2", "HTML", "Markdown", "None"}:
        parse_mode = "MarkdownV2"

    return Settings(
        telegram_token=tg_token,
        openai_api_key=oaikey,
        telegram_proxy_url=tg_proxy_url,
        telegram_proxy_user=tg_proxy_user,
        telegram_proxy_pass=tg_proxy_pass,
        telegram_api_base=tg_api_base,
        openai_model=model,
        openai_temperature=temp,
        openai_max_tokens=max_tokens,
        openai_verbosity=verbosity,
        openai_reasoning_effort=effort,
        openai_proxy_url=oai_proxy_url,
        openai_proxy_user=oai_proxy_user,
        openai_proxy_pass=oai_proxy_pass,
        parse_mode=parse_mode,
        log_json=log_json,
        min_question_length=min_len,
        max_requests_per_hour=max_per_hour,
        history_size=history_size,
    )
