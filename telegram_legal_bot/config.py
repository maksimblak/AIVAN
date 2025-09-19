from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


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
    Ищет .env в:
      • текущая рабочая директория
      • каталог этого файла
      • родитель каталога (корень проекта)
    Подставляет только те ключи, которых нет в os.environ.
    Формат: KEY=VALUE. Строки с # игнорируются.
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
            # best-effort
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

    # обязательные
    telegram_token: str
    openai_api_key: str

    # Telegram proxy (опционально)
    telegram_proxy_url: Optional[str] = None
    telegram_proxy_user: Optional[str] = None
    telegram_proxy_pass: Optional[str] = None

    # Кастомный Bot API сервер (self-hosted), напр. http://127.0.0.1:8081
    telegram_api_base: Optional[str] = None

    # OpenAI: базовые параметры генерации
    openai_model: str
    openai_temperature: float = 0.3
    openai_max_tokens: int = 1500
    openai_verbosity: str = "low"             # low|medium|high (для логгирования/уровня трассировки)
    openai_reasoning_effort: str = "medium"   # low|medium|high (оставлено для совместимости)

    # Доп. параметры генерации (под ask_ivan)
    top_p: float = 1.0
    seed: int = 7
    max_output_tokens: int = 1800              # специально под Responses API (может отличаться от openai_max_tokens)
    reasoning_effort: str = "medium"           # дубль значения openai_reasoning_effort (удобнее в коде)

    # OpenAI proxy (опционально)
    openai_proxy_url: Optional[str] = None
    openai_proxy_user: Optional[str] = None
    openai_proxy_pass: Optional[str] = None

    # App
    parse_mode: Optional[str] = "MarkdownV2"   # нормализуется в None при "none"/""/ "null"
    log_json: bool = True
    min_question_length: int = 20
    max_requests_per_hour: int = 10
    history_size: int = 5                      # число пар «вопрос-ответ» рекомендуется * 2 в deque

    # Поиск/инструменты
    search_domains: Optional[str] = None       # CSV: "kad.arbitr.ru,sudrf.ru,vsrf.ru,ksrf.ru,publication.pravo.gov.ru"
    web_search_recency_days: int = 3650
    web_search_max_results: int = 8
    file_search_enabled: bool = True
    tool_choice: str = "auto"              # "required" | "auto"
    web_search_enabled: bool = False


def load_settings() -> Settings:
    """Собирает настройки. Пытается подхватить .env, если переменных нет в окружении."""
    _maybe_load_dotenv()

    # Telegram token (поддержка альтернативных имён)
    tg_token = _env_first(
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_TOKEN",
        "BOT_TOKEN",
        "TG_BOT_TOKEN",
        "TELEGRAM_BOT_API_TOKEN",
    )
    # OpenAI key (поддержка альтернативных имён)
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

    # Telegram proxy / Bot API
    tg_proxy_url = os.getenv("TELEGRAM_PROXY_URL") or None
    tg_proxy_user = os.getenv("TELEGRAM_PROXY_USER") or None
    tg_proxy_pass = os.getenv("TELEGRAM_PROXY_PASS") or None
    tg_api_base = os.getenv("TELEGRAM_API_BASE") or os.getenv("BOT_API_BASE") or None

    # OpenAI base
    openai_model = (os.getenv("OPENAI_MODEL", "") or "").strip()
    temp = _get_float("OPENAI_TEMPERATURE", 0.3)
    oai_max_tokens = _get_int("OPENAI_MAX_TOKENS", 1500)
    verbosity = (os.getenv("OPENAI_VERBOSITY", "low") or "low").lower()
    effort = (os.getenv("OPENAI_REASONING_EFFORT", "medium") or "medium").lower()

    # Доп. генерация (Responses API)
    top_p = _get_float("TOP_P", 1.0)
    seed = _get_int("SEED", 7)
    max_output_tokens = _get_int("MAX_OUTPUT_TOKENS", 1800)
    reasoning_effort = (os.getenv("REASONING_EFFORT", effort) or effort).lower()  # синхронизируем с OPENAI_REASONING_EFFORT

    # OpenAI proxy
    oai_proxy_url = os.getenv("OPENAI_PROXY_URL") or None
    oai_proxy_user = os.getenv("OPENAI_PROXY_USER") or None
    oai_proxy_pass = os.getenv("OPENAI_PROXY_PASS") or None

    # App
    parse_mode_raw = os.getenv("PARSE_MODE", "MarkdownV2")
    parse_mode_norm = parse_mode_raw.strip().lower()
    parse_mode: Optional[str]
    if parse_mode_norm in {"none", "", "null"}:
        parse_mode = None
    else:
        parse_mode = parse_mode_raw

    log_json = _get_bool("LOG_JSON", True)
    min_len = _get_int("MIN_QUESTION_LENGTH", 20)
    max_per_hour = _get_int("MAX_REQUESTS_PER_HOUR", 10)
    history_size = _get_int("HISTORY_SIZE", 5)

    # Поиск/инструменты
    search_domains = os.getenv("SEARCH_DOMAINS") or None
    web_search_recency_days = _get_int("WEB_SEARCH_RECENCY_DAYS", 3650)
    web_search_max_results = _get_int("WEB_SEARCH_MAX_RESULTS", 8)
    file_search_enabled = _get_bool("FILE_SEARCH_ENABLED", True)
    tool_choice = (os.getenv("TOOL_CHOICE", "required") or "required").lower()

    # Валидации
    if not (0.0 <= temp <= 2.0):
        temp = 0.3
    if oai_max_tokens < 1:
        oai_max_tokens = 1500
    if max_output_tokens < 1:
        max_output_tokens = 1800
    if verbosity not in {"low", "medium", "high"}:
        verbosity = "low"
    if effort not in {"low", "medium", "high"}:
        effort = "medium"
    if reasoning_effort not in {"low", "medium", "high"}:
        reasoning_effort = effort
    if tool_choice not in {"required", "auto"}:
        tool_choice = "required"
    if not (0.0 < top_p <= 1.0):
        top_p = 1.0

    if not openai_model:
        raise RuntimeError(
            "OPENAI_MODEL не задан. Укажите доступную модель (например, 'gpt-4o', 'gpt-4.1-mini')."
        )

    return Settings(
        telegram_token=tg_token,
        openai_api_key=oaikey,

        telegram_proxy_url=tg_proxy_url,
        telegram_proxy_user=tg_proxy_user,
        telegram_proxy_pass=tg_proxy_pass,
        telegram_api_base=tg_api_base,

        openai_model=openai_model,
        openai_temperature=temp,
        openai_max_tokens=oai_max_tokens,
        openai_verbosity=verbosity,
        openai_reasoning_effort=effort,

        top_p=top_p,
        seed=seed,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,

        openai_proxy_url=oai_proxy_url,
        openai_proxy_user=oai_proxy_user,
        openai_proxy_pass=oai_proxy_pass,

        parse_mode=parse_mode,
        log_json=log_json,
        min_question_length=min_len,
        max_requests_per_hour=max_per_hour,
        history_size=history_size,

        search_domains=search_domains,
        web_search_recency_days=web_search_recency_days,
        web_search_max_results=web_search_max_results,
        file_search_enabled=file_search_enabled,
        tool_choice=tool_choice,
        web_search_enabled=_get_bool("WEB_SEARCH_ENABLED", False),
    )
