from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _parse_int(val: str | None, default: int) -> int:
    try:
        return int(val) if val is not None and val != "" else default
    except ValueError:
        return default


def _parse_int_list(val: str | None) -> list[int]:
    if not val:
        return []
    out: list[int] = []
    for part in val.replace(" ", "").split(","):
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _parse_float(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


@dataclass
class Config:
    telegram_bot_token: str
    openai_api_key: str

    use_status_animation: bool

    db_path: str
    trial_requests: int
    sub_duration_days: int

    telegram_provider_token_rub: str
    subscription_price_rub: int
    telegram_provider_token_stars: str
    subscription_price_xtr: int

    admin_ids: list[int]

    user_sessions_max: int
    user_session_ttl_seconds: int

    redis_url: str | None
    rate_limit_requests: int
    rate_limit_window_seconds: int
    rub_per_xtr: float | None


_cached_config: Config | None = None


def load_config() -> Config:
    """Load and validate environment configuration once per process."""
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    load_dotenv()

    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    cfg = Config(
        telegram_bot_token=telegram_bot_token,
        openai_api_key=openai_api_key,
        use_status_animation=_bool(os.getenv("USE_STATUS_ANIMATION", "1"), True),
        db_path=os.getenv("DB_PATH", "src/core/data/bot.sqlite3"),
        trial_requests=_parse_int(os.getenv("TRIAL_REQUESTS", "10"), 10),
        sub_duration_days=_parse_int(os.getenv("SUB_DURATION_DAYS", "30"), 30),
        telegram_provider_token_rub=os.getenv("TELEGRAM_PROVIDER_TOKEN_RUB", "").strip(),
        subscription_price_rub=_parse_int(os.getenv("SUBSCRIPTION_PRICE_RUB", "300"), 300),
        telegram_provider_token_stars=os.getenv("TELEGRAM_PROVIDER_TOKEN_STARS", "STARS").strip(),
        subscription_price_xtr=_parse_int(os.getenv("SUBSCRIPTION_PRICE_XTR", "3000"), 3000),
        admin_ids=_parse_int_list(os.getenv("ADMIN_IDS", "")),
        user_sessions_max=_parse_int(os.getenv("USER_SESSIONS_MAX", "10000"), 10000),
        user_session_ttl_seconds=_parse_int(os.getenv("USER_SESSION_TTL_SECONDS", "3600"), 3600),
        redis_url=os.getenv("REDIS_URL", "").strip() or None,
        rate_limit_requests=_parse_int(os.getenv("RATE_LIMIT_REQUESTS", "10"), 10),
        rate_limit_window_seconds=_parse_int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"), 60),
        rub_per_xtr=_parse_float(os.getenv("RUB_PER_XTR")),
    )

    _cached_config = cfg
    return cfg
