from __future__ import annotations

import os
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AppSettings(BaseModel):
    """Centralised application configuration loaded from environment variables."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    raw_env: dict[str, str] = Field(default_factory=dict, exclude=True)

    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    telegram_support_username: str | None = Field(default=None, alias="TELEGRAM_SUPPORT_USERNAME")

    use_status_animation: bool = Field(default=True, alias="USE_STATUS_ANIMATION")
    use_streaming: bool = Field(default=True, alias="USE_STREAMING")
    welcome_media_file_id: str | None = Field(default=None, alias="WELCOME_MEDIA_FILE_ID")
    welcome_media_type: str | None = Field(default=None, alias="WELCOME_MEDIA_TYPE")

    db_path: str = Field(default="src/core/data/bot.sqlite3", alias="DB_PATH")
    db_max_connections: int = Field(default=5, alias="DB_MAX_CONNECTIONS")

    trial_requests: int = Field(default=10, alias="TRIAL_REQUESTS")
    sub_duration_days: int = Field(default=30, alias="SUB_DURATION_DAYS")

    telegram_provider_token_rub: str = Field(default="", alias="TELEGRAM_PROVIDER_TOKEN_RUB")
    subscription_price_rub: int = Field(default=300, alias="SUBSCRIPTION_PRICE_RUB")

    telegram_provider_token_stars: str = Field(default="STARS", alias="TELEGRAM_PROVIDER_TOKEN_STARS")
    subscription_price_xtr: int = Field(default=3000, alias="SUBSCRIPTION_PRICE_XTR")

    admin_ids: list[int] = Field(default_factory=list, alias="ADMIN_IDS")

    robokassa_merchant_login: str = Field(default="", alias="ROBOKASSA_MERCHANT_LOGIN")
    robokassa_password1: str = Field(default="", alias="ROBOKASSA_PASSWORD1")
    robokassa_password2: str = Field(default="", alias="ROBOKASSA_PASSWORD2")
    robokassa_is_test: bool = Field(default=False, alias="ROBOKASSA_IS_TEST")
    robokassa_result_url: str | None = Field(default=None, alias="ROBOKASSA_RESULT_URL")
    robokassa_success_url: str | None = Field(default=None, alias="ROBOKASSA_SUCCESS_URL")
    robokassa_fail_url: str | None = Field(default=None, alias="ROBOKASSA_FAIL_URL")

    yookassa_shop_id: str = Field(default="", alias="YOOKASSA_SHOP_ID")
    yookassa_secret_key: str = Field(default="", alias="YOOKASSA_SECRET_KEY")
    yookassa_return_url: str | None = Field(default=None, alias="YOOKASSA_RETURN_URL")
    yookassa_require_email: bool = Field(default=True, alias="YOOKASSA_REQUIRE_EMAIL")
    yookassa_require_phone: bool = Field(default=False, alias="YOOKASSA_REQUIRE_PHONE")
    yookassa_tax_system_code: int | None = Field(default=None, alias="YOOKASSA_TAX_SYSTEM_CODE")
    yookassa_vat_code: int | None = Field(default=None, alias="YOOKASSA_VAT_CODE")
    yookassa_payment_mode: str | None = Field(default="full_payment", alias="YOOKASSA_PAYMENT_MODE")
    yookassa_payment_subject: str | None = Field(default="service", alias="YOOKASSA_PAYMENT_SUBJECT")

    user_sessions_max: int = Field(default=10_000, alias="USER_SESSIONS_MAX")
    user_session_ttl_seconds: int = Field(default=3600, alias="USER_SESSION_TTL_SECONDS")

    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    rate_limit_requests: int = Field(default=10, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")
    rub_per_xtr: float | None = Field(default=None, alias="RUB_PER_XTR")

    voice_mode_enabled: bool = Field(default=False, alias="ENABLE_VOICE_MODE")
    voice_stt_model: str = Field(default="gpt-4o-mini-transcribe", alias="VOICE_STT_MODEL")
    voice_tts_model: str = Field(default="gpt-4o-mini-tts", alias="VOICE_TTS_MODEL")
    voice_tts_voice: str = Field(default="alloy", alias="VOICE_TTS_VOICE")
    voice_tts_voice_male: str | None = Field(default="verse", alias="VOICE_TTS_VOICE_MALE")
    voice_tts_format: str = Field(default="ogg", alias="VOICE_TTS_FORMAT")
    voice_tts_chunk_char_limit: int = Field(default=6000, alias="VOICE_TTS_CHUNK_CHAR_LIMIT")
    voice_tts_speed: float | None = Field(default=0.95, alias="VOICE_TTS_SPEED")
    voice_tts_style: str | None = Field(default="formal", alias="VOICE_TTS_STYLE")
    voice_tts_sample_rate: int | None = Field(default=None, alias="VOICE_TTS_SAMPLE_RATE")
    voice_tts_backend: str = Field(default="auto", alias="VOICE_TTS_BACKEND")
    voice_max_duration_seconds: int = Field(default=120, alias="VOICE_MAX_DURATION_SECONDS")

    telegram_proxy_url: str | None = Field(default=None, alias="TELEGRAM_PROXY_URL")
    telegram_proxy_user: str | None = Field(default=None, alias="TELEGRAM_PROXY_USER")
    telegram_proxy_pass: str | None = Field(default=None, alias="TELEGRAM_PROXY_PASS")

    enable_prometheus: bool = Field(default=True, alias="ENABLE_PROMETHEUS")
    prometheus_port: int | None = Field(default=None, alias="PROMETHEUS_PORT")

    cache_max_size: int = Field(default=1000, alias="CACHE_MAX_SIZE")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")

    garant_api_enabled: bool = Field(default=False, alias="GARANT_API_ENABLED")
    garant_api_base_url: str = Field(default="https://api.garant.ru", alias="GARANT_API_BASE_URL")
    garant_api_env: str = Field(default="internet", alias="GARANT_API_ENV")
    garant_api_token: str | None = Field(default=None, alias="GARANT_API_TOKEN")
    garant_api_timeout: float = Field(default=15.0, alias="GARANT_API_TIMEOUT")
    garant_api_result_limit: int = Field(default=3, alias="GARANT_API_RESULT_LIMIT")
    garant_api_snippet_limit: int = Field(default=2, alias="GARANT_API_SNIPPET_LIMIT")
    garant_document_base_url: str | None = Field(default="https://d.garant.ru", alias="GARANT_DOCUMENT_BASE_URL")
    garant_api_verify_ssl: bool = Field(default=True, alias="GARANT_API_VERIFY_SSL")
    garant_api_use_query_language: bool = Field(default=True, alias="GARANT_API_USE_QUERY_LANGUAGE")
    garant_api_sutyazhnik_enabled: bool = Field(
        default=True,
        alias="GARANT_API_SUTYAZHNIK_ENABLED",
    )
    garant_api_sutyazhnik_kinds: list[str] = Field(
        default_factory=lambda: ["301", "302"],
        alias="GARANT_API_SUTYAZHNIK_KINDS",
    )
    garant_api_sutyazhnik_count: int = Field(
        default=5,
        alias="GARANT_API_SUTYAZHNIK_COUNT",
    )
    cache_compression: bool = Field(default=True, alias="CACHE_COMPRESSION")

    crypto_asset: str = Field(default="USDT", alias="CRYPTO_ASSET")

    enable_scaling: bool = Field(default=False, alias="ENABLE_SCALING")
    heartbeat_interval: float = Field(default=15.0, alias="HEARTBEAT_INTERVAL")
    session_affinity_ttl: int = Field(default=3600, alias="SESSION_AFFINITY_TTL")

    health_check_interval: float = Field(default=30.0, alias="HEALTH_CHECK_INTERVAL")
    enable_system_monitoring: bool = Field(default=True, alias="ENABLE_SYSTEM_MONITORING")

    db_cleanup_interval: float = Field(default=3600.0, alias="DB_CLEANUP_INTERVAL")
    db_cleanup_days: int = Field(default=90, alias="DB_CLEANUP_DAYS")

    cache_cleanup_interval: float = Field(default=300.0, alias="CACHE_CLEANUP_INTERVAL")
    session_cleanup_interval: float = Field(default=600.0, alias="SESSION_CLEANUP_INTERVAL")
    health_check_task_interval: float = Field(default=120.0, alias="HEALTH_CHECK_TASK_INTERVAL")
    metrics_collection_interval: float = Field(default=30.0, alias="METRICS_COLLECTION_INTERVAL")

    document_storage_quota_mb: int | None = Field(default=None, alias="DOCUMENTS_STORAGE_QUOTA_MB")
    document_cleanup_hours: int = Field(default=24, alias="DOCUMENTS_CLEANUP_HOURS")
    document_cleanup_interval_seconds: float = Field(default=3600.0, alias="DOCUMENTS_CLEANUP_INTERVAL_SECONDS")
    documents_s3_bucket: str | None = Field(default=None, alias="DOCUMENTS_S3_BUCKET")
    documents_s3_prefix: str = Field(default="documents", alias="DOCUMENTS_S3_PREFIX")
    documents_s3_region: str | None = Field(default=None, alias="DOCUMENTS_S3_REGION")
    documents_s3_endpoint: str | None = Field(default=None, alias="DOCUMENTS_S3_ENDPOINT")
    documents_s3_public_url: str | None = Field(default=None, alias="DOCUMENTS_S3_PUBLIC_URL")
    documents_s3_acl: str | None = Field(default=None, alias="DOCUMENTS_S3_ACL")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")

    def get_str(self, key: str, default: str | None = None) -> str | None:
        return self.raw_env.get(key, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.raw_env.get(key)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def get_int(self, key: str, default: int = 0) -> int:
        value = self.raw_env.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        value = self.raw_env.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @field_validator("admin_ids", mode="before")
    @classmethod
    def _parse_admin_ids(cls, value: Any) -> list[int]:
        if value in (None, "", []):
            return []
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            parsed: list[int] = []
            for item in items:
                try:
                    parsed.append(int(item))
                except ValueError:
                    continue
            return parsed
        return [int(value)]

    @field_validator(
        "redis_url",
        "telegram_proxy_url",
        "telegram_proxy_user",
        "telegram_proxy_pass",
        "robokassa_result_url",
        "robokassa_success_url",
        "robokassa_fail_url",
        "yookassa_return_url",
        "yookassa_payment_mode",
        "yookassa_payment_subject",
        "welcome_media_file_id",
        "welcome_media_type",
        "garant_api_token",
        "garant_document_base_url",
        mode="before",
    )
    @classmethod
    def _empty_string_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator(
        "prometheus_port",
        "rub_per_xtr",
        "yookassa_tax_system_code",
        "yookassa_vat_code",
        mode="before",
    )
    @classmethod
    def _convert_optional_numbers(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        return value

    @staticmethod
    def _parse_code_list(value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
        elif isinstance(value, (list, tuple, set)):
            items = [str(item).strip() for item in value]
        else:
            items = [str(value).strip()]
        return [item for item in items if item]

    @field_validator("garant_api_env", mode="before")
    @classmethod
    def _normalize_garant_env(cls, value: Any) -> str:
        if value is None:
            return "internet"
        if isinstance(value, str):
            trimmed = value.strip().lower()
        else:
            trimmed = str(value).strip().lower()
        return trimmed or "internet"

    @field_validator("garant_api_sutyazhnik_kinds", mode="before")
    @classmethod
    def _parse_sutyazhnik_kinds(cls, value: Any) -> list[str]:
        return cls._parse_code_list(value)

    @model_validator(mode="after")
    def _fill_voice_defaults(self) -> "AppSettings":
        if not self.voice_tts_voice_male:
            self.voice_tts_voice_male = self.voice_tts_voice
        if self.voice_tts_speed is not None and self.voice_tts_speed <= 0:
            self.voice_tts_speed = None
        if self.voice_tts_style:
            stripped = self.voice_tts_style.strip()
            self.voice_tts_style = stripped or None
        if self.voice_tts_sample_rate is not None and self.voice_tts_sample_rate <= 0:
            self.voice_tts_sample_rate = None
        backend = (self.voice_tts_backend or "auto").strip().lower()
        if backend not in {"auto", "speech", "responses"}:
            backend = "auto"
        self.voice_tts_backend = backend

        self.garant_api_result_limit = max(1, int(self.garant_api_result_limit or 1))
        self.garant_api_snippet_limit = max(0, int(self.garant_api_snippet_limit or 0))
        self.garant_api_sutyazhnik_count = max(1, int(self.garant_api_sutyazhnik_count or 1))
        if not self.garant_api_base_url:
            self.garant_api_enabled = False
        else:
            trimmed = self.garant_api_base_url.strip()
            if trimmed.endswith("/"):
                trimmed = trimmed.rstrip("/")
            self.garant_api_base_url = trimmed or "https://api.garant.ru"

        env = (self.garant_api_env or "internet").strip().lower()
        if env not in {"internet", "arbitr"}:
            env = "internet"
        self.garant_api_env = env

        if self.garant_document_base_url:
            doc_base = self.garant_document_base_url.strip()
            if doc_base.endswith("/"):
                doc_base = doc_base.rstrip("/")
            self.garant_document_base_url = doc_base or None

        return self

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> "AppSettings":
        """Load settings from provided mapping or OS environment."""
        source: Mapping[str, str] = env or os.environ
        source_map = {str(key): str(value) for key, value in source.items()}
        settings = cls.model_validate(source_map)
        settings.raw_env = source_map
        return settings
