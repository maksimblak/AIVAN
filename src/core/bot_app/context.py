from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.app_context import set_settings
from src.core.runtime import AppRuntime, DerivedRuntime, SubscriptionPlanPricing, WelcomeMedia
from src.core.settings import AppSettings

if TYPE_CHECKING:
    from src.bot.ratelimit import RateLimiter
    from src.bot.stream_manager import StreamManager
    from src.core.access import AccessService
    from src.core.audio_service import AudioService
    from src.core.background_tasks import BackgroundTaskManager
    from src.core.db_advanced import DatabaseAdvanced
    from src.core.exceptions import ErrorHandler
    from src.core.health import HealthChecker
    from src.core.metrics import MetricsCollector
    from src.core.openai_service import OpenAIService
    from src.core.payments import CryptoPayProvider, RoboKassaProvider, YooKassaProvider
    from src.core.cache import ResponseCache
    from src.core.session_store import SessionStore

    ServiceRegistry = Any
    ResponseCacheType = ResponseCache
else:
    RateLimiter = Any  # type: ignore[assignment]
    StreamManager = Any  # type: ignore[assignment]
    AccessService = Any  # type: ignore[assignment]
    AudioService = Any  # type: ignore[assignment]
    BackgroundTaskManager = Any  # type: ignore[assignment]
    DatabaseAdvanced = Any  # type: ignore[assignment]
    ErrorHandler = Any  # type: ignore[assignment]
    HealthChecker = Any  # type: ignore[assignment]
    MetricsCollector = Any  # type: ignore[assignment]
    OpenAIService = Any  # type: ignore[assignment]
    CryptoPayProvider = Any  # type: ignore[assignment]
    RoboKassaProvider = Any  # type: ignore[assignment]
    YooKassaProvider = Any  # type: ignore[assignment]
    SessionStore = Any  # type: ignore[assignment]
    ServiceRegistry = Any
    ResponseCacheType = Any

__all__ = [
    "set_runtime",
    "get_runtime",
    "refresh_runtime_globals",
    "settings",
    "derived",
    "WELCOME_MEDIA",
    "BOT_TOKEN",
    "BOT_USERNAME",
    "SUPPORT_USERNAME",
    "USE_ANIMATION",
    "USE_STREAMING",
    "SAFE_LIMIT",
    "MAX_MESSAGE_LENGTH",
    "DB_PATH",
    "TRIAL_REQUESTS",
    "SUB_DURATION_DAYS",
    "RUB_PROVIDER_TOKEN",
    "SUB_PRICE_RUB",
    "SUB_PRICE_RUB_KOPEKS",
    "STARS_PROVIDER_TOKEN",
    "SUB_PRICE_XTR",
    "DYNAMIC_PRICE_XTR",
    "SUBSCRIPTION_PLANS",
    "SUBSCRIPTION_PLAN_MAP",
    "DEFAULT_SUBSCRIPTION_PLAN",
    "ADMIN_IDS",
    "USER_SESSIONS_MAX",
    "USER_SESSION_TTL_SECONDS",
    "db",
    "rate_limiter",
    "access_service",
    "openai_service",
    "audio_service",
    "session_store",
    "crypto_provider",
    "robokassa_provider",
    "yookassa_provider",
    "error_handler",
    "document_manager",
    "response_cache",
    "stream_manager",
    "metrics_collector",
    "task_manager",
    "health_checker",
    "scaling_components",
    "judicial_rag",
    "garant_client",
]

_runtime: AppRuntime | None = None

WELCOME_MEDIA: WelcomeMedia | None = None
BOT_TOKEN = ""
BOT_USERNAME = ""
SUPPORT_USERNAME = ""
USE_ANIMATION = True
USE_STREAMING = True
SAFE_LIMIT = 3900
MAX_MESSAGE_LENGTH = 4000
DB_PATH = ""
TRIAL_REQUESTS = 0
SUB_DURATION_DAYS = 0
RUB_PROVIDER_TOKEN = ""
SUB_PRICE_RUB = 0
SUB_PRICE_RUB_KOPEKS = 0
STARS_PROVIDER_TOKEN = ""
SUB_PRICE_XTR = 0
DYNAMIC_PRICE_XTR = 0
SUBSCRIPTION_PLANS: tuple[SubscriptionPlanPricing, ...] = ()
SUBSCRIPTION_PLAN_MAP: dict[str, SubscriptionPlanPricing] = {}
DEFAULT_SUBSCRIPTION_PLAN: SubscriptionPlanPricing | None = None
ADMIN_IDS: set[int] = set()
USER_SESSIONS_MAX = 0
USER_SESSION_TTL_SECONDS = 0

db: DatabaseAdvanced | None = None
rate_limiter: RateLimiter | None = None
access_service: AccessService | None = None
openai_service: OpenAIService | None = None
audio_service: AudioService | None = None
session_store: SessionStore | None = None
crypto_provider: CryptoPayProvider | None = None
robokassa_provider: RoboKassaProvider | None = None
yookassa_provider: YooKassaProvider | None = None
error_handler: ErrorHandler | None = None
document_manager: Any | None = None
response_cache: ResponseCacheType | None = None
stream_manager: StreamManager | None = None
metrics_collector: MetricsCollector | None = None
task_manager: BackgroundTaskManager | None = None
health_checker: HealthChecker | None = None
scaling_components: dict[str, Any] | None = None
judicial_rag: Any | None = None
garant_client: Any | None = None


def set_runtime(runtime: AppRuntime) -> None:
    """Store runtime instance and propagate settings to dependent globals."""
    global _runtime
    _runtime = runtime
    set_settings(runtime.settings)
    _sync_runtime_globals()


def get_runtime() -> AppRuntime:
    if _runtime is None:
        raise RuntimeError("Application runtime is not initialized")
    return _runtime


def settings() -> AppSettings:
    return get_runtime().settings


def derived() -> DerivedRuntime:
    return get_runtime().derived


def refresh_runtime_globals() -> None:
    _sync_runtime_globals()


def _sync_runtime_globals() -> None:
    if _runtime is None:
        return

    cfg = _runtime.settings
    drv = _runtime.derived

    global WELCOME_MEDIA, BOT_TOKEN, BOT_USERNAME, SUPPORT_USERNAME, USE_ANIMATION, USE_STREAMING
    global SAFE_LIMIT, MAX_MESSAGE_LENGTH, DB_PATH, TRIAL_REQUESTS, SUB_DURATION_DAYS
    global RUB_PROVIDER_TOKEN, SUB_PRICE_RUB, SUB_PRICE_RUB_KOPEKS
    global STARS_PROVIDER_TOKEN, SUB_PRICE_XTR, DYNAMIC_PRICE_XTR
    global SUBSCRIPTION_PLANS, SUBSCRIPTION_PLAN_MAP, DEFAULT_SUBSCRIPTION_PLAN
    global ADMIN_IDS, USER_SESSIONS_MAX, USER_SESSION_TTL_SECONDS
    global db, rate_limiter, access_service, openai_service, audio_service
    global session_store, crypto_provider, robokassa_provider, yookassa_provider
    global error_handler, document_manager, response_cache, stream_manager
    global metrics_collector, task_manager, health_checker, scaling_components
    global judicial_rag, garant_client

    WELCOME_MEDIA = drv.welcome_media
    BOT_TOKEN = cfg.telegram_bot_token
    username_attr = getattr(cfg, "telegram_bot_username", "")
    BOT_USERNAME = (username_attr or "").strip()
    support_attr = getattr(cfg, "telegram_support_username", "")
    SUPPORT_USERNAME = (support_attr or "").strip()
    USE_ANIMATION = cfg.use_status_animation
    USE_STREAMING = cfg.use_streaming
    SAFE_LIMIT = drv.safe_limit
    MAX_MESSAGE_LENGTH = drv.max_message_length
    DB_PATH = cfg.db_path
    TRIAL_REQUESTS = cfg.trial_requests
    SUB_DURATION_DAYS = cfg.sub_duration_days
    RUB_PROVIDER_TOKEN = cfg.telegram_provider_token_rub
    SUB_PRICE_RUB = cfg.subscription_price_rub
    SUB_PRICE_RUB_KOPEKS = drv.subscription_price_rub_kopeks
    STARS_PROVIDER_TOKEN = cfg.telegram_provider_token_stars
    SUB_PRICE_XTR = cfg.subscription_price_xtr
    DYNAMIC_PRICE_XTR = drv.dynamic_price_xtr
    SUBSCRIPTION_PLANS = drv.subscription_plans
    SUBSCRIPTION_PLAN_MAP = drv.subscription_plan_map
    DEFAULT_SUBSCRIPTION_PLAN = drv.default_subscription_plan
    ADMIN_IDS = drv.admin_ids
    USER_SESSIONS_MAX = cfg.user_sessions_max
    USER_SESSION_TTL_SECONDS = cfg.user_session_ttl_seconds

    db = _runtime.db
    rate_limiter = _runtime.rate_limiter
    access_service = _runtime.access_service
    openai_service = _runtime.openai_service
    audio_service = _runtime.audio_service
    session_store = _runtime.session_store
    crypto_provider = _runtime.crypto_provider
    robokassa_provider = _runtime.robokassa_provider
    yookassa_provider = _runtime.yookassa_provider
    error_handler = _runtime.error_handler
    document_manager = _runtime.document_manager
    response_cache = _runtime.response_cache
    stream_manager = _runtime.stream_manager
    metrics_collector = _runtime.metrics_collector
    task_manager = _runtime.task_manager
    health_checker = _runtime.health_checker
    scaling_components = _runtime.scaling_components
    judicial_rag = _runtime.get_dependency("judicial_rag")
    garant_client = _runtime.get_dependency("garant_client")
