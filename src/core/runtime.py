from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

from src.core.settings import AppSettings
from src.core.subscription_plans import SubscriptionPlan

if False:  # pragma: no cover - hints only
    from core.bot_app.ratelimit import RateLimiter
    from core.bot_app.stream_manager import StreamManager
    from src.core.access import AccessService
    from src.core.audio_service import AudioService
    from src.core.background_tasks import BackgroundTaskManager
    from src.core.cache import ResponseCache
    from src.core.db_advanced import DatabaseAdvanced
    from src.core.exceptions import ErrorHandler
    from src.core.health import HealthChecker
    from src.core.metrics import MetricsCollector
    from src.core.openai_service import OpenAIService
    from src.core.payments import CryptoPayProvider, RoboKassaProvider, YooKassaProvider
    from src.core.session_store import SessionStore
    from src.documents.document_manager import DocumentManager


@dataclass(frozen=True)
class WelcomeMedia:
    media_type: str
    path: Path | None = None
    file_id: str | None = None


@dataclass(frozen=True)
class SubscriptionPlanPricing:
    """Runtime view of a subscription plan with calculated prices."""

    plan: SubscriptionPlan
    price_rub_kopeks: int
    price_stars: int


@dataclass
class DerivedRuntime:
    welcome_media: WelcomeMedia | None
    subscription_price_rub_kopeks: int
    dynamic_price_xtr: int
    admin_ids: set[int]
    subscription_plans: Tuple[SubscriptionPlanPricing, ...] = ()
    default_subscription_plan: SubscriptionPlanPricing | None = None
    subscription_plan_map: dict[str, SubscriptionPlanPricing] = field(default_factory=dict)
    max_message_length: int = 4000
    safe_limit: int = 3900
    models: Dict[str, str] = field(default_factory=dict)


@dataclass
class AppRuntime:
    settings: AppSettings
    logger: logging.Logger
    derived: DerivedRuntime
    db: "DatabaseAdvanced" | None = None
    rate_limiter: "RateLimiter" | None = None
    access_service: "AccessService" | None = None
    openai_service: "OpenAIService" | None = None
    audio_service: "AudioService" | None = None
    session_store: "SessionStore" | None = None
    crypto_provider: "CryptoPayProvider" | None = None
    robokassa_provider: "RoboKassaProvider" | None = None
    yookassa_provider: "YooKassaProvider" | None = None
    error_handler: "ErrorHandler" | None = None
    document_manager: "DocumentManager" | None = None
    response_cache: "ResponseCache" | None = None
    stream_manager: "StreamManager" | None = None
    metrics_collector: "MetricsCollector" | None = None
    task_manager: "BackgroundTaskManager" | None = None
    health_checker: "HealthChecker" | None = None
    scaling_components: Dict[str, Any] | None = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def set_dependency(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            self.extras[name] = value

    def get_dependency(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        return self.extras.get(name)
