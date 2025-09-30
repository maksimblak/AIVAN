from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from src.core.di_container import create_container
from src.core.payments import convert_rub_to_xtr
from src.core.runtime import AppRuntime, DerivedRuntime, WelcomeMedia
from src.core.settings import AppSettings


def _discover_welcome_media() -> WelcomeMedia | None:
    images_dir = Path(__file__).resolve().parents[2] / "images"
    if not images_dir.exists():
        return None
    for candidate in sorted(images_dir.iterdir()):
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix in {".mp4", ".mov", ".m4v", ".webm"}:
            return WelcomeMedia(candidate, "video")
        if suffix in {".gif"}:
            return WelcomeMedia(candidate, "animation")
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return WelcomeMedia(candidate, "photo")
    return None


def build_runtime(settings: AppSettings, *, logger: logging.Logger | None = None) -> Tuple[AppRuntime, object]:
    """Construct base runtime context and DI container."""
    logger = logger or logging.getLogger("ai-ivan.simple")

    derived = DerivedRuntime(
        welcome_media=_discover_welcome_media(),
        subscription_price_rub_kopeks=int(float(settings.subscription_price_rub) * 100),
        dynamic_price_xtr=convert_rub_to_xtr(
            amount_rub=float(settings.subscription_price_rub),
            rub_per_xtr=settings.rub_per_xtr,
            default_xtr=settings.subscription_price_xtr,
        ),
        admin_ids=set(settings.admin_ids),
    )

    container = create_container(settings)
    runtime = AppRuntime(settings=settings, logger=logger, derived=derived)

    from src.core.db_advanced import DatabaseAdvanced
    from src.core.access import AccessService
    from src.core.audio_service import AudioService
    from src.core.openai_service import OpenAIService
    from src.core.payments import CryptoPayProvider
    from src.core.session_store import SessionStore
    from src.telegram_legal_bot.ratelimit import RateLimiter

    runtime.db = container.get(DatabaseAdvanced)
    runtime.access_service = container.get(AccessService)
    runtime.audio_service = container.get(AudioService)
    runtime.openai_service = container.get(OpenAIService)
    runtime.rate_limiter = container.get(RateLimiter)
    runtime.session_store = container.get(SessionStore)
    runtime.crypto_provider = container.get(CryptoPayProvider)

    runtime.set_dependency("container", container)
    return runtime, container
