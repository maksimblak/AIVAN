from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from src.core.di_container import create_container
from src.core.payments import convert_rub_to_xtr
from src.core.runtime import AppRuntime, DerivedRuntime, SubscriptionPlanPricing, WelcomeMedia
from src.core.subscription_plans import get_default_subscription_plans
from src.core.settings import AppSettings
from src.core.rag.judicial_rag import JudicialPracticeRAG
from src.core.garant_api import GarantAPIClient


def _discover_welcome_media(settings: AppSettings) -> WelcomeMedia | None:
    file_id = (settings.welcome_media_file_id or "").strip()
    if file_id:
        media_type = (settings.welcome_media_type or "video").strip().lower()
        if media_type not in {"video", "animation", "photo"}:
            media_type = "video"
        return WelcomeMedia(media_type=media_type, file_id=file_id)

    images_dir = Path(__file__).resolve().parents[2] / "images"
    if not images_dir.exists():
        return None
    for candidate in sorted(images_dir.iterdir()):
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower()
        if suffix in {".mp4", ".mov", ".m4v", ".webm"}:
            return WelcomeMedia(media_type="video", path=candidate)
        if suffix in {".gif"}:
            return WelcomeMedia(media_type="animation", path=candidate)
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return WelcomeMedia(media_type="photo", path=candidate)
    return None


def _calculate_plan_stars(price_rub: float, settings: AppSettings) -> int:
    """Convert RUB price to Telegram Stars using configured ratio."""
    default_xtr = None
    try:
        base_price = float(settings.subscription_price_rub)
        base_stars = float(settings.subscription_price_xtr)
        if base_price > 0 and base_stars > 0:
            default_xtr = int(round(price_rub * (base_stars / base_price)))
    except (TypeError, ValueError, ZeroDivisionError):
        default_xtr = settings.subscription_price_xtr

    return convert_rub_to_xtr(
        amount_rub=price_rub,
        rub_per_xtr=settings.rub_per_xtr,
        default_xtr=default_xtr,
    )


def build_runtime(settings: AppSettings, *, logger: logging.Logger | None = None) -> Tuple[AppRuntime, object]:
    """Construct base runtime context and DI container."""
    logger = logger or logging.getLogger("ai-ivan.simple")

    plan_catalog = get_default_subscription_plans()
    plan_infos = tuple(
        SubscriptionPlanPricing(
            plan=plan,
            price_rub_kopeks=plan.price_rub_kopeks,
            price_stars=_calculate_plan_stars(float(plan.price_rub), settings),
        )
        for plan in plan_catalog
    )
    plan_map = {info.plan.plan_id: info for info in plan_infos}
    default_plan = plan_map.get('base_1m') or (plan_infos[0] if plan_infos else None)

    fallback_price_rub_kopeks = int(float(settings.subscription_price_rub) * 100)
    fallback_price_stars = _calculate_plan_stars(float(settings.subscription_price_rub), settings)

    derived = DerivedRuntime(
        welcome_media=_discover_welcome_media(settings),
        subscription_price_rub_kopeks=(default_plan.price_rub_kopeks if default_plan else fallback_price_rub_kopeks),
        dynamic_price_xtr=(default_plan.price_stars if default_plan else fallback_price_stars),
        admin_ids=set(settings.admin_ids),
        subscription_plans=plan_infos,
        default_subscription_plan=default_plan,
        subscription_plan_map=plan_map,
    )

    container = create_container(settings)
    runtime = AppRuntime(settings=settings, logger=logger, derived=derived)

    from src.core.db_advanced import DatabaseAdvanced
    from src.core.access import AccessService
    from src.core.audio_service import AudioService
    from src.core.openai_service import OpenAIService
    from src.core.payments import CryptoPayProvider, RoboKassaProvider, YooKassaProvider
    from src.core.session_store import SessionStore
    from src.bot.ratelimit import RateLimiter

    runtime.db = container.get(DatabaseAdvanced)
    runtime.access_service = container.get(AccessService)
    runtime.audio_service = container.get(AudioService)
    runtime.openai_service = container.get(OpenAIService)
    runtime.rate_limiter = container.get(RateLimiter)
    runtime.session_store = container.get(SessionStore)
    runtime.crypto_provider = container.get(CryptoPayProvider)
    runtime.robokassa_provider = container.get(RoboKassaProvider)
    runtime.yookassa_provider = container.get(YooKassaProvider)

    try:
        rag_service = container.get(JudicialPracticeRAG)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judicial RAG service unavailable: %s", exc)
        rag_service = None
    runtime.set_dependency("judicial_rag", rag_service)

    garant_client = None
    if settings.garant_api_enabled:
        try:
            garant_client = container.get(GarantAPIClient)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Garant API client unavailable: %s", exc)
            garant_client = None
    runtime.set_dependency("garant_client", garant_client)

    runtime.set_dependency("container", container)
    return runtime, container
