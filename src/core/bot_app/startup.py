from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from aiogram import Bot, Dispatcher

from core.bot_app.ratelimit import RateLimiter
from core.bot_app.retention_notifier import RetentionNotifier
from src.core.access import AccessService
from src.core.admin_modules.admin_commands import setup_admin_commands
from src.core.audio_service import AudioService
from src.core.background_tasks import (
    BackgroundTaskManager,
    CacheCleanupTask,
    DatabaseCleanupTask,
    DocumentStorageCleanupTask,
    HealthCheckTask,
    MetricsCollectionTask,
    SessionCleanupTask,
)
from src.core.bot_app import context as simple_context
from src.core.cache import ResponseCache, create_cache_backend
from src.core.db_advanced import DatabaseAdvanced
from src.core.exceptions import ErrorHandler, ErrorType
from src.core.health import (
    DatabaseHealthCheck,
    HealthChecker,
    OpenAIHealthCheck,
    RateLimiterHealthCheck,
    SessionStoreHealthCheck,
    SystemResourcesHealthCheck,
)
from src.core.metrics import init_metrics, set_system_status
from src.core.middlewares.error_middleware import ErrorHandlingMiddleware
from src.core.openai_service import OpenAIService
from src.core.payments import CryptoPayProvider
from src.core.scaling import LoadBalancer, ScalingManager, ServiceRegistry, SessionAffinity
from src.core.session_store import SessionStore
from src.documents.document_manager import DocumentManager

__all__ = ["RuntimeBundle", "maybe_call", "setup_bot_runtime"]


@dataclass(slots=True)
class RuntimeBundle:
    metrics_collector: Any
    cache_backend: Any
    response_cache: ResponseCache
    db: DatabaseAdvanced
    rate_limiter: RateLimiter
    access_service: AccessService
    openai_service: OpenAIService
    audio_service: AudioService | None
    session_store: SessionStore
    crypto_provider: CryptoPayProvider
    error_handler: ErrorHandler
    document_manager: DocumentManager
    scaling_components: Optional[dict[str, Any]]
    health_checker: HealthChecker
    task_manager: BackgroundTaskManager
    retention_notifier: RetentionNotifier


async def maybe_call(coro_or_func: Any) -> Any:
    """Call sync/async callables uniformly."""
    if coro_or_func is None:
        return None
    try:
        result = coro_or_func()
    except TypeError:
        result = coro_or_func
    if asyncio.iscoroutine(result):
        return await result
    return result


async def setup_bot_runtime(
    *,
    dispatcher: Dispatcher,
    bot: Bot,
    ctx: Any,
    cfg: Any,
    container: Any,
    logger: logging.Logger,
    admin_ids: set[int],
) -> RuntimeBundle:
    """Initialize runtime services and background tasks for the simple bot."""
    prometheus_port = cfg.prometheus_port
    metrics_collector = init_metrics(
        enable_prometheus=cfg.enable_prometheus,
        prometheus_port=prometheus_port,
    )
    ctx.metrics_collector = metrics_collector
    simple_context.metrics_collector = metrics_collector
    set_system_status("starting")

    logger.info("🚀 Starting AI-Ivan (simple)")

    # Database
    logger.info("Using advanced database with connection pooling")
    db = ctx.db or container.get(DatabaseAdvanced)
    ctx.db = db
    simple_context.db = db
    await db.init()

    setup_admin_commands(dispatcher, db, admin_ids)

    cache_backend = await create_cache_backend(
        redis_url=cfg.redis_url,
        fallback_to_memory=True,
        memory_max_size=cfg.cache_max_size,
    )

    response_cache = ResponseCache(
        backend=cache_backend,
        default_ttl=cfg.cache_ttl,
        enable_compression=cfg.cache_compression,
    )
    ctx.response_cache = response_cache
    simple_context.response_cache = response_cache

    rate_limiter: RateLimiter = ctx.rate_limiter or container.get(RateLimiter)
    ctx.rate_limiter = rate_limiter
    simple_context.rate_limiter = rate_limiter
    await rate_limiter.init()

    access_service: AccessService = ctx.access_service or container.get(AccessService)
    ctx.access_service = access_service
    simple_context.access_service = access_service

    openai_service: OpenAIService = ctx.openai_service or container.get(OpenAIService)
    openai_service.cache = response_cache
    ctx.openai_service = openai_service
    simple_context.openai_service = openai_service

    if cfg.voice_mode_enabled:
        audio_service = AudioService(
            stt_model=cfg.voice_stt_model,
            tts_model=cfg.voice_tts_model,
            tts_voice=cfg.voice_tts_voice,
            tts_format=cfg.voice_tts_format,
            max_duration_seconds=cfg.voice_max_duration_seconds,
            tts_voice_male=cfg.voice_tts_voice_male,
            tts_chunk_char_limit=cfg.voice_tts_chunk_char_limit,
            tts_speed=cfg.voice_tts_speed,
            tts_style=cfg.voice_tts_style,
            tts_sample_rate=cfg.voice_tts_sample_rate,
            tts_backend=cfg.voice_tts_backend,
        )
        ctx.audio_service = audio_service
        simple_context.audio_service = audio_service
        logger.info(
            "Voice mode enabled (stt=%s, tts=%s, voice=%s, male_voice=%s, format=%s, chunk_limit=%s)",
            cfg.voice_stt_model,
            cfg.voice_tts_model,
            cfg.voice_tts_voice,
            cfg.voice_tts_voice_male,
            cfg.voice_tts_format,
            cfg.voice_tts_chunk_char_limit,
        )
    else:
        audio_service = None
        ctx.audio_service = None
        simple_context.audio_service = None
        logger.info("Voice mode disabled")

    session_store: SessionStore = ctx.session_store or container.get(SessionStore)
    ctx.session_store = session_store
    simple_context.session_store = session_store

    crypto_provider: CryptoPayProvider = ctx.crypto_provider or container.get(CryptoPayProvider)
    ctx.crypto_provider = crypto_provider
    simple_context.crypto_provider = crypto_provider

    error_handler = ErrorHandler(logger=logger)
    ctx.error_handler = error_handler
    simple_context.error_handler = error_handler

    dispatcher.update.middleware(ErrorHandlingMiddleware(error_handler, logger=logger))

    document_manager = DocumentManager(openai_service=openai_service, settings=cfg)
    ctx.document_manager = document_manager
    simple_context.document_manager = document_manager
    logger.info("Document processing system initialized")

    simple_context.refresh_runtime_globals()

    # Recovery handler for database issues
    async def database_recovery_handler(exc: Exception) -> None:
        if db is not None and hasattr(db, "init"):
            try:
                await maybe_call(db.init)
                logger.info("Database recovery completed")
            except Exception as recovery_error:  # noqa: BLE001
                logger.error("Database recovery failed: %s", recovery_error)

    try:
        error_handler.register_recovery_handler(ErrorType.DATABASE, database_recovery_handler)
    except Exception:
        logger.debug("Recovery handler registration skipped", exc_info=True)

    # Optional scaling components
    scaling_components: Optional[dict[str, Any]] = None
    ctx.scaling_components = None
    simple_context.scaling_components = None
    if cfg.enable_scaling:
        try:
            service_registry = ServiceRegistry(
                redis_url=cfg.redis_url,
                heartbeat_interval=cfg.heartbeat_interval,
            )
            await service_registry.initialize()
            await service_registry.start_background_tasks()

            load_balancer = LoadBalancer(service_registry)
            session_affinity = SessionAffinity(
                redis_client=getattr(cache_backend, "_redis", None),
                ttl=cfg.session_affinity_ttl,
            )
            scaling_manager = ScalingManager(
                service_registry=service_registry,
                load_balancer=load_balancer,
                session_affinity=session_affinity,
            )

            scaling_components = {
                "service_registry": service_registry,
                "load_balancer": load_balancer,
                "session_affinity": session_affinity,
                "scaling_manager": scaling_manager,
            }
            ctx.scaling_components = scaling_components
            simple_context.scaling_components = scaling_components
            logger.info("🌐 Scaling components initialized")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize scaling components: %s", exc)

    # Health checks
    health_checker = HealthChecker(check_interval=cfg.health_check_interval)
    ctx.health_checker = health_checker
    simple_context.health_checker = health_checker
    health_checker.register_check(DatabaseHealthCheck(db))
    health_checker.register_check(OpenAIHealthCheck(openai_service))
    health_checker.register_check(SessionStoreHealthCheck(session_store))
    health_checker.register_check(RateLimiterHealthCheck(rate_limiter))
    if cfg.enable_system_monitoring:
        health_checker.register_check(SystemResourcesHealthCheck())
    await health_checker.start_background_checks()

    # Background tasks
    task_manager = BackgroundTaskManager(error_handler)
    ctx.task_manager = task_manager
    simple_context.task_manager = task_manager
    task_manager.register_task(
        DatabaseCleanupTask(
            db,
            interval_seconds=cfg.db_cleanup_interval,
            max_old_transactions_days=cfg.db_cleanup_days,
        )
    )
    task_manager.register_task(
        CacheCleanupTask(
            [openai_service],
            interval_seconds=cfg.cache_cleanup_interval,
        )
    )
    task_manager.register_task(
        SessionCleanupTask(
            session_store,
            interval_seconds=cfg.session_cleanup_interval,
        )
    )
    task_manager.register_task(
        DocumentStorageCleanupTask(
            document_manager.storage,
            max_age_hours=document_manager.storage.cleanup_max_age_hours,
            interval_seconds=document_manager.storage.cleanup_interval_seconds,
        )
    )

    main_components: dict[str, Any] = {
        "database": db,
        "openai_service": openai_service,
        "rate_limiter": rate_limiter,
        "session_store": session_store,
        "error_handler": error_handler,
        "health_checker": health_checker,
    }
    if scaling_components:
        main_components.update(scaling_components)

    task_manager.register_task(
        HealthCheckTask(
            main_components,
            interval_seconds=cfg.health_check_task_interval,
        )
    )
    if getattr(metrics_collector, "enable_prometheus", False):
        task_manager.register_task(
            MetricsCollectionTask(
                main_components,
                interval_seconds=cfg.metrics_collection_interval,
            )
        )
    await task_manager.start_all()
    logger.info("Started %s background tasks", len(task_manager.tasks))

    retention_notifier = RetentionNotifier(bot, db)
    await retention_notifier.start()
    logger.info("🔔 Retention notifier started")

    simple_context.refresh_runtime_globals()

    # Optional: Log Garant API monthly limits (diagnostics)
    try:
        garant_client = getattr(simple_context, "garant_client", None)
        if getattr(garant_client, "enabled", False):
            limits = await garant_client.get_limits()  # type: ignore[attr-defined]
            if limits:
                summary = ", ".join(f"{item.title}: {item.value}" for item in limits[:5])
                logger.info("Garant API limits: %s", summary)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to fetch Garant limits: %s", exc)

    return RuntimeBundle(
        metrics_collector=metrics_collector,
        cache_backend=cache_backend,
        response_cache=response_cache,
        db=db,
        rate_limiter=rate_limiter,
        access_service=access_service,
        openai_service=openai_service,
        audio_service=audio_service,
        session_store=session_store,
        crypto_provider=crypto_provider,
        error_handler=error_handler,
        document_manager=document_manager,
        scaling_components=scaling_components,
        health_checker=health_checker,
        task_manager=task_manager,
        retention_notifier=retention_notifier,
    )
