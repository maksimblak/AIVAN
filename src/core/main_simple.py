"""
Простая версия Telegram бота ИИ-Иван
Только /start и обработка вопросов, никаких кнопок и лишних команд
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import tempfile
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Optional, Sequence

from src.core.excel_export import build_practice_excel
from src.core.safe_telegram import send_html_text
from src.documents.document_manager import DocumentManager

if TYPE_CHECKING:
    from src.core.db_advanced import DatabaseAdvanced, TransactionStatus

import re
from html import escape as html_escape

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    BotCommand,
    BotCommandScopeChat,
    ErrorEvent,
    FSInputFile,
    Message,
    User,
)

from src.bot.promt import JUDICIAL_PRACTICE_SEARCH_PROMPT, LEGAL_SYSTEM_PROMPT
from src.bot.status_manager import ProgressStatus, progress_router
from src.bot.retention_notifier import RetentionNotifier
from src.bot.stream_manager import StreamingCallback, StreamManager
from src.bot.ui_components import Emoji
from src.core.attachments import QuestionAttachment
from src.core.audio_service import AudioService
from src.core.access import AccessService
from src.core.db_advanced import DatabaseAdvanced, TransactionStatus
from src.core.exceptions import (
    ErrorContext,
    ErrorHandler,
    ErrorType,
    NetworkException,
    OpenAIException,
    SystemException,
    ValidationException,
)
from src.core.middlewares.error_middleware import ErrorHandlingMiddleware
from src.core.openai_service import OpenAIService
from src.core.payments import CryptoPayProvider
from src.core.admin_modules.admin_commands import setup_admin_commands
from src.core.session_store import SessionStore, UserSession
from src.core.validation import InputValidator, ValidationSeverity
from src.core.runtime import SubscriptionPlanPricing, WelcomeMedia
from src.documents.base import ProcessingError
from src.bot.ratelimit import RateLimiter
from src.bot.typing_indicator import send_typing_once, typing_action

from src.core.simple_bot.menus import register_menu_handlers, cmd_start
from src.core.simple_bot.documents import register_document_handlers
from src.core.simple_bot.feedback import (
    ensure_rating_snapshot,
    handle_pending_feedback,
    register_feedback_handlers,
    send_rating_request,
)
from src.core.simple_bot.retention import register_retention_handlers
from src.core.simple_bot import context as simple_context
from src.core.simple_bot.common import (ensure_valid_user_id, get_user_session, get_safe_db_method)
from src.core.simple_bot.formatting import (
    _format_currency,
    _format_datetime,
    _format_hour_label,
    _format_number,
    _format_progress_extras,
    _format_risk_count,
    _format_response_time,
    _format_stat_row,
    _format_trend_value,
    _split_plain_text,
)
from src.core.simple_bot.stats import (
    FEATURE_LABELS,
    DAY_NAMES,
    build_stats_keyboard,
    describe_primary_summary,
    describe_secondary_summary,
    generate_user_stats_response,
    normalize_stats_period,
    peak_summary,
    progress_line,
    translate_payment_status,
    translate_plan_name,
)
from src.core.simple_bot.payments import get_plan_pricing, register_payment_handlers
from src.core.simple_bot.voice import register_voice_handlers
QUESTION_ATTACHMENT_MAX_BYTES = 4 * 1024 * 1024  # 4MB per attachment (base64-safe)

SECTION_DIVIDER = "<code>────────────────────</code>"


retention_notifier = None


# ============ КОНФИГУРАЦИЯ ============

logger = logging.getLogger("ai-ivan.simple")

set_runtime = simple_context.set_runtime
get_runtime = simple_context.get_runtime
settings = simple_context.settings
derived = simple_context.derived

WELCOME_MEDIA: WelcomeMedia | None = None
BOT_TOKEN = ""
BOT_USERNAME = ""
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
robokassa_provider: Any | None = None
yookassa_provider: Any | None = None
error_handler: ErrorHandler | None = None
document_manager: DocumentManager | None = None
response_cache: Any | None = None
stream_manager: StreamManager | None = None
metrics_collector: Any | None = None
task_manager: Any | None = None
health_checker: Any | None = None
scaling_components: dict[str, Any] | None = None
judicial_rag: Any | None = None

_SYNCED_ATTRS = (
    "WELCOME_MEDIA",
    "BOT_TOKEN",
    "BOT_USERNAME",
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
)


def _sync_local_globals() -> None:
    for attr in _SYNCED_ATTRS:
        globals()[attr] = getattr(simple_context, attr, None)


_sync_local_globals()


def refresh_runtime_globals() -> None:
    simple_context.refresh_runtime_globals()
    _sync_local_globals()


def __getattr__(name: str) -> Any:
    return getattr(simple_context, name)

class ResponseTimer:
    def __init__(self) -> None:
        self._t0: float | None = None
        self.duration = 0.0

    def start(self) -> None:
        self._t0 = time.monotonic()

    def stop(self) -> None:
        if self._t0 is not None:
            self.duration = max(0.0, time.monotonic() - self._t0)

    def get_duration_text(self) -> str:
        s = int(self.duration)
        return f"{s//60:02d}:{s%60:02d}"


async def process_question_with_attachments(message: Message) -> None:
    caption = (message.caption or "").strip()
    if not caption:
        warning_msg = "\n\n".join([
            f"{Emoji.WARNING} <b>Добавьте текст вопроса</b>",
            "Напишите короткое описание ситуации в подписи к файлу и отправьте снова.",
        ])
        await message.answer(warning_msg, parse_mode=ParseMode.HTML)
        return

    try:
        attachments = await _collect_question_attachments(message)
    except ValueError as exc:
        error_msg = "\n\n".join([
            f"{Emoji.WARNING} <b>Не удалось обработать вложение</b>",
            html_escape(str(exc)),
        ])
        await message.answer(error_msg, parse_mode=ParseMode.HTML)
        return

    if not attachments:
        await process_question(message, text_override=caption)
        return

    await process_question(message, text_override=caption, attachments=attachments)


async def process_question(
    message: Message,
    *,
    text_override: str | None = None,
    attachments: Sequence[QuestionAttachment] | None = None,
):
    """Главный обработчик юридических вопросов"""
    if not message.from_user:
        return



    # Выбираем тип индикатора в зависимости от контента
    if attachments:
        action = "upload_document"
    else:
        action = "typing"

    await send_typing_once(message.bot, message.chat.id, action)

    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="process_question")
    except ValidationException as exc:
        context = ErrorContext(
            chat_id=message.chat.id if message.chat else None,
            function_name="process_question",
        )
        if error_handler is not None:
            await error_handler.handle_exception(exc, context)
        else:
            logger.warning("Validation error in process_question: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} <b>Ошибка идентификатора пользователя</b>\n\nПопробуйте перезапустить диалог.",
            parse_mode=ParseMode.HTML,
        )
        return

    chat_id = message.chat.id
    message_id = message.message_id

    error_context = ErrorContext(
        user_id=user_id, chat_id=chat_id, message_id=message_id, function_name="process_question"
    )

    user_session = get_user_session(user_id)
    question_text = ((text_override if text_override is not None else (message.text or ""))).strip()
    attachments_list = list(attachments or [])
    quota_msg_to_send: str | None = None

    # Если ждём текстовый комментарий к рейтингу — обрабатываем его
    if not hasattr(user_session, "pending_feedback_request_id"):
        user_session.pending_feedback_request_id = None
    if user_session.pending_feedback_request_id is not None:
        await handle_pending_feedback(message, user_session, question_text)
        return

    # Игнорим команды
    if question_text.startswith("/"):
        return

    # Валидация
    if error_handler is None:
        raise SystemException("Error handler not initialized", error_context)
    cleaned = await _validate_question_or_reply(message, question_text, user_id)
    if not cleaned:
        return
    question_text = cleaned

    request_text = question_text
    if attachments_list:
        attachment_lines = []
        for idx, item in enumerate(attachments_list, start=1):
            size_kb = max(1, item.size // 1024)
            attachment_lines.append(
                f"{idx}. {item.filename} ({item.mime_type}, {size_kb} KB)"
            )
        request_text = (
            f"{question_text}\n\n[Attachments]\n" + "\n".join(attachment_lines)
        )

    logger.info("Processing question from user %s: %s", user_id, question_text[:100])
    timer = ResponseTimer()
    timer.start()
    use_streaming = USE_STREAMING and not attachments_list

    # Прогресс-бар
    status: ProgressStatus | None = None

    ok_flag = False
    request_error_type = None
    stream_manager: StreamManager | None = None
    had_stream_content = False
    stream_final_text: str | None = None
    final_answer_text: str | None = None
    result: dict[str, Any] = {}
    request_start_time = time.time()
    request_record_id: int | None = None

    try:
        # Rate limit
        if not await _rate_limit_guard(user_id, message):
            return

        # Доступ/квота
        quota_text = ""
        quota_is_trial: bool = False
        if access_service is not None:
            decision = await access_service.check_and_consume(user_id)
            if not decision.allowed:
                if decision.has_subscription and decision.subscription_plan:
                    plan_info = get_plan_pricing(decision.subscription_plan)
                    plan_name = plan_info.plan.name if plan_info else decision.subscription_plan
                    limit_lines = [
                        f"{Emoji.WARNING} <b>Лимит подписки исчерпан</b>",
                        f"Тариф: {plan_name}",
                    ]
                    if plan_info is not None:
                        limit_lines.append(
                            f"Использовано: {plan_info.plan.request_quota}/{plan_info.plan.request_quota} запросов."
                        )
                    limit_lines.append("Оформите новый пакет — /buy.")
                    await message.answer("\n".join(limit_lines), parse_mode=ParseMode.HTML)
                else:
                    await message.answer(
                        f"{Emoji.WARNING} <b>Лимит бесплатных запросов исчерпан</b>\n\nОформите подписку — /buy.",
                        parse_mode=ParseMode.HTML,
                    )
                return
            if decision.is_admin:
                quota_text = f"\n\n{Emoji.STATS} <b>Статус: безлимитный доступ</b>"
            elif decision.has_subscription:
                plan_info = get_plan_pricing(decision.subscription_plan) if decision.subscription_plan else None
                plan_name = plan_info.plan.name if plan_info else "Подписка"
                parts: list[str] = []
                if decision.subscription_requests_remaining is not None:
                    parts.append(
                        f"{Emoji.STATS} <b>{plan_name}:</b> осталось {decision.subscription_requests_remaining} запросов"
                    )
                if decision.subscription_until:
                    until_dt = datetime.fromtimestamp(decision.subscription_until)
                    parts.append(f"{Emoji.CALENDAR} <b>Активна до:</b> {until_dt:%Y-%m-%d}")
                quota_text = "\n\n" + "\n".join(parts) if parts else ""
            elif decision.trial_used is not None and decision.trial_remaining is not None:
                quota_is_trial = True
                quota_msg_core = html_escape(
                    f"Бесплатные запросы: {decision.trial_used}/{TRIAL_REQUESTS}. Осталось: {decision.trial_remaining}"
                )
                quota_msg_to_send = f"{Emoji.STATS} <b>{quota_msg_core}</b>"

        # Прогресс-бар
        status = await _start_status_indicator(message)

        ok_flag = False
        request_error_type = None
        stream_manager: StreamManager | None = None
        had_stream_content = False
        stream_final_text: str | None = None
        final_answer_text: str | None = None
        result: dict[str, Any] = {}
        request_start_time = time.time()
        request_record_id: int | None = None

        try:
            # Имитация первых этапов (если анимация отключена)
            if not USE_ANIMATION and hasattr(status, "update_stage"):
                await asyncio.sleep(0.5)
                await status.update_stage(1, f"{Emoji.SEARCH} Анализирую ваш вопрос...")
                await asyncio.sleep(1.0)
                await status.update_stage(2, f"{Emoji.LOADING} Ищу релевантную судебную практику...")

            # Выбор промпта
            selected_prompt = LEGAL_SYSTEM_PROMPT
            practice_mode = getattr(user_session, "practice_search_mode", False)
            rag_context = ""
            rag_fragments = []

            if practice_mode:
                selected_prompt = JUDICIAL_PRACTICE_SEARCH_PROMPT
                user_session.practice_search_mode = False

                # Интеграция RAG для поиска судебной практики
                if judicial_rag is not None and judicial_rag.enabled:
                    try:
                        if hasattr(status, "update_stage"):
                            await status.update_stage(2, f"{Emoji.LOADING} Ищу релевантную судебную практику в базе...")
                        rag_context, rag_fragments = await judicial_rag.build_context(question_text)
                        if rag_context:
                            logger.info(f"RAG found {len(rag_fragments)} relevant cases for question")
                    except Exception as rag_error:
                        logger.warning(f"RAG search failed: {rag_error}", exc_info=True)

            if text_override is not None and getattr(message, "voice", None):
                selected_prompt = (
                    selected_prompt
                    + "\n\nГолосовой режим: сохрани указанную структуру блоков, обязательно перечисли нормативные акты с точными реквизитами и уточни, что текстовый ответ уже предоставлен в чате."
                )

            # Добавляем контекст RAG в промпт
            if rag_context:
                selected_prompt = (
                    selected_prompt
                    + f"\n\n<judicial_practice_context>\nВот релевантная судебная практика из базы данных:\n\n{rag_context}\n</judicial_practice_context>\n\n"
                    + "ВАЖНО: Используй эту судебную практику в своём ответе, ссылайся на конкретные дела с указанием ссылок."
                )

            # --- Запрос к OpenAI (стрим/нестрим) ---
            if openai_service is None:
                raise SystemException("OpenAI service not initialized", error_context)

            if use_streaming and message.bot:
                stream_manager = StreamManager(
                    bot=message.bot,
                    chat_id=message.chat.id,
                    update_interval=1.5,
                    buffer_size=120,
                )
                await stream_manager.start_streaming(f"{Emoji.ROBOT} Обдумываю ваш вопрос...")
                callback = StreamingCallback(stream_manager)

                try:
                    # 1) Стриминговый запрос
                    result = await openai_service.ask_legal_stream(
                        selected_prompt, request_text, callback=callback
                    )
                    had_stream_content = bool((stream_manager.pending_text or "").strip())
                    if had_stream_content:
                        stream_final_text = stream_manager.pending_text or ""

                    # 2) Успех, если API вернул ok ИЛИ уже показывали текст пользователю
                    ok_flag = bool(isinstance(result, dict) and result.get("ok")) or had_stream_content

                    # 3) Фолбэк — если стрим не дал результата и текста нет
                    if not ok_flag:
                        with suppress(Exception):
                            await stream_manager.stop()
                            if stream_manager.message and message.bot:
                                await message.bot.delete_message(message.chat.id, stream_manager.message.message_id)
                        result = await openai_service.ask_legal(
                            selected_prompt,
                            request_text,
                            attachments=attachments_list or None,
                        )
                        ok_flag = bool(result.get("ok"))

                except Exception as e:
                    # Если что-то упало, но буфер уже есть — считаем успехом и завершаем стрим
                    had_stream_content = bool((stream_manager.pending_text or "").strip())
                    if had_stream_content:
                        logger.warning("Streaming failed, but content exists — using buffered text: %s", e)
                        stream_final_text = stream_manager.pending_text or ""
                        result = {"ok": True, "text": stream_final_text}
                        ok_flag = True
                    else:
                        with suppress(Exception):
                            await stream_manager.stop()
                        low = str(e).lower()
                        if "rate limit" in low or "quota" in low:
                            raise OpenAIException(str(e), error_context, is_quota_error=True)
                        elif "timeout" in low or "network" in low:
                            raise NetworkException(f"OpenAI network error: {str(e)}", error_context)
                        else:
                            raise OpenAIException(f"OpenAI API error: {str(e)}", error_context)
            else:
                # Нестриминговый путь
                result = await openai_service.ask_legal(
                    selected_prompt,
                    request_text,
                    attachments=attachments_list or None,
                )
                ok_flag = bool(result.get("ok"))

        except Exception as e:
            request_error_type = type(e).__name__
            if stream_manager:
                with suppress(Exception):
                    await stream_manager.stop()
            raise
        finally:
            # Всегда закрываем прогресс-бар
            with suppress(Exception):
                await _stop_status_indicator(status, ok=ok_flag)

        # ----- Постобработка результата -----
        timer.stop()

        if not ok_flag:
            error_text = (isinstance(result, dict) and (result.get("error") or "")) or ""
            logger.error("OpenAI error or empty result for user %s: %s", user_id, error_text)
            await message.answer(
                (
                    f"{Emoji.ERROR} <b>Произошла ошибка</b>\n\n"
                    f"Не удалось получить ответ. Попробуйте ещё раз чуть позже.\n\n"
                    f"{Emoji.HELP} <i>Подсказка</i>: Проверьте формулировку вопроса"
                    + (f"\n\n<code>{html_escape(error_text[:300])}</code>" if error_text else "")
                ),
                parse_mode=ParseMode.HTML,
            )
            return

        # Добавляем время ответа к финальному сообщению
        time_footer_raw = f"{Emoji.CLOCK} Время ответа: {timer.get_duration_text()} "

        # Формируем футер с найденными делами (если есть)
        sources_footer = ""
        if rag_fragments and practice_mode:
            sources_lines = ["\n\n📚 <b>Использованные дела из базы:</b>"]
            for idx, fragment in enumerate(rag_fragments[:5], start=1):
                header = fragment.header or f"Дело #{idx}"
                sources_lines.append(f"{idx}. {header}")
            sources_footer = "\n".join(sources_lines)

        text_to_send = ""
        if use_streaming and had_stream_content and stream_manager is not None:
            final_stream_text = stream_final_text or ((isinstance(result, dict) and (result.get("text") or "")) or "")
            combined_stream_text = (final_stream_text.rstrip() + sources_footer + f"\n\n{time_footer_raw}") if final_stream_text else time_footer_raw
            final_answer_text = combined_stream_text
            await stream_manager.finalize(combined_stream_text)
        else:
            text_to_send = (isinstance(result, dict) and (result.get("text") or "")) or ""
            if text_to_send:
                combined_text = f"{text_to_send.rstrip()}{sources_footer}\n\n{time_footer_raw}"
                final_answer_text = combined_text
                await send_html_text(
                    bot=message.bot,
                    chat_id=message.chat.id,
                    raw_text=combined_text,
                    reply_to_message_id=message.message_id,
                )

        if practice_mode:
            practice_excel_path: Path | None = None
            try:
                structured_payload = (
                    result.get("structured") if isinstance(result, dict) and isinstance(result.get("structured"), Mapping) else None
                )
                excel_source = final_answer_text or text_to_send or ""
                if excel_source or rag_fragments:
                    practice_excel_path = await asyncio.to_thread(
                        build_practice_excel,
                        summary_html=excel_source,
                        fragments=rag_fragments,
                        structured=structured_payload,
                        file_stub="practice_report",
                    )
                    await message.answer_document(
                        FSInputFile(str(practice_excel_path)),
                        caption="📊 Отчёт по судебной практике (XLSX)",
                        parse_mode=ParseMode.HTML,
                    )
            except Exception as excel_error:  # noqa: BLE001
                logger.warning("Failed to build practice Excel", exc_info=True)
            finally:
                if practice_excel_path is not None:
                    practice_excel_path.unlink(missing_ok=True)

        # Сообщения про квоту/подписку
            with suppress(Exception):
                await message.answer(quota_text, parse_mode=ParseMode.HTML)
        if quota_msg_to_send:
            with suppress(Exception):
                await message.answer(quota_msg_to_send, parse_mode=ParseMode.HTML)

        # Статистика в сессии/БД
        user_session.add_question_stats(timer.duration)
        if db is not None and hasattr(db, "record_request"):
            with suppress(Exception):
                request_time_ms = int((time.time() - request_start_time) * 1000)
                record_request_fn = get_safe_db_method("record_request", default_return=None)
                if record_request_fn:
                    request_record_id = await record_request_fn(
                        user_id=user_id,
                        request_type="legal_question",
                        tokens_used=0,
                        response_time_ms=request_time_ms,
                        success=True,
                        error_type=None,
                    )
        if request_record_id:
            if final_answer_text and message.from_user:
                with suppress(Exception):
                    await ensure_rating_snapshot(
                        request_record_id,
                        message.from_user,
                        final_answer_text or "",
                    )
            with suppress(Exception):
                await send_rating_request(
                    message,
                    request_record_id,
                    answer_snapshot=final_answer_text or "",
                )

        logger.info("Successfully processed question for user %s in %.2fs", user_id, timer.duration)
        return (isinstance(result, dict) and (result.get("text") or "")) or ""

    except Exception as e:
        # Централизованная обработка
        if error_handler is not None:
            try:
                custom_exc = await error_handler.handle_exception(e, error_context)
                user_message = getattr(
                    custom_exc, "user_message", "Произошла системная ошибка. Попробуйте позже."
                )
            except Exception:
                logger.exception("Error handler failed for user %s", user_id)
                user_message = "Произошла системная ошибка. Попробуйте позже."
        else:
            logger.exception("Error processing question for user %s (no error handler)", user_id)
            user_message = "Произошла ошибка. Попробуйте позже."

        # Статистика о неудаче
        if db is not None and hasattr(db, "record_request"):
            with suppress(Exception):
                request_time_ms = (
                    int((time.time() - request_start_time) * 1000)
                    if "request_start_time" in locals() else 0
                )
                err_type = request_error_type if "request_error_type" in locals() else type(e).__name__
                record_request_fn = get_safe_db_method("record_request", default_return=None)
                if record_request_fn:
                    await record_request_fn(
                        user_id=user_id,
                        request_type="legal_question",
                        tokens_used=0,
                        response_time_ms=request_time_ms,
                        success=False,
                        error_type=str(err_type),
                    )

        # Ответ пользователю
        with suppress(Exception):
            await message.answer(
                "❌ <b>Ошибка обработки запроса</b>\n\n"
                f"{user_message}\n\n"
                "💡 <b>Рекомендации:</b>\n"
                "• Переформулируйте вопрос\n"
                "• Попробуйте через несколько минут\n"
                "• Обратитесь в поддержку, если проблема повторяется",
                parse_mode=ParseMode.HTML,
            )


# ============ СИСТЕМА РЕЙТИНГА ============

async def cmd_askdoc(message: Message) -> None:
    if document_manager is None or not message.from_user:
        await message.answer(f"{Emoji.WARNING} Сессия документа не найдена. Загрузите документ с режимом \"Чат\".")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer(f"{Emoji.WARNING} Укажите вопрос после команды, например: /askdoc Какой срок?")
        return

    question = parts[1].strip()
    try:
        async with typing_action(message.bot, message.chat.id, "typing"):
            result = await document_manager.answer_chat_question(message.from_user.id, question)
    except ProcessingError as exc:
        await message.answer(f"{Emoji.WARNING} {html_escape(exc.message)}", parse_mode=ParseMode.HTML)
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Document chat failed: %s", exc, exc_info=True)
        await message.answer(
            f"{Emoji.ERROR} Не удалось получить ответ. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    formatted = document_manager.format_chat_answer_for_telegram(result)
    await message.answer(formatted, parse_mode=ParseMode.HTML)


async def cmd_enddoc(message: Message) -> None:
    if document_manager is None or not message.from_user:
        await message.answer(f"{Emoji.WARNING} Активная сессия не найдена.")
        return

    closed = document_manager.end_chat_session(message.from_user.id)
    if closed:
        await message.answer(f"{Emoji.SUCCESS} Чат с документом завершён.")
    else:
        await message.answer(f"{Emoji.WARNING} Активная сессия не найдена.")




async def cmd_ratings_stats(message: Message):
    """Команда для просмотра статистики рейтингов (только для админов)"""
    if not message.from_user:
        await message.answer("❌ Команда доступна только в диалоге с ботом")
        return

    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="cmd_ratings_stats")
    except ValidationException as exc:
        logger.warning("Некорректный пользователь id in cmd_ratings_stats: %s", exc)
        await message.answer("❌ Ошибка идентификатора пользователя")
        return

    if user_id not in ADMIN_IDS:
        await message.answer("❌ Команда доступна только администраторам")
        return

    stats_fn = get_safe_db_method("get_ratings_statistics", default_return={})
    low_rated_fn = get_safe_db_method("get_low_rated_requests", default_return=[])
    if not stats_fn or not low_rated_fn:
        await message.answer("❌ Статистика рейтингов недоступна")
        return

    try:
        stats_7d = await stats_fn(7)
        stats_30d = await stats_fn(30)
        low_rated = await low_rated_fn(5)

        stats_text = f"""📊 <b>Статистика рейтингов</b>

📅 <b>За 7 дней:</b>
• Всего оценок: {stats_7d.get('total_ratings', 0)}
• 👍 Лайков: {stats_7d.get('total_likes', 0)}
• 👎 Дизлайков: {stats_7d.get('total_dislikes', 0)}
• 📈 Рейтинг лайков: {stats_7d.get('like_rate', 0):.1f}%
• 💬 С комментариями: {stats_7d.get('feedback_count', 0)}

📅 <b>За 30 дней:</b>
• Всего оценок: {stats_30d.get('total_ratings', 0)}
• 👍 Лайков: {stats_30d.get('total_likes', 0)}
• 👎 Дизлайков: {stats_30d.get('total_dislikes', 0)}
• 📈 Рейтинг лайков: {stats_30d.get('like_rate', 0):.1f}%
• 💬 С комментариями: {stats_30d.get('feedback_count', 0)}"""

        if low_rated:
            stats_text += "\n\n⚠️ <b>Запросы для улучшения:</b>\n"
            for req in low_rated[:3]:
                stats_text += f"• ID {req['request_id']}: рейтинг {req['avg_rating']:.1f} ({req['rating_count']} оценок)\n"

        await message.answer(stats_text, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Error in cmd_ratings_stats: {e}")
        await message.answer("❌ Ошибка получения статистики рейтингов")


async def cmd_error_stats(message: Message):
    """Краткая сводка ошибок из ErrorHandler (админы)."""
    if not message.from_user:
        await message.answer("❌ Команда доступна только в диалоге с ботом")
        return

    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="cmd_error_stats")
    except ValidationException as exc:
        logger.warning("Некорректный пользователь id in cmd_error_stats: %s", exc)
        await message.answer("❌ Ошибка идентификатора пользователя")
        return

    if user_id not in ADMIN_IDS:
        await message.answer("❌ Команда доступна только администраторам")
        return

    if not error_handler:
        await message.answer("❌ Система мониторинга ошибок не инициализирована")
        return

    stats = error_handler.get_error_stats()
    if not stats:
        await message.answer("✅ Критических ошибок не зафиксировано")
        return

    lines = ["🚨 <b>Статистика ошибок</b>"]
    for error_type, count in sorted(stats.items(), key=lambda item: item[0]):
        lines.append(f"• {error_type}: {count}")

    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)

# ============ ОБРАБОТКА ОШИБОК ============


async def log_only_aiogram_error(event: ErrorEvent):
    """Глобальный обработчик ошибок"""
    logger.exception("Critical error in bot: %s", event.exception)


# ============ ГЛАВНАЯ ФУНКЦИЯ ============


async def _maybe_call(coro_or_func):
    """Вспомогательный вызов: поддерживает sync/async методы init()/close()."""
    if coro_or_func is None:
        return
    try:
        res = coro_or_func()
    except TypeError:
        # Если передали уже корутину
        res = coro_or_func
    if asyncio.iscoroutine(res):
        return await res
    return res


async def run_bot() -> None:
    """Main coroutine launching the bot."""
    global BOT_USERNAME
    global metrics_collector, db, response_cache, rate_limiter, access_service, openai_service
    global audio_service, session_store, crypto_provider, error_handler, document_manager
    global scaling_components, health_checker, task_manager
    ctx = get_runtime()
    cfg = ctx.settings
    container = ctx.get_dependency('container')
    if container is None:
        raise RuntimeError('DI container is not available')

    refresh_runtime_globals()

    if not cfg.telegram_bot_token:
        raise RuntimeError('TELEGRAM_BOT_TOKEN is required')

    session = None
    proxy_url = (cfg.telegram_proxy_url or '').strip()
    if proxy_url:
        logger.info('Using proxy: %s', proxy_url.split('@')[-1])
        proxy_user = (cfg.telegram_proxy_user or '').strip()
        proxy_pass = (cfg.telegram_proxy_pass or '').strip()
        if proxy_user and proxy_pass:
            from urllib.parse import quote, urlparse, urlunparse

            if '://' not in proxy_url:
                proxy_url = 'http://' + proxy_url
            u = urlparse(proxy_url)
            userinfo = f"{quote(proxy_user, safe='')}:{quote(proxy_pass, safe='')}"
            netloc = f"{userinfo}@{u.hostname}{':' + str(u.port) if u.port else ''}"
            proxy_url = urlunparse((u.scheme, netloc, u.path or '', u.params, u.query, u.fragment))
        session = AiohttpSession(proxy=proxy_url)

    bot = Bot(cfg.telegram_bot_token, session=session)
    try:
        bot_info = await bot.get_me()
        simple_context.BOT_USERNAME = (bot_info.username or '').strip()
        BOT_USERNAME = simple_context.BOT_USERNAME
    except Exception as exc:
        logger.warning('Could not fetch bot username: %s', exc)
    dp = Dispatcher()
    register_progressbar(dp)

    # Инициализация системы метрик/кэша/т.п.
    from src.core.background_tasks import (
        BackgroundTaskManager,
        CacheCleanupTask,
        DatabaseCleanupTask,
        DocumentStorageCleanupTask,
        HealthCheckTask,
        MetricsCollectionTask,
        SessionCleanupTask,
    )
    from src.core.cache import ResponseCache, create_cache_backend
    from src.core.health import (
        DatabaseHealthCheck,
        HealthChecker,
        OpenAIHealthCheck,
        RateLimiterHealthCheck,
        SessionStoreHealthCheck,
        SystemResourcesHealthCheck,
    )
    from src.core.metrics import init_metrics, set_system_status
    from src.core.scaling import LoadBalancer, ScalingManager, ServiceRegistry, SessionAffinity

    prometheus_port = cfg.prometheus_port
    metrics_collector = init_metrics(
        enable_prometheus=cfg.enable_prometheus,
        prometheus_port=prometheus_port,
    )
    ctx.metrics_collector = metrics_collector
    simple_context.metrics_collector = metrics_collector
    set_system_status("starting")

    logger.info("🚀 Starting AI-Ivan (simple)")

    # Используем продвинутую базу данных с connection pooling
    logger.info("Using advanced database with connection pooling")
    db = ctx.db or container.get(DatabaseAdvanced)
    ctx.db = db
    simple_context.db = db
    await db.init()

    setup_admin_commands(dp, db, ADMIN_IDS)

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

    rate_limiter = ctx.rate_limiter or container.get(RateLimiter)
    ctx.rate_limiter = rate_limiter
    simple_context.rate_limiter = rate_limiter
    await rate_limiter.init()

    access_service = ctx.access_service or container.get(AccessService)
    ctx.access_service = access_service
    simple_context.access_service = access_service

    openai_service = ctx.openai_service or container.get(OpenAIService)
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

    session_store = ctx.session_store or container.get(SessionStore)
    ctx.session_store = session_store
    simple_context.session_store = session_store
    crypto_provider = ctx.crypto_provider or container.get(CryptoPayProvider)
    ctx.crypto_provider = crypto_provider
    simple_context.crypto_provider = crypto_provider

    error_handler = ErrorHandler(logger=logger)
    ctx.error_handler = error_handler
    simple_context.error_handler = error_handler

    dp.update.middleware(ErrorHandlingMiddleware(error_handler, logger=logger))

    document_manager = DocumentManager(openai_service=openai_service, settings=cfg)
    ctx.document_manager = document_manager
    simple_context.document_manager = document_manager
    logger.info("Document processing system initialized")

    refresh_runtime_globals()

    # Регистрируем recovery handler для БД
    async def database_recovery_handler(exc):
        if db is not None and hasattr(db, "init"):
            try:
                await _maybe_call(db.init)
                logger.info("Database recovery completed")
            except Exception as recovery_error:
                logger.error(f"Database recovery failed: {recovery_error}")

    try:
        error_handler.register_recovery_handler(ErrorType.DATABASE, database_recovery_handler)
    except Exception:
        # Если ErrorType/handler не поддерживает регистрацию — просто логируем
        logger.debug("Recovery handler registration skipped")

    # Инициализация компонентов масштабирования (опционально)
    scaling_components = None
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
            logger.info("🔄 Scaling components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize scaling components: {e}")

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

    # Фоновые задачи
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
            [openai_service], interval_seconds=cfg.cache_cleanup_interval
        )
    )
    task_manager.register_task(
        SessionCleanupTask(
            session_store, interval_seconds=cfg.session_cleanup_interval
        )
    )
    task_manager.register_task(
        DocumentStorageCleanupTask(
            document_manager.storage,
            max_age_hours=document_manager.storage.cleanup_max_age_hours,
            interval_seconds=document_manager.storage.cleanup_interval_seconds,
        )
    )

    all_components = {
        "database": db,
        "openai_service": openai_service,
        "rate_limiter": rate_limiter,
        "session_store": session_store,
        "error_handler": error_handler,
        "health_checker": health_checker,
    }
    if scaling_components:
        all_components.update(scaling_components)

    task_manager.register_task(
        HealthCheckTask(
            all_components, interval_seconds=cfg.health_check_task_interval
        )
    )
    if getattr(metrics_collector, "enable_prometheus", False):
        task_manager.register_task(
            MetricsCollectionTask(
                all_components,
                interval_seconds=cfg.metrics_collection_interval,
            )
        )
    await task_manager.start_all()
    logger.info("Started %s background tasks", len(task_manager.tasks))

    # Запускаем retention notifier
    global retention_notifier
    retention_notifier = RetentionNotifier(bot, db)
    await retention_notifier.start()
    logger.info("✉️ Retention notifier started")

    refresh_runtime_globals()

    # Команды
    base_commands = [
        BotCommand(command="start", description=f"{Emoji.ROBOT} Начать работу"),
        BotCommand(command="buy", description=f"{Emoji.MAGIC} Оформить подписку"),
        BotCommand(command="status", description=f"{Emoji.STATS} Статус подписки"),
        BotCommand(command="mystats", description="📊 Моя статистика"),

    ]
    await bot.set_my_commands(base_commands)

    if ADMIN_IDS:
        admin_commands = base_commands + [
            BotCommand(command="ratings", description="📈 Статистика рейтингов (админ)"),
            BotCommand(command="errors", description="🚨 Статистика ошибок (админ)"),
        ]
        for admin_id in ADMIN_IDS:
            try:
                await bot.set_my_commands(
                    admin_commands,
                    scope=BotCommandScopeChat(chat_id=admin_id),
                )
            except TelegramBadRequest as exc:
                logger.warning(
                    "Failed to set admin command list for %s: %s",
                    admin_id,
                    exc,
                )

    # Роутинг
    register_payment_handlers(dp)

    register_menu_handlers(dp)
    register_document_handlers(dp)
    register_retention_handlers(dp)
    register_feedback_handlers(dp)

    dp.message.register(cmd_ratings_stats, Command("ratings"))
    dp.message.register(cmd_error_stats, Command("errors"))
    dp.message.register(cmd_askdoc, Command("askdoc"))
    dp.message.register(cmd_enddoc, Command("enddoc"))

    if settings().voice_mode_enabled:
        register_voice_handlers(dp, process_question)

    dp.message.register(process_question_with_attachments, F.photo | F.document)
    dp.message.register(process_question, F.text & ~F.text.startswith("/"))

    # Глобальный обработчик ошибок aiogram (с интеграцией ErrorHandler при наличии)
    async def telegram_error_handler(event: ErrorEvent):
        if error_handler:
            try:
                context = ErrorContext(
                    function_name="telegram_error_handler",
                    additional_data={
                        "update": str(event.update) if event.update else None,
                        "exception_type": type(event.exception).__name__,
                    },
                )
                await error_handler.handle_exception(event.exception, context)
            except Exception as handler_error:
                logger.error(f"Error handler failed: {handler_error}")
        logger.exception("Critical error in bot: %s", event.exception)

    dp.error.register(telegram_error_handler)

    # Лог старта
    set_system_status("running")
    startup_info = [
        "🤖 AI-Ivan (simple) successfully started!",
        f"🎞 Animation: {'enabled' if USE_ANIMATION else 'disabled'}",
        f"🗄️ Database: advanced",
        f"🔄 Cache: {cache_backend.__class__.__name__}",
        f"📈 Metrics: {'enabled' if getattr(metrics_collector, 'enable_prometheus', False) else 'disabled'}",
        f"🏥 Health checks: {len(health_checker.checks)} registered",
        f"⚙️ Background tasks: {len(task_manager.tasks)} running",
        f"🔄 Scaling: {'enabled' if scaling_components else 'disabled'}",
    ]
    for info in startup_info:
        logger.info(info)
    if prometheus_port:
        logger.info(
            f"📊 Prometheus metrics available at http://localhost:{prometheus_port}/metrics"
        )

    try:
        logger.info("🚀 Starting bot polling...")
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("🛑 AI-Ivan stopped by user")
        set_system_status("stopping")
    except Exception as e:
        logger.exception("💥 Fatal error in main loop: %s", e)
        set_system_status("stopping")
        raise
    finally:
        logger.info("🔧 Shutting down services...")
        set_system_status("stopping")

        # Останавливаем retention notifier
        if retention_notifier:
            try:
                await retention_notifier.stop()
            except Exception as e:
                logger.error(f"Error stopping retention notifier: {e}")

        # Останавливаем фоновые задачи
        try:
            await task_manager.stop_all()
        except Exception as e:
            logger.error(f"Error stopping background tasks: {e}")

        # Останавливаем health checks
        try:
            await health_checker.stop_background_checks()
        except Exception as e:
            logger.error(f"Error stopping health checks: {e}")

        # Останавливаем компоненты масштабирования
        if scaling_components:
            try:
                await scaling_components["service_registry"].stop_background_tasks()
            except Exception as e:
                logger.error(f"Error stopping scaling components: {e}")

        # Закрываем основные сервисы (поддержка sync/async close)
        services_to_close = [
            ("Bot session", lambda: bot.session.close()),
            ("Database", lambda: getattr(db, "close", None) and db.close()),
            ("Rate limiter", lambda: getattr(rate_limiter, "close", None) and rate_limiter.close()),
            (
                "OpenAI service",
                lambda: getattr(openai_service, "close", None) and openai_service.close(),
            ),
            (
                "Audio service",
                lambda: getattr(audio_service, "aclose", None) and audio_service.aclose(),
            ),
            (
                "Response cache",
                lambda: getattr(response_cache, "close", None) and response_cache.close(),
            ),
        ]
        for service_name, close_func in services_to_close:
            try:
                await _maybe_call(close_func)
                logger.debug(f"✅ {service_name} closed")
            except Exception as e:
                logger.error(f"❌ Error closing {service_name}: {e}")

        logger.info("👋 AI-Ivan shutdown complete")
