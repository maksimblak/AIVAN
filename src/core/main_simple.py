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
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BotCommand,
    BotCommandScopeChat,
    CallbackQuery,
    ErrorEvent,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    User,
)

from src.bot.promt import JUDICIAL_PRACTICE_SEARCH_PROMPT, LEGAL_SYSTEM_PROMPT
from src.documents.document_drafter import (
    DocumentDraftingError,
    build_docx_from_markdown,
    format_plan_summary,
    generate_document,
    plan_document,
)
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
_NUMBERED_ANSWER_RE = re.compile(r"^\s*(\d+)[\).:-]\s*(.*)")
_BULLET_ANSWER_RE = re.compile(r"^\s*[-\u2022]\s*(.*)")
_HEADING_PATTERN_RE = re.compile(
    r"^\s*(?![-\u2022])(?!\d+[\).:-])([A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451\u0030-\u0039][^:]{0,80}):\s*(.*)$"
)
QUESTION_ATTACHMENT_MAX_BYTES = 4 * 1024 * 1024  # 4MB per attachment (base64-safe)

SECTION_DIVIDER = "<code>────────────────────</code>"


def _format_user_display(user: User | None) -> str:
    if user is None:
        return ""
    parts: list[str] = []
    if user.username:
        parts.append(f"@{user.username}")
    name = " ".join(filter(None, [user.first_name, user.last_name])).strip()
    if name and name not in parts:
        parts.append(name)
    if not parts and user.first_name:
        parts.append(user.first_name)
    if not parts:
        parts.append(str(user.id))
    return " ".join(parts)

retention_notifier = None


async def _ensure_rating_snapshot(request_id: int, telegram_user: User | None, answer_text: str) -> None:
    if db is None or not answer_text.strip():
        return
    if telegram_user is None:
        return
    add_rating_fn = get_safe_db_method("add_rating", default_return=False)
    if not add_rating_fn:
        return
    username = _format_user_display(telegram_user)
    await add_rating_fn(
        request_id,
        telegram_user.id,
        0,
        None,
        username=username,
        answer_text=answer_text,
    )

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


class DocumentProcessingStates(StatesGroup):
    waiting_for_document = State()
    processing_document = State()




class DocumentDraftStates(StatesGroup):
    waiting_for_request = State()
    asking_details = State()
    generating = State()


# ============ УПРАВЛЕНИЕ СОСТОЯНИЕМ ============


# ============ УТИЛИТЫ ============




async def _download_telegram_file(bot: Bot, file_id: str) -> bytes:
    file_info = await bot.get_file(file_id)
    file_path = getattr(file_info, "file_path", None)
    if not file_path:
        raise ValueError("Не удалось получить путь к файлу в Telegram.")
    file_content = await bot.download_file(file_path)
    try:
        return await asyncio.to_thread(file_content.read)
    finally:
        close_method = getattr(file_content, "close", None)
        if callable(close_method):
            close_method()

async def _collect_question_attachments(message: Message) -> list[QuestionAttachment]:
    bot = message.bot
    if bot is None:
        raise ValueError("Бот временно недоступен. Попробуйте ещё раз немного позже.")

    if message.media_group_id:
        raise ValueError("Сейчас можно отправить только один файл вместе с вопросом.")

    attachments: list[QuestionAttachment] = []
    size_limit_mb = max(1, QUESTION_ATTACHMENT_MAX_BYTES // (1024 * 1024))

    if message.document:
        document = message.document
        declared_size = document.file_size or 0
        if declared_size > QUESTION_ATTACHMENT_MAX_BYTES:
            raise ValueError(
                f"Файл '{document.file_name or document.file_unique_id}' больше {size_limit_mb} МБ."
            )
        data = await _download_telegram_file(bot, document.file_id)
        if len(data) > QUESTION_ATTACHMENT_MAX_BYTES:
            raise ValueError(
                f"Файл '{document.file_name or document.file_unique_id}' больше {size_limit_mb} МБ."
            )
        attachments.append(
            QuestionAttachment(
                filename=document.file_name or f"document_{document.file_unique_id}",
                mime_type=document.mime_type or "application/octet-stream",
                data=data,
            )
        )
        return attachments

    if message.photo:
        photo = message.photo[-1]
        declared_size = photo.file_size or 0
        if declared_size > QUESTION_ATTACHMENT_MAX_BYTES:
            raise ValueError(f"Фото больше {size_limit_mb} МБ.")
        data = await _download_telegram_file(bot, photo.file_id)
        if len(data) > QUESTION_ATTACHMENT_MAX_BYTES:
            raise ValueError(f"Фото больше {size_limit_mb} МБ.")
        attachments.append(
            QuestionAttachment(
                filename=f"photo_{photo.file_unique_id}.jpg",
                mime_type="image/jpeg",
                data=data,
            )
        )
    return attachments


async def _validate_question_or_reply(message: Message, text: str, user_id: int) -> str | None:
    result = InputValidator.validate_question(text, user_id)
    if not result.is_valid:
        bullet = "\n• "
        error_msg = bullet.join(result.errors)
        if result.severity == ValidationSeverity.CRITICAL:
            await message.answer(
                f"{Emoji.ERROR} <b>Критическая ошибка валидации</b>\n\n• {error_msg}\n\n<i>Попробуйте переформулировать запрос</i>",
                parse_mode=ParseMode.HTML,
            )
        else:
            await message.answer(
                f"{Emoji.WARNING} <b>Ошибка в запросе</b>\n\n• {error_msg}",
                parse_mode=ParseMode.HTML,
            )
        return None

    if result.warnings:
        bullet = "\n• "
        logger.warning("Validation warnings for user %s: %s", user_id, bullet.join(result.warnings))

    cleaned = (result.cleaned_data or "").strip()
    if not cleaned:
        await message.answer(
            f"{Emoji.WARNING} <b>Пустой запрос</b>\n\nПожалуйста, опишите вопрос подробнее.",
            parse_mode=ParseMode.HTML,
        )
        return None
    return cleaned


async def _rate_limit_guard(user_id: int, message: Message) -> bool:
    if rate_limiter is None:
        return True
    allowed = await rate_limiter.allow(user_id)
    if allowed:
        return True
    await message.answer(
        f"{Emoji.WARNING} <b>Превышен лимит запросов</b>\n\nПопробуйте позже.",
        parse_mode=ParseMode.HTML,
    )
    return False




async def _delete_status_message(bot: Bot, chat_id: int, message_id: int, attempts: int = 3) -> bool:
    """Attempt to remove the progress message with basic retry/backoff."""
    for attempt in range(max(1, attempts)):
        try:
            await bot.delete_message(chat_id, message_id)
            return True
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
        except TelegramBadRequest as exc:
            low = str(exc).lower()
            if "message to delete not found" in low or "message can't be deleted" in low:
                return False
            if attempt == attempts - 1:
                logger.warning("Unable to delete status message: %s", exc)
            await asyncio.sleep(0.2 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1:
                logger.warning("Unexpected error deleting status message: %s", exc)
            await asyncio.sleep(0.2 * (attempt + 1))
    return False


async def _start_status_indicator(message):
    status = ProgressStatus(
        message.bot,
        message.chat.id,
        show_checklist=True,
        show_context_toggle=False,   # ⟵ прячем кнопку
        min_edit_interval=0.9,
        auto_advance_stages=True,
        # percent_thresholds=[1, 10, 30, 50, 70, 85, 95],
    )
    await status.start(auto_cycle=True, interval=2.7)  # approx. 4 min until auto-complete
    return status

async def _stop_status_indicator(status: ProgressStatus | None, ok: bool) -> None:
    if status is None:
        return

    message_id = getattr(status, "message_id", None)

    try:
        if ok:
            if hasattr(status, "_last_edit_ts"):
                status._last_edit_ts = 0.0  # allow immediate update on completion
            await status.complete()
        else:
            await status.fail("Ошибка при формировании ответа")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to finalize status indicator: %s", exc)
        return

    if ok and message_id:
        await _delete_status_message(status.bot, status.chat_id, message_id)

# ============ ФУНКЦИИ РЕЙТИНГА И UI ============










def create_rating_keyboard(request_id: int) -> InlineKeyboardMarkup:
    """Создает клавиатуру с кнопками рейтинга для оценки ответа"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👍", callback_data=f"rate_like_{request_id}"),
                InlineKeyboardButton(text="👎", callback_data=f"rate_dislike_{request_id}"),
            ]
        ]
    )


def _build_ocr_reply_markup(output_format: str) -> InlineKeyboardMarkup:
    """Создаёт клавиатуру для возврата и повторной загрузки режима "распознание текста"."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text=f"{Emoji.BACK} Назад", callback_data="back_to_menu"),
            InlineKeyboardButton(text=f"{Emoji.DOCUMENT} Загрузить ещё", callback_data=f"ocr_upload_more:{output_format}")
        ]]
    )


_BASE_STAGE_LABELS: dict[str, tuple[str, str]] = {
    "start": ("Подготавливаем обработку", "🚀"),
    "downloading": ("Скачиваем документ", "⬇️"),
    "uploaded": ("Файл сохранён", "💾"),
    "processing": ("Обрабатываем документ", "⏳"),
    "finalizing": ("Формируем результат", "🧾"),
    "completed": ("Готово", "✅"),
    "failed": ("Ошибка обработки", "❌"),
}

_STAGE_LABEL_OVERRIDES: dict[str, dict[str, tuple[str, str]]] = {
    "summarize": {
        "processing": ("Анализируем структуру", "🧠"),
        "finalizing": ("Собираем саммари", "📄"),
    },
    "analyze_risks": {
        "processing": ("Анализируем риски", "⚠️"),
        "pattern_scan": ("Ищем шаблоны рисков", "🧭"),
        "ai_analysis": ("ИИ анализирует документ", "🤖"),
        "compliance_check": ("Проверяем требования закона", "⚖️"),
        "aggregation": ("Сводим результаты", "🗂️"),
        "highlighting": ("Готовим подсветку", "🔍"),
    },
    "lawsuit_analysis": {
        "processing": ("Анализируем иск", "⚖️"),
        "model_request": ("Анализируем правовые аспекты", "🔍"),
        "analysis_ready": ("Формируем итоговое заключение", "✅"),
    },
    "anonymize": {
        "processing": ("Ищем персональные данные", "🕵️"),
        "finalizing": ("Формируем обезличенную версию", "🧾"),
    },
    "translate": {
        "processing": ("Переводим текст", "🌐"),
        "finalizing": ("Готовим итоговый перевод", "📝"),
    },
    "ocr": {
        "processing": ("Распознаём текст", "🖨️"),
        "finalizing": ("Очищаем результат", "🧼"),
        "ocr_page": ("Распознаём страницы", "📑"),
    },
    "chat": {
        "processing": ("Индексируем документ", "🧠"),
        "finalizing": ("Готовим чаты", "💬"),
        "chunking": ("Режем документ на блоки", "🧩"),
        "indexing": ("Создаём поисковый индекс", "📚"),
    },
}


def _get_stage_labels(operation: str) -> dict[str, tuple[str, str]]:
    labels = _BASE_STAGE_LABELS.copy()
    labels.update(_STAGE_LABEL_OVERRIDES.get(operation, {}))
    return labels






def _build_completion_payload(op: str, result_obj) -> dict[str, Any]:
    data = getattr(result_obj, "data", None) or {}
    payload: dict[str, Any] = {}
    if op == "analyze_risks":
        pattern = len(data.get("pattern_risks", []) or [])
        ai_risks = len(((data.get("ai_analysis") or {}).get("risks")) or [])
        payload["risks_found"] = pattern + ai_risks
        payload["violations"] = len(((data.get("legal_compliance") or {}).get("violations")) or [])
        payload["overall"] = data.get("overall_risk_level")
    elif op == "summarize":
        summary_struct = ((data.get("summary") or {}).get("structured")) or {}
        payload["words"] = len(((summary_struct.get("summary")) or "").split())
        payload["chunks_total"] = len(summary_struct.get("key_points") or [])
    elif op == "anonymize":
        report = data.get("anonymization_report") or {}
        masked = report.get("processed_items")
        if masked is None:
            stats = report.get("statistics") or {}
            masked = sum(int(v) for v in stats.values()) if stats else 0
        payload["masked"] = int(masked or 0)
    elif op == "translate":
        meta = data.get("translation_metadata") or {}
        payload["language_pair"] = meta.get("language_pair")
        payload["chunks_total"] = meta.get("chunks_processed")
        payload["mode"] = meta.get("mode")
    elif op == "ocr":
        payload["confidence"] = data.get("confidence_score")
        processing = data.get("processing_info") or {}
        payload["pages_total"] = processing.get("pages_processed") or len(data.get("pages", []) or [])
        payload["mode"] = processing.get("file_type")
    elif op == "chat":
        info = data.get("document_info") or {}
        payload["chunks_total"] = info.get("chunks_count")
    return {k: v for k, v in payload.items() if v not in (None, "", [])}


def _make_progress_updater(
    message: Message,
    status_msg: Message,
    *,
    file_name: str,
    operation_name: str,
    file_size_kb: int,
    stage_labels: dict[str, tuple[str, str]],
) -> tuple[Callable[[dict[str, Any]], Awaitable[None]], dict[str, Any]]:
    progress_state: dict[str, Any] = {"percent": 0, "stage": "start", "started_at": time.monotonic()}

    async def send_progress(update: dict[str, Any]) -> None:
        nonlocal progress_state, status_msg
        if not status_msg or not status_msg.message_id:
            return
        stage = str(update.get("stage") or progress_state["stage"] or "processing")
        percent_val = update.get("percent")
        if percent_val is None:
            percent = progress_state["percent"]
        else:
            percent = max(0, min(100, int(round(float(percent_val)))))
        if percent < progress_state["percent"] and stage != "failed":
            percent = progress_state["percent"]

        progress_state["stage"] = stage
        progress_state["percent"] = percent

        label, icon = stage_labels.get(stage, stage_labels.get("processing", ("Обработка", "⏳")))
        extras_line = _format_progress_extras(update)
        elapsed = time.monotonic() - progress_state["started_at"]
        elapsed_text = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

        lines = [
            f"{icon} {label}: {percent}%",
            f"🗂️ Файл: <b>{html_escape(file_name)}</b>",
            f"🛠️ Операция: {html_escape(operation_name)}",
            f"📊 Размер: {file_size_kb} КБ",
            f"⏱️ Время: {elapsed_text}",
        ]
        if extras_line:
            lines.append(extras_line)

        try:
            await message.bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=status_msg.message_id,
                text="\n".join(lines),
                parse_mode=ParseMode.HTML,
            )
        except TelegramBadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                logger.debug("Progress edit failed: %s", exc)
        except Exception as exc:  # pragma: no cover
            logger.debug("Unexpected progress update error: %s", exc)

    return send_progress, progress_state


async def send_rating_request(message: Message, request_id: int):
    """Отправляет сообщение с запросом на оценку ответа, если пользователь ещё не голосовал."""
    if not message.from_user:
        return


    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="send_rating_request")
    except ValidationException as exc:
        logger.debug("Skip rating request due to invalid user id: %s", exc)
        return

    get_rating_fn = get_safe_db_method("get_rating", default_return=None)
    if get_rating_fn:
        existing_rating = await get_rating_fn(request_id, user_id)
        if existing_rating and getattr(existing_rating, "rating", 0) in (1, -1):
            return

    rating_keyboard = create_rating_keyboard(request_id)
    try:
        await message.answer(
            f"{Emoji.STAR} <b>Оцените качество ответа</b>\n\n"
            "Ваша оценка поможет нам улучшить сервис!",
            parse_mode=ParseMode.HTML,
            reply_markup=rating_keyboard,
        )
    except Exception as e:
        logger.error(f"Failed to send rating request: {e}")
        # Не критично, если не удалось отправить запрос на рейтинг


# ============ КОМАНДЫ ============


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
                    await _ensure_rating_snapshot(request_record_id, message.from_user, final_answer_text)
            with suppress(Exception):
                await send_rating_request(message, request_record_id)

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

async def handle_ocr_upload_more(callback: CallbackQuery, state: FSMContext):
    """Prepare state for another "распознание текста" upload after a result message."""
    output_format = "txt"
    data = callback.data or ""
    if ":" in data:
        _, payload = data.split(":", 1)
        if payload:
            output_format = payload
    try:
        with suppress(Exception):
            await callback.message.edit_reply_markup()

        await state.clear()
        await state.update_data(
            document_operation="ocr",
            operation_options={"output_format": output_format},
        )
        await state.set_state(DocumentProcessingStates.waiting_for_document)

        await callback.message.answer(
            f"{Emoji.DOCUMENT} Отправьте следующий файл или фото для режима \"распознание текста\".",
            parse_mode=ParseMode.HTML,
        )
        await callback.answer("Готов к загрузке нового документа")
    except Exception as exc:
        logger.error(f"Error in handle_ocr_upload_more: {exc}", exc_info=True)
        await callback.answer("Не удалось подготовить повторную загрузку", show_alert=True)


async def handle_pending_feedback(message: Message, user_session: UserSession, text_override: str | None = None):
    """Обработка текстового комментария после оценки"""
    if not message.from_user:
        return

    feedback_source = text_override if text_override is not None else (message.text or "")
    if not feedback_source or not user_session.pending_feedback_request_id:
        return

    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="handle_pending_feedback")
    except ValidationException as exc:
        logger.warning("Ignore feedback: invalid user id (%s)", exc)
        user_session.pending_feedback_request_id = None
        return

    request_id = user_session.pending_feedback_request_id
    feedback_text = feedback_source.strip()

    # Сбрасываем ожидание комментария после обработки
    user_session.pending_feedback_request_id = None

    add_rating_fn = get_safe_db_method("add_rating", default_return=False)
    if not add_rating_fn:
        await message.answer("❌ Сервис отзывов временно недоступен")
        return

    get_rating_fn = get_safe_db_method("get_rating", default_return=None)
    existing_rating = await get_rating_fn(request_id, user_id) if get_rating_fn else None

    rating_value = -1
    answer_snapshot = ""
    if existing_rating:
        if existing_rating.rating not in (None, 0):
            rating_value = existing_rating.rating
        if getattr(existing_rating, 'answer_text', None):
            answer_snapshot = existing_rating.answer_text or ""

    if not answer_snapshot:
        session_snapshot = getattr(user_session, "last_answer_snapshot", None)
        if session_snapshot:
            answer_snapshot = session_snapshot

    username_display = _format_user_display(message.from_user)

    success = await add_rating_fn(
        request_id,
        user_id,
        rating_value,
        feedback_text,
        username=username_display,
        answer_text=answer_snapshot,
    )
    if success:
        await message.answer(
            "Спасибо за подробный отзыв!\n\n"
            "Ваш комментарий поможет нам сделать ответы лучше.",
            parse_mode=ParseMode.HTML,
        )
        logger.info(
            "Received feedback for request %s from user %s: %s",
            request_id,
            user_id,
            feedback_text,
        )
        user_session.last_answer_snapshot = None
    else:
        await message.answer("❌ Не удалось сохранить отзыв")


async def handle_rating_callback(callback: CallbackQuery):
    """Обработка кнопок рейтинга и запрос текстового комментария"""
    if not callback.data or not callback.from_user:
        await callback.answer("Некорректные данные")
        return

    try:
        user_id = ensure_valid_user_id(callback.from_user.id, context="handle_rating_callback")
    except ValidationException as exc:
        logger.warning("Некорректный пользователь id in rating callback: %s", exc)
        await callback.answer("Некорректный пользователь", show_alert=True)
        return

    user_session = get_user_session(user_id)

    try:
        parts = callback.data.split("_")
        if len(parts) != 3:
            await callback.answer("Некорректный формат данных")
            return
        action = parts[1]
        if action not in {"like", "dislike"}:
            await callback.answer("Неизвестное действие")
            return
        request_id = int(parts[2])
    except (ValueError, IndexError):
        await callback.answer("Некорректный формат данных")
        return

    get_rating_fn = get_safe_db_method("get_rating", default_return=None)
    existing_rating = await get_rating_fn(request_id, user_id) if get_rating_fn else None

    if existing_rating and existing_rating.rating not in (None, 0):
        await callback.answer("По этому ответу уже собрана обратная связь")
        return

    add_rating_fn = get_safe_db_method("add_rating", default_return=False)
    if not add_rating_fn:
        await callback.answer("Сервис рейтингов временно недоступен")
        return

    rating_value = 1 if action == "like" else -1
    answer_snapshot = ""
    if existing_rating and getattr(existing_rating, 'answer_text', None):
        answer_snapshot = existing_rating.answer_text or ""
    if not answer_snapshot:
        session_snapshot = getattr(user_session, "last_answer_snapshot", None)
        if session_snapshot:
            answer_snapshot = session_snapshot
    username_display = _format_user_display(callback.from_user)

    success = await add_rating_fn(
        request_id,
        user_id,
        rating_value,
        None,
        username=username_display,
        answer_text=answer_snapshot,
    )
    if not success:
        await callback.answer("Не удалось сохранить оценку")
        return

    if action == "like":
        await callback.answer("Спасибо за оценку! Рады, что ответ оказался полезным.")
        await callback.message.edit_text(
            "💬 <b>Спасибо за оценку!</b> ✅ Отмечено как полезное",
            parse_mode=ParseMode.HTML,
        )
        return

    await callback.answer("Спасибо за обратную связь!")
    feedback_keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📝 Написать комментарий",
                    callback_data=f"feedback_{request_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    text="❌ Пропустить",
                    callback_data=f"skip_feedback_{request_id}",
                )
            ],
        ]
    )
    await callback.message.edit_text(
        "💬 <b>Что можно улучшить?</b>\n\nВаша обратная связь поможет нам стать лучше:",
        reply_markup=feedback_keyboard,
        parse_mode=ParseMode.HTML,
    )


async def handle_feedback_callback(callback: CallbackQuery):
    """Обработчик запроса обратной связи"""
    if not callback.data or not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        user_id = ensure_valid_user_id(callback.from_user.id, context="handle_feedback_callback")
    except ValidationException as exc:
        logger.warning("Некорректный пользователь id in feedback callback: %s", exc)
        await callback.answer("❌ Ошибка пользователя", show_alert=True)
        return

    try:
        data = callback.data

        if data.startswith("feedback_"):
            action = "feedback"
            request_id = int(data.removeprefix("feedback_"))
        elif data.startswith("skip_feedback_"):
            action = "skip"
            request_id = int(data.removeprefix("skip_feedback_"))
        else:
            await callback.answer("❌ Неверный формат данных")
            return

        if action == "skip":
            await callback.message.edit_text(
                "💬 <b>Спасибо за оценку!</b> 👎 Отмечено для улучшения", parse_mode=ParseMode.HTML
            )
            await callback.answer("✅ Спасибо за обратную связь!")
            return

        # action == "feedback"
        user_session = get_user_session(user_id)
        if not hasattr(user_session, "pending_feedback_request_id"):
            user_session.pending_feedback_request_id = None
        user_session.pending_feedback_request_id = request_id

        await callback.message.edit_text(
            "💬 <b>Напишите ваш комментарий:</b>\n\n"
            "<i>Что можно улучшить в ответе? Отправьте текстовое сообщение.</i>",
            parse_mode=ParseMode.HTML,
        )
        await callback.answer("✏️ Напишите комментарий следующим сообщением")

    except Exception as e:
        logger.error(f"Error in handle_feedback_callback: {e}")
        await callback.answer("❌ Произошла ошибка")

async def handle_search_practice_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Поиск и аналитика судебной практики'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        instruction_keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🏠 Главное меню", callback_data="back_to_main")],
                [InlineKeyboardButton(text="👤 Мой профиль", callback_data="my_profile")],
            ]
        )

        await callback.message.edit_text(
            "🔍 <b>Поиск судебной практики</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "⚖️ <i>Найду релевантную судебную практику\n"
            "   для вашего юридического вопроса</i>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📋 <b>Что вы получите:</b>\n\n"
            "💡 <b>Краткая консультация</b>\n"
            "   └ 2 ссылки на судебную практику\n"
            "   └ Быстрый анализ ситуации\n\n"
            "📊 <b>Углубленный анализ</b>\n"
            "   └ 6+ примеров из практики\n"
            "   └ Детальные рекомендации\n\n"
            "📄 <b>Подготовка документов</b>\n"
            "   └ На основе найденной практики\n"
            "   └ С учетом актуальных решений\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "✍️ <i>Напишите ваш юридический вопрос\n"
            "   следующим сообщением...</i>",
            parse_mode=ParseMode.HTML,
            reply_markup=instruction_keyboard,
        )

        # Устанавливаем режим поиска практики для пользователя
        user_session = get_user_session(callback.from_user.id)
        if not hasattr(user_session, "practice_search_mode"):
            user_session.practice_search_mode = False
        user_session.practice_search_mode = True

    except Exception as e:
        logger.error(f"Error in handle_search_practice_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_prepare_documents_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Подготовка документов'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        await callback.message.answer(
            "📄 <b>Подготовка документов</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📑 <i>Помогу составить процессуальные\n"
            "   и юридические документы</i>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📋 <b>Типы документов:</b>\n\n"
            "⚖️ <b>Исковые заявления</b>\n"
            "   └ С учетом судебной практики\n\n"
            "📝 <b>Ходатайства</b>\n"
            "   └ Процессуальные запросы\n\n"
            "📧 <b>Жалобы и возражения</b>\n"
            "   └ На решения и действия\n\n"
            "📜 <b>Договоры и соглашения</b>\n"
            "   └ Правовая защита интересов\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "✍️ <i>Опишите какой документ нужен\n"
            "   и приложите детали дела...</i>",
            parse_mode=ParseMode.HTML,
        )

        # Режим подготовки документов
        user_session = get_user_session(callback.from_user.id)
        if not hasattr(user_session, "document_preparation_mode"):
            user_session.document_preparation_mode = False
        user_session.document_preparation_mode = True

    except Exception as e:
        logger.error(f"Error in handle_prepare_documents_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_doc_draft_start(callback: CallbackQuery, state: FSMContext) -> None:
    """Запуск режима подготовки нового документа."""
    if not callback.from_user:
        await callback.answer("❌ Не удалось определить пользователя")
        return

    try:
        # Показываем typing indicator для лучшего UX
        await send_typing_once(callback.bot, callback.message.chat.id, "typing")

        await state.clear()
        await state.set_state(DocumentDraftStates.waiting_for_request)

        intro_text = (
            f"✨ <b>Создание юридического документа</b>\n"
            f"<code>{'━' * 35}</code>\n\n"

            f"📋 <b>Как это работает:</b>\n\n"

            f"<b>1️⃣ Опишите задачу</b>\n"
            f"   └ Расскажите, какой документ нужен\n\n"

            f"<b>2️⃣ Отвечайте на вопросы</b>\n"
            f"   └ Я уточню детали для точности\n\n"

            f"<b>3️⃣ Получите DOCX</b>\n"
            f"   └ Готовый документ за минуту\n\n"

            f"<code>{'━' * 35}</code>\n\n"

            f"💡 <i>Совет: Опишите ситуацию максимально подробно — "
            f"это поможет создать точный документ с первого раза</i>\n\n"

            f"<b>Примеры запросов:</b>\n"
            f"• Исковое заявление о взыскании долга\n"
            f"• Договор оказания юридических услуг\n"
            f"• Жалоба в Роспотребнадзор\n\n"

            f"👇 <b>Опишите, что нужно создать:</b>"
        )

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text=f"{Emoji.BACK} Отмена", callback_data="doc_draft_cancel")]]
        )
        await callback.message.answer(intro_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
        await callback.answer()
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось запустить конструктор документа: %s", exc, exc_info=True)
        await callback.answer("Не удалось запустить конструктор", show_alert=True)


async def handle_doc_draft_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    """Отмена процесса создания документа."""
    await state.clear()
    with suppress(Exception):
        await callback.message.answer(
            f"🚫 <b>Создание документа отменено</b>\n"
            f"<code>{'─' * 30}</code>\n\n"
            f"💡 Вы можете начать заново в любой момент",
            parse_mode=ParseMode.HTML
        )
    with suppress(Exception):
        await callback.answer("Отменено")


async def handle_doc_draft_request(
    message: Message,
    state: FSMContext,
    *,
    text_override: str | None = None,
) -> None:
    """Обработка исходного запроса юриста."""
    source_text = text_override if text_override is not None else message.text
    request_text = (source_text or "").strip()
    if not request_text:
        await message.answer(
            f"⚠️ <b>Пустой запрос</b>\n"
            f"<code>{'─' * 30}</code>\n\n"
            f"📝 Пожалуйста, опишите какой документ нужен\n\n"
            f"<i>Например:</i>\n"
            f"• Договор аренды квартиры\n"
            f"• Исковое заявление о возврате товара\n"
            f"• Претензия в управляющую компанию",
            parse_mode=ParseMode.HTML
        )
        return

    if openai_service is None:
        await message.answer(
            f"❌ <b>Сервис недоступен</b>\n"
            f"<code>{'─' * 30}</code>\n\n"
            f"⚠️ Генерация документов временно недоступна\n"
            f"🔄 Попробуйте позже или обратитесь к администратору",
            parse_mode=ParseMode.HTML
        )
        await state.clear()
        return

    # Показываем индикатор "печатает"
    await send_typing_once(message.bot, message.chat.id, "typing")

    # Динамический прогресс-бар с автообновлением
    progress = ProgressStatus(
        message.bot,
        message.chat.id,
        steps=[
            {"label": "🔍 Определяю тип документа"},
            {"label": "📝 Формирую план вопросов"},
            {"label": "✨ Подготавливаю структуру"},
        ],
        show_context_toggle=False,
        show_checklist=True,
        auto_advance_stages=True,
        percent_thresholds=[0, 50, 90],
    )

    await progress.start(auto_cycle=True, interval=1.5)

    try:
        plan = await plan_document(openai_service, request_text)

        # Завершаем прогресс успешно
        await progress.complete()
        await asyncio.sleep(0.3)  # Короткая пауза для визуального эффекта
    except DocumentDraftingError as err:
        await progress.fail(note=str(err))
        await state.clear()
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка планирования документа: %s", exc, exc_info=True)
        await progress.fail(note="Попробуйте еще раз")
        await state.clear()
        return
    else:
        with suppress(Exception):
            # Удаляем сообщение прогресса после завершения
            if progress.message_id:
                await message.bot.delete_message(message.chat.id, progress.message_id)

    await state.update_data(
        draft_request=request_text,
        draft_plan={"title": plan.title, "questions": plan.questions, "notes": plan.notes},
        draft_answers=[],
        current_question_index=0,
    )

    summary = format_plan_summary(plan)
    for chunk in _split_plain_text(summary):
        await message.answer(chunk, parse_mode=ParseMode.HTML)

    if plan.questions:
        await state.set_state(DocumentDraftStates.asking_details)
        await _send_questions_prompt(
            message,
            plan.questions,
            title="Вопросы для подготовки документа",
        )
    else:
        await state.set_state(DocumentDraftStates.generating)
        await message.answer(
            f"✅ <b>Информации достаточно!</b>\n"
            f"<code>{'▰' * 20}</code>\n\n"
            f"🚀 Приступаю к формированию документа\n"
            f"⏱ Это займет около минуты",
            parse_mode=ParseMode.HTML
        )
        await _finalize_draft(message, state)


async def handle_doc_draft_answer(
    message: Message,
    state: FSMContext,
    *,
    text_override: str | None = None,
) -> None:
    """Обработка ответов юриста на уточняющие вопросы."""
    # Показываем typing indicator при обработке ответов
    await send_typing_once(message.bot, message.chat.id, "typing")

    data = await state.get_data()
    plan = data.get("draft_plan") or {}
    questions = plan.get("questions") or []
    index = data.get("current_question_index", 0)

    if index >= len(questions):
        await message.answer(
            f"✅ <b>Ответы получены</b>\n"
            f"<code>{'▰' * 20}</code>\n\n"
            f"🚀 Приступаю к формированию документа",
            parse_mode=ParseMode.HTML
        )
        await state.set_state(DocumentDraftStates.generating)
        await _finalize_draft(message, state)
        return

    source_text = text_override if text_override is not None else message.text
    answer_text = (source_text or "").strip()
    if not answer_text:
        await message.answer(
            f"⚠️ <b>Пустой ответ</b>\n\n"
            f"📝 Пожалуйста, введите ваш ответ на вопрос",
            parse_mode=ParseMode.HTML
        )
        return

    answers = data.get("draft_answers") or []
    remaining_questions = questions[index:]
    question_headings = [str(q.get("text", "") or "") for q in remaining_questions]
    bulk_answers = _extract_answer_chunks(
        answer_text,
        expected_count=len(remaining_questions),
        question_headings=question_headings,
    )

    if bulk_answers:
        used_count = 0
        for offset, chunk in enumerate(bulk_answers):
            if offset >= len(remaining_questions):
                break
            question = remaining_questions[offset]
            answers.append({"question": question.get("text", ""), "answer": chunk})
            used_count += 1

        if used_count > 0:
            if used_count < len(bulk_answers):
                extra = "\n".join(bulk_answers[used_count:]).strip()
                if extra and answers:
                    answers[-1]["answer"] = f"{answers[-1]['answer']}\n{extra}"
            index += used_count

            await state.update_data(draft_answers=answers, current_question_index=index)
            if index < len(questions):
                missing_numbers = ", ".join(str(i) for i in range(index + 1, len(questions) + 1))
                await message.answer(
                    f"⚠️ <b>Неполные ответы</b>\n"
                    f"<code>{'─' * 30}</code>\n\n"
                    f"✅ Получено ответов: <b>{index}</b>\n"
                    f"❌ Осталось вопросов: <b>{len(questions) - index}</b>\n"
                    f"📝 Номера вопросов: {missing_numbers}\n\n"
                    f"<b>Как дополнить:</b>\n"
                    f"• Отправьте недостающие ответы одним сообщением\n"
                    f"• Отделяйте пустой строкой или нумеруйте",
                    parse_mode=ParseMode.HTML,
                )
            else:
                await state.set_state(DocumentDraftStates.generating)
                await message.answer(
                    f"⚙️ <b>Формирование документа...</b>\n"
                    f"<code>{'▰' * 20}</code>\n\n"

                    f"✅ Все ответы получены\n"
                    f"🔄 Анализирую информацию\n"
                    f"📝 Подготавливаю текст\n"
                    f"📄 Формирую DOCX файл\n\n"

                    f"<i>⏱ Обычно занимает 30-60 секунд</i>",
                    parse_mode=ParseMode.HTML,
                )
                await _finalize_draft(message, state)
            return
        # если не удалось сопоставить ни одного ответа — переходим к обычной обработке

    if index < len(questions):
        current_question = questions[index]
        answers.append({"question": current_question.get("text", ""), "answer": answer_text})
        index += 1

        await state.update_data(draft_answers=answers, current_question_index=index)

        if index < len(questions):
            next_question = questions[index]
            next_text = html_escape(next_question.get("text", ""))
            purpose = next_question.get("purpose")

            lines = [
                f"{Emoji.SUCCESS} <b>Ответ принят</b>",
                f"<code>{'▰' * 20}</code>",
                "",
                f"{Emoji.QUESTION} <b>Вопрос {index + 1} из {len(questions)}</b>",
                next_text,
            ]
            if purpose:
                lines.append(f"<i>💡 {html_escape(str(purpose))}</i>")
            await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)
        else:
            await state.set_state(DocumentDraftStates.generating)
            await message.answer(
                f"⚙️ <b>Формирование документа...</b>\n"
                f"<code>{'▰' * 20}</code>\n\n"
                f"✅ Все ответы получены\n"
                f"🔄 Анализирую информацию\n"
                f"📝 Подготавливаю текст\n"
                f"📄 Формирую DOCX файл\n\n"
                f"<i>⏱ Обычно занимает 30-60 секунд</i>",
                parse_mode=ParseMode.HTML,
            )
            await _finalize_draft(message, state)
        return

    await message.answer(
        f"{Emoji.WARNING} Не удалось обработать ответ. Попробуйте ещё раз.",
        parse_mode=ParseMode.HTML,
    )




async def _extract_doc_voice_text(message: Message) -> str | None:
    """Распознать голосовое сообщение в сценарии составления документа."""
    if not message.voice:
        return None

    if audio_service is None:
        await message.answer(f"{Emoji.WARNING} Голосовой режим недоступен, отправьте текстовое сообщение.")
        return None

    try:
        voice_enabled = settings().voice_mode_enabled
    except RuntimeError:
        voice_enabled = settings().voice_mode_enabled

    if not voice_enabled:
        await message.answer(f"{Emoji.WARNING} Голосовой режим сейчас выключен. Пришлите ответ текстом.")
        return None

    if not message.bot:
        await message.answer(f"{Emoji.WARNING} Не удалось получить доступ к боту. Ответьте текстом.")
        return None

    temp_voice_path: Path | None = None
    try:
        await audio_service.ensure_short_enough(message.voice.duration)

        async with typing_action(message.bot, message.chat.id, "record_voice"):
            temp_voice_path = await _download_voice_to_temp(message)
            transcript = await audio_service.transcribe(temp_voice_path)
    except ValueError as duration_error:
        logger.warning("Document draft voice input too long: %s", duration_error)
        await message.answer(
            f"{Emoji.WARNING} Голосовое сообщение слишком длинное. Максимальная длительность — {audio_service.max_duration_seconds} секунд."
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to transcribe voice for document draft: %s", exc)
        await message.answer(
            f"{Emoji.ERROR} Не получилось распознать голос. Отправьте ответ текстом, пожалуйста."
        )
        return None
    finally:
        with suppress(Exception):
            if temp_voice_path:
                temp_voice_path.unlink()

    preview = html_escape(transcript[:500])
    if len(transcript) > 500:
        preview += "…"
    await message.answer(
        f"{Emoji.MICROPHONE} Распознанный текст:\n<i>{preview}</i>",
        parse_mode=ParseMode.HTML,
    )
    return transcript


async def handle_doc_draft_request_voice(message: Message, state: FSMContext) -> None:
    """Обработать голосовой запрос на составление документа."""
    transcript = await _extract_doc_voice_text(message)
    if transcript is None:
        return
    await handle_doc_draft_request(message, state, text_override=transcript)


async def handle_doc_draft_answer_voice(message: Message, state: FSMContext) -> None:
    """Обработать голосовой ответ в сценарии уточняющих вопросов."""
    transcript = await _extract_doc_voice_text(message)
    if transcript is None:
        return
    await handle_doc_draft_answer(message, state, text_override=transcript)


def _extract_answer_chunks(
    answer_text: str,
    *,
    expected_count: int | None = None,
    question_headings: Sequence[str] | None = None,
) -> list[str] | None:
    """Split a combined answer message into separate answers."""
    text = (answer_text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (
        text.replace("\u00A0", " ")
        .replace("\u202F", " ")
        .replace("\u2007", " ")
        .replace("\u2060", "")
        .replace("\ufeff", "")
    )
    text = text.strip()
    if not text:
        return None

    lines = text.split("\n")
    numbered_pattern = _NUMBERED_ANSWER_RE
    answers: list[str] = []
    current: list[str] | None = None
    has_numbers = False

    for line in lines:
        match = numbered_pattern.match(line)
        if match:
            has_numbers = True
            if current is not None:
                chunk = "\n".join(current).strip()
                if chunk:
                    answers.append(chunk)
            current = [match.group(2)]
        else:
            if current is not None:
                current.append(line)
    if has_numbers:
        if current:
            chunk = "\n".join(current).strip()
            if chunk:
                answers.append(chunk)
        if len(answers) > 1:
            return answers

    bullet_pattern = _BULLET_ANSWER_RE
    first_nonempty = next((line for line in lines if line.strip()), "")
    if bullet_pattern.match(first_nonempty):
        answers = []
        current = None
        for line in lines:
            match = bullet_pattern.match(line)
            if match:
                if current:
                    chunk = "\n".join(current).strip()
                    if chunk:
                        answers.append(chunk)
                current = [match.group(1)]
            else:
                if current:
                    current.append(line)
        if current:
            chunk = "\n".join(current).strip()
            if chunk:
                answers.append(chunk)
        if len(answers) > 1:
            return answers

    chunks = [
        chunk.strip()
        for chunk in re.split(r"(?:\n[ \t\u00A0\u2007\u202F\u2060]*){2,}", text)
        if chunk.strip()
    ]
    if len(chunks) > 1:
        if question_headings:
            normalized = [
                re.sub(r"\s+", " ", (heading or "")).strip().lower()
                for heading in question_headings
            ]

            def _normalize_line(line: str) -> str:
                return re.sub(r"\s+", " ", (line or "")).strip().lower()

            ordered: list[str] = []
            idx = 0
            for chunk in chunks:
                first_line = chunk.split("\n", 1)[0]
                first_norm = _normalize_line(first_line)
                if idx < len(normalized) and normalized[idx] and first_norm == normalized[idx]:
                    ordered.append(chunk)
                    idx += 1
                else:
                    ordered.append(chunk)
            return ordered
        return chunks

    if expected_count is not None and expected_count < 2:
        return None

    heading_pattern = _HEADING_PATTERN_RE
    candidates: list[str] = []
    current: list[str] = []
    heading_boundaries = 0

    for line in lines:
        match = heading_pattern.match(line)
        is_heading = False
        if match:
            heading_text = match.group(1).strip()
            if heading_text and len(heading_text) <= 80 and len(heading_text.split()) <= 8 and not re.search(r"[.!?]", heading_text):
                is_heading = True

        if is_heading and current:
            chunk = "\n".join(current).strip()
            if chunk:
                candidates.append(chunk)
            current = [line]
            heading_boundaries += 1
            continue

        if not current and not line.strip():
            continue

        if not current:
            current = [line]
        else:
            current.append(line)

    if current:
        chunk = "\n".join(current).strip()
        if chunk:
            candidates.append(chunk)

    if heading_boundaries >= 1 and len(candidates) > 1:
        max_allowed = expected_count + 3 if expected_count is not None else None
        if max_allowed is None or len(candidates) <= max_allowed:
            return candidates

    return None

async def _send_questions_prompt(
    message: Message,
    questions: list[dict[str, Any]],
    *,
    title: str,
) -> None:
    if not questions:
        return

    # Показываем typing indicator перед отправкой вопросов
    await send_typing_once(message.bot, message.chat.id, "typing")

    # Форматируем вопросы с улучшенным дизайном
    question_blocks: list[str] = []
    for idx, question in enumerate(questions, 1):
        text = html_escape(question.get("text", ""))
        purpose = question.get("purpose")

        # Чистый и читаемый дизайн без лишних линий
        block_lines = [
            f"<b>{idx}. {text}</b>",  # Вопрос жирным шрифтом
        ]

        if purpose:
            block_lines.append(f"<i>   💡 {html_escape(purpose)}</i>")  # Цель с отступом

        question_blocks.append("\n".join(block_lines))

    if not question_blocks:
        return

    # Только список вопросов (инструкция уже в сообщении выше)
    max_len = 3500
    chunk_lines: list[str] = [
        "📋 <b>Вопросы:</b>",
        f"<code>{'─' * 35}</code>",
        ""
    ]

    for block in question_blocks:
        candidate = chunk_lines + [block, ""]  # Пустая строка между вопросами
        candidate_text = "\n".join(candidate)
        if len(candidate_text) > max_len and len(chunk_lines) > 3:
            await message.answer("\n".join(chunk_lines), parse_mode=ParseMode.HTML)
            chunk_lines = [
                "📋 <b>Вопросы (продолжение):</b>",
                f"<code>{'─' * 35}</code>",
                "",
                block,
                ""
            ]
        else:
            if len(candidate_text) > max_len:
                # блок слишком большой сам по себе — отправим отдельно
                await message.answer("\n".join(chunk_lines), parse_mode=ParseMode.HTML)
                chunk_lines = [
                    "📋 <b>Вопросы (продолжение):</b>",
                    f"<code>{'─' * 35}</code>",
                    "",
                    block,
                    ""
                ]
            else:
                chunk_lines.append(block)
                chunk_lines.append("")  # Пустая строка между вопросами

    if len(chunk_lines) > 3:
        await message.answer("\n".join(chunk_lines), parse_mode=ParseMode.HTML)



_TITLE_SANITIZE_RE = re.compile(r"[\\/:*?\"<>|\r\n]+")
_TITLE_WHITESPACE_RE = re.compile(r"\s+")


def _prepare_document_titles(raw_title: str | None) -> tuple[str, str, str]:
    base = (raw_title or "").strip()
    if not base:
        base = "Документ"
    if base.endswith(")") and "(" in base:
        simplified = re.sub(r"\s*\([^)]*\)\s*$", "", base).strip()
        if simplified:
            base = simplified
    display_title = _TITLE_WHITESPACE_RE.sub(" ", base).strip()
    if not display_title:
        display_title = "Документ"
    caption = f"{Emoji.DOCUMENT} {display_title}"

    file_stub = _TITLE_SANITIZE_RE.sub("_", display_title).strip("._ ")
    if not file_stub:
        file_stub = "Документ"
    max_len = 80
    if len(file_stub) > max_len:
        file_stub = file_stub[:max_len].rstrip("._ ")
        if not file_stub:
            file_stub = "Документ"
    filename = f"{file_stub}.docx"
    return display_title, caption, filename


async def _finalize_draft(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    request_text = data.get("draft_request", "")
    plan = data.get("draft_plan") or {}
    answers = data.get("draft_answers") or []
    title = plan.get("title", "Документ")

    if openai_service is None:
        await message.answer(f"{Emoji.ERROR} Сервис генерации документов временно недоступен. Попробуйте позже.")
        await state.clear()
        return

    progress: ProgressStatus | None = None
    try:
        progress = ProgressStatus(
            message.bot,
            message.chat.id,
            steps=[
                {"label": "Готовим черновик"},
                {"label": "Проверяем структуру"},
                {"label": "Формируем DOCX"},
                {"label": "Отправляем файл"},
            ],
            show_context_toggle=False,
            show_checklist=True,
            auto_advance_stages=True,
            min_edit_interval=0.5,
            percent_thresholds=[0, 55, 80, 95],
        )
        await progress.start(auto_cycle=True, interval=1.4)
        await progress.update_stage(percent=5, step=1)
    except Exception as progress_err:  # pragma: no cover - индикатор не критичен
        logger.debug("Failed to start document drafting progress: %s", progress_err)
        progress = None

    # Показываем индикатор "отправляет документ" во время генерации
    try:
        async with typing_action(message.bot, message.chat.id, "upload_document"):
            result = await generate_document(openai_service, request_text, title, answers)
    except DocumentDraftingError as err:
        if progress:
            await progress.fail(note=str(err))
        await message.answer(f"{Emoji.ERROR} Не удалось подготовить документ: {err}")
        await state.clear()
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка генерации документа: %s", exc, exc_info=True)
        if progress:
            await progress.fail(note="Сбой при генерации документа")
        await message.answer(f"{Emoji.ERROR} Произошла ошибка при генерации документа")
        await state.clear()
        return

    if progress:
        await progress.update_stage(percent=65, step=2)

    if result.status != "ok":
        if progress:
            note = "Нужны дополнительные уточнения" if result.follow_up_questions else "Не удалось завершить документ"
            await progress.fail(note=note)
        if result.follow_up_questions:
            extra_questions = [
                {"id": f"f{i+1}", "text": item, "purpose": "Дополнительное уточнение"}
                for i, item in enumerate(result.follow_up_questions)
            ]
            await state.update_data(
                draft_plan={
                    "title": result.title or title,
                    "questions": extra_questions,
                    "notes": plan.get("notes", []),
                },
                current_question_index=0,
                draft_answers=answers,
            )
            await state.set_state(DocumentDraftStates.asking_details)
            await message.answer(f"{Emoji.WARNING} Нужно несколько уточнений, чтобы завершить документ.")
            await _send_questions_prompt(
                message,
                extra_questions,
                title="Дополнительные вопросы",
            )
            return

        issues_text = "\n".join(result.issues) or "Модель не смогла подготовить документ."
        await message.answer(f"{Emoji.WARNING} Документ не готов. Причина:\n{issues_text}")
        await state.clear()
        return

    # Показываем информацию о проверке и предупреждения
    summary_sections: list[str] = []
    if result.validated:
        validated_lines = "\n".join(
            f"• {html_escape(str(item).strip())}"
            for item in result.validated
            if str(item).strip()
        )
        if validated_lines:
            summary_sections.append(
                f"{Emoji.SUCCESS} <b>Проверено</b>\n{validated_lines}"
            )

    if result.issues:
        issue_lines = "\n".join(
            f"• {html_escape(str(item).strip())}"
            for item in result.issues
            if str(item).strip()
        )
        if issue_lines:
            summary_sections.append(
                f"{Emoji.WARNING} <b>На что обратить внимание</b>\n{issue_lines}"
            )

    if summary_sections:
        await message.answer(
            "\n\n".join(summary_sections),
            parse_mode=ParseMode.HTML,
        )

    # Создаем и отправляем документ
    if progress:
        await progress.update_stage(percent=85, step=3)
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        await asyncio.to_thread(build_docx_from_markdown, result.markdown, str(tmp_path))
        display_title, caption, filename = _prepare_document_titles(result.title or title)

        # Красивое сообщение с документом
        final_caption = (
            f"📄 <b>{display_title}</b>\n"
            f"<code>{'─' * 30}</code>\n\n"
            f"✨ Документ успешно создан!\n"
            f"📎 Формат: DOCX\n\n"
            f"<i>💡 Проверьте содержимое и при необходимости внесите правки</i>"
        )

        if progress:
            await progress.update_stage(percent=95, step=4)

        await message.answer_document(
            FSInputFile(str(tmp_path), filename=filename),
            caption=final_caption,
            parse_mode=ParseMode.HTML
        )
        if progress:
            await progress.complete(note="Документ готов")
            await asyncio.sleep(0.3)
            with suppress(Exception):
                if progress.message_id:
                    await message.bot.delete_message(message.chat.id, progress.message_id)
    except DocumentDraftingError as err:
        if progress:
            await progress.fail(note=str(err))
        await message.answer(
            f"❌ <b>Ошибка формирования DOCX</b>\n"
            f"<code>{'─' * 30}</code>\n\n"
            f"⚠️ {err}",
            parse_mode=ParseMode.HTML
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    await state.clear()


async def handle_document_processing(callback: CallbackQuery):
    """Обработка кнопки работы с документами"""
    try:
        operations = document_manager.get_supported_operations()

        # Создаем кнопки в удобном порядке (по 2 в ряд)
        buttons = []

        # Получаем операции и создаем кнопки
        buttons.append([
            InlineKeyboardButton(
                text="⚖️ Анализ искового заявления",
                callback_data="doc_operation_lawsuit_analysis",
            )
        ])
        buttons.append([
            InlineKeyboardButton(
                text=f"{Emoji.MAGIC} Создание юридического документа",
                callback_data="doc_draft_start",
            )
        ])

        secondary_buttons = []
        for op_key, op_info in operations.items():
            if op_key in {"translate", "chat", "lawsuit_analysis"}:
                continue
            emoji = op_info.get("emoji", "📄")
            name = op_info.get("name", op_key)
            secondary_buttons.append(
                InlineKeyboardButton(text=f"{emoji} {name}", callback_data=f"doc_operation_{op_key}")
            )

        for i in range(0, len(secondary_buttons), 2):
            row = secondary_buttons[i:i+2]
            buttons.append(row)

        # Кнопка "Назад" в отдельном ряду
        buttons.append([InlineKeyboardButton(text="◀️ Назад в меню", callback_data="back_to_menu")])

        message_text = (
            "🗂️ <b>Работа с документами</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🤖 <i>Автоматическая обработка и анализ\n"
            "   ваших документов с помощью ИИ</i>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📋 <b>Доступные операции:</b>\n\n"
            "📄 <b>Краткая выжимка</b>\n"
            "   └ Превращает объёмные документы\n"
            "      в короткие выжимки\n\n"
            "⚠️ <b>Риск-анализ</b>\n"
            "   └ Находит опасные формулировки\n"
            "      и проблемные места в договорах\n\n"
            "⚖️ <b>Анализ искового заявления</b>\n"
            "   └ Оценивает правовую позицию,\n"
            "      риски и рекомендации\n\n"
            "🔒 <b>Обезличивание</b>\n"
            "   └ Скрывает персональные данные\n"
            "      и конфиденциальные сведения\n\n"
            "🔍 <b>Распознавание текста</b>\n"
            "   └ Извлекает текст со сканов\n"
            "      и фотографий\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "👇 Выберите нужную операцию:"
        )

        await callback.message.answer(
            message_text,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        )
        await callback.answer()

    except Exception as e:
        await callback.answer(f"Ошибка: {e}")
        logger.error(f"Ошибка в handle_document_processing: {e}", exc_info=True)


async def handle_document_operation(callback: CallbackQuery, state: FSMContext):
    """Обработка выбора операции с документом"""
    try:
        # Показываем typing indicator
        await send_typing_once(callback.bot, callback.message.chat.id, "typing")

        operation = callback.data.replace("doc_operation_", "")
        operation_info = document_manager.get_operation_info(operation)

        if not operation_info:
            await callback.answer("Неизвестная операция")
            return

        # Сохраняем выбранную операцию и настройки в состояние
        state_data = await state.get_data()
        operation_options = dict(state_data.get("operation_options") or {})
        if operation == "lawsuit_analysis":
            operation_options.setdefault("output_format", "md")
        await state.update_data(document_operation=operation, operation_options=operation_options)

        emoji = operation_info.get("emoji", "📄")
        name = operation_info.get("name", operation)
        description = operation_info.get("description", "")
        upload_formats = operation_info.get("upload_formats")
        if upload_formats:
            formats = ", ".join(upload_formats)
        else:
            formats = ", ".join(operation_info.get("formats", []))

        # Создаем подробное описание для каждой операции
        detailed_descriptions = {
            "summarize": (
                "📋 <b>Краткая выжимка</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⚙️ <b>Как это работает:</b>\n\n"
                "🔍 <b>Анализ</b>\n"
                "   └ Изучает содержание документа\n\n"
                "📌 <b>Выделение ключевого</b>\n"
                "   └ Основные положения и идеи\n\n"
                "📝 <b>Структурирование</b>\n"
                "   └ Создает краткую выжимку\n\n"
                "💾 <b>Сохранение деталей</b>\n"
                "   └ Важные цифры и факты\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📊 <b>Что вы получите:</b>\n\n"
                "   ✓ Выжимка на 1-3 страницы\n"
                "   ✓ Основные выводы\n"
                "   ✓ Рекомендации\n"
                "   ✓ Экспорт в DOCX/PDF\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "💼 <b>Полезно для:</b>\n"
                "   • Договоры\n"
                "   • Отчеты\n"
                "   • Исследования\n"
                "   • Техническая документация"
            ),
            "analyze_risks": (
                "⚠️ <b>Риск-анализ</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⚙️ <b>Как это работает:</b>\n\n"
                "🔎 <b>Сканирование</b>\n"
                "   └ Поиск потенциальных рисков\n\n"
                "⚡ <b>Выявление проблем</b>\n"
                "   └ Опасные формулировки\n\n"
                "📖 <b>Правовой анализ</b>\n"
                "   └ Соответствие нормам права\n\n"
                "📈 <b>Оценка рисков</b>\n"
                "   └ Общий уровень опасности\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📊 <b>Что вы получите:</b>\n\n"
                "   ✓ Детальный отчет по рискам\n"
                "   ✓ Маркировка по опасности\n"
                "   ✓ Рекомендации по устранению\n"
                "   ✓ Ссылки на нормативы\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "💼 <b>Полезно для:</b>\n"
                "   • Договоры\n"
                "   • Соглашения\n"
                "   • Корпоративные документы"
            ),
            "lawsuit_analysis": (
                "⚖️ <b>Анализ иска</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⚙️ <b>Как это работает:</b>\n\n"
                "📋 <b>Определение требований</b>\n"
                "   └ Иск и правовая позиция\n\n"
                "🔍 <b>Оценка доказательств</b>\n"
                "   └ Сильные и слабые стороны\n\n"
                "⚠️ <b>Процессуальные риски</b>\n"
                "   └ Пробелы и недостатки\n\n"
                "💡 <b>Рекомендации</b>\n"
                "   └ Предложения по доработке\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📊 <b>Что вы получите:</b>\n\n"
                "   ✓ Краткое резюме иска\n"
                "   ✓ Правовое обоснование\n"
                "   ✓ Оценка рисков\n"
                "   ✓ Практические советы\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "💼 <b>Полезно для:</b>\n"
                "   • Проверка перед подачей\n"
                "   • Подготовка к суду\n\n"
                "📝 <b>Перед загрузкой:</b>\n"
                "   1. Финальная версия (PDF/DOCX)\n"
                "   2. Ключевые доказательства\n"
                "   3. Описание ситуации"
            ),
            "chat": (
                "💬 <b>Чат с документом</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⚙️ <b>Как это работает:</b>\n\n"
                "❓ <b>Задавайте вопросы</b>\n"
                "   └ По содержанию документа\n\n"
                "🔍 <b>Поиск информации</b>\n"
                "   └ Релевантные фрагменты\n\n"
                "💭 <b>Развернутые ответы</b>\n"
                "   └ Со ссылками на текст\n\n"
                "🔄 <b>Контекст беседы</b>\n"
                "   └ Учитывает предыдущие вопросы\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📊 <b>Что вы получите:</b>\n\n"
                "   ✓ Точные ответы\n"
                "   ✓ Цитаты из документа\n"
                "   ✓ Уточняющие вопросы\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "💼 <b>Полезно для:</b>\n"
                "   • Изучение сложных текстов\n"
                "   • Поиск конкретной информации"
            ),
            "anonymize": (
                "🔒 <b>Обезличивание</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⚙️ <b>Как это работает:</b>\n\n"
                "🔍 <b>Поиск данных</b>\n"
                "   └ Находит персональные данные\n\n"
                "🔄 <b>Замена</b>\n"
                "   └ Безопасные заглушки\n\n"
                "🗑️ <b>Удаление</b>\n"
                "   └ Конфиденциальная информация\n\n"
                "📋 <b>Сохранение</b>\n"
                "   └ Структура и смысл документа\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📊 <b>Что обрабатывается:</b>\n\n"
                "   ✓ ФИО, адреса, телефоны\n"
                "   ✓ Email, номера документов\n"
                "   ✓ Банковские реквизиты\n"
                "   ✓ Персональные идентификаторы\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "💼 <b>Полезно для:</b>\n"
                "   • Передача третьим лицам\n"
                "   • Публичное использование"
            ),
            "translate": (
                "🌍 <b>Перевод документов</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⚙️ <b>Как это работает:</b>\n\n"
                "📄 <b>Перевод текста</b>\n"
                "   └ С сохранением структуры\n\n"
                "⚖️ <b>Терминология</b>\n"
                "   └ Юридическая и техническая\n\n"
                "📐 <b>Форматирование</b>\n"
                "   └ Сохраняет разметку\n\n"
                "🌐 <b>Языки</b>\n"
                "   └ Основные языки мира\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📊 <b>Возможности:</b>\n\n"
                "   ✓ Высокое качество\n"
                "   ✓ Специализированная терминология\n"
                "   ✓ Экспорт в DOCX и TXT\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "💼 <b>Полезно для:</b>\n"
                "   • Международные договоры\n"
                "   • Документооборот с партнерами"
            ),
            "ocr": (
                "🔍 <b>Распознавание текста</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "⚙️ <b>Как это работает:</b>\n\n"
                "📷 <b>Извлечение текста</b>\n"
                "   └ Из сканированных документов\n\n"
                "🖼️ <b>Распознавание</b>\n"
                "   └ Изображения и PDF\n\n"
                "✍️ <b>Типы текста</b>\n"
                "   └ Рукописный и печатный\n\n"
                "🔄 <b>Восстановление</b>\n"
                "   └ Структура документа\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "📊 <b>Что вы получите:</b>\n\n"
                "   ✓ Текстовая версия\n"
                "   ✓ Оценка качества\n"
                "   ✓ Экспорт в форматы\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "💼 <b>Полезно для:</b>\n"
                "   • Старые документы\n"
                "   • Сканы и фотографии"
            )
        }

        detailed_description = detailed_descriptions.get(operation, f"{html_escape(description)}")

        message_text = (
            f"{detailed_description}\n\n"
            f"📄 <b>Поддерживаемые форматы:</b> {html_escape(formats)}\n\n"
            "📎 <b>Загрузите документ для обработки</b>"
        )

        await callback.message.answer(
            message_text,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="◀️ Назад к операциям", callback_data="document_processing")]
                ]
            ),
        )
        await callback.answer()

        # Переходим в состояние ожидания документа
        await state.set_state(DocumentProcessingStates.waiting_for_document)

    except Exception as e:
        await callback.answer(f"Ошибка: {e}")
        logger.error(f"Ошибка в handle_document_operation: {e}", exc_info=True)

async def handle_back_to_menu(callback: CallbackQuery, state: FSMContext):
    """Возврат в главное меню"""
    try:
        if document_manager is not None and callback.from_user:
            document_manager.end_chat_session(callback.from_user.id)

        # Очищаем состояние FSM
        await state.clear()

        # Отправляем главное меню
        await cmd_start(callback.message)
        await callback.answer()

    except Exception as e:
        await callback.answer(f"Ошибка: {e}")
        logger.error(f"Ошибка в handle_back_to_menu: {e}", exc_info=True)


async def handle_retention_quick_question(callback: CallbackQuery):
    """Обработка кнопки 'Задать вопрос' из retention уведомления"""
    try:
        await callback.answer()
        await callback.message.answer(
            f"{Emoji.ROBOT} <b>Отлично!</b>\n\n"
            "Просто напиши свой вопрос, и я отвечу на него.\n\n"
            f"{Emoji.INFO} <i>Пример:</i> Что делать, если нарушили права потребителя?",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error(f"Error in handle_retention_quick_question: {e}", exc_info=True)


async def handle_retention_show_features(callback: CallbackQuery):
    """Обработка кнопки 'Все возможности' из retention уведомления"""
    try:
        await callback.answer()

        features_text = (
            f"{Emoji.ROBOT} <b>Что я умею:</b>\n\n"
            f"{Emoji.QUESTION} <b>Юридические консультации</b>\n"
            "Отвечаю на вопросы по любым правовым темам\n\n"
            f"📄 <b>Работа с документами</b>\n"
            "• Анализ договоров и документов\n"
            "• Поиск рисков и проблем\n"
            "• Режим \"распознание текста\" — извлечение текста из фото\n"
            "• Составление документов\n\n"
            f"📚 <b>Судебная практика</b>\n"
            "Поиск релевантных судебных решений\n\n"
            f"{Emoji.MICROPHONE} <b>Голосовые сообщения</b>\n"
            "Отправь голосовое — получишь голосовой ответ\n\n"
            f"{Emoji.INFO} Просто напиши вопрос или выбери действие!"
        )

        await callback.message.answer(features_text, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Error in handle_retention_show_features: {e}", exc_info=True)

# --- progress router hookup ---
def register_progressbar(dp: Dispatcher) -> None:
    dp.include_router(progress_router)






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


GENERIC_INTERNAL_ERROR_HTML = "<i>Произошла внутренняя ошибка. Попробуйте позже.</i>"
GENERIC_INTERNAL_ERROR_TEXT = "Произошла внутренняя ошибка. Попробуйте позже."


async def handle_document_upload(message: Message, state: FSMContext):
    """Обработка загружённого документа"""
    try:
        if not message.document:
            await message.answer("❌ Ошибка: документ не найден")
            return

        # НОВОЕ: Показываем индикатор "отправляет документ"
        async with typing_action(message.bot, message.chat.id, "upload_document"):
            # Получаем данные из состояния
            data = await state.get_data()
            operation = data.get("document_operation")
            options = dict(data.get("operation_options") or {})
            output_format = str(options.get("output_format", "txt"))
            output_format = str(options.get("output_format", "txt"))

            if not operation:
                await message.answer("❌ Операция не выбрана. Начните заново с /start")
                await state.clear()
                return

            # Переходим в состояние обработки
            await state.set_state(DocumentProcessingStates.processing_document)

            # Информация о файле
            file_name = message.document.file_name or "unknown"
            file_size = message.document.file_size or 0
            mime_type = message.document.mime_type or "application/octet-stream"

            # Проверяем размер файла (максимум 50MB)
            max_size = 50 * 1024 * 1024
            if file_size > max_size:
                reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                await message.answer(
                    f"{Emoji.ERROR} Файл слишком большой. Максимальный размер: {max_size // (1024*1024)} МБ",
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
                await state.clear()
                return

            # Показываем статус обработки
            operation_info = document_manager.get_operation_info(operation) or {}
            operation_name = operation_info.get("name", operation)
            file_size_kb = max(1, file_size // 1024)

            stage_labels = _get_stage_labels(operation)

            status_msg = await message.answer("⏳ Подготавливаем обработку…", parse_mode=ParseMode.HTML)

            send_progress, progress_state = _make_progress_updater(
                message,
                status_msg,
                file_name=file_name,
                operation_name=operation_name,
                file_size_kb=file_size_kb,
                stage_labels=stage_labels,
            )

            await send_progress({"stage": "start", "percent": 5})

            try:
                await send_progress({"stage": "downloading", "percent": 18})
                # Скачиваем файл
                file_info = await message.bot.get_file(message.document.file_id)
                file_path = file_info.file_path

                if not file_path:
                    raise ProcessingError("Не удалось получить путь к файлу", "FILE_ERROR")

                file_content = await message.bot.download_file(file_path)
                try:
                    file_bytes = await asyncio.to_thread(file_content.read)
                finally:
                    close_method = getattr(file_content, "close", None)
                    if callable(close_method):
                        close_method()
                await send_progress({"stage": "uploaded", "percent": 32})

                await send_progress({"stage": "processing", "percent": 45})
                result = await document_manager.process_document(
                    user_id=message.from_user.id,
                    file_content=file_bytes,
                    original_name=file_name,
                    mime_type=mime_type,
                    operation=operation,
                    progress_callback=send_progress,
                    **options,
                )
                await send_progress({"stage": "finalizing", "percent": 90})

                if result.success:
                    # Форматируем результат для Telegram
                    formatted_result = document_manager.format_result_for_telegram(result, operation)

                    # Отправляем результат
                    reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                    await message.answer(formatted_result, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

                    exports = result.data.get("exports") or []
                    for export in exports:
                        export_path = export.get("path")
                        if not export_path:
                            error_msg = export.get("error")
                            if error_msg:
                                await message.answer(f"{Emoji.WARNING} {error_msg}")
                            continue
                        label = export.get("label") or export.get("name")
                        file_name = Path(export_path).name
                        format_tag = str(export.get("format", "file")).upper()
                        parts = [f"📄 {format_tag}"]
                        if label:
                            parts.append(str(label))
                        parts.append(file_name)
                        caption = " • ".join(part for part in parts if part)
                        try:
                            await message.answer_document(FSInputFile(export_path), caption=caption)
                        except Exception as send_error:
                            logger.error(
                                f"Не удалось отправить файл {export_path}: {send_error}", exc_info=True
                            )
                            await message.answer(
                                f"Не удалось отправить файл {file_name}"
                            )
                        finally:
                            with suppress(Exception):
                                Path(export_path).unlink(missing_ok=True)

                    completion_payload = _build_completion_payload(operation, result)
                    await send_progress({'stage': 'completed', 'percent': 100, **completion_payload})
                    with suppress(Exception):
                        await asyncio.sleep(0.6)
                        await status_msg.delete()

                    logger.info(
                        f"Successfully processed document {file_name} for user {message.from_user.id}"
                    )
                else:
                    await send_progress({'stage': 'failed', 'percent': progress_state['percent'], 'note': result.message})
                    reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                    await message.answer(
                        f"{Emoji.ERROR} <b>Ошибка обработки документа</b>\n\n{html_escape(str(result.message))}",
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup,
                    )
                    with suppress(Exception):
                        await status_msg.delete()

            except Exception as e:
                # Удаляем статусное сообщение в случае ошибки
                try:
                    await status_msg.delete()
                except:
                    pass

                reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                await message.answer(
                    f"{Emoji.ERROR} <b>Ошибка обработки документа</b>\n\n{GENERIC_INTERNAL_ERROR_HTML}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
                logger.error(f"Error processing document {file_name}: {e}", exc_info=True)

            finally:
                # Очищаем состояние
                await state.clear()

    except Exception as e:
        reply_markup = None
        if 'operation' in locals() and operation == "ocr":
            reply_markup = _build_ocr_reply_markup(locals().get('output_format', 'txt'))
        await message.answer(
            f"{Emoji.ERROR} <b>Произошла ошибка</b>\n\n{GENERIC_INTERNAL_ERROR_HTML}",
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
        logger.error(f"Error in handle_document_upload: {e}", exc_info=True)
        await state.clear()


async def handle_photo_upload(message: Message, state: FSMContext):
    """Обработка загруженной фотографии для режима "распознание текста"."""
    try:
        if not message.photo:
            await message.answer("❌ Ошибка: фотография не найдена")
            return

        # Показываем индикатор "отправляет фото"
        async with typing_action(message.bot, message.chat.id, "upload_photo"):
            # Получаем данные из состояния
            data = await state.get_data()
            operation = data.get("document_operation")
            options = dict(data.get("operation_options") or {})
            output_format = str(options.get("output_format", "txt"))
            output_format = str(options.get("output_format", "txt"))

            if not operation:
                await message.answer("❌ Операция не выбрана. Начните заново с /start")
                await state.clear()
                return

            # Переходим в состояние обработки
            await state.set_state(DocumentProcessingStates.processing_document)

            # Получаем самую большую версию фотографии
            photo = message.photo[-1]
            file_name = f"photo_{photo.file_id}.jpg"
            file_size = photo.file_size or 0
            mime_type = "image/jpeg"

            # Проверяем размер файла (максимум 20MB для фотографий)
            max_size = 20 * 1024 * 1024
            if file_size > max_size:
                await message.answer(
                    f"❌ Фотография слишком большая. Максимальный размер: {max_size // (1024*1024)} МБ"
                )
                await state.clear()
                return

            # Показываем статус обработки
            operation_info = document_manager.get_operation_info(operation) or {}
            operation_name = operation_info.get("name", operation)

            file_size_kb = max(1, file_size // 1024)
            stage_labels = _get_stage_labels(operation)

            status_msg = await message.answer(
                f"📷 Обрабатываем фотографию для режима \"распознание текста\"...\n\n"
                f"⏳ Операция: {html_escape(operation_name)}\n"
                f"📏 Размер: {file_size_kb} КБ",
                parse_mode=ParseMode.HTML,
            )

            send_progress, progress_state = _make_progress_updater(
                message,
                status_msg,
                file_name=file_name,
                operation_name=operation_name,
                file_size_kb=file_size_kb,
                stage_labels=stage_labels,
            )

            try:
                await send_progress({"stage": "start", "percent": 5})

                # Скачиваем фотографию
                file_info = await message.bot.get_file(photo.file_id)
                file_path = file_info.file_path

                if not file_path:
                    raise ProcessingError("Не удалось получить путь к фотографии", "FILE_ERROR")

                file_content = await message.bot.download_file(file_path)
                try:
                    file_bytes = await asyncio.to_thread(file_content.read)
                finally:
                    close_method = getattr(file_content, "close", None)
                    if callable(close_method):
                        close_method()
                await send_progress({"stage": "uploaded", "percent": 32})

                await send_progress({"stage": "processing", "percent": 45})
                result = await document_manager.process_document(
                    user_id=message.from_user.id,
                    file_content=file_bytes,
                    original_name=file_name,
                    mime_type=mime_type,
                    operation=operation,
                    progress_callback=send_progress,
                    **options,
                )
                await send_progress({"stage": "finalizing", "percent": 90})

                if result.success:
                    # Форматируем результат для Telegram
                    formatted_result = document_manager.format_result_for_telegram(result, operation)

                    # Отправляем результат
                    reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                    await message.answer(formatted_result, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

                    # Отправляем экспортированные файлы, если есть
                    exports = result.data.get("exports") or []
                    for export in exports:
                        export_path = export.get("path")
                        if not export_path:
                            error_msg = export.get("error")
                            if error_msg:
                                await message.answer(f"{Emoji.WARNING} {error_msg}")
                            continue
                        label = export.get("label") or export.get("name")
                        file_name = Path(export_path).name
                        format_tag = str(export.get("format", "file")).upper()
                        parts = [f"📄 {format_tag}"]
                        if label:
                            parts.append(str(label))
                        parts.append(file_name)
                        caption = " • ".join(part for part in parts if part)
                        try:
                            await message.answer_document(FSInputFile(export_path), caption=caption)
                        except Exception as send_error:
                            logger.error(
                                f"Не удалось отправить файл {export_path}: {send_error}", exc_info=True
                            )
                            await message.answer(
                                f"Не удалось отправить файл {file_name}"
                            )
                        finally:
                            with suppress(Exception):
                                Path(export_path).unlink(missing_ok=True)

                    completion_payload = _build_completion_payload(operation, result)
                    await send_progress({"stage": "completed", "percent": 100, **completion_payload})
                    with suppress(Exception):
                        await asyncio.sleep(0.6)
                        await status_msg.delete()

                    logger.info(
                        f"Successfully processed photo {file_name} for user {message.from_user.id}"
                    )
                else:
                    await send_progress(
                        {"stage": "failed", "percent": progress_state["percent"], "note": result.message}
                    )
                    reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                    await message.answer(
                        f"{Emoji.ERROR} <b>Ошибка обработки фотографии</b>\n\n{html_escape(str(result.message))}",
                        parse_mode=ParseMode.HTML,
                        reply_markup=reply_markup,
                    )

            except Exception as e:
                # Удаляем статусное сообщение
                try:
                    await send_progress(
                        {"stage": "failed", "percent": progress_state["percent"], "note": GENERIC_INTERNAL_ERROR_TEXT}
                    )
                    await status_msg.delete()
                except:
                    pass

                reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                await message.answer(
                    f"{Emoji.ERROR} <b>Ошибка обработки фотографии</b>\n\n{GENERIC_INTERNAL_ERROR_HTML}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
                logger.error(f"Error processing photo {file_name}: {e}", exc_info=True)

            finally:
                # Очищаем состояние
                await state.clear()

    except Exception as e:
        await message.answer("❌ Произошла внутренняя ошибка. Попробуйте позже.")
        logger.error(f"Error in handle_photo_upload: {e}", exc_info=True)
        await state.clear()


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
    dp.message.register(cmd_ratings_stats, Command("ratings"))
    dp.message.register(cmd_error_stats, Command("errors"))
    dp.message.register(cmd_askdoc, Command("askdoc"))
    dp.message.register(cmd_enddoc, Command("enddoc"))

    dp.callback_query.register(handle_rating_callback, F.data.startswith("rate_"))
    dp.callback_query.register(
        handle_feedback_callback, F.data.startswith(("feedback_", "skip_feedback_"))
    )

    # Обработчики кнопок главного меню
    dp.callback_query.register(handle_search_practice_callback, F.data == "search_practice")
    dp.callback_query.register(handle_prepare_documents_callback, F.data == "prepare_documents")
    
    # Обработчики профиля

    # Обработчики retention уведомлений
    dp.callback_query.register(handle_retention_quick_question, F.data == "quick_question")
    dp.callback_query.register(handle_retention_show_features, F.data == "show_features")

    # Обработчики системы документооборота
    dp.callback_query.register(handle_doc_draft_start, F.data == "doc_draft_start")
    dp.callback_query.register(handle_doc_draft_cancel, F.data == "doc_draft_cancel")
    dp.callback_query.register(handle_document_processing, F.data == "document_processing")
    dp.callback_query.register(handle_document_operation, F.data.startswith("doc_operation_"))
    dp.callback_query.register(handle_ocr_upload_more, F.data.startswith("ocr_upload_more:"))
    dp.callback_query.register(handle_back_to_menu, F.data == "back_to_menu")
    dp.message.register(
        handle_doc_draft_request, DocumentDraftStates.waiting_for_request, F.text
    )
    dp.message.register(
        handle_doc_draft_request_voice, DocumentDraftStates.waiting_for_request, F.voice
    )
    dp.message.register(
        handle_doc_draft_answer, DocumentDraftStates.asking_details, F.text
    )
    dp.message.register(
        handle_doc_draft_answer_voice, DocumentDraftStates.asking_details, F.voice
    )
    dp.message.register(
        handle_document_upload, DocumentProcessingStates.waiting_for_document, F.document
    )
    dp.message.register(
        handle_photo_upload, DocumentProcessingStates.waiting_for_document, F.photo
    )

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
