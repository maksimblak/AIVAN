"""
Простая версия Telegram бота ИИ-Иван
Только /start и обработка вопросов, никаких кнопок и лишних команд
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import tempfile
from contextlib import suppress
from datetime import datetime
from pathlib import Path
import uuid
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
    LabeledPrice,
    Message,
    PreCheckoutQuery,
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
from src.bot.ui_components import Emoji, sanitize_telegram_html
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
from src.core.payments import CryptoPayProvider, convert_rub_to_xtr
from src.core.subscription_payments import (
    build_subscription_payload,
    parse_subscription_payload,
    SubscriptionPayloadError,
)
from src.core.admin_modules.admin_commands import setup_admin_commands
from src.core.session_store import SessionStore, UserSession
from src.core.validation import InputValidator, ValidationSeverity
from src.core.runtime import AppRuntime, DerivedRuntime, SubscriptionPlanPricing, WelcomeMedia
from src.core.settings import AppSettings
from src.core.app_context import set_settings
from src.documents.base import ProcessingError
from src.bot.ratelimit import RateLimiter
from src.bot.typing_indicator import send_typing_once, typing_action

SAFE_LIMIT = 3900  # Buffer below Telegram 4096 character limit
QUESTION_ATTACHMENT_MAX_BYTES = 4 * 1024 * 1024  # 4MB per attachment (base64-safe)

VOICE_REPLY_CAPTION = (
    f"{Emoji.MICROPHONE} <b>Голосовой ответ готов</b>"
    f"\n{Emoji.INFO} Нажмите, чтобы прослушать."
)

PERIOD_OPTIONS = (7, 30, 90)
PROGRESS_BAR_LENGTH = 10
FEATURE_LABELS = {
    "legal_question": "Юридические вопросы",
    "document_processing": "Обработка документов",
    "judicial_practice": "Судебная практика",
    "document_draft": "Составление документов",
    "voice_message": "Голосовые сообщения",
    "ocr_processing": "распознание текста",
    "document_chat": "Чат с документом",
}

SECTION_DIVIDER = "<code>────────────────────</code>"

def _build_stats_keyboard(has_subscription: bool) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    if not has_subscription:
        buttons.append([InlineKeyboardButton(text="💳 Оформить подписку", callback_data="get_subscription")])
    buttons.append([InlineKeyboardButton(text="🔙 Назад к профилю", callback_data="my_profile")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


DAY_NAMES = {
    "0": "Вс",
    "1": "Пн",
    "2": "Вт",
    "3": "Ср",
    "4": "Чт",
    "5": "Пт",
    "6": "Сб",
}

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

# Runtime-managed globals (synchronised via refresh_runtime_globals)
# Defaults keep module usable before runtime initialisation.
WELCOME_MEDIA: WelcomeMedia | None = None
BOT_TOKEN = ""
BOT_USERNAME = ""
USE_ANIMATION = True
USE_STREAMING = True
config: Any | None = None
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

# Service-like dependencies
db = None
rate_limiter = None
access_service = None
openai_service = None
audio_service = None
session_store = None
crypto_provider = None
robokassa_provider = None
yookassa_provider = None
error_handler = None
document_manager = None
response_cache = None
stream_manager = None
metrics_collector = None
task_manager = None
health_checker = None
scaling_components = None
judicial_rag = None
retention_notifier = None


async def _ensure_rating_snapshot(request_id: int, telegram_user: User | None, answer_text: str) -> None:
    if db is None or not answer_text.strip():
        return
    if telegram_user is None:
        return
    add_rating_fn = _get_safe_db_method("add_rating", default_return=False)
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

_runtime: AppRuntime | None = None


def set_runtime(runtime: AppRuntime) -> None:
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


def _sync_runtime_globals() -> None:
    if _runtime is None:
        return
    cfg = _runtime.settings
    drv = _runtime.derived
    g = globals()
    g.update({
        'WELCOME_MEDIA': drv.welcome_media,
        'BOT_TOKEN': cfg.telegram_bot_token,
        'USE_ANIMATION': cfg.use_status_animation,
        'USE_STREAMING': cfg.use_streaming,
        'MAX_MESSAGE_LENGTH': drv.max_message_length,
        'SAFE_LIMIT': drv.safe_limit,
        'DB_PATH': cfg.db_path,
        'TRIAL_REQUESTS': cfg.trial_requests,
        'SUB_DURATION_DAYS': cfg.sub_duration_days,
        'RUB_PROVIDER_TOKEN': cfg.telegram_provider_token_rub,
        'SUB_PRICE_RUB': cfg.subscription_price_rub,
        'SUB_PRICE_RUB_KOPEKS': drv.subscription_price_rub_kopeks,
        'STARS_PROVIDER_TOKEN': cfg.telegram_provider_token_stars,
        'SUB_PRICE_XTR': cfg.subscription_price_xtr,
        'DYNAMIC_PRICE_XTR': drv.dynamic_price_xtr,
        'SUBSCRIPTION_PLANS': drv.subscription_plans,
        'SUBSCRIPTION_PLAN_MAP': drv.subscription_plan_map,
        'DEFAULT_SUBSCRIPTION_PLAN': drv.default_subscription_plan,
        'ADMIN_IDS': drv.admin_ids,
        'USER_SESSIONS_MAX': cfg.user_sessions_max,
        'USER_SESSION_TTL_SECONDS': cfg.user_session_ttl_seconds,
        'db': _runtime.db,
        'rate_limiter': _runtime.rate_limiter,
        'access_service': _runtime.access_service,
        'openai_service': _runtime.openai_service,
        'audio_service': _runtime.audio_service,
        'session_store': _runtime.session_store,
        'crypto_provider': _runtime.crypto_provider,
        'robokassa_provider': _runtime.robokassa_provider,
        'yookassa_provider': _runtime.yookassa_provider,
        'error_handler': _runtime.error_handler,
        'document_manager': _runtime.document_manager,
        'response_cache': _runtime.response_cache,
        'stream_manager': _runtime.stream_manager,
        'metrics_collector': _runtime.metrics_collector,
        'task_manager': _runtime.task_manager,
        'health_checker': _runtime.health_checker,
        'scaling_components': _runtime.scaling_components,
        'judicial_rag': _runtime.get_dependency('judicial_rag'),
    })


def refresh_runtime_globals() -> None:
    _sync_runtime_globals()

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


def get_user_session(user_id: int) -> UserSession:
    if session_store is None:
        raise RuntimeError("Session store not initialized")
    return session_store.get_or_create(user_id)


def _ensure_valid_user_id(raw_user_id: int | None, *, context: str) -> int:
    """Validate and normalise user id, raising ValidationException when invalid."""

    result = InputValidator.validate_user_id(raw_user_id)
    if result.is_valid and result.cleaned_data:
        return int(result.cleaned_data)

    errors = ', '.join(result.errors or ['Недопустимый идентификатор пользователя'])
    try:
        normalized_user_id = int(raw_user_id) if raw_user_id is not None else None
    except (TypeError, ValueError):
        normalized_user_id = None

    raise ValidationException(
        errors,
        ErrorContext(user_id=normalized_user_id, function_name=context),
    )


def _get_safe_db_method(method_name: str, default_return=None):
    """Return DB coroutine when available."""

    _ = default_return  # backward compatibility

    if db is None or not hasattr(db, method_name):
        return None

    return getattr(db, method_name)

# ============ УТИЛИТЫ ============


def chunk_text(text: str, max_length: int | None = None) -> list[str]:
    """Split long Telegram messages into chunks respecting limits."""
    limit = max_length or derived().max_message_length
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current_chunk = ''

    for paragraph in text.split('\n\n'):
        if len(current_chunk + paragraph + '\n\n') <= limit:
            current_chunk += paragraph + '\n\n'
        elif current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n\n'
        else:
            while len(paragraph) > limit:
                chunks.append(paragraph[:limit])
                paragraph = paragraph[limit:]
            current_chunk = paragraph + '\n\n'

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def _resolve_bot_username() -> str:
    """Return cached bot username or fallback from settings."""
    username = (BOT_USERNAME or '').strip()
    if username.startswith('@'):
        username = username[1:]
    if username:
        return username
    try:
        env_username = settings().get_str('TELEGRAM_BOT_USERNAME')
    except Exception:
        env_username = None
    if env_username:
        env_username = env_username.strip()
        if env_username.startswith('https://t.me/'):
            env_username = env_username[len('https://t.me/'):]
        elif env_username.startswith('t.me/'):
            env_username = env_username[len('t.me/'):]
        if env_username.startswith('@'):
            env_username = env_username[1:]
        if env_username:
            return env_username
    return ''


def _build_referral_link(referral_code: str | None) -> tuple[str | None, str | None]:
    """Compose deep link for referral code; returns (link, code)."""
    if not referral_code or referral_code == 'SYSTEM_ERROR':
        return None, None
    safe_code = html_escape(referral_code)
    username = _resolve_bot_username()
    if username:
        return f'https://t.me/{username}?start=ref_{safe_code}', referral_code
    try:
        fallback_base = settings().get_str('TELEGRAM_REFERRAL_BASE_URL')
    except Exception:
        fallback_base = None
    if fallback_base:
        base = fallback_base.strip().rstrip('/')
        if base:
            if not base.startswith('http'):
                base = f"https://{base.lstrip('/')}"
            return f"{base}?start=ref_{safe_code}", referral_code
    return None, referral_code


# Удалено: используется md_links_to_anchors из ui_components



    

def _split_plain_text(text: str, limit: int = SAFE_LIMIT) -> list[str]:
    if not text:
        return []

    if len(text) <= limit:
        return [text]

    line_sep = chr(10)
    paragraph_sep = line_sep * 2
    chunks: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            chunks.append(paragraph_sep.join(current))
            current.clear()

    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines():
        if line.strip():
            buffer.append(line)
        else:
            if buffer:
                paragraphs.append(line_sep.join(buffer))
                buffer.clear()
    if buffer:
        paragraphs.append(line_sep.join(buffer))

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        joined = paragraph_sep.join(current + [paragraph])
        if len(joined) <= limit:
            current.append(paragraph)
            continue
        flush_current()
        if len(paragraph) > limit:
            for i in range(0, len(paragraph), limit):
                chunks.append(paragraph[i : i + limit])
        else:
            current.append(paragraph)
    flush_current()

    return chunks


def _split_html_safely(html: str, hard_limit: int = SAFE_LIMIT) -> list[str]:
    """
    Режем уже готовый HTML аккуратно:
      1) по пустой строке  -> <br><br>
      2) по одиночному <br>
      3) по предложениям (. ! ?)
      4) в крайнем случае — жёсткая нарезка.
    Стараемся не разрывать теги; режем только на границах <br> или текста.
    """
    if not html:
        return []

    # нормализуем варианты переносов
    h = re.sub(r"<br\s*/?>", "<br>", html, flags=re.IGNORECASE)

    chunks: list[str] = []

    def _pack(parts: list[str], sep: str) -> list[str]:
        out, cur, ln = [], [], 0
        for p in parts:
            add = p
            # учтём разделитель между элементами
            sep_len = len(sep) if cur else 0
            if ln + sep_len + len(add) <= hard_limit:
                if cur:
                    cur.append(sep)
                cur.append(add)
                ln += sep_len + len(add)
            else:
                if cur:
                    out.append("".join(cur))
                cur, ln = [add], len(add)
        if cur:
            out.append("".join(cur))
        return out

    # 1) крупные параграфы: <br><br>
    paras = re.split(r"(?:<br>\s*){2,}", h)
    tmp = _pack(paras, "<br><br>")

    # 2) если что-то всё ещё длиннее лимита — режем по одиночным <br>
    next_stage: list[str] = []
    for block in tmp:
        if len(block) <= hard_limit:
            next_stage.append(block)
            continue
        lines = block.split("<br>")
        next_stage.extend(_pack(lines, "<br>"))

    # 3) если всё ещё длинно — режем по предложениям
    final: list[str] = []
    sent_re = re.compile(r"(?<=[\.\!\?])\s+")
    for block in next_stage:
        if len(block) <= hard_limit:
            final.append(block)
            continue
        sentences = sent_re.split(block)
        if len(sentences) > 1:
            final.extend(_pack(sentences, " "))
        else:
            # 4) крайний случай — жёсткая нарезка без учёта предложений
            for i in range(0, len(block), hard_limit):
                final.append(block[i : i + hard_limit])

    return [b.strip() for b in final if b.strip()]




async def _download_telegram_file(bot: Bot, file_id: str) -> bytes:
    file_info = await bot.get_file(file_id)
    file_path = getattr(file_info, "file_path", None)
    if not file_path:
        raise ValueError("Не удалось получить путь к файлу в Telegram.")
    file_content = await bot.download_file(file_path)
    return file_content.read()


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


def _format_datetime(ts: int | None, *, default: str = "Никогда") -> str:
    if not ts or ts <= 0:
        return default
    try:
        return datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return default


def _format_response_time(ms: int) -> str:
    if ms <= 0:
        return "—"
    if ms < 1000:
        return f"{ms} мс"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes = seconds / 60
    return f"{minutes:.1f} мин"


def _format_number(value: int | float) -> str:
    if value is None:
        return "0"
    if isinstance(value, float):
        return f"{value:,.1f}".replace(",", " ")
    return f"{int(value):,}".replace(",", " ")


def _format_trend_value(current: int, previous: int) -> str:
    delta = current - previous
    if previous <= 0:
        if current == 0:
            return "0"
        return f"{current} (▲)"
    if delta == 0:
        return f"{current} (=)"
    sign = "+" if delta > 0 else ""
    pct = (delta / previous) * 100
    pct_sign = "+" if pct > 0 else ""
    return f"{current} ({sign}{delta} / {pct_sign}{pct:.0f}%)"


def _normalize_stats_period(days: int) -> int:
    if days <= 0:
        return PERIOD_OPTIONS[0]
    for option in PERIOD_OPTIONS:
        if days <= option:
            return option
    return PERIOD_OPTIONS[-1]


def _build_progress_bar(used: int, total: int) -> str:
    if total is None or total <= 0:
        return "<code>[██████████]</code> ∞ / <b>Безлимит</b>"

    total = max(total, 0)
    used = max(0, min(used, total))

    ratio = used / total if total else 0.0
    filled = min(PROGRESS_BAR_LENGTH, max(0, int(round(ratio * PROGRESS_BAR_LENGTH))))
    bar = f"[{'█' * filled}{'░' * (PROGRESS_BAR_LENGTH - filled)}]"
    bar_markup = f"<code>{bar}</code>"

    remaining = max(0, total - used)
    if total:
        remaining_pct = max(0, min(100, int(round((remaining / total) * 100))))
    else:
        remaining_pct = 0

    return f"{bar_markup} {used}/{total} · осталось <b>{remaining}</b> ({remaining_pct}%)"


def _progress_line(label: str, used: int, total: int) -> str:
    return f"<b>{label}</b> {_build_progress_bar(used, total)}"


def _format_stat_row(label: str, value: str) -> str:
    return f"<b>{label}</b> · {value}"


def _translate_payment_status(status: str) -> str:
    """Переводит статус платежа на русский язык"""
    status_map = {
        "pending": "⏳ Ожидание",
        "processing": "🔄 Обработка",
        "succeeded": "✅ Успешно",
        "success": "✅ Успешно",
        "completed": "✅ Завершён",
        "failed": "❌ Ошибка",
        "cancelled": "🚫 Отменён",
        "canceled": "🚫 Отменён",
        "refunded": "↩️ Возврат",
        "unknown": "❓ Неизвестно",
    }
    return status_map.get(status.lower(), status)


def _translate_plan_name(plan_id: str) -> str:
    """Переводит название тарифа на русский язык"""
    # Словарь для перевода базовых названий тарифов
    plan_map = {
        "basic": "Базовый",
        "standard": "Стандарт",
        "premium": "Премиум",
        "pro": "Про",
        "trial": "Триал",
    }

    # Словарь для перевода периодов
    period_map = {
        "1m": "1 месяц",
        "3m": "3 месяца",
        "6m": "6 месяцев",
        "12m": "1 год",
        "1y": "1 год",
    }

    # Разбираем plan_id (например, "standard_1m" -> "Стандарт • 1 месяц")
    parts = plan_id.split("_")
    if len(parts) >= 2:
        plan_name = plan_map.get(parts[0].lower(), parts[0].capitalize())
        period = period_map.get(parts[1].lower(), parts[1])
        return f"{plan_name} • {period}"

    # Если не удалось разобрать, пробуем найти в словаре целиком
    return plan_map.get(plan_id.lower(), plan_id)


def _describe_primary_summary(summary: str, unit: str) -> str:
    if not summary or summary == "—":
        return "нет данных"
    if "(" in summary and summary.endswith(")"):
        label, count = summary.rsplit("(", 1)
        label = label.strip()
        count = count[:-1].strip()
        if count.isdigit():
            return f"{label} — {count} {unit}"
        return f"{label} — {count}"
    return summary


def _describe_secondary_summary(summary: str, unit: str) -> str:
    if not summary:
        return "нет данных"
    parts = []
    for raw in summary.split(","):
        item = raw.strip()
        if not item:
            continue
        tokens = item.split()
        if len(tokens) >= 2 and tokens[-1].isdigit():
            count = tokens[-1]
            label = " ".join(tokens[:-1])
            parts.append(f"{label} — {count}")
        else:
            parts.append(item)
    if not parts:
        return "нет данных"
    return "; ".join(parts)


def _peak_summary(
    counts: dict[str, int],
    *,
    mapping: dict[str, str] | None = None,
    formatter: Callable[[str], str] | None = None,
    secondary_limit: int = 3,
) -> tuple[str, str]:
    if not counts:
        return "—", ""

    sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    def _render(raw_key: str) -> str:
        label = mapping.get(raw_key, raw_key) if mapping else raw_key
        return formatter(label) if formatter else label

    primary_key, primary_count = sorted_items[0]
    primary_label = _render(str(primary_key))
    primary = f"{primary_label} ({primary_count})"

    secondary_parts: list[str] = []
    for key, count in sorted_items[1:secondary_limit]:
        secondary_parts.append(f"{_render(str(key))} {count}")

    secondary = ", ".join(secondary_parts)
    return primary, secondary

def _top_labels(
    counts: dict[str, int],
    *,
    mapping: dict[str, str] | None = None,
    limit: int = 3,
    formatter: Callable[[str], str] | None = None,
) -> str:
    if not counts:
        return "—"
    top_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    labels: list[str] = []
    for key, value in top_items:
        label = mapping.get(key, key) if mapping else key
        if formatter:
            label = formatter(label)
        labels.append(f"{label}×{value}")
    return ", ".join(labels)


def _format_hour_label(hour: str) -> str:
    if not hour:
        return hour
    try:
        hour_int = int(hour)
        return f"{hour_int:02d}:00"
    except ValueError:
        return hour


def _format_currency(amount_minor: int | None, currency: str) -> str:
    if amount_minor is None:
        return f"0 {currency.upper()}"
    if currency.upper() == "RUB":
        value = amount_minor / 100
        return f"{value:,.2f} ₽".replace(",", " ")
    return f"{amount_minor} {currency.upper()}"


def _build_recommendations(
    *,
    trial_remaining: int,
    has_subscription: bool,
    subscription_days_left: int,
    period_requests: int,
    previous_requests: int,
) -> list[str]:
    tips: list[str] = []
    if not has_subscription:
        if trial_remaining > 0:
            tips.append(
                f"Используйте оставшиеся {trial_remaining} запросов триала и оформите подписку через /buy."
            )
        else:
            tips.append("Подключите подписку для безлимитного доступа — /buy.")
    else:
        if subscription_days_left <= 5:
            tips.append("Продлите подписку заранее, чтобы не потерять доступ — кнопка ниже.")

    if period_requests == 0:
        tips.append("Задайте боту первый вопрос — начните с /start или загрузите документ.")
    elif period_requests < previous_requests:
        tips.append("Активность снизилась — попробуйте использовать подборки или вопросы к документам.")

    if not tips:
        tips.append("Продолжайте спрашивать — бот быстрее реагирует на частые запросы.")
    return tips[:3]


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


def _format_risk_count(count: int) -> str:
    count = int(count)
    suffix = "рисков"
    if count % 10 == 1 and count % 100 != 11:
        suffix = "риск"
    elif count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        suffix = "риска"
    return f"Найдено {count} {suffix}"


def _format_progress_extras(update: dict[str, Any]) -> str:
    parts: list[str] = []
    if update.get("risks_found") is not None:
        parts.append(_format_risk_count(update["risks_found"]))
    if update.get("violations") is not None:
        parts.append(f"⚖️ Нарушений: {int(update['violations'])}")
    if update.get("chunks_total") and update.get("chunk_index"):
        parts.append(f"🧩 Блок {int(update['chunk_index'])}/{int(update['chunks_total'])}")
    elif update.get("chunks_total") is not None:
        parts.append(f"🧩 Блоков: {int(update['chunks_total'])}")
    if update.get("language_pair"):
        parts.append(f"🌐 {html_escape(str(update['language_pair']))}")
    if update.get("mode"):
        parts.append(f"⚙️ Режим: {html_escape(str(update['mode']))}")
    if update.get("pages_total") is not None:
        done = int(update.get("pages_done") or 0)
        total = int(update["pages_total"])
        parts.append(f"📑 Страницы: {done}/{total}")
    if update.get("masked") is not None:
        parts.append(f"🔐 Заменено: {int(update['masked'])}")
    if update.get("words") is not None:
        parts.append(f"📝 Слов: {int(update['words'])}")
    if update.get("confidence") is not None:
        parts.append(f"🎯 Точность: {float(update['confidence']):.1f}%")
    if update.get("note"):
        parts.append(f"⚠️ {html_escape(str(update['note']))}")
    return " | ".join(parts)


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
        user_id = _ensure_valid_user_id(message.from_user.id, context="send_rating_request")
    except ValidationException as exc:
        logger.debug("Skip rating request due to invalid user id: %s", exc)
        return

    get_rating_fn = _get_safe_db_method("get_rating", default_return=None)
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


async def _try_send_welcome_media(
    message: Message,
    caption_html: str,
    keyboard: Optional[InlineKeyboardMarkup],
) -> bool:
    """Send welcome media via cached file id or local file when available."""
    if not WELCOME_MEDIA:
        return False

    media_type = (WELCOME_MEDIA.media_type or "video").lower()
    media_source = None
    supports_streaming = False
    media_caption = caption_html

    if WELCOME_MEDIA.file_id:
        media_source = WELCOME_MEDIA.file_id
        supports_streaming = media_type == "video"
    elif WELCOME_MEDIA.path and WELCOME_MEDIA.path.exists():
        media_source = FSInputFile(WELCOME_MEDIA.path)
        supports_streaming = media_type == "video"
    else:
        return False

    try:
        if media_type == "animation":
            await message.answer_animation(
                animation=media_source,
                caption=media_caption,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
        elif media_type == "photo":
            await message.answer_photo(
                photo=media_source,
                caption=media_caption,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
        else:
            await message.answer_video(
                video=media_source,
                caption=media_caption,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
                supports_streaming=supports_streaming,
            )
        return True
    except Exception as media_error:  # noqa: BLE001
        logger.warning("Failed to send welcome media: %s", media_error)
        return False


def _profile_menu_text(
    user: User | None = None,
    *,
    status_text: str | None = None,
    tariff_text: str | None = None,
    hint_text: str | None = None,
) -> str:
    """Build profile menu header with a compact card-style layout."""

    def _display_name(person: User | None) -> str:
        if person is None:
            return "—"
        parts = [part.strip() for part in (person.first_name or "", person.last_name or "") if part.strip()]
        if parts:
            return " ".join(parts)
        if person.username:
            return f"@{person.username}"
        try:
            return str(person.id)
        except Exception:
            return "—"

    name_html = html_escape(_display_name(user))

    card_lines: list[str] = [
        "┏━━━━━━━━━━━━━━━━━━━━━┓",
        "┃   📇 <b>Ваш профиль</b>      ┃",
        "┗━━━━━━━━━━━━━━━━━━━━━┛",
        "",
        f"👤 <b>Пользователь</b>",
        f"   └ {name_html}",
    ]

    if status_text or tariff_text:
        card_lines.append("")
        card_lines.append("📊 <b>Подписка</b>")

    if status_text:
        card_lines.append(f"   ├ Статус: {status_text}")
    if tariff_text:
        prefix = "   └" if not status_text else "   └"
        card_lines.append(f"{prefix} Тариф: {tariff_text}")

    if hint_text:
        card_lines.append("")
        card_lines.append(f"💡 <b>Совет:</b> {html_escape(hint_text)}")

    card_lines.extend([
        "",
        "─────────────────────",
        "",
        "🎯 <b>Что можно сделать:</b>",
        "   • Посмотреть статистику",
        "   • Управлять подпиской",
        "   • Реферальная программа",
        "",
        "💼 <i>Обучение работе с ИИ-ИВАНОМ — /help</i>",
    ])
    return "\n".join(card_lines)


def _profile_menu_keyboard(subscribe_label: str | None = None, *, has_subscription: bool = False) -> InlineKeyboardMarkup:
    if has_subscription:
        change_button = InlineKeyboardButton(text="🔄 Сменить тариф", callback_data="buy_catalog")
        cancel_label = subscribe_label or "❌ Отменить подписку"
        cancel_button = InlineKeyboardButton(text=cancel_label, callback_data="cancel_subscription")
        return InlineKeyboardMarkup(inline_keyboard=[[change_button], [cancel_button]])

    first_label = subscribe_label or "💳 Оформить подписку"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=first_label, callback_data="get_subscription"),
            ],
            [
                InlineKeyboardButton(text="📊 Моя статистика", callback_data="my_stats"),
                InlineKeyboardButton(text="👥 Реферальная программа", callback_data="referral_program"),
            ],
            [
                InlineKeyboardButton(text="↩️ Назад в меню", callback_data="back_to_main"),
            ],
        ]
    )


def _main_menu_text() -> str:
    return (
        "┏━━━━━━━━━━━━━━━━━━━━━┓\n"
        "┃  🏠 <b>Главное меню</b>    ┃\n"
        "┗━━━━━━━━━━━━━━━━━━━━━┛\n\n"
        "⚖️ <b>ИИ-ИВАН</b> — ваш виртуальный юридический ассистент\n\n"
        "🎯 <b>Доступные возможности:</b>\n"
        "   • Поиск судебной практики\n"
        "   • Работа с документами\n"
        "   • Юридические консультации\n\n"
        "Выберите действие:"
    )


def _main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🔍 Поиск судебной практики", callback_data="search_practice"),
            ],
            [
                InlineKeyboardButton(text="🗂️ Работа с документами", callback_data="document_processing"),
            ],
            [
                InlineKeyboardButton(text="👤 Мой профиль", callback_data="my_profile"),
                InlineKeyboardButton(text="💬 Поддержка", callback_data="help_info"),
            ],
        ]
    )


async def cmd_start(message: Message):
    """Единственная команда - приветствие"""
    if not message.from_user:
        return

    try:
        user_id = _ensure_valid_user_id(message.from_user.id, context="cmd_start")
    except ValidationException as exc:
        context = ErrorContext(function_name="cmd_start", chat_id=message.chat.id if message.chat else None)
        if error_handler:
            await error_handler.handle_exception(exc, context)
        else:
            logger.warning("Validation error in cmd_start: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} <b>Не удалось инициализировать сессию.</b>\nПопробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    user_session = get_user_session(user_id)  # noqa: F841 (инициализация)
    # Обеспечим запись в БД
    if db is not None and hasattr(db, "ensure_user"):
        await db.ensure_user(
            user_id,
            default_trial=TRIAL_REQUESTS,
            is_admin=user_id in ADMIN_IDS,
        )
    user_name = message.from_user.first_name or "Пользователь"

    # Подробное приветствие

    welcome_raw = f"""<b>Добро пожаловать, {user_name}!</b>

Меня зовут <b>ИИ-ИВАН</b>, я ваш виртуальный юридический ассистент.

<b>ЧТО Я УМЕЮ:</b>

<b>Юридические вопросы</b>
— составляю выигрышные стратегии, даю быстрые консультации, проверяю аргументы на ошибки
и «человеческий фактор».

<b>Поиск и анализ судебной практики</b>
— анализирую миллионы дел и подбираю релевантные решения: какова вероятность успеха и как суд
трактует норму.

<b>Работа с документами</b>
— подготавливаю (в том числе голосом) процессуальные документы, проверяю договоры на риски,
делаю саммари.

<b>ПРИМЕРЫ ОБРАЩЕНИЙ:</b>
💬 "Администрация отказала в согласовании — подбери стратегию обжалования со ссылками на
судебную практику".
💬 "Проанализируй различия между статьями 228 и 228.1 УК РФ".
💬 "Найди судебную практику по взысканию неустойки с застройщика".
💬 "Могут ли наследники оспорить завещание после 6 месяцев?".

<b> ПОПРОБУЙТЕ ПРЯМО СЕЙЧАС </b>👇👇👇"""

    welcome_html = sanitize_telegram_html(welcome_raw)

    main_menu_keyboard = _main_menu_keyboard()

    media_sent = await _try_send_welcome_media(
        message=message,
        caption_html=welcome_html,
        keyboard=None,
    )

    if not media_sent:
        await message.answer(
            welcome_html,
            parse_mode=ParseMode.HTML,
        )

    await message.answer(
        _main_menu_text(),
        parse_mode=ParseMode.HTML,
        reply_markup=main_menu_keyboard,
    )
    logger.info("User %s started bot", message.from_user.id)




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
        user_id = _ensure_valid_user_id(message.from_user.id, context="process_question")
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
                    plan_info = _get_plan_pricing(decision.subscription_plan)
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
                plan_info = _get_plan_pricing(decision.subscription_plan) if decision.subscription_plan else None
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
                    practice_excel_path = build_practice_excel(
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
                record_request_fn = _get_safe_db_method("record_request", default_return=None)
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
                record_request_fn = _get_safe_db_method("record_request", default_return=None)
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


# ============ ПОДПИСКИ И ПЛАТЕЖИ ============


def _format_rub(amount_rub: int) -> str:
    return f"{amount_rub:,}".replace(",", " ")


def _plan_stars_amount(plan_info: SubscriptionPlanPricing) -> int:
    amount = int(plan_info.price_stars or 0)
    if amount <= 0:
        cfg = settings()
        amount = convert_rub_to_xtr(
            amount_rub=float(plan_info.plan.price_rub),
            rub_per_xtr=cfg.rub_per_xtr,
            default_xtr=cfg.subscription_price_xtr,
        )
    return max(int(amount), 0)


def _get_plan_pricing(plan_id: str | None) -> SubscriptionPlanPricing | None:
    if not plan_id:
        return DEFAULT_SUBSCRIPTION_PLAN
    return SUBSCRIPTION_PLAN_MAP.get(plan_id)


_catalog_header_lines = [
    "✨ <b>Каталог подписок AIVAN</b>",
    "━━━━━━━━━━━━━━━━━━━━",
    "",
    "💡 <b>Выберите идеальный тариф для себя</b>",
    "🎯 Доступ ко всем функциям AI-юриста",
    "⚡ Мгновенные ответы на юридические вопросы",
    "📄 Анализ и составление документов",
    "",
]


def _plan_catalog_text() -> str:
    if not SUBSCRIPTION_PLANS:
        return f"{Emoji.WARNING} Подписки временно недоступны. Попробуйте позже."

    lines: list[str] = list(_catalog_header_lines)

    for idx, plan_info in enumerate(SUBSCRIPTION_PLANS, 1):
        plan = plan_info.plan
        stars_amount = _plan_stars_amount(plan_info)

        # Рамка для плана
        lines.append("╔═══════════════════════╗")

        # Название тарифа с emoji
        plan_emoji = "💎" if idx == 1 else "👑" if idx == 2 else "✨"
        lines.append(f"║ {plan_emoji} <b>{html_escape(plan.name).upper()}</b>")
        lines.append("╠═══════════════════════╣")

        # Основная информация
        lines.append(f"║ ⏰ <b>Срок:</b> {plan.duration_days} дней")
        lines.append(f"║ 📊 <b>Запросов:</b> {plan.request_quota}")

        # Описание если есть
        if plan.description:
            lines.append(f"║ 💬 {html_escape(plan.description)}")

        # Цена
        price_line = f"║ 💰 <b>Цена:</b> {_format_rub(plan.price_rub)} ₽"
        if stars_amount > 0:
            price_line += f" / {stars_amount} ⭐"
        lines.append(price_line)

        # Нижняя граница
        lines.append("╚═══════════════════════╝")
        lines.append("")

    lines.append("👇 <b>Выберите тариф для оплаты</b>")
    return "\n".join(lines)


_def_no_plans_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_main")]]
)


def _build_plan_catalog_keyboard() -> InlineKeyboardMarkup:
    if not SUBSCRIPTION_PLANS:
        return _def_no_plans_keyboard

    rows: list[list[InlineKeyboardButton]] = []

    for idx, plan_info in enumerate(SUBSCRIPTION_PLANS, 1):
        stars_amount = _plan_stars_amount(plan_info)

        # Emoji для каждого плана
        plan_emoji = "💎" if idx == 1 else "👑" if idx == 2 else "✨"

        # Формируем красивую метку
        price_label = f"{_format_rub(plan_info.plan.price_rub)} ₽"
        if stars_amount > 0:
            price_label += f" • {stars_amount} ⭐"

        # Название + цена в одной строке
        label = f"{plan_emoji} {plan_info.plan.name} — {price_label}"

        rows.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=f"select_plan:{plan_info.plan.plan_id}",
                )
            ]
        )

    # Кнопка назад
    rows.append([InlineKeyboardButton(text="⬅️ Вернуться в меню", callback_data="back_to_main")])

    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _send_plan_catalog(message: Message, *, edit: bool = False) -> None:
    text = _plan_catalog_text()
    keyboard = _build_plan_catalog_keyboard()
    kwargs = dict(parse_mode=ParseMode.HTML, reply_markup=keyboard)
    if edit:
        try:
            await message.edit_text(text, **kwargs)
        except TelegramBadRequest:
            await message.answer(text, **kwargs)
    else:
        await message.answer(text, **kwargs)


async def _generate_user_stats_response(
    user_id: int,
    days: int,
    *,
    stats: dict[str, Any] | None = None,
    user: Any | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    if db is None:
        raise RuntimeError("Database is not available")

    normalized_days = _normalize_stats_period(days)
    if user is None:
        user = await db.ensure_user(
            user_id, default_trial=TRIAL_REQUESTS, is_admin=user_id in ADMIN_IDS
        )

    if stats is None:
        stats = await db.get_user_statistics(user_id, days=normalized_days)
    if stats.get("error"):
        raise RuntimeError(stats.get("error"))

    plan_id = stats.get("subscription_plan") or getattr(user, "subscription_plan", None)
    plan_info = _get_plan_pricing(plan_id) if plan_id else None

    subscription_until_ts = int(stats.get("subscription_until", 0) or 0)
    now_ts = int(time.time())
    has_subscription = subscription_until_ts > now_ts
    subscription_days_left = (
        max(0, math.ceil((subscription_until_ts - now_ts) / 86400)) if has_subscription else 0
    )

    subscription_status_text = "❌ Не активна"
    if has_subscription:
        until_text = _format_datetime(subscription_until_ts, default="—")
        subscription_status_text = f"✅ Активна до {until_text} (≈{subscription_days_left} дн.)"
    elif subscription_until_ts:
        until_text = _format_datetime(subscription_until_ts, default="—")
        subscription_status_text = f"⏰ Истекла {until_text}"

    trial_remaining = int(stats.get("trial_remaining", getattr(user, "trial_remaining", 0)) or 0)

    if plan_info:
        plan_label = plan_info.plan.name
    elif plan_id:
        plan_label = plan_id
    elif trial_remaining > 0:
        plan_label = "Триал"
    else:
        plan_label = "—"
    period_requests = int(stats.get("period_requests", 0) or 0)
    previous_requests = int(stats.get("previous_period_requests", 0) or 0)
    period_successful = int(stats.get("period_successful", 0) or 0)
    previous_successful = int(stats.get("previous_period_successful", 0) or 0)
    period_tokens = int(stats.get("period_tokens", 0) or 0)
    avg_response_time_ms = int(stats.get("avg_response_time_ms", 0) or 0)

    success_rate = (period_successful / period_requests * 100) if period_requests else 0.0

    day_counts = stats.get("day_of_week_counts") or {}
    hour_counts = stats.get("hour_of_day_counts") or {}
    type_stats = stats.get("request_types") or {}

    day_primary, day_secondary = _peak_summary(day_counts, mapping=DAY_NAMES)
    hour_primary, hour_secondary = _peak_summary(
        hour_counts, formatter=_format_hour_label
    )

    last_transaction = stats.get("last_transaction")

    created_at_ts = stats.get("created_at") or getattr(user, "created_at", 0)
    updated_at_ts = stats.get("updated_at") or getattr(user, "updated_at", 0)
    last_request_ts = stats.get("last_request_at", 0)

    subscription_balance_raw = stats.get("subscription_requests_balance")
    if subscription_balance_raw is None:
        subscription_balance_raw = getattr(user, "subscription_requests_balance", None)
    subscription_balance = int(subscription_balance_raw or 0)

    divider = SECTION_DIVIDER

    lines = [
        "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
        f"┃  {Emoji.STATS} <b>Моя статистика</b>       ┃",
        "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛",
        "",
        f"📅 <i>Период: последние {normalized_days} дней</i>",
        "",
        divider,
        "",
        "👤 <b>Профиль</b>",
        "",
        _format_stat_row("  📆 Регистрация", _format_datetime(created_at_ts)),
        _format_stat_row("  🕐 Последний запрос", _format_datetime(last_request_ts)),
        _format_stat_row("  💳 Подписка", subscription_status_text),
        _format_stat_row("  🏷️ План", plan_label),
    ]

    lines.append("")
    lines.append(divider)
    lines.append("")
    lines.append("🔋 <b>Лимиты</b>")
    lines.append("")
    if TRIAL_REQUESTS > 0:
        trial_used = max(0, TRIAL_REQUESTS - trial_remaining)
        lines.append(_progress_line("Триал", trial_used, TRIAL_REQUESTS))
    else:
        lines.append(_format_stat_row("Триал", "недоступен"))

    if plan_info and plan_info.plan.request_quota > 0:
        used = max(0, plan_info.plan.request_quota - subscription_balance)
        lines.append(_progress_line("  📊 Подписка", used, plan_info.plan.request_quota))
    elif has_subscription:
        lines.append(_format_stat_row("  📊 Подписка", "безлимит ♾️"))

    lines.extend([
        "",
        divider,
        "",
        "📈 <b>Активность</b>",
        "",
        _format_stat_row("  📝 Запросов", _format_trend_value(period_requests, previous_requests)),
        "",
    ])
    if day_primary != "—":
        lines.append(_format_stat_row("  📅 Активный день", _describe_primary_summary(day_primary, "обращений")))
        if day_secondary:
            lines.append(_format_stat_row("  📆 Другие дни", _describe_secondary_summary(day_secondary, "обращений")))
    else:
        lines.append(_format_stat_row("  📅 Активный день", "нет данных"))

    if hour_primary != "—":
        lines.append(_format_stat_row("  🕐 Активный час", _describe_primary_summary(hour_primary, "обращений")))
        if hour_secondary:
            lines.append(_format_stat_row("  🕑 Другие часы", _describe_secondary_summary(hour_secondary, "обращений")))
    else:
        lines.append(_format_stat_row("  🕐 Активный час", "нет данных"))

    lines.append("")
    lines.append(divider)
    lines.append("")
    lines.append("📋 <b>Типы запросов</b>")
    lines.append("")
    if type_stats:
        top_types = sorted(type_stats.items(), key=lambda item: item[1], reverse=True)[:5]
        for req_type, count in top_types:
            share_pct = (count / period_requests * 100) if period_requests else 0.0
            label = FEATURE_LABELS.get(req_type, req_type)
            lines.append(_format_stat_row(f"  • {label}", f"{count} ({share_pct:.0f}%)"))
    else:
        lines.append(_format_stat_row("  • Типы", "нет данных"))

    if last_transaction:
        lines.append("")
        lines.append(divider)
        lines.append("")
        lines.append("💳 <b>Последний платёж</b>")
        lines.append("")
        currency = last_transaction.get("currency", "RUB") or "RUB"
        amount_minor = last_transaction.get("amount_minor_units")
        if amount_minor is None:
            amount_minor = last_transaction.get("amount")
        lines.append(_format_stat_row("  💰 Сумма", _format_currency(amount_minor, currency)))

        # Переводим статус на русский
        status = last_transaction.get("status", "unknown")
        translated_status = _translate_payment_status(status)
        lines.append(_format_stat_row("  📊 Статус", translated_status))

        lines.append(_format_stat_row("  📅 Дата", _format_datetime(last_transaction.get("created_at"))))
        payload_raw = last_transaction.get("payload")
        if payload_raw:
            try:
                payload = parse_subscription_payload(payload_raw)
                if payload.plan_id:
                    # Переводим название тарифа на русский
                    translated_plan = _translate_plan_name(payload.plan_id)
                    lines.append(_format_stat_row("  🏷️ Тариф", translated_plan))
            except SubscriptionPayloadError:
                pass

    text = "\n".join(lines)
    keyboard = _build_stats_keyboard(has_subscription)
    return text, keyboard


async def cmd_buy(message: Message):
    await _send_plan_catalog(message, edit=False)


async def handle_ignore_callback(callback: CallbackQuery):
    """Обработчик для декоративных кнопок-разделителей"""
    await callback.answer()


async def handle_buy_catalog_callback(callback: CallbackQuery):
    if not callback.message:
        await callback.answer("Ошибка: нет данных сообщения", show_alert=True)
        return
    await callback.answer()
    await _send_plan_catalog(callback.message, edit=True)


async def handle_get_subscription_callback(callback: CallbackQuery):
    if not callback.from_user or not callback.message:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return
    try:
        await callback.answer()
        await _send_plan_catalog(callback.message, edit=False)
    except TelegramBadRequest:
        await callback.message.answer(
            f"{Emoji.WARNING} Не удалось показать каталог подписок. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )



async def handle_cancel_subscription_callback(callback: CallbackQuery):
    """Показывает инструкции по отмене активной подписки."""
    if not callback.from_user or callback.message is None:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return

    try:
        await callback.answer()

        if db is None:
            message_text = (
                f"{Emoji.DIAMOND} <b>Отмена подписки</b>\n\n"
                "Сервис управления подписками временно недоступен. Напишите в поддержку — команда /help."
            )
        else:
            user_id = callback.from_user.id
            user_record = await db.ensure_user(
                user_id,
                default_trial=TRIAL_REQUESTS,
                is_admin=user_id in ADMIN_IDS,
            )
            has_subscription = await db.has_active_subscription(user_id)

            if has_subscription:
                cancellation_applied = await db.cancel_subscription(user_id)
                updated_record = await db.get_user(user_id)
                if updated_record is not None:
                    user_record = updated_record
                until_ts = int(getattr(user_record, "subscription_until", 0) or 0)
                until_text = (
                    datetime.fromtimestamp(until_ts).strftime("%d.%m.%Y") if until_ts else "—"
                )
                if cancellation_applied:
                    message_text = (
                        f"{Emoji.DIAMOND} <b>Отмена подписки</b>\n\n"
                        f"Отмена оформлена. Доступ сохранится до {until_text}, после чего подписка отключится.\n"
                        "Если передумали, выберите `🔄 Сменить тариф`, чтобы продлить доступ."
                    )
                else:
                    message_text = (
                        f"{Emoji.DIAMOND} <b>Отмена подписки</b>\n\n"
                        f"Отмена уже оформлена. Доступ сохранится до {until_text}."
                    )
            else:
                message_text = (
                    f"{Emoji.DIAMOND} <b>Отмена подписки</b>\n\n"
                    "Подписка уже не активна. Вы можете подключить новый тариф в каталоге."
                )

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="📦 Каталог тарифов", callback_data="buy_catalog")],
                [InlineKeyboardButton(text="👤 Мой профиль", callback_data="my_profile")],
                [InlineKeyboardButton(text="💬 Поддержка", callback_data="help_info")],
                [InlineKeyboardButton(text="↩️ Назад в меню", callback_data="back_to_main")],
            ]
        )

        await callback.message.edit_text(
            message_text,
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Error in handle_cancel_subscription_callback: %s", exc, exc_info=True)
        await callback.answer("❌ Не удалось обработать запрос")


async def _send_rub_invoice(message: Message, plan_info: SubscriptionPlanPricing, user_id: int) -> None:
    if not message.bot or not message.chat:
        return
    if not RUB_PROVIDER_TOKEN:
        await message.answer(
            f"{Emoji.WARNING} Оплата картами временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        return
    payload = build_subscription_payload(plan_info.plan.plan_id, "rub", user_id)
    cfg = settings()
    amount_value = f"{float(plan_info.plan.price_rub):.2f}"
    receipt_item: dict[str, Any] = {
        "description": plan_info.plan.name[:128],
        "quantity": 1,
        "amount": {
            "value": amount_value,
            "currency": "RUB",
        },
    }
    if cfg.yookassa_vat_code is not None:
        receipt_item["vat_code"] = int(cfg.yookassa_vat_code)
    if cfg.yookassa_payment_mode:
        receipt_item["payment_mode"] = cfg.yookassa_payment_mode
    if cfg.yookassa_payment_subject:
        receipt_item["payment_subject"] = cfg.yookassa_payment_subject

    receipt: dict[str, Any] = {
        "items": [receipt_item],
    }
    if cfg.yookassa_tax_system_code is not None:
        receipt["tax_system_code"] = int(cfg.yookassa_tax_system_code)

    provider_data = json.dumps({"receipt": receipt}, ensure_ascii=False)
    prices = [
        LabeledPrice(
            label=f"{plan_info.plan.name}",
            amount=plan_info.price_rub_kopeks,
        )
    ]
    description = (
        f"Доступ к ИИ-Иван на {plan_info.plan.duration_days} дн.\n"
        f"Квота: {plan_info.plan.request_quota} запросов."
    )
    await message.bot.send_invoice(
        chat_id=message.chat.id,
        title=f"Подписка • {plan_info.plan.name}",
        description=description,
        payload=payload,
        provider_token=RUB_PROVIDER_TOKEN,
        currency="RUB",
        prices=prices,
        is_flexible=False,
        need_email=cfg.yookassa_require_email,
        need_phone_number=cfg.yookassa_require_phone,
        send_email_to_provider=cfg.yookassa_require_email,
        send_phone_number_to_provider=cfg.yookassa_require_phone,
        provider_data=provider_data,
    )


async def _send_stars_invoice(message: Message, plan_info: SubscriptionPlanPricing, user_id: int) -> None:
    if not message.bot or not message.chat:
        return
    if not STARS_PROVIDER_TOKEN:
        await message.answer(
            f"{Emoji.WARNING} Telegram Stars временно недоступны.",
            parse_mode=ParseMode.HTML,
        )
        return
    stars_amount = _plan_stars_amount(plan_info)
    if stars_amount <= 0:
        await message.answer(
            f"{Emoji.WARNING} Не удалось рассчитать стоимость в Stars, попробуйте другой способ.",
            parse_mode=ParseMode.HTML,
        )
        return
    payload = build_subscription_payload(plan_info.plan.plan_id, "stars", user_id)
    prices = [LabeledPrice(label=f"{plan_info.plan.name}", amount=stars_amount)]
    description = (
        f"Оплата в Telegram Stars. Срок: {plan_info.plan.duration_days} дн.\n"
        f"Квота: {plan_info.plan.request_quота} запросов."
    )
    await message.bot.send_invoice(
        chat_id=message.chat.id,
        title=f"Подписка • {plan_info.plan.name} (Stars)",
        description=description,
        payload=payload,
        provider_token=STARS_PROVIDER_TOKEN,
        currency="XTR",
        prices=prices,
        is_flexible=False,
    )


async def _record_pending_transaction(
    *,
    user_id: int,
    provider: str,
    amount_minor_units: int,
    payload: str,
    provider_payment_charge_id: str | None = None,
) -> int:
    if db is None:
        raise RuntimeError("Database not initialized")
    return await db.record_transaction(
        user_id=user_id,
        provider=provider,
        currency="RUB",
        amount=amount_minor_units,
        amount_minor_units=amount_minor_units,
        payload=payload,
        status=TransactionStatus.PENDING,
        provider_payment_charge_id=provider_payment_charge_id,
    )


def _external_payment_keyboard(provider: str, payment_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Проверить оплату",
                    callback_data=f"verify_payment:{provider}:{payment_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"{Emoji.BACK} Назад к тарифам",
                    callback_data="buy_catalog",
                )
            ],
        ]
    )


async def _send_robokassa_invoice(
    message: Message, plan_info: SubscriptionPlanPricing, user_id: int
) -> None:
    if robokassa_provider is None or not getattr(robokassa_provider, "is_available", False):
        await message.answer(
            f"{Emoji.WARNING} Оплата через RoboKassa временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        return

    amount_minor = plan_info.price_rub_kopeks
    payload = build_subscription_payload(plan_info.plan.plan_id, "robokassa", user_id)
    try:
        transaction_id = await _record_pending_transaction(
            user_id=user_id,
            provider="robokassa",
            amount_minor_units=amount_minor,
            payload=payload,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to record RoboKassa transaction: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} Не удалось начать оплату. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        creation = await robokassa_provider.create_invoice(
            amount_rub=float(plan_info.plan.price_rub),
            description=f"Подписка {plan_info.plan.name} на {plan_info.plan.duration_days} дн.",
            payload=payload,
            invoice_id=transaction_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("RoboKassa invoice error: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} Не удалось создать счет RoboKassa.",
            parse_mode=ParseMode.HTML,
        )
        with suppress(Exception):
            await db.update_transaction(transaction_id, status=TransactionStatus.FAILED)
        return

    if not creation.ok or not creation.url or not creation.payment_id:
        logger.warning("RoboKassa invoice creation failed: %s", creation.error or creation.raw)
        await message.answer(
            f"{Emoji.WARNING} Оплата через RoboKassa временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        with suppress(Exception):
            await db.update_transaction(transaction_id, status=TransactionStatus.FAILED)
        return

    with suppress(Exception):
        await db.update_transaction(
            transaction_id,
            provider_payment_charge_id=str(creation.payment_id),
        )

    payment_text = (
        f"🏦 <b>RoboKassa</b>\n\n"
        f"1. Нажмите на ссылку и оплатите счет картой или через СБП.\n"
        f"2. После оплаты вернитесь и нажмите кнопку \"Проверить оплату\".\n\n"
        f"{creation.url}"
    )
    await message.answer(
        payment_text,
        parse_mode=ParseMode.HTML,
        reply_markup=_external_payment_keyboard("robokassa", str(creation.payment_id)),
        disable_web_page_preview=True,
    )


async def _send_yookassa_invoice(
    message: Message, plan_info: SubscriptionPlanPricing, user_id: int
) -> None:
    if yookassa_provider is None or not getattr(yookassa_provider, "is_available", False):
        await message.answer(
            f"{Emoji.WARNING} Оплата через YooKassa временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        return

    amount_minor = plan_info.price_rub_kopeks
    payload = build_subscription_payload(plan_info.plan.plan_id, "yookassa", user_id)
    try:
        transaction_id = await _record_pending_transaction(
            user_id=user_id,
            provider="yookassa",
            amount_minor_units=amount_minor,
            payload=payload,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to record YooKassa transaction: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} Не удалось начать оплату. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        creation = await yookassa_provider.create_invoice(
            amount_rub=float(plan_info.plan.price_rub),
            description=f"Подписка {plan_info.plan.name} на {plan_info.plan.duration_days} дн.",
            payload=payload,
            metadata={"transaction_id": transaction_id, "plan_id": plan_info.plan.plan_id},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("YooKassa invoice error: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} Не удалось создать счет YooKassa.",
            parse_mode=ParseMode.HTML,
        )
        with suppress(Exception):
            await db.update_transaction(transaction_id, status=TransactionStatus.FAILED)
        return

    if not creation.ok or not creation.url or not creation.payment_id:
        logger.warning("YooKassa invoice creation failed: %s", creation.error or creation.raw)
        await message.answer(
            f"{Emoji.WARNING} Оплата через YooKassa временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        with suppress(Exception):
            await db.update_transaction(transaction_id, status=TransactionStatus.FAILED)
        return

    with suppress(Exception):
        await db.update_transaction(
            transaction_id,
            provider_payment_charge_id=str(creation.payment_id),
        )

    payment_text = (
        f"💳 <b>YooKassa</b>\n\n"
        f"1. Оплатите подписку на защищенной странице YooKassa.\n"
        f"2. После оплаты вернитесь и нажмите кнопку \"Проверить оплату\".\n\n"
        f"{creation.url}"
    )
    await message.answer(
        payment_text,
        parse_mode=ParseMode.HTML,
        reply_markup=_external_payment_keyboard("yookassa", str(creation.payment_id)),
        disable_web_page_preview=True,
    )


async def _send_crypto_invoice(message: Message, plan_info: SubscriptionPlanPricing, user_id: int) -> None:
    if crypto_provider is None:
        await message.answer(
            f"{Emoji.IDEA} Криптовалюта временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        return
    payload = build_subscription_payload(plan_info.plan.plan_id, "crypto", user_id)
    try:
        invoice = await crypto_provider.create_invoice(
            amount_rub=float(plan_info.plan.price_rub),
            description=f"Подписка {plan_info.plan.name} на {plan_info.plan.duration_days} дн.",
            payload=payload,
        )
    except Exception as exc:
        logger.warning("Crypto invoice failed: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} Не удалось создать крипто-счет. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return
    url = invoice.get("url") if isinstance(invoice, dict) else None
    if invoice and invoice.get("ok") and url:
        await message.answer(
            f"{Emoji.DOWNLOAD} Оплата криптовалютой: перейдите по ссылке\n{url}",
            parse_mode=ParseMode.HTML,
        )
    else:
        await message.answer(
            f"{Emoji.IDEA} Криптовалюта временно недоступна.",
            parse_mode=ParseMode.HTML,
        )


def _plan_details_keyboard(plan_info: SubscriptionPlanPricing) -> tuple[InlineKeyboardMarkup, list[str]]:
    rows: list[list[InlineKeyboardButton]] = []
    unavailable: list[str] = []

    rub_label = f"💳 Карта • {_format_rub(plan_info.plan.price_rub)} ₽"
    if RUB_PROVIDER_TOKEN:
        rows.append(
            [
                InlineKeyboardButton(
                    text=rub_label,
                    callback_data=f"pay_plan:{plan_info.plan.plan_id}:rub",
                )
            ]
        )
    else:
        unavailable.append("💳 Оплата картой — временно недоступна")

    stars_amount = _plan_stars_amount(plan_info)
    stars_label = f"⭐ Telegram Stars • {stars_amount}"
    if stars_amount > 0 and STARS_PROVIDER_TOKEN:
        rows.append(
            [
                InlineKeyboardButton(
                    text=stars_label,
                    callback_data=f"pay_plan:{plan_info.plan.plan_id}:stars",
                )
            ]
        )
    else:
        unavailable.append("⭐ Telegram Stars — временно недоступно")

    crypto_label = f"🪙 Криптовалюта • {_format_rub(plan_info.plan.price_rub)} ₽"
    if crypto_provider is not None:
        rows.append(
            [
                InlineKeyboardButton(
                    text=crypto_label,
                    callback_data=f"pay_plan:{plan_info.plan.plan_id}:crypto",
                )
            ]
        )
    else:
        unavailable.append("🪙 Криптовалюта — временно недоступна")

    robo_label = f"🏦 RoboKassa • {_format_rub(plan_info.plan.price_rub)} ₽"
    if robokassa_provider is not None and getattr(robokassa_provider, "is_available", False):
        rows.append(
            [
                InlineKeyboardButton(
                    text=robo_label,
                    callback_data=f"pay_plan:{plan_info.plan.plan_id}:robokassa",
                )
            ]
        )

    yk_label = f"💳 YooKassa • {_format_rub(plan_info.plan.price_rub)} ₽"
    if yookassa_provider is not None and getattr(yookassa_provider, "is_available", False):
        rows.append(
            [
                InlineKeyboardButton(
                    text=yk_label,
                    callback_data=f"pay_plan:{plan_info.plan.plan_id}:yookassa",
                )
            ]
        )

    rows.append(
        [
            InlineKeyboardButton(
                text=f"{Emoji.BACK} Назад к тарифам",
                callback_data="buy_catalog",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows), unavailable


async def handle_select_plan_callback(callback: CallbackQuery):
    data = callback.data or ""
    parts = data.split(":", 1)
    if len(parts) != 2:
        await callback.answer("❌ Некорректный тариф", show_alert=True)
        return
    plan_id = parts[1]
    plan_info = _get_plan_pricing(plan_id)
    if not plan_info:
        await callback.answer("❌ Тариф недоступен", show_alert=True)
        return
    if not callback.message:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return
    await callback.answer()

    plan = plan_info.plan
    stars_amount = _plan_stars_amount(plan_info)
    lines = [f"{Emoji.DIAMOND} <b>{html_escape(plan.name)}</b>"]
    if plan.description:
        lines.append(f"<i>{html_escape(plan.description)}</i>")

    lines.extend(
        [
            "",
            f"{Emoji.CALENDAR} Период доступа: {plan.duration_days} дней",
            f"{Emoji.DOCUMENT} Лимит запросов: {plan.request_quota}",
        ]
    )

    price_line = f"💳 {_format_rub(plan.price_rub)} ₽"
    if stars_amount > 0:
        price_line += f" • {stars_amount} ⭐"
    lines.append(price_line)

    lines.extend(
        [
            "",
            f"{Emoji.MAGIC} Выберите удобный способ оплаты ниже.",
        ]
    )

    keyboard, unavailable = _plan_details_keyboard(plan_info)
    if unavailable:
        lines.append("")
        lines.append(f"{Emoji.WARNING} Временно недоступно:")
        lines.extend(f"• {item}" for item in unavailable)

    text = "\n".join(lines)
    try:
        await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    except TelegramBadRequest:
        await callback.message.answer(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


async def handle_pay_plan_callback(callback: CallbackQuery):
    data = callback.data or ""
    parts = data.split(":")
    if len(parts) != 3:
        await callback.answer("❌ Некорректные параметры оплаты", show_alert=True)
        return
    _, plan_id, method = parts
    plan_info = _get_plan_pricing(plan_id)
    if not plan_info:
        await callback.answer("❌ Тариф недоступен", show_alert=True)
        return
    if not callback.message or not callback.from_user:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return
    await callback.answer()
    user_id = callback.from_user.id
    if method == "rub":
        await _send_rub_invoice(callback.message, plan_info, user_id)
    elif method in {"stars", "xtr"}:
        await _send_stars_invoice(callback.message, plan_info, user_id)
    elif method == "crypto":
        await _send_crypto_invoice(callback.message, plan_info, user_id)
    elif method == "robokassa":
        await _send_robokassa_invoice(callback.message, plan_info, user_id)
    elif method == "yookassa":
        await _send_yookassa_invoice(callback.message, plan_info, user_id)
    else:
        await callback.message.answer("❌ Этот способ оплаты не поддерживается")


async def handle_verify_payment_callback(callback: CallbackQuery):
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return

    data = callback.data or ""
    parts = data.split(":")
    if len(parts) != 3:
        await callback.answer("❌ Некорректные параметры", show_alert=True)
        return

    _, provider_code, payment_id = parts
    provider_code = provider_code.lower()
    payment_id = payment_id.strip()
    await callback.answer()

    if not payment_id:
        await callback.message.answer(
            f"{Emoji.WARNING} Платеж не найден. Попробуйте ещё раз.",
            parse_mode=ParseMode.HTML,
        )
        return

    provider_obj = None
    if provider_code == "robokassa":
        provider_obj = robokassa_provider
    elif provider_code == "yookassa":
        provider_obj = yookassa_provider

    if provider_obj is None:
        await callback.message.answer(
            f"{Emoji.WARNING} Этот способ оплаты недоступен.",
            parse_mode=ParseMode.HTML,
        )
        return

    if db is None:
        await callback.message.answer(
            f"{Emoji.WARNING} Сервис временно недоступен. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        transaction = await db.get_transaction_by_provider_charge_id(provider_code, payment_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load transaction for provider %s: %s", provider_code, exc)
        transaction = None

    if transaction is None:
        await callback.message.answer(
            f"{Emoji.WARNING} Платеж не найден. Убедитесь, что вы использовали последнюю ссылку оплаты.",
            parse_mode=ParseMode.HTML,
        )
        return

    if transaction.user_id != callback.from_user.id:
        await callback.message.answer(
            f"{Emoji.WARNING} Проверка доступна только владельцу платежа.",
            parse_mode=ParseMode.HTML,
        )
        return

    current_status = TransactionStatus.from_value(transaction.status)
    if current_status == TransactionStatus.COMPLETED:
        await callback.message.answer(
            f"{Emoji.SUCCESS} Платёж уже подтвержден. Статус подписки можно посмотреть через /status.",
            parse_mode=ParseMode.HTML,
        )
        return

    poll_method = getattr(provider_obj, "poll_payment", None)
    if poll_method is None:
        await callback.message.answer(
            f"{Emoji.WARNING} Проверка оплаты для этого провайдера пока не реализована.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        result = await poll_method(payment_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Payment polling failed (%s): %s", provider_code, exc)
        await callback.message.answer(
            f"{Emoji.WARNING} Не удалось проверить оплату. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    if result.status == TransactionStatus.PENDING:
        await callback.message.answer(
            f"{Emoji.WARNING} Платёж ещё обрабатывается. Попробуйте снова через минуту.",
            parse_mode=ParseMode.HTML,
        )
        return

    if result.status in {TransactionStatus.CANCELLED, TransactionStatus.FAILED}:
        await db.update_transaction(transaction.id, status=result.status)
        reason = result.description or "Провайдер сообщил об отмене"
        await callback.message.answer(
            f"{Emoji.ERROR} Оплата не прошла: {html_escape(reason)}",
            parse_mode=ParseMode.HTML,
        )
        return

    payload_raw = transaction.payload or ""
    try:
        payload = parse_subscription_payload(payload_raw)
    except SubscriptionPayloadError as exc:
        logger.error("Failed to parse payload for transaction %s: %s", transaction.id, exc)
        await callback.message.answer(
            f"{Emoji.ERROR} Ошибка обработки платежа. Свяжитесь с поддержкой.",
            parse_mode=ParseMode.HTML,
        )
        return

    plan_info = _get_plan_pricing(payload.plan_id) if payload.plan_id else DEFAULT_SUBSCRIPTION_PLAN
    if plan_info is None:
        await callback.message.answer(
            f"{Emoji.ERROR} Не удалось определить тариф. Свяжитесь с поддержкой.",
            parse_mode=ParseMode.HTML,
        )
        return

    expected_amount = transaction.amount
    if result.paid_amount is not None and expected_amount not in (0, result.paid_amount):
        logger.warning(
            "Paid amount mismatch for transaction %s: expected %s, got %s",
            transaction.id,
            expected_amount,
            result.paid_amount,
        )

    try:
        await db.update_transaction(transaction.id, status=TransactionStatus.COMPLETED)
        new_until, new_balance = await db.apply_subscription_purchase(
            user_id=transaction.user_id,
            plan_id=plan_info.plan.plan_id,
            duration_days=plan_info.plan.duration_days,
            request_quota=plan_info.plan.request_quota,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to finalize subscription for transaction %s: %s", transaction.id, exc)
        await callback.message.answer(
            f"{Emoji.ERROR} Не удалось активировать подписку, напишите поддержке.",
            parse_mode=ParseMode.HTML,
        )
        return

    until_dt = datetime.fromtimestamp(new_until)
    balance_text = f"Остаток запросов: {max(0, new_balance)}" if plan_info.plan.request_quota else "Безлимит"
    success_text = (
        f"{Emoji.SUCCESS} Оплата подтверждена!\n\n"
        f"План: {plan_info.plan.name}\n"
        f"Доступ до: {until_dt:%d.%m.%Y %H:%M}\n"
        f"{balance_text}"
    )
    await callback.message.answer(success_text, parse_mode=ParseMode.HTML)

async def cmd_status(message: Message):
    if db is None:
        await message.answer("Статус временно недоступен")
        return

    if not message.from_user:
        await message.answer("Статус доступен только для авторизованных пользователей")
        return

    try:
        user_id = _ensure_valid_user_id(message.from_user.id, context="cmd_status")
    except ValidationException as exc:
        context = ErrorContext(function_name="cmd_status", chat_id=message.chat.id if message.chat else None)
        if error_handler:
            await error_handler.handle_exception(exc, context)
        else:
            logger.warning("Validation error in cmd_status: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} <b>Не удалось получить статус.</b>\nПопробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    user = await db.ensure_user(
        user_id,
        default_trial=TRIAL_REQUESTS,
        is_admin=user_id in ADMIN_IDS,
    )

    until_ts = int(getattr(user, "subscription_until", 0) or 0)
    now_ts = int(time.time())
    has_active = until_ts > now_ts
    plan_id = getattr(user, "subscription_plan", None)
    plan_info = _get_plan_pricing(plan_id) if plan_id else None
    if plan_info:
        plan_label = plan_info.plan.name
    elif plan_id:
        plan_label = plan_id
    elif has_active:
        plan_label = "Безлимит"
    else:
        plan_label = "нет"

    if until_ts > 0:
        until_dt = datetime.fromtimestamp(until_ts)
        if has_active:
            left_days = max(0, (until_dt - datetime.now()).days)
            until_text = f"{until_dt:%Y-%m-%d} (≈{left_days} дн.)"
        else:
            until_text = f"Истекла {until_dt:%Y-%m-%d}"
    else:
        until_text = "Не активна"

    quota_balance_raw = getattr(user, "subscription_requests_balance", None)
    quota_balance = int(quota_balance_raw) if quota_balance_raw is not None else None

    lines = [
        f"{Emoji.STATS} <b>Статус</b>",
        "",
        f"ID: <code>{user_id}</code>",
        f"Роль: {'админ' if getattr(user, 'is_admin', False) else 'пользователь'}",
        f"Триал: {getattr(user, 'trial_remaining', 0)} запрос(ов)",
        "Подписка:",
    ]
    if plan_info or plan_id or until_ts:
        lines.append(f"• План: {plan_label}")
        lines.append(f"• Доступ до: {until_text}")
        if plan_info and quota_balance is not None:
            lines.append(f"• Остаток запросов: {max(0, quota_balance)}")
        elif plan_id and quota_balance is not None:
            lines.append(f"• Остаток запросов: {max(0, quota_balance)}")
        elif has_active and not plan_id:
            lines.append("• Лимит: без ограничений")
    else:
        lines.append("• Не активна")

    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_mystats(message: Message):
    """Показать детальную статистику пользователя"""
    if db is None:
        await message.answer("Статистика временно недоступна")
        return

    if not message.from_user:
        await message.answer("Статистика доступна только авторизованным пользователям")
        return

    days = 30
    if message.text:
        parts = message.text.strip().split()
        if len(parts) >= 2:
            try:
                days = int(parts[1])
            except ValueError:
                days = 30

    days = _normalize_stats_period(days)

    try:
        stats_text, keyboard = await _generate_user_stats_response(message.from_user.id, days)
        await message.answer(stats_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in cmd_mystats: %s", exc)
        await message.answer("❌ Ошибка получения статистики. Попробуйте позже.")


async def _download_voice_to_temp(message: Message) -> Path:
    """Download Telegram voice message into a temporary file."""
    if not message.bot:
        raise RuntimeError("Bot instance is not available for voice download")
    if not message.voice:
        raise RuntimeError("Voice payload is missing")

    file_info = await message.bot.get_file(message.voice.file_id)
    file_path = file_info.file_path
    if not file_path:
        raise RuntimeError("Telegram did not return a file path for the voice message")

    temp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    temp_path = Path(temp.name)
    temp.close()

    file_stream = await message.bot.download_file(file_path)
    try:
        temp_path.write_bytes(file_stream.read())
    finally:
        close_method = getattr(file_stream, "close", None)
        if callable(close_method):
            close_method()

    return temp_path


async def process_voice_message(message: Message):
    """Handle incoming Telegram voice messages via STT -> processing -> TTS."""
    if not message.voice:
        return

    # НОВОЕ: Показываем индикатор "записывает голосовое"
    try:
        voice_enabled = settings().voice_mode_enabled
    except RuntimeError:
        voice_enabled = bool(getattr(config, "voice_mode_enabled", False))

    if audio_service is None or not voice_enabled:
        await message.answer("Voice mode is currently unavailable. Please send text.")
        return

    if not message.bot:
        await message.answer("Unable to access bot context for processing the voice message.")
        return

    temp_voice_path: Path | None = None
    tts_paths: list[Path] = []

    try:
        await audio_service.ensure_short_enough(message.voice.duration)

        # Показываем индикатор во время транскрипции и обработки
        async with typing_action(message.bot, message.chat.id, "record_voice"):
            temp_voice_path = await _download_voice_to_temp(message)
            transcript = await audio_service.transcribe(temp_voice_path)

        preview = html_escape(transcript[:500])
        if len(transcript) > 500:
            preview += "..."
        await message.answer(
            f"{Emoji.ROBOT} Recognized: <i>{preview}</i>",
            parse_mode=ParseMode.HTML,
        )

        response_text = await process_question(message, text_override=transcript)
        if not response_text:
            return

        # Показываем индикатор "отправляет голосовое" во время генерации TTS
        async with typing_action(message.bot, message.chat.id, "upload_voice"):
            try:
                tts_paths = await audio_service.synthesize(response_text, prefer_male=True)
            except Exception as tts_error:
                logger.warning("Text-to-speech failed: %s", tts_error)
                return

            if not tts_paths:
                logger.warning("Text-to-speech returned no audio chunks")
                return

            for idx, generated_path in enumerate(tts_paths):
                caption = VOICE_REPLY_CAPTION if idx == 0 else None
                await message.answer_voice(
                    FSInputFile(generated_path),
                    caption=caption,
                    parse_mode=ParseMode.HTML if caption else None,
                )

    except ValueError as duration_error:
        logger.warning("Voice message duration exceeded: %s", duration_error)
        await message.answer(
            f"{Emoji.WARNING} Voice message is too long. Maximum duration is {audio_service.max_duration_seconds} seconds.",
            parse_mode=ParseMode.HTML,
        )
    except Exception as exc:
        logger.exception("Failed to process voice message: %s", exc)
        await message.answer(
            f"{Emoji.ERROR} Could not process the voice message. Please try again later.",
            parse_mode=ParseMode.HTML,
        )
    finally:
        with suppress(Exception):
            if temp_voice_path:
                temp_voice_path.unlink()
        with suppress(Exception):
            for generated_path in tts_paths:
                generated_path.unlink()


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
        user_id = _ensure_valid_user_id(message.from_user.id, context="handle_pending_feedback")
    except ValidationException as exc:
        logger.warning("Ignore feedback: invalid user id (%s)", exc)
        user_session.pending_feedback_request_id = None
        return

    request_id = user_session.pending_feedback_request_id
    feedback_text = feedback_source.strip()

    # Сбрасываем ожидание комментария после обработки
    user_session.pending_feedback_request_id = None

    add_rating_fn = _get_safe_db_method("add_rating", default_return=False)
    if not add_rating_fn:
        await message.answer("❌ Сервис отзывов временно недоступен")
        return

    get_rating_fn = _get_safe_db_method("get_rating", default_return=None)
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
        user_id = _ensure_valid_user_id(callback.from_user.id, context="handle_rating_callback")
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

    get_rating_fn = _get_safe_db_method("get_rating", default_return=None)
    existing_rating = await get_rating_fn(request_id, user_id) if get_rating_fn else None

    if existing_rating and existing_rating.rating not in (None, 0):
        await callback.answer("По этому ответу уже собрана обратная связь")
        return

    add_rating_fn = _get_safe_db_method("add_rating", default_return=False)
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
        user_id = _ensure_valid_user_id(callback.from_user.id, context="handle_feedback_callback")
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
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃  🔍 <b>Поиск судебной практики</b>  ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
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


async def handle_my_profile_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Мой профиль'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        status_text = None
        tariff_text = None
        hint_text = None
        subscribe_label = "💳 Оформить подписку"
        has_subscription = False

        if db is not None:
            try:
                user_id = callback.from_user.id
                user_record = await db.ensure_user(
                    user_id,
                    default_trial=TRIAL_REQUESTS,
                    is_admin=user_id in ADMIN_IDS,
                )
                has_subscription = await db.has_active_subscription(user_id)
                cancel_flag = bool(getattr(user_record, "subscription_cancelled", 0))

                plan_id = getattr(user_record, "subscription_plan", None)
                plan_info = _get_plan_pricing(plan_id) if plan_id else None
                if plan_info:
                    tariff_text = plan_info.plan.name
                elif plan_id and plan_id not in (None, "—"):
                    tariff_text = str(plan_id)
                else:
                    tariff_text = "триал"

                if has_subscription and getattr(user_record, "subscription_until", 0):
                    until_dt = datetime.fromtimestamp(int(user_record.subscription_until))
                    purchase_ts = int(getattr(user_record, "subscription_last_purchase_at", 0) or 0)
                    if purchase_ts:
                        purchase_dt = datetime.fromtimestamp(purchase_ts)
                        status_text = (
                            f"подписка оформлена {purchase_dt:%d.%m.%y} (доступ до {until_dt:%d.%m.%y})"
                        )
                    else:
                        status_text = f"подписка активна до {until_dt:%d.%m.%y}"

                    if cancel_flag:
                        hint_text = "Отмена оформлена — доступ сохранится до даты окончания."
                        subscribe_label = "✅ Отмена оформлена"
                    else:
                        hint_text = "Пополнить пакет — команда /buy"
                        subscribe_label = "❌ Отменить подписку"
                else:
                    trial_remaining = int(getattr(user_record, "trial_remaining", 0) or 0)
                    status_text = "⭕ <i>нет активной подписки</i>"
                    tariff_text = f" <b>Триал</b> • <i>{trial_remaining} запросов</i>"
                    hint_text = ""
            except Exception as profile_error:  # pragma: no cover - fallback
                logger.debug("Failed to build profile header: %s", profile_error, exc_info=True)

        await callback.message.edit_text(
            _profile_menu_text(
                callback.from_user,
                status_text=status_text,
                tariff_text=tariff_text,
                hint_text=hint_text,
            ),
            parse_mode=ParseMode.HTML,
            reply_markup=_profile_menu_keyboard(subscribe_label, has_subscription=has_subscription),
        )

    except Exception as e:
        logger.error(f"Error in handle_my_profile_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_my_stats_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Моя статистика'"""
    if not callback.from_user or callback.message is None:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return

    try:
        await callback.answer()

        if db is None:
            await callback.message.edit_text(
                "Статистика временно недоступна",
                parse_mode=ParseMode.HTML,
                reply_markup=_profile_menu_keyboard(),
            )
            return

        user_id = callback.from_user.id
        user = await db.ensure_user(
            user_id, default_trial=TRIAL_REQUESTS, is_admin=user_id in ADMIN_IDS
        )
        stats = await db.get_user_statistics(user_id, days=30)

        try:
            status_text, keyboard = await _generate_user_stats_response(
                user_id,
                days=30,
                stats=stats,
                user=user,
            )
        except RuntimeError as stats_error:
            logger.error("Failed to build user stats: %s", stats_error)
            await callback.message.edit_text(
                "Статистика временно недоступна",
                parse_mode=ParseMode.HTML,
                reply_markup=_profile_menu_keyboard(),
            )
            return

        def generate_activity_graph(daily_data: Sequence[int]) -> tuple[str, int]:
            window = list(daily_data)[-7:]
            if not window:
                return "", 0
            max_val = max(window)
            if max_val <= 0:
                return "▁" * len(window), sum(window)
            bars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
            graph = "".join(
                bars[min(int((value / max_val) * (len(bars) - 1)), len(bars) - 1)]
                if value > 0
                else bars[0]
                for value in window
            )
            return graph, sum(window)

        def format_feature_name(feature: str | None) -> str:
            feature_names = {
                "legal_question": "⚖️ Юридические вопросы",
                "document_processing": "📄 Обработка документов",
                "judicial_practice": "📚 Судебная практика",
                "document_draft": "📝 Составление документов",
                "voice_message": "🎙️ Голосовые сообщения",
                "ocr_processing": "🔍 распознание текста",
                "document_chat": "💬 Чат с документом",
            }
            if not feature:
                return "Другие функции"
            return feature_names.get(feature, feature)

        extra_sections: list[str] = []
        divider = "──────────"

        def append_section(title: str) -> None:
            if not extra_sections:
                extra_sections.append(divider)
            extra_sections.append(title)

        daily_activity = stats.get("daily_activity") or []
        activity_graph, activity_total = generate_activity_graph(daily_activity)
        if activity_graph:
            append_section("📈 Активность (7 дн.)")
            extra_sections.append(f"• {activity_graph} — {activity_total} запросов")

        feature_stats = stats.get("feature_stats") or []
        if feature_stats:
            append_section("✨ Популярные функции")
            for feature_data in feature_stats[:5]:
                feature_name = format_feature_name(feature_data.get("feature"))
                count = feature_data.get("count", 0)
                extra_sections.append(f"• {feature_name}: {count}")

        if extra_sections:
            status_text = f"{status_text}\n\n" + "\n".join(extra_sections)

        await callback.message.edit_text(
            status_text,
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )

    except Exception as e:
        logger.error(f"Error in handle_my_stats_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


        await callback.answer('❌ Произошла ошибка', show_alert=True)

async def handle_back_to_main_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Назад' - возврат в главное меню"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        await callback.message.edit_text(
            _main_menu_text(),
            parse_mode=ParseMode.HTML,
            reply_markup=_main_menu_keyboard(),
        )

    except Exception as e:
        logger.error(f"Error in handle_back_to_main_callback: {e}")
        await callback.answer("❌ Произошла ошибка")

        await callback.answer("❌ Произошла ошибка")


async def handle_referral_program_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Реферальная программа'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        if db is None:
            await callback.message.edit_text(
                "Сервис временно недоступен",
                parse_mode=ParseMode.HTML,
                reply_markup=_profile_menu_keyboard(),
            )
            return

        user_id = callback.from_user.id
        user = await db.get_user(user_id)

        if not user:
            await callback.message.edit_text(
                "Ошибка получения данных пользователя",
                parse_mode=ParseMode.HTML,
                reply_markup=_profile_menu_keyboard(),
            )
            return

        referral_code: str | None = None
        stored_code = (getattr(user, 'referral_code', None) or '').strip()

        if stored_code and stored_code != 'SYSTEM_ERROR':
            referral_code = stored_code
        else:
            try:
                generated_code = (await db.generate_referral_code(user_id) or '').strip()
            except Exception as e:
                logger.error(f"Error with referral code: {e}")
                generated_code = ''
            if generated_code and generated_code != 'SYSTEM_ERROR':
                referral_code = generated_code
                try:
                    setattr(user, 'referral_code', referral_code)
                except Exception:
                    pass
            else:
                referral_code = None

        referral_link, share_code = _build_referral_link(referral_code)

        # Получаем список рефералов
        try:
            referrals = await db.get_user_referrals(user_id)
        except Exception as e:
            logger.error(f"Error getting referrals: {e}")
            referrals = []

        # Подсчитываем статистику
        total_referrals = len(referrals)
        active_referrals = sum(1 for ref in referrals if ref.get('has_active_subscription', False))

        # Безопасные значения для старых пользователей
        referral_bonus_days = getattr(user, 'referral_bonus_days', 0)
        referrals_count = getattr(user, 'referrals_count', 0)

        referral_lines: list[str] = [
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
            "┃  👥 <b>Реферальная программа</b>  ┃",
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛",
            "",
            "🎁 <b>Ваши бонусы</b>",
            "",
            f"  🎉 Бонусных дней: <b>{referral_bonus_days}</b>",
            f"  👫 Приглашено друзей: <b>{referrals_count}</b>",
            f"  ✅ С активной подпиской: <b>{active_referrals}</b>",
            "",
        ]

        if referral_link:
            referral_lines.extend([
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                "🔗 <b>Ваша реферальная ссылка</b>",
                "",
                f"<code>{referral_link}</code>",
                "",
            ])
        elif share_code:
            safe_code = html_escape(share_code)
            referral_lines.extend([
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                "🔗 <b>Ваш реферальный код</b>",
                "",
                f"<code>ref_{safe_code}</code>",
                "",
                "<i>Отправьте его друзьям, чтобы они\nуказали код при запуске бота</i>",
                "",
            ])
        else:
            referral_lines.extend([
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                "⚠️ <b>Ссылка временно недоступна</b>",
                "",
                "<i>Попробуйте позже или обратитесь\nв поддержку</i>",
                "",
            ])

        referral_lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "💡 <b>Как это работает</b>",
            "",
            "  1️⃣ Поделитесь ссылкой с друзьями",
            "  2️⃣ За каждого друга получите 3 дня",
            "  3️⃣ Друг получит скидку 20%",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "📈 <b>Ваши рефералы</b>",
            "",
        ])

        if referrals:
            referral_lines.append(f"  📊 Всего: <b>{total_referrals}</b>")
            referral_lines.append(f"  💎 С подпиской: <b>{active_referrals}</b>")

            # Показываем последних рефералов
            recent_referrals = referrals[:5]
            for ref in recent_referrals:
                join_date = datetime.fromtimestamp(ref['joined_at']).strftime('%d.%m.%Y')
                status = "💎" if ref['has_active_subscription'] else "👤"
                referral_lines.append(f"{status} Пользователь #{ref['user_id']} - {join_date}")
        else:
            referral_lines.append("• Пока никого нет")

        referral_text = "\n".join(referral_lines)

        # Создаем клавиатуру
        keyboard_buttons: list[list[InlineKeyboardButton]] = []
        if share_code:
            copy_text = "📋 Скопировать ссылку" if referral_link else "📋 Скопировать код"
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=copy_text,
                    callback_data=f"copy_referral_{share_code}",
                )
            ])

        # Кнопка назад к профилю
        keyboard_buttons.append([
            InlineKeyboardButton(text="🔙 Назад к профилю", callback_data="my_profile")
        ])

        referral_keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)

        await callback.message.edit_text(
            referral_text,
            parse_mode=ParseMode.HTML,
            reply_markup=referral_keyboard
        )

    except Exception as e:
        logger.error(f"Error in handle_referral_program_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_copy_referral_callback(callback: CallbackQuery):
    """Обработчик кнопки копирования реферальной ссылки"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        # Получаем код из callback_data
        callback_data = callback.data
        if callback_data and callback_data.startswith("copy_referral_"):
            referral_code = callback_data.replace("copy_referral_", "")
            referral_link, share_code = _build_referral_link(referral_code)

            if referral_link:
                await callback.answer(f"📋 Ссылка скопирована!\n{referral_link}", show_alert=True)
                return
            if share_code:
                await callback.answer(f"📋 Код скопирован!\nref_{share_code}", show_alert=True)
                return

            await callback.answer("❌ Реферальная ссылка временно недоступна", show_alert=True)
            return

        await callback.answer("❌ Ошибка получения кода")

    except Exception as e:
        logger.error(f"Error in handle_copy_referral_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_prepare_documents_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Подготовка документов'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        await callback.message.answer(
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃  📄 <b>Подготовка документов</b>  ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
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


async def handle_help_info_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Поддержка'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        support_text = (
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃  🔧 <b>Техническая поддержка</b>  ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
            "📞 <b>Контакты поддержки</b>\n"
            "   ├ Telegram: @support_username\n"
            "   └ Email: support@example.com\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "❓ <b>Часто задаваемые вопросы</b>\n\n"
            "🤖 <b>Бот не отвечает</b>\n"
            "   ├ Попробуйте команду /start\n"
            "   └ Проверьте интернет-соединение\n\n"
            "📄 <b>Ошибка при обработке документа</b>\n"
            "   ├ Форматы: PDF, DOCX, DOC, TXT\n"
            "   ├ Максимальный размер: 20 МБ\n"
            "   └ Проверьте целостность файла\n\n"
            "⏳ <b>Долгое ожидание ответа</b>\n"
            "   ├ Сложные запросы: 2-3 минуты\n"
            "   └ Большие документы: до 5 минут\n\n"
            "💬 <b>Как задать вопрос боту?</b>\n"
            "   ├ Напишите свой вопрос\n"
            "   ├ Можете прикрепить документ\n"
            "   └ Бот учитывает контекст беседы\n\n"
            "🔄 <b>Как начать новую беседу?</b>\n"
            "   ├ Используйте команду /start\n"
            "   └ Или кнопку \"Новый диалог\"\n\n"
            "💰 <b>Как проверить баланс?</b>\n"
            "   └ Откройте раздел \"Профиль\"\n\n"
            "🎯 <b>Какие запросы понимает бот?</b>\n"
            "   ├ Вопросы на любые темы\n"
            "   ├ Анализ документов и текстов\n"
            "   ├ Генерация контента\n"
            "   └ Помощь с кодом и задачами\n\n"
            "🔒 <b>Безопасны ли мои данные?</b>\n"
            "   ├ Все данные зашифрованы\n"
            "   └ Не передаем данные третьим лицам"
        )

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="↩️ Назад в меню", callback_data="back_to_main")]
            ]
        )

        await callback.message.answer(support_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

        logger.info(f"Support info requested by user {callback.from_user.id}")

    except Exception as e:
        logger.error(f"Error in handle_help_info_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


# ============ ОБРАБОТЧИКИ СИСТЕМЫ ДОКУМЕНТООБОРОТА ============


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
        voice_enabled = bool(getattr(config, "voice_mode_enabled", False))

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
    numbered_pattern = re.compile(r"^\s*(\d+)[\).:-]\s*(.*)")
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

    bullet_pattern = re.compile(r"^\s*[-\u2022]\s*(.*)")
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

    heading_pattern = re.compile(r"^\s*(?![-\u2022])(?!\d+[\).:-])([A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451\u0030-\u0039][^:]{0,80}):\s*(.*)$")
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

    # Показываем индикатор "отправляет документ" во время генерации
    try:
        async with typing_action(message.bot, message.chat.id, "upload_document"):
            result = await generate_document(openai_service, request_text, title, answers)
    except DocumentDraftingError as err:
        await message.answer(f"{Emoji.ERROR} Не удалось подготовить документ: {err}")
        await state.clear()
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка генерации документа: %s", exc, exc_info=True)
        await message.answer(f"{Emoji.ERROR} Произошла ошибка при генерации документа")
        await state.clear()
        return

    if result.status != "ok":
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
    notes: list[str] = []
    if result.validated:
        validated_items = "\n".join([f"  ✓ {item}" for item in result.validated])
        notes.append(
            f"✅ <b>Проверено:</b>\n{validated_items}"
        )

    if result.issues:
        issues_items = "\n".join([f"  ⚠️ {item}" for item in result.issues])
        notes.append(
            f"⚠️ <b>На что обратить внимание:</b>\n{issues_items}"
        )

    if notes:
        info_text = (
            f"<code>{'━' * 35}</code>\n"
            f"{chr(10).join(notes)}\n"
            f"<code>{'━' * 35}</code>"
        )
        await message.answer(info_text, parse_mode=ParseMode.HTML)

    # Создаем и отправляем документ
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        build_docx_from_markdown(result.markdown, str(tmp_path))
        display_title, caption, filename = _prepare_document_titles(result.title or title)

        # Красивое сообщение с документом
        final_caption = (
            f"📄 <b>{display_title}</b>\n"
            f"<code>{'─' * 30}</code>\n\n"
            f"✨ Документ успешно создан!\n"
            f"📎 Формат: DOCX\n\n"
            f"<i>💡 Проверьте содержимое и при необходимости внесите правки</i>"
        )

        await message.answer_document(
            FSInputFile(str(tmp_path), filename=filename),
            caption=final_caption,
            parse_mode=ParseMode.HTML
        )
    except DocumentDraftingError as err:
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
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
            "┃  🗂️ <b>Работа с документами</b>  ┃\n"
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
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
                "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                "┃  📋 <b>Краткая выжимка</b>       ┃\n"
                "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
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
                "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                "┃  ⚠️ <b>Риск-анализ</b>           ┃\n"
                "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
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
                "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                "┃  ⚖️ <b>Анализ иска</b>             ┃\n"
                "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
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
                "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                "┃  💬 <b>Чат с документом</b>      ┃\n"
                "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
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
                "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                "┃  🔒 <b>Обезличивание</b>         ┃\n"
                "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
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
                "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                "┃  🌍 <b>Перевод документов</b>    ┃\n"
                "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
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
                "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
                "┃  🔍 <b>Распознавание текста</b>  ┃\n"
                "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
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

                documents_dir = Path("documents")
                documents_dir.mkdir(parents=True, exist_ok=True)
                safe_name = Path(file_name).name or "document"
                unique_name = f"{uuid.uuid4().hex}_{safe_name}"
                stored_path = documents_dir / unique_name

                file_bytes = file_content.read()
                stored_path.write_bytes(file_bytes)
                await send_progress({"stage": "uploaded", "percent": 32})

                try:
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
                finally:
                    with suppress(Exception):
                        stored_path.unlink(missing_ok=True)

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

                documents_dir = Path("documents")
                documents_dir.mkdir(parents=True, exist_ok=True)
                safe_name = Path(file_name).name or "photo.jpg"
                unique_name = f"{uuid.uuid4().hex}_{safe_name}"
                stored_path = documents_dir / unique_name

                file_bytes = file_content.read()
                stored_path.write_bytes(file_bytes)
                await send_progress({"stage": "uploaded", "percent": 32})

                try:
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
                finally:
                    with suppress(Exception):
                        stored_path.unlink(missing_ok=True)

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
        user_id = _ensure_valid_user_id(message.from_user.id, context="cmd_ratings_stats")
    except ValidationException as exc:
        logger.warning("Некорректный пользователь id in cmd_ratings_stats: %s", exc)
        await message.answer("❌ Ошибка идентификатора пользователя")
        return

    if user_id not in ADMIN_IDS:
        await message.answer("❌ Команда доступна только администраторам")
        return

    stats_fn = _get_safe_db_method("get_ratings_statistics", default_return={})
    low_rated_fn = _get_safe_db_method("get_low_rated_requests", default_return=[])
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
        user_id = _ensure_valid_user_id(message.from_user.id, context="cmd_error_stats")
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

async def pre_checkout(pre: PreCheckoutQuery):
    try:
        payload_raw = pre.invoice_payload or ""
        parsed = None
        try:
            parsed = parse_subscription_payload(payload_raw)
        except SubscriptionPayloadError:
            parsed = None

        plan_info = _get_plan_pricing(parsed.plan_id if parsed else None)
        if plan_info is None:
            plan_info = DEFAULT_SUBSCRIPTION_PLAN
        if plan_info is None:
            await pre.answer(ok=False, error_message="Подписка недоступна")
            return

        method = (parsed.method if parsed else "").lower()
        if method == "xtr":
            method = "stars"

        if parsed and pre.from_user and parsed.user_id and parsed.user_id != pre.from_user.id:
            await pre.answer(ok=False, error_message="Счёт предназначен для другого пользователя")
            return

        if method == "rub":
            expected_currency = "RUB"
            expected_amount = plan_info.price_rub_kopeks
        elif method == "stars":
            expected_currency = "XTR"
            expected_amount = _plan_stars_amount(plan_info)
        else:
            expected_currency = pre.currency.upper()
            expected_amount = pre.total_amount

        if expected_amount <= 0:
            await pre.answer(ok=False, error_message="Некорректная сумма оплаты")
            return

        if pre.currency.upper() != expected_currency or int(pre.total_amount) != int(expected_amount):
            await pre.answer(ok=False, error_message="Некорректные параметры оплаты")
            return

        amount_major = pre.total_amount / 100 if expected_currency == "RUB" else pre.total_amount
        amount_check = InputValidator.validate_payment_amount(amount_major, expected_currency)
        if not amount_check.is_valid:
            await pre.answer(ok=False, error_message="Сумма оплаты вне допустимого диапазона")
            return

        await pre.answer(ok=True)
    except Exception:
        await pre.answer(ok=False, error_message="Ошибка проверки оплаты, попробуйте позже")



async def on_successful_payment(message: Message):
    try:
        sp = message.successful_payment
        if sp is None or message.from_user is None:
            return

        currency_up = sp.currency.upper()
        if currency_up == "RUB":
            provider_name = "telegram_rub"
        elif currency_up == "XTR":
            provider_name = "telegram_stars"
        else:
            provider_name = f"telegram_{currency_up.lower()}"

        payload_raw = sp.invoice_payload or ""
        parsed_payload = None
        try:
            parsed_payload = parse_subscription_payload(payload_raw)
        except SubscriptionPayloadError:
            parsed_payload = None

        plan_info = _get_plan_pricing(parsed_payload.plan_id if parsed_payload else None)
        if plan_info is None:
            plan_info = DEFAULT_SUBSCRIPTION_PLAN

        cfg = settings()
        duration_days = plan_info.plan.duration_days if plan_info else max(1, int(cfg.sub_duration_days or 30))
        quota = plan_info.plan.request_quota if plan_info else 0
        plan_id = plan_info.plan.plan_id if plan_info else (parsed_payload.plan_id if parsed_payload and parsed_payload.plan_id else "legacy")

        user_id = message.from_user.id
        new_until = None
        new_balance: int | None = None

        if db is not None and sp.telegram_payment_charge_id:
            exists = await db.transaction_exists_by_telegram_charge_id(sp.telegram_payment_charge_id)
            if exists:
                return

        if db is not None:
            await db.record_transaction(
                user_id=user_id,
                provider=provider_name,
                currency=sp.currency,
                amount=sp.total_amount,
                amount_minor_units=sp.total_amount,
                payload=payload_raw,
                status=TransactionStatus.COMPLETED.value,
                telegram_payment_charge_id=sp.telegram_payment_charge_id,
                provider_payment_charge_id=sp.provider_payment_charge_id,
            )

            if plan_info is not None:
                new_until, new_balance = await db.apply_subscription_purchase(
                    user_id,
                    plan_id=plan_id,
                    duration_days=duration_days,
                    request_quota=quota,
                )
            else:
                await db.extend_subscription_days(user_id, duration_days)
                user = await db.get_user(user_id)
                if user and user.subscription_until:
                    new_until = int(user.subscription_until)
                if user and getattr(user, "subscription_requests_balance", None) is not None:
                    new_balance = int(getattr(user, "subscription_requests_balance"))

        response_lines = [f"{Emoji.SUCCESS} <b>Оплата получена!</b>"]
        if plan_info is not None:
            response_lines.append(f"Тариф: <b>{plan_info.plan.name}</b>")
            response_lines.append(f"Срок действия: {duration_days} дней")
            response_lines.append(f"Квота: {plan_info.plan.request_quota} запросов")
        elif parsed_payload and parsed_payload.plan_id:
            response_lines.append(f"Тариф: {parsed_payload.plan_id}")
            response_lines.append(f"Срок действия: {duration_days} дней")

        if new_until:
            until_text = datetime.fromtimestamp(new_until).strftime("%Y-%m-%d")
            response_lines.append(f"Доступ до: {until_text}")

        if plan_info is not None and new_balance is not None:
            response_lines.append(f"Остаток запросов: {new_balance}")

        response_lines.append("Проверить подписку — команда /status.")

        await message.answer("\n".join(response_lines), parse_mode=ParseMode.HTML)
    except Exception:
        logger.exception("Failed to handle successful payment")



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
    global BOT_USERNAME
    try:
        bot_info = await bot.get_me()
        BOT_USERNAME = (bot_info.username or '').strip()
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
    set_system_status("starting")

    logger.info("🚀 Starting AI-Ivan (simple)")

    # Инициализация глобальных переменных
    global db, openai_service, audio_service, rate_limiter, access_service, session_store, crypto_provider, error_handler, document_manager

    # Используем продвинутую базу данных с connection pooling
    logger.info("Using advanced database with connection pooling")
    db = ctx.db or container.get(DatabaseAdvanced)
    ctx.db = db
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

    rate_limiter = ctx.rate_limiter or container.get(RateLimiter)
    ctx.rate_limiter = rate_limiter
    await rate_limiter.init()

    access_service = ctx.access_service or container.get(AccessService)
    ctx.access_service = access_service

    openai_service = ctx.openai_service or container.get(OpenAIService)
    openai_service.cache = response_cache
    ctx.openai_service = openai_service

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
        logger.info("Voice mode disabled")

    session_store = ctx.session_store or container.get(SessionStore)
    ctx.session_store = session_store
    crypto_provider = ctx.crypto_provider or container.get(CryptoPayProvider)
    ctx.crypto_provider = crypto_provider

    error_handler = ErrorHandler(logger=logger)
    ctx.error_handler = error_handler

    dp.update.middleware(ErrorHandlingMiddleware(error_handler, logger=logger))

    document_manager = DocumentManager(openai_service=openai_service, settings=cfg)
    ctx.document_manager = document_manager
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
            logger.info("🔄 Scaling components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize scaling components: {e}")

    # Health checks
    health_checker = HealthChecker(check_interval=cfg.health_check_interval)
    ctx.health_checker = health_checker
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
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_buy, Command("buy"))
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(cmd_mystats, Command("mystats"))
    dp.message.register(cmd_ratings_stats, Command("ratings"))
    dp.message.register(cmd_error_stats, Command("errors"))
    dp.message.register(cmd_askdoc, Command("askdoc"))
    dp.message.register(cmd_enddoc, Command("enddoc"))

    dp.callback_query.register(handle_ignore_callback, F.data == "ignore")
    dp.callback_query.register(handle_rating_callback, F.data.startswith("rate_"))
    dp.callback_query.register(
        handle_feedback_callback, F.data.startswith(("feedback_", "skip_feedback_"))
    )

    # Обработчики кнопок главного меню
    dp.callback_query.register(handle_search_practice_callback, F.data == "search_practice")
    dp.callback_query.register(handle_prepare_documents_callback, F.data == "prepare_documents")
    dp.callback_query.register(handle_help_info_callback, F.data == "help_info")
    dp.callback_query.register(handle_my_profile_callback, F.data == "my_profile")
    
    # Обработчики профиля
    dp.callback_query.register(handle_my_stats_callback, F.data == "my_stats")
    dp.callback_query.register(handle_get_subscription_callback, F.data == "get_subscription")
    dp.callback_query.register(handle_cancel_subscription_callback, F.data == "cancel_subscription")
    dp.callback_query.register(handle_buy_catalog_callback, F.data == "buy_catalog")
    dp.callback_query.register(handle_verify_payment_callback, F.data.startswith("verify_payment:"))
    dp.callback_query.register(handle_select_plan_callback, F.data.startswith("select_plan:"))
    dp.callback_query.register(handle_pay_plan_callback, F.data.startswith("pay_plan:"))
    dp.callback_query.register(handle_referral_program_callback, F.data == "referral_program")
    dp.callback_query.register(handle_copy_referral_callback, F.data.startswith("copy_referral_"))
    dp.callback_query.register(handle_back_to_main_callback, F.data == "back_to_main")

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
        dp.message.register(process_voice_message, F.voice)

    dp.message.register(on_successful_payment, F.successful_payment)
    dp.pre_checkout_query.register(pre_checkout)
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
