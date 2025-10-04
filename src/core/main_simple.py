"""
Простая версия Telegram бота ИИ-Иван
Только /start и обработка вопросов, никаких кнопок и лишних команд
"""

from __future__ import annotations

import asyncio
import logging
import time
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid
from typing import TYPE_CHECKING, Any

from src.core.safe_telegram import send_html_text
from src.documents.document_manager import DocumentManager

if TYPE_CHECKING:
    from src.core.db_advanced import DatabaseAdvanced, TransactionStatus

import re
from html import escape as html_escape

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BotCommand,
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
from src.bot.document_drafter import (
    DocumentDraftingError,
    build_docx_from_markdown,
    format_plan_summary,
    generate_document,
    plan_document,
)
from src.bot.status_manager import ProgressStatus, progress_router

from src.bot.stream_manager import StreamingCallback, StreamManager
from src.bot.ui_components import Emoji, sanitize_telegram_html, render_legal_html
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
from src.telegram_legal_bot.ratelimit import RateLimiter

SAFE_LIMIT = 3900  # чуть меньше телеграмного 4096 (запас на теги)

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
error_handler = None
document_manager = None
response_cache = None
stream_manager = None
metrics_collector = None
task_manager = None
health_checker = None
scaling_components = None


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
        'error_handler': _runtime.error_handler,
        'document_manager': _runtime.document_manager,
        'response_cache': _runtime.response_cache,
        'stream_manager': _runtime.stream_manager,
        'metrics_collector': _runtime.metrics_collector,
        'task_manager': _runtime.task_manager,
        'health_checker': _runtime.health_checker,
        'scaling_components': _runtime.scaling_components,
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
    await status.start(auto_cycle=True, interval=1.0)  # см. пункт 2 ниже
    return status

async def _stop_status_indicator(status: ProgressStatus | None, ok: bool) -> None:
    if status is None:
        return

    try:
        if ok:
            await status.complete()  # ставит «выполнено», фиксирует время
        else:
            await status.fail("Ошибка при формировании ответа")
    except Exception:
        return  # опционально залогируй

    # по запросу: удаляем прогресс-бар, когда ответ пришёл
    if ok and getattr(status, "message_id", None):
        with suppress(Exception):
            await status.bot.delete_message(status.chat_id, status.message_id)

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
    """Создаёт клавиатуру для возврата и повторной загрузки OCR."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text=f"{Emoji.BACK} Назад", callback_data="back_to_menu"),
            InlineKeyboardButton(text=f"{Emoji.DOCUMENT} Загрузить ещё", callback_data=f"ocr_upload_more:{output_format}")
        ]]
    )


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
        if existing_rating:
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

•  <b>Анализ судебной практики</b> — поиск релевантных решений судов по вашему делу
•  <b>Составление документов</b> — иски, жалобы, договоры
•  <b>Юридическая стратегия</b> — рекомендации по защите интересов и разрешению споров
•  <b>Консультации без ограничений</b> — ответы на любые правовые вопросы
•  <b>Проверка документов</b> — выявление юридических рисков и подводных камней Анализ судебной практики — поиск релевантных решений судов по вашему делу
•  <b>Составление документов</b> — иски, жалобы, договоры
•  <b>Юридическая стратегия</b> — рекомендации по защите интересов и разрешению споров
•  <b>Консультации без ограничений</b> — ответы на любые правовые вопросы
•  <b>Проверка документов</b> — выявление юридических рисков и подводных камней

<b>ПРИМЕРЫ ОБРАЩЕНИЙ:</b>

💬 "Администрация отказала в согласовании — подбери стратегию обжалования со ссылками на судебную практику"

💬 "Проанализируй различия между статьями 228 и 228.1 УК РФ"

💬 "Найди судебную практику по взысканию неустойки с застройщика"

💬 "Могут ли наследники оспорить завещание после 6 месяцев?"

<b>ПОПРОБУЙ ПРЯМО СЕЙЧАС 👇</b>"""

    # Создаем inline клавиатуру с кнопками (компактное размещение)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🔍 Поиск судебной практики", callback_data="search_practice"),
            ],
            [
                InlineKeyboardButton(text="🗂️ Работа с документами", callback_data="document_processing" ),
            ],
            [
                InlineKeyboardButton(text="👤 Мой профиль", callback_data="my_profile"),
                InlineKeyboardButton(text="💬 Поддержка", callback_data="help_info"),
            ],
        ]
    )

    if WELCOME_MEDIA and WELCOME_MEDIA.path.exists():
        try:
            await message.answer_video(
                video=FSInputFile(WELCOME_MEDIA.path),
                caption=sanitize_telegram_html(welcome_raw),  # текст под видео
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
                supports_streaming=True  # чтобы можно было смотреть без полного скачивания
            )
            return
        except Exception as video_error:
            logger.warning("Failed to send welcome video: %s", video_error)


    # финальный фолбэк — просто текст
    await message.answer(
        sanitize_telegram_html(welcome_raw),
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard
    )
    logger.info("User %s started bot", message.from_user.id)


async def process_question(message: Message, *, text_override: str | None = None):
    """Главный обработчик юридических вопросов"""
    if not message.from_user:
        return

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

    # Таймер ответа
    timer = ResponseTimer()
    timer.start()

    logger.info("Processing question from user %s: %s", user_id, question_text[:100])

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
            if getattr(user_session, "practice_search_mode", False):
                selected_prompt = JUDICIAL_PRACTICE_SEARCH_PROMPT
                user_session.practice_search_mode = False

            if text_override is not None and getattr(message, "voice", None):
                selected_prompt = (
                    selected_prompt
                    + "\n\nГолосовой режим: сохрани указанную структуру блоков, обязательно перечисли нормативные акты с точными реквизитами и уточни, что текстовый ответ уже предоставлен в чате."
                )

            # --- Запрос к OpenAI (стрим/нестрим) ---
            if openai_service is None:
                raise SystemException("OpenAI service not initialized", error_context)

            if USE_STREAMING and message.bot:
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
                        selected_prompt, question_text, callback=callback
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
                        result = await openai_service.ask_legal(selected_prompt, question_text)
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
                result = await openai_service.ask_legal(selected_prompt, question_text)
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
        if USE_STREAMING and had_stream_content and stream_manager is not None:
            final_stream_text = stream_final_text or ((isinstance(result, dict) and (result.get("text") or "")) or "")
            combined_stream_text = (final_stream_text.rstrip() + f"\n\n{time_footer_raw}") if final_stream_text else time_footer_raw
            final_answer_text = combined_stream_text
            await stream_manager.finalize(combined_stream_text)
        else:
            text_to_send = (isinstance(result, dict) and (result.get("text") or "")) or ""
            if text_to_send:
                combined_text = f"{text_to_send.rstrip()}\n\n{time_footer_raw}"
                final_answer_text = combined_text
                await send_html_text(
                    bot=message.bot,
                    chat_id=message.chat.id,
                    raw_text=combined_text,
                    reply_to_message_id=message.message_id,
                )

        # Сообщения про квоту/подписку
        if final_answer_text:
            user_session.last_answer_snapshot = final_answer_text

        if quota_text and not quota_is_trial:
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
    else:
        await callback.message.answer("❌ Этот способ оплаты не поддерживается")
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

    try:
        user_id = message.from_user.id
        user = await db.ensure_user(
            user_id, default_trial=TRIAL_REQUESTS, is_admin=user_id in ADMIN_IDS
        )

        # Получаем детальную статистику
        stats = await db.get_user_statistics(user_id, days=30)

        plan_id = stats.get('subscription_plan') or getattr(user, 'subscription_plan', None)
        plan_info = _get_plan_pricing(plan_id) if plan_id else None
        plan_label = plan_info.plan.name if plan_info else (plan_id or '—')
        quota_balance_raw = stats.get('subscription_requests_balance')
        quota_balance = int(quota_balance_raw) if quota_balance_raw is not None else None

        # Форматируем даты
        def format_timestamp(ts):
            if not ts or ts == 0:
                return 'Никогда'
            return datetime.fromtimestamp(ts).strftime('%d.%m.%Y %H:%M')

        def format_subscription_status(until_ts):
            if not until_ts or until_ts == 0:
                return '❌ Не активна'
            until_dt = datetime.fromtimestamp(until_ts)
            if until_dt < datetime.now():
                return '❌ Истекла'
            days_left = (until_dt - datetime.now()).days
            return f"✔ До {until_dt.strftime('%d.%m.%Y')} ({days_left} дн.)"

        status_lines = [
            '📊 <b>Моя статистика</b>',
            '',
            '👤 <b>Профиль</b>',
            f"• ID: <code>{user_id}</code>",
            f"• Статус: {'🛡️ Администратор' if stats.get('is_admin') else '👤 Пользователь'}",
            f"• Регистрация: {format_timestamp(getattr(user, 'created_at', 0))}",
            '',
            '💰 <b>Баланс и доступ</b>',
            f"• Пробные запросы: {stats.get('trial_remaining', 0)} из {TRIAL_REQUESTS}",
        ]
        if plan_id or plan_info:
            status_lines.append(f"• Тариф: {plan_label}")
        status_lines.append(f"• Подписка: {format_subscription_status(stats.get('subscription_until', 0))}")
        if quota_balance is not None and (plan_id or plan_info):
            status_lines.append(f"• Остаток запросов: {max(0, quota_balance)}")

        status_lines.extend([
            '',
            '📈 <b>Общая статистика</b>',
            f"• Всего запросов: {stats.get('total_requests', 0)}",
            f"• Успешных: {stats.get('successful_requests', 0)} ✔",
            f"• Неудачных: {stats.get('failed_requests', 0)} ✖",
            f"• Последний запрос: {format_timestamp(stats.get('last_request_at', 0))}",
            '',
            '📅 <b>За последние 30 дней</b>',
            f"• Запросов: {stats.get('period_requests', 0)}",
            f"• Успешных: {stats.get('period_successful', 0)}",
            f"• Потрачено токенов: {stats.get('period_tokens', 0)}",
            f"• Среднее время ответа: {stats.get('avg_response_time_ms', 0)} мс",
        ])

        if stats.get('request_types'):
            status_lines.extend(['', '📊 <b>Типы запросов (30 дней)</b>'])
            for req_type, count in stats['request_types'].items():
                emoji = '⚖️' if req_type == 'legal_question' else '🤔'
                status_lines.append(f"• {emoji} {req_type}: {count}")

        status_text = "\n".join(status_lines)

        await message.answer(status_text, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Error in cmd_mystats: {e}")
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

    if audio_service is None or not settings().voice_mode_enabled:
        await message.answer("Voice mode is currently unavailable. Please send text.")
        return

    if not message.bot:
        await message.answer("Unable to access bot context for processing the voice message.")
        return

    temp_voice_path: Path | None = None
    tts_path: Path | None = None

    try:
        await audio_service.ensure_short_enough(message.voice.duration)

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

        try:
            tts_path = await audio_service.synthesize(response_text)
        except Exception as tts_error:
            logger.warning("Text-to-speech failed: %s", tts_error)
            return

        await message.answer_voice(
            FSInputFile(tts_path),
            caption=f"{Emoji.ROBOT} Voice reply",
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
            if tts_path:
                tts_path.unlink()


# ============ СИСТЕМА РЕЙТИНГА ============

async def handle_ocr_upload_more(callback: CallbackQuery, state: FSMContext):
    """Prepare state for another OCR upload after a result message."""
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
            f"{Emoji.DOCUMENT} Отправьте следующий файл или фото для OCR.",
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

        # Создаем сообщение для запроса вопроса
        await callback.message.answer(
            "🔍 <b>Поиск и аналитика судебной практики</b>\n\n"
            "📝 Опишите ваш юридический вопрос, и я найду релевантную судебную практику:\n\n"
            "• Получите краткую консультацию с 2 ссылками на практику\n"
            "• Возможность углубленного анализа с 6+ примерами\n"
            "• Подготовка документов на основе практики\n\n"
            "<i>Напишите ваш вопрос следующим сообщением...</i>",
            parse_mode=ParseMode.HTML,
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

        # Создаем клавиатуру с кнопками профиля
        profile_keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="📊 Моя статистика", callback_data="my_stats"),
                    InlineKeyboardButton(text="💎 Статус подписки", callback_data="subscription_status"),
                ],
                [
                    InlineKeyboardButton(text="💳 История платежей", callback_data="payment_history"),
                    InlineKeyboardButton(text="👥 Реферальная программа", callback_data="referral_program"),
                ],
                [
                    InlineKeyboardButton(text="🔙 Назад", callback_data="back_to_main"),
                ],
            ]
        )

        await callback.message.answer(
            "👤 <b>Мой профиль</b>\n\n"
            "Выберите действие:",
            parse_mode=ParseMode.HTML,
            reply_markup=profile_keyboard
        )

    except Exception as e:
        logger.error(f"Error in handle_my_profile_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_my_stats_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Моя статистика'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        # Используем существующую логику из cmd_mystats
        if db is None:
            await callback.message.answer("Статистика временно недоступна")
            return

        user_id = callback.from_user.id
        user = await db.ensure_user(
            user_id, default_trial=TRIAL_REQUESTS, is_admin=user_id in ADMIN_IDS
        )

        # Получаем детальную статистику
        stats = await db.get_user_statistics(user_id, days=30)

        # Форматируем даты
        def format_timestamp(ts):
            if not ts or ts == 0:
                return "Никогда"
            return datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")

        def format_subscription_status(until_ts):
            if not until_ts or until_ts == 0:
                return "❌ Не активна"
            now = int(time.time())
            if until_ts > now:
                dt = datetime.fromtimestamp(until_ts)
                return f"✅ До {dt.strftime('%d.%m.%Y')}"
            else:
                return "⏰ Истекла"

        status_text = f"""📊 <b>Моя статистика</b>

👤 <b>Профиль</b>
• ID: {user_id}
• Триал: {stats.get('trial_remaining', 0)} запросов
• Админ: {"✅" if stats.get('is_admin', False) else "❌"}
• Создан: {format_timestamp(stats.get('created_at', 0))}
• Обновлён: {format_timestamp(stats.get('updated_at', 0))}
• Подписка: {format_subscription_status(stats.get('subscription_until', 0))}

📈 <b>Общая статистика</b>
• Всего запросов: {stats.get('total_requests', 0)}
• За 30 дней: {stats.get('recent_requests', 0)}
• Последний запрос: {format_timestamp(stats.get('last_request_at', 0))}

📋 <b>По типам запросов (30 дней)</b>"""

        # Добавляем статистику по типам
        type_stats = stats.get('request_types', {})
        if type_stats:
            for req_type, count in type_stats.items():
                status_text += f"\n• {req_type}: {count}"
        else:
            status_text += "\n• Нет данных"

        # Добавляем кнопку "Назад"
        back_keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад к профилю", callback_data="my_profile")],
            ]
        )

        await callback.message.answer(
            status_text,
            parse_mode=ParseMode.HTML,
            reply_markup=back_keyboard
        )

    except Exception as e:
        logger.error(f"Error in handle_my_stats_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_subscription_status_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Статус подписки'"""
    if not callback.from_user or callback.message is None:
        await callback.answer('❌ Ошибка данных', show_alert=True)
        return

    try:
        await callback.answer()

        if db is None:
            await callback.message.answer('Сервис временно недоступен')
            return

        user_id = callback.from_user.id
        user = await db.ensure_user(
            user_id, default_trial=TRIAL_REQUESTS, is_admin=user_id in ADMIN_IDS
        )

        has_subscription = await db.has_active_subscription(user_id)
        plan_id = getattr(user, 'subscription_plan', None)
        plan_info = _get_plan_pricing(plan_id) if plan_id else None
        plan_label = plan_info.plan.name if plan_info else (plan_id or '—')
        quota_balance_raw = getattr(user, 'subscription_requests_balance', None)
        quota_balance = int(quota_balance_raw) if quota_balance_raw is not None else None

        keyboard_buttons: list[list[InlineKeyboardButton]] = []

        if has_subscription and user.subscription_until:
            until_dt = datetime.fromtimestamp(user.subscription_until)
            left_days = max(0, (until_dt - datetime.now()).days)
            status_lines = [
                f"{Emoji.DIAMOND} <b>Статус подписки</b>",
                '',
                '✅ <b>Подписка активна</b>',
                f"• План: {plan_label}",
                f"• Доступ до: {until_dt:%d.%m.%Y %H:%M} (≈{left_days} дн.)",
            ]
            if plan_info and quota_balance is not None:
                status_lines.append(f"• Остаток запросов: {max(0, quota_balance)}")
            elif plan_id and quota_balance is not None:
                status_lines.append(f"• Остаток запросов: {max(0, quota_balance)}")
            elif plan_id is None:
                status_lines.append('• Лимит: без ограничений')
            status_lines.append('')
            status_lines.append('🛠 Пополнить пакет — команда /buy.')
            status_text = "\n".join(status_lines)
            keyboard_buttons.append([InlineKeyboardButton(text='🛒 Каталог тарифов', callback_data='buy_catalog')])
        else:
            trial_remaining = getattr(user, 'trial_remaining', 0)
            status_lines = [
                f"{Emoji.DIAMOND} <b>Статус подписки</b>",
                '',
                '❌ <b>Подписка не активна</b>',
                f"🔓 Триал: {trial_remaining} запросов",
                '',
                '💡 Оформите подписку, чтобы получить дополнительные запросы, приоритет и поддержку.',
            ]
            status_text = "\n".join(status_lines)
            keyboard_buttons.append([InlineKeyboardButton(text='💳 Оформить подписку', callback_data='get_subscription')])

        keyboard_buttons.append([InlineKeyboardButton(text='🔙 Назад к профилю', callback_data='my_profile')])
        subscription_keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)

        await callback.message.answer(
            status_text,
            parse_mode=ParseMode.HTML,
            reply_markup=subscription_keyboard,
        )

    except Exception as e:
        logger.error(f"Error in handle_subscription_status_callback: {e}")
        await callback.answer('❌ Произошла ошибка', show_alert=True)

async def handle_back_to_main_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Назад' - возврат в главное меню"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        # Отправляем главное меню (как в команде /start)
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="🔍 Поиск судебной практики", callback_data="search_practice"),
                ],
                [
                    InlineKeyboardButton(text="🗂️ Работа с документами", callback_data="document_processing" ),
                ],
                [
                    InlineKeyboardButton(text="👤 Мой профиль", callback_data="my_profile"),
                    InlineKeyboardButton(text="💬 Поддержка", callback_data="help_info"),
                ],
            ]
        )

        await callback.message.answer(
            "🏠 <b>Главное меню</b>\n\nВыберите действие:",
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard
        )

    except Exception as e:
        logger.error(f"Error in handle_back_to_main_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_payment_history_callback(callback: CallbackQuery):
    """Обработчик кнопки 'История платежей'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        if db is None:
            await callback.message.answer("Сервис временно недоступен")
            return

        user_id = callback.from_user.id

        # Получаем историю транзакций
        transactions = await db.get_user_transactions(user_id, limit=15)
        transaction_stats = await db.get_transaction_stats(user_id)

        if not transactions:
            history_text = """💳 <b>История платежей</b>

📊 <b>Статистика</b>
• Всего транзакций: 0
• Потрачено: 0 ₽

❌ <b>История пуста</b>
У вас пока нет платежей."""
        else:
            def format_transaction_date(timestamp):
                if timestamp:
                    return datetime.fromtimestamp(timestamp).strftime("%d.%m.%Y %H:%M")
                return "Неизвестно"

            def format_transaction_status(status):
                status_map = {
                    "completed": "✅ Завершен",
                    "pending": "⏳ В обработке",
                    "failed": "❌ Отклонен",
                    "cancelled": "🚫 Отменен"
                }
                return status_map.get(status, f"❓ {status}")

            def format_amount(amount, currency):
                if currency == "RUB":
                    return f"{amount} ₽"
                elif currency == "XTR":
                    return f"{amount} ⭐"
                else:
                    return f"{amount} {currency}"

            history_text = f"""💳 <b>История платежей</b>

📊 <b>Статистика</b>
• Всего транзакций: {transaction_stats.get('total_transactions', 0)}
• Потрачено: {transaction_stats.get('total_spent', 0)} ₽

📝 <b>Последние операции</b>"""

            for transaction in transactions:
                history_text += f"""

💰 {format_amount(transaction.amount, transaction.currency)}
├ {format_transaction_status(transaction.status)}
├ {transaction.provider}
└ {format_transaction_date(transaction.created_at)}"""

        # Кнопка назад к профилю
        back_keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад к профилю", callback_data="my_profile")],
            ]
        )

        await callback.message.answer(
            history_text,
            parse_mode=ParseMode.HTML,
            reply_markup=back_keyboard
        )

    except Exception as e:
        logger.error(f"Error in handle_payment_history_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_referral_program_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Реферальная программа'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        if db is None:
            await callback.message.answer("Сервис временно недоступен")
            return

        user_id = callback.from_user.id
        user = await db.get_user(user_id)

        if not user:
            await callback.message.answer("Ошибка получения данных пользователя")
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
            "👥 <b>Реферальная программа</b>",
            "",
            "🎁 <b>Ваши бонусы</b>",
            f"• Бонусных дней: {referral_bonus_days}",
            f"• Приглашено друзей: {referrals_count}",
            f"• С активной подпиской: {active_referrals}",
            "",
        ]

        if referral_link:
            referral_lines.extend([
                "🔗 <b>Ваша реферальная ссылка</b>",
                f"<code>{referral_link}</code>",
            ])
        elif share_code:
            safe_code = html_escape(share_code)
            referral_lines.extend([
                "🔗 <b>Ваш реферальный код</b>",
                f"<code>ref_{safe_code}</code>",
                "Отправьте его друзьям, чтобы они указали код при запуске бота.",
            ])
        else:
            referral_lines.extend([
                "🔗 <b>Реферальная ссылка временно недоступна</b>",
                "Попробуйте позже или обратитесь в поддержку.",
            ])

        referral_lines.extend([
            "",
            "💡 <b>Как это работает</b>",
            "• Поделитесь ссылкой с друзьями",
            "• За каждого друга получите 3 дня подписки",
            "• Друг получит скидку 20% на первую покупку",
            "",
            "📈 <b>Ваши рефералы</b>",
        ])

        if referrals:
            referral_lines.append(f"• Всего: {total_referrals}")
            referral_lines.append(f"• С подпиской: {active_referrals}")

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

        await callback.message.answer(
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
            "📄 <b>Подготовка документов</b>\n\n"
            "📑 Я помогу составить процессуальные документы:\n\n"
            "• Исковые заявления\n"
            "• Ходатайства\n"
            "• Жалобы и возражения\n"
            "• Договоры и соглашения\n\n"
            "<i>Опишите какой документ нужно подготовить и приложите детали дела...</i>",
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

        support_text = """<b>💬 Техническая поддержка</b>

Если у вас возникли технические проблемы с работой бота или есть предложения по улучшению, обращайтесь к администраторам:

<b>Контакты поддержки:</b>
• Telegram: @support_username
• Email: support@example.com

<b>Часто возникающие вопросы:</b>

<b>Бот не отвечает</b>
Попробуйте отправить команду /start

<b>Ошибка при обработке документа</b>
Проверьте формат файла (поддерживаются PDF, DOCX, DOC, TXT)

<b>Долгое ожидание ответа</b>
Сложные запросы могут обрабатываться до 2-3 минут"""

        await callback.message.answer(support_text, parse_mode=ParseMode.HTML)

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
        await state.clear()
        await state.set_state(DocumentDraftStates.waiting_for_request)
        intro_text = (
            f"{Emoji.MAGIC} <b>Создание юридического документа</b>\n\n"
            "Кратко опишите, какой документ нужен и для какой ситуации. "
            "После этого я задам нужные уточнения и подготовлю проект в формате DOCX."
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
        await callback.message.answer(f"{Emoji.BACK} Создание документа отменено")
    with suppress(Exception):
        await callback.answer()


async def handle_doc_draft_request(message: Message, state: FSMContext) -> None:
    """Обработка исходного запроса юриста."""
    request_text = (message.text or "").strip()
    if not request_text:
        await message.answer(f"{Emoji.WARNING} Опишите, пожалуйста, какой документ нужен")
        return

    if openai_service is None:
        await message.answer(f"{Emoji.ERROR} Сервис генерации документов временно недоступен. Попробуйте позже.")
        await state.clear()
        return

    status_msg = await message.answer(f"{Emoji.LOADING} Анализирую запрос…")
    try:
        plan = await plan_document(openai_service, request_text)
    except DocumentDraftingError as err:
        with suppress(Exception):
            await status_msg.edit_text(f"{Emoji.ERROR} Не получилось составить план вопросов: {err}")
        await state.clear()
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка планирования документа: %s", exc, exc_info=True)
        with suppress(Exception):
            await status_msg.edit_text(f"{Emoji.ERROR} Что-то пошло не так при обращении к модели")
        await state.clear()
        return
    else:
        with suppress(Exception):
            await status_msg.delete()

    await state.update_data(
        draft_request=request_text,
        draft_plan={"title": plan.title, "questions": plan.questions, "notes": plan.notes},
        draft_answers=[],
        current_question_index=0,
    )

    summary = format_plan_summary(plan)
    for chunk in _split_plain_text(summary):
        await message.answer(chunk)

    if plan.questions:
        await state.set_state(DocumentDraftStates.asking_details)
        await _send_next_question(message, state, prefix="Вопрос")
    else:
        await state.set_state(DocumentDraftStates.generating)
        await message.answer(f"{Emoji.LOADING} Деталей достаточно, формирую документ…")
        await _finalize_draft(message, state)


async def handle_doc_draft_answer(message: Message, state: FSMContext) -> None:
    """Обработка ответов юриста на уточняющие вопросы."""
    data = await state.get_data()
    plan = data.get("draft_plan") or {}
    questions = plan.get("questions") or []
    index = data.get("current_question_index", 0)

    if index >= len(questions):
        await message.answer(f"{Emoji.WARNING} Все ответы уже получены, приступаю к формированию документа")
        await state.set_state(DocumentDraftStates.generating)
        await _finalize_draft(message, state)
        return

    answer_text = (message.text or "").strip()
    if not answer_text:
        await message.answer(f"{Emoji.WARNING} Пожалуйста, введите ответ")
        return

    answers = data.get("draft_answers") or []
    answers.append({"question": questions[index]["text"], "answer": answer_text})
    index += 1

    await state.update_data(draft_answers=answers, current_question_index=index)

    if index < len(questions):
        await _send_next_question(message, state, prefix="Вопрос")
    else:
        await state.set_state(DocumentDraftStates.generating)
        await message.answer(f"{Emoji.LOADING} Спасибо! Формирую документ…")
        await _finalize_draft(message, state)


async def _send_next_question(message: Message, state: FSMContext, *, prefix: str) -> None:
    data = await state.get_data()
    plan = data.get("draft_plan") or {}
    questions = plan.get("questions") or []
    index = data.get("current_question_index", 0)

    if index >= len(questions):
        return

    question = questions[index]
    purpose = question.get("purpose")
    text = question.get("text", "")
    parts = [f"{Emoji.MAGIC} {prefix} {index + 1}: {text}"]
    if purpose:
        parts.append(f"<i>Цель: {purpose}</i>")
    await message.answer("\n".join(parts), parse_mode=ParseMode.HTML)


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

    try:
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
            await _send_next_question(message, state, prefix="Дополнительный вопрос")
            return

        issues_text = "\n".join(result.issues) or "Модель не смогла подготовить документ."
        await message.answer(f"{Emoji.WARNING} Документ не готов. Причина:\n{issues_text}")
        await state.clear()
        return

    notes: list[str] = []
    if result.validated:
        notes.append("Проверено:\n- " + "\n- ".join(result.validated))
    if result.issues:
        notes.append(f"{Emoji.WARNING} На что обратить внимание:\n- " + "\n- ".join(result.issues))
    if notes:
        await message.answer("\n\n".join(notes))


    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        build_docx_from_markdown(result.markdown, str(tmp_path))
        caption = f"{Emoji.DOCUMENT} {result.title}" if result.title else f"{Emoji.DOCUMENT} Документ"
        await message.answer_document(FSInputFile(str(tmp_path)), caption=caption)
    except DocumentDraftingError as err:
        await message.answer(f"{Emoji.ERROR} Не удалось сформировать DOCX: {err}")
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
        operation_buttons = []
        for op_key, op_info in operations.items():
            emoji = op_info.get("emoji", "📄")
            name = op_info.get("name", op_key)
            operation_buttons.append(
                InlineKeyboardButton(text=f"{emoji} {name}", callback_data=f"doc_operation_{op_key}")
            )

        # Размещаем кнопки по 2 в ряд
        for i in range(0, len(operation_buttons), 2):
            row = operation_buttons[i:i+2]
            buttons.append(row)

        # Запуск конструктора документа отдельной кнопкой
        buttons.append([InlineKeyboardButton(text=f"{Emoji.MAGIC} Создать документ", callback_data="doc_draft_start")])

        # Кнопка "Назад" в отдельном ряду
        buttons.append([InlineKeyboardButton(text="◀️ Назад в меню", callback_data="back_to_menu")])

        message_text = (
            "🗂️ <b>Работа с документами</b>\n\n"
            "Автоматическая обработка и анализ ваших документов с помощью ИИ\n\n"
            "🔹 <b>Что можно делать:</b>\n"
            "• Создавать краткие выжимки больших документов\n"
            "• Находить риски и проблемы в договорах\n"
            "• Задавать вопросы по содержанию файлов\n"
            "• Переводить на другие языки\n"
            "• Обезличивать персональные данные\n"
            "• Распознавать текст со сканов и фото\n\n"
            "👇 <b>Выберите нужную операцию:</b>"
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
        operation = callback.data.replace("doc_operation_", "")
        operation_info = document_manager.get_operation_info(operation)

        if not operation_info:
            await callback.answer("Неизвестная операция")
            return

        # Сохраняем выбранную операцию в состояние
        await state.update_data(document_operation=operation)

        emoji = operation_info.get("emoji", "📄")
        name = operation_info.get("name", operation)
        description = operation_info.get("description", "")
        formats = ", ".join(operation_info.get("formats", []))

        # Создаем подробное описание для каждой операции
        detailed_descriptions = {
            "summarize": (
                "📋 <b>Саммаризация документов</b>\n\n"
                "⚙️ <b>Как это работает:</b>\n"
                "• Анализирует содержание документа\n"
                "• Выделяет ключевые положения и основные идеи\n"
                "• Создает структурированную краткую выжимку\n"
                "• Сохраняет важные детали и цифры\n\n"
                "📊 <b>Результат:</b>\n"
                "• Краткое резюме документа (1-3 страницы)\n"
                "• Основные выводы и рекомендации\n"
                "• Экспорт в форматы DOCX и PDF\n\n"
                "💼 <b>Полезно для:</b>\n"
                "Договоров, отчетов, исследований, технической документации"
            ),
            "analyze_risks": (
                "⚠️ <b>Анализ рисков и проблем</b>\n\n"
                "⚙️ <b>Как это работает:</b>\n"
                "• Сканирует документ на предмет потенциальных рисков\n"
                "• Выявляет проблемные формулировки и условия\n"
                "• Анализирует соответствие нормам права\n"
                "• Оценивает общий уровень риска\n\n"
                "📊 <b>Результат:</b>\n"
                "• Детальный отчет с найденными рисками\n"
                "• Цветовая маркировка по уровню опасности\n"
                "• Рекомендации по устранению проблем\n"
                "• Ссылки на нормативные документы\n\n"
                "💼 <b>Полезно для:</b>\n"
                "Договоров, соглашений, корпоративных документов"
            ),
            "chat": (
                "💬 <b>Интерактивный чат с документом</b>\n\n"
                "⚙️ <b>Как это работает:</b>\n"
                "• Позволяет задавать вопросы по содержанию\n"
                "• Находит релевантные фрагменты текста\n"
                "• Дает развернутые ответы со ссылками\n"
                "• Поддерживает контекст беседы\n\n"
                "📊 <b>Результат:</b>\n"
                "• Точные ответы на ваши вопросы\n"
                "• Цитаты из исходного документа\n"
                "• Возможность уточняющих вопросов\n\n"
                "💼 <b>Полезно для:</b>\n"
                "Изучения сложных документов, поиска конкретной информации"
            ),
            "anonymize": (
                "🔐 <b>Обезличивание документов</b>\n\n"
                "⚙️ <b>Как это работает:</b>\n"
                "• Автоматически находит персональные данные\n"
                "• Заменяет их на безопасные заглушки\n"
                "• Удаляет конфиденциальную информацию\n"
                "• Сохраняет структуру и смысл документа\n\n"
                "📊 <b>Обрабатывает:</b>\n"
                "• ФИО, адреса, телефоны, email\n"
                "• Номера документов и банковские реквизиты\n"
                "• Другие персональные идентификаторы\n\n"
                "💼 <b>Полезно для:</b>\n"
                "Подготовки документов к передаче третьим лицам"
            ),
            "translate": (
                "🌍 <b>Перевод документов</b>\n\n"
                "⚙️ <b>Как это работает:</b>\n"
                "• Переводит текст с сохранением структуры\n"
                "• Учитывает юридическую и техническую терминологию\n"
                "• Сохраняет форматирование и разметку\n"
                "• Поддерживает основные языки мира\n\n"
                "📊 <b>Возможности:</b>\n"
                "• Высокое качество перевода\n"
                "• Специализированная терминология\n"
                "• Экспорт в DOCX и TXT форматы\n\n"
                "💼 <b>Полезно для:</b>\n"
                "Международных договоров, документооборота с зарубежными партнерами"
            ),
            "ocr": (
                "👁️ <b>OCR - распознавание текста</b>\n\n"
                "⚙️ <b>Как это работает:</b>\n"
                "• Извлекает текст из сканированных документов\n"
                "• Распознает текст на изображениях и PDF\n"
                "• Поддерживает рукописный и печатный текст\n"
                "• Восстанавливает структуру документа\n\n"
                "📊 <b>Результат:</b>\n"
                "• Полностью текстовая версия документа\n"
                "• Оценка качества распознавания\n"
                "• Экспорт в различные форматы\n\n"
                "💼 <b>Полезно для:</b>\n"
                "Старых документов, сканов, фотографий документов"
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
        # Очищаем состояние FSM
        await state.clear()

        # Отправляем главное меню
        await cmd_start(callback.message)
        await callback.answer()

    except Exception as e:
        await callback.answer(f"Ошибка: {e}")
        logger.error(f"Ошибка в handle_back_to_menu: {e}", exc_info=True)

# --- progress router hookup ---
def register_progressbar(dp: Dispatcher) -> None:
    dp.include_router(progress_router)


async def handle_document_upload(message: Message, state: FSMContext):
    """Обработка загружённого документа"""
    try:
        if not message.document:
            await message.answer("❌ Ошибка: документ не найден")
            return

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

        status_msg = await message.answer(
            f"📄 Обрабатываем документ <b>{html_escape(file_name)}</b>...\n\n"
            f"⏳ Операция: {html_escape(operation_name)}\n"
            f"📊 Размер: {file_size // 1024} КБ",
            parse_mode=ParseMode.HTML,
        )

        try:
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

            try:
                result = await document_manager.process_document(
                    user_id=message.from_user.id,
                    file_content=file_bytes,
                    original_name=file_name,
                    mime_type=mime_type,
                    operation=operation,
                    **options,
                )
            finally:
                with suppress(Exception):
                    stored_path.unlink(missing_ok=True)

            # Удаляем статусное сообщение
            try:
                await status_msg.delete()
            except:
                pass

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

                logger.info(
                    f"Successfully processed document {file_name} for user {message.from_user.id}"
                )
            else:
                reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                await message.answer(
                    f"{Emoji.ERROR} <b>Ошибка обработки документа</b>\n\n{html_escape(str(result.message))}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )

        except Exception as e:
            # Удаляем статусное сообщение в случае ошибки
            try:
                await status_msg.delete()
            except:
                pass

            reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
            await message.answer(
                f"{Emoji.ERROR} <b>Ошибка обработки документа</b>\n\n{html_escape(str(e))}",
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
            f"{Emoji.ERROR} <b>Произошла ошибка</b>\n\n{html_escape(str(e))}",
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
        logger.error(f"Error in handle_document_upload: {e}", exc_info=True)
        await state.clear()


async def handle_photo_upload(message: Message, state: FSMContext):
    """Обработка загруженной фотографии для OCR"""
    try:
        if not message.photo:
            await message.answer("❌ Ошибка: фотография не найдена")
            return

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

        status_msg = await message.answer(
            f"📷 Обрабатываем фотографию для OCR...\n\n"
            f"⏳ Операция: {html_escape(operation_name)}\n"
            f"📏 Размер: {file_size // 1024} КБ",
            parse_mode=ParseMode.HTML,
        )

        try:
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

            try:
                result = await document_manager.process_document(
                    user_id=message.from_user.id,
                    file_content=file_bytes,
                    original_name=file_name,
                    mime_type=mime_type,
                    operation=operation,
                    **options,
                )
            finally:
                with suppress(Exception):
                    stored_path.unlink(missing_ok=True)

            # Удаляем статусное сообщение
            try:
                await status_msg.delete()
            except:
                pass

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

                logger.info(
                    f"Successfully processed photo {file_name} for user {message.from_user.id}"
                )
            else:
                reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
                await message.answer(
                    f"{Emoji.ERROR} <b>Ошибка обработки фотографии</b>\n\n{html_escape(str(result.message))}",
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )

        except Exception as e:
            # Удаляем статусное сообщение
            try:
                await status_msg.delete()
            except:
                pass

            reply_markup = _build_ocr_reply_markup(output_format) if operation == "ocr" else None
            await message.answer(
                f"{Emoji.ERROR} <b>Ошибка обработки фотографии</b>\n\n{html_escape(str(e))}",
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup,
            )
            logger.error(f"Error processing photo {file_name}: {e}", exc_info=True)

        finally:
            # Очищаем состояние
            await state.clear()

    except Exception as e:
        await message.answer(f"❌ Произошла ошибка: {str(e)}")
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
        )
        ctx.audio_service = audio_service
        logger.info(
            "Voice mode enabled (stt=%s, tts=%s, voice=%s, format=%s)",
            cfg.voice_stt_model,
            cfg.voice_tts_model,
            cfg.voice_tts_voice,
            cfg.voice_tts_format,
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

    refresh_runtime_globals()

    # Команды
    await bot.set_my_commands(
        [
            BotCommand(command="start", description=f"{Emoji.ROBOT} Начать работу"),
            BotCommand(command="buy", description=f"{Emoji.MAGIC} Купить подписку"),
            BotCommand(command="status", description=f"{Emoji.STATS} Статус подписки"),
            BotCommand(command="mystats", description="📊 Моя статистика"),
            BotCommand(command="ratings", description="📈 Статистика рейтингов (админ)"),
            BotCommand(command="errors", description="🚨 Статистика ошибок (админ)"),
        ]
    )

    # Роутинг
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_buy, Command("buy"))
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(cmd_mystats, Command("mystats"))
    dp.message.register(cmd_ratings_stats, Command("ratings"))
    dp.message.register(cmd_error_stats, Command("errors"))

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
    dp.callback_query.register(handle_subscription_status_callback, F.data == "subscription_status")
    dp.callback_query.register(handle_get_subscription_callback, F.data == "get_subscription")
    dp.callback_query.register(handle_payment_history_callback, F.data == "payment_history")
    dp.callback_query.register(handle_referral_program_callback, F.data == "referral_program")
    dp.callback_query.register(handle_copy_referral_callback, F.data.startswith("copy_referral_"))
    dp.callback_query.register(handle_back_to_main_callback, F.data == "back_to_main")

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
        handle_doc_draft_answer, DocumentDraftStates.asking_details, F.text
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


