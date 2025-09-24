"""
Простая версия Telegram бота ИИ-Иван
Только /start и обработка вопросов, никаких кнопок и лишних команд
"""

from __future__ import annotations
import asyncio
import os
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Union, TYPE_CHECKING

from src.documents.document_manager import DocumentManager

if TYPE_CHECKING:
    from src.core.db_advanced import DatabaseAdvanced

from html import escape as html_escape
import re

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.types import (
    Message,
    BotCommand,
    ErrorEvent,
    LabeledPrice,
    PreCheckoutQuery,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    Document,
    ContentType,
    FSInputFile
)
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from src.bot.logging_setup import setup_logging
from src.bot.promt import LEGAL_SYSTEM_PROMPT, JUDICIAL_PRACTICE_SEARCH_PROMPT
from src.bot.ui_components import Emoji, escape_markdown_v2
from src.bot.stream_manager import StreamManager, StreamingCallback
from src.bot.status_manager import AnimatedStatus, ProgressStatus, ResponseTimer, TypingContext
from src.core.db import Database
from src.telegram_legal_bot.config import load_config
from src.telegram_legal_bot.ratelimit import RateLimiter
from src.core.access import AccessService
from src.core.openai_service import OpenAIService
from src.core.session_store import SessionStore, UserSession
from src.core.payments import CryptoPayProvider, convert_rub_to_xtr
from src.core.validation import InputValidator, ValidationSeverity
from src.core.exceptions import (
    ErrorHandler,
    ErrorContext,
    ErrorType,
    ValidationException,
    DatabaseException,
    OpenAIException,
    TelegramException,
    NetworkException,
    PaymentException,
    AuthException,
    RateLimitException,
    SystemException,
)
from src.documents.base import ProcessingError

SAFE_LIMIT = 3900  # чуть меньше телеграмного 4096 (запас на теги)
# ============ КОНФИГУРАЦИЯ ============

load_dotenv()
setup_logging()
logger = logging.getLogger("ai-ivan.simple")

config = load_config()
BOT_TOKEN = config.telegram_bot_token
USE_ANIMATION = config.use_status_animation
USE_STREAMING = os.getenv("USE_STREAMING", "1").lower() in ("1", "true", "yes", "on")
MAX_MESSAGE_LENGTH = 4000

# Подписки и платежи
DB_PATH = config.db_path
TRIAL_REQUESTS = config.trial_requests
SUB_DURATION_DAYS = config.sub_duration_days

# RUB платеж через Telegram Payments (провайдер-эквайринг)
RUB_PROVIDER_TOKEN = config.telegram_provider_token_rub
SUB_PRICE_RUB = config.subscription_price_rub  # руб.
SUB_PRICE_RUB_KOPEKS = SUB_PRICE_RUB * 100

# Telegram Stars (XTR)
STARS_PROVIDER_TOKEN = config.telegram_provider_token_stars
SUB_PRICE_XTR = config.subscription_price_xtr  # XTR
# Динамическая цена в XTR, рассчитанная на старте по курсу (если задан RUB_PER_XTR)
DYNAMIC_PRICE_XTR = convert_rub_to_xtr(
    amount_rub=float(SUB_PRICE_RUB),
    rub_per_xtr=getattr(config, "rub_per_xtr", None),
    default_xtr=SUB_PRICE_XTR,
)

# Админы
ADMIN_IDS = set(config.admin_ids)

# Глобальная БД/лимитер
db: Optional[Union[Database, DatabaseAdvanced]] = None
rate_limiter: Optional[RateLimiter] = None
access_service: Optional[AccessService] = None
openai_service: Optional[OpenAIService] = None
session_store: Optional[SessionStore] = None
crypto_provider: Optional[CryptoPayProvider] = None
error_handler: Optional[ErrorHandler] = None
document_manager: Optional[Any] = None  # DocumentManager будет инициализирован позже

# Политика сессий
USER_SESSIONS_MAX = int(getattr(config, "user_sessions_max", 10000) or 10000)
USER_SESSION_TTL_SECONDS = int(getattr(config, "user_session_ttl_seconds", 3600) or 3600)

# ============ СОСТОЯНИЯ ДЛЯ РАБОТЫ С ДОКУМЕНТАМИ ============

class DocumentProcessingStates(StatesGroup):
    waiting_for_document = State()
    processing_document = State()

# ============ УПРАВЛЕНИЕ СОСТОЯНИЕМ ============


def get_user_session(user_id: int) -> UserSession:
    if session_store is None:
        raise RuntimeError("Session store not initialized")
    return session_store.get_or_create(user_id)


# ============ УТИЛИТЫ ============


def chunk_text(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Разбивает текст на части для отправки в Telegram"""
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""

    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        if len(current_chunk + paragraph + "\n\n") <= max_length:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
            else:
                # Параграф слишком длинный, разбиваем принудительно
                while len(paragraph) > max_length:
                    chunks.append(paragraph[:max_length])
                    paragraph = paragraph[max_length:]
                current_chunk = paragraph + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def _md_links_to_anchors(line: str) -> str:
    """Convert markdown links [text](url) into safe HTML anchors.

    Both link text and URL are escaped; only http/https URLs are allowed.
    """
    pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
    result_parts: list[str] = []
    last = 0
    for m in pattern.finditer(line):
        # escape non-link part
        result_parts.append(html_escape(line[last : m.start()]))
        text = html_escape(m.group(1))
        url = html_escape(m.group(2), quote=True)
        result_parts.append(f'<a href="{url}">{text}</a>')
        last = m.end()
    # tail
    result_parts.append(html_escape(line[last:]))
    return "".join(result_parts)


def sanitize_telegram_html(raw: str) -> str:
    """Allow only Telegram-supported HTML tags; escape the rest.

    Allowed: b, i, u, s, code, pre, a[href=http/https], br
    """
    if not raw:
        return ""
    # Start from fully escaped text
    esc = html_escape(raw, quote=True)
    # Restore <br>, <br/>, <br />
    esc = re.sub(r"&lt;br\s*/?&gt;", "<br>", esc, flags=re.IGNORECASE)
    # Restore simple tags exactly
    for tag in ("b", "i", "u", "s", "code", "pre"):
        esc = re.sub(rf"&lt;{tag}&gt;", rf"<{tag}>", esc, flags=re.IGNORECASE)
        esc = re.sub(rf"&lt;/{tag}&gt;", rf"</{tag}>", esc, flags=re.IGNORECASE)
    # Restore anchors with http(s) only; keep entities like &amp; inside href
    esc = re.sub(
        r"&lt;a href=&quot;(https?://[^&quot;]+)&quot;&gt;",
        r'<a href="\1">',
        esc,
        flags=re.IGNORECASE,
    )
    esc = re.sub(r"&lt;/a&gt;", "</a>", esc, flags=re.IGNORECASE)
    return esc


def render_legal_html(raw: str) -> str:
    """Beautify plain model text into simple, safe HTML.

    - Escapes HTML by default
    - Converts [text](url) markdown links to <a>
    - Bolds headings (lines ending with ':' or starting with 'N) ' or 'TL;DR')
    - Normalizes bullets (leading '-', '—', '•') to an em dash '— '
    - Replaces newlines with <br>
    """
    if not raw:
        return ""

    # If looks like HTML from the model, sanitize and keep structure
    if "<" in raw and re.search(r"<\s*(b|i|u|s|code|pre|a|br)\b", raw, re.IGNORECASE):
        return sanitize_telegram_html(raw)

    def _auto_paragraph_breaks(text: str) -> str:
        # Normalize spaces but preserve intentional structure
        t = re.sub(r"[ \t]+", " ", text)  # Only normalize spaces/tabs, keep newlines

        # Insert breaks before numbered items like "1) ", "2) ", "1.", "2."
        t = re.sub(r"(?<!\n)(?=\b\d+[\.)]\s)", "\n\n", t)

        # Insert breaks before section markers
        t = re.sub(r"(?<!\n)(?=\b(?:Коротко|Далее|Вариант|Итак|Резюме|Заключение)\b)", "\n\n", t)

        # Break after sentence end before em dash bullets or numbers
        t = re.sub(r"(?<=[\.!?])\s+(?=(?:—|•|-|\d+[\.)]\s))", "\n", t)

        # Insert breaks before article references like "ст. 304", "Статья 222"
        t = re.sub(r"(?<=[\.!?])\s+(?=(?:—\s*)?(?:ст\.|Статья)\s*\d+)", "\n", t)

        # Break long sentences with semicolons into separate lines
        t = re.sub(r";\s+(?=и\s+\d+\))", ";\n— ", t)

        return t

    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Always apply auto paragraph breaks for better structure
    text = _auto_paragraph_breaks(text)

    lines = text.split("\n")
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped == "":
            out.append("<br>")
            continue

        # Enhanced bullet detection
        if re.match(r"^\s*[-•—]\s+", line):
            line = re.sub(r"^\s*[-•—]\s+", "— ", line)

        # Transform md links and escape other parts FIRST
        html_line = _md_links_to_anchors(line)

        # Numbered lists with proper formatting AFTER escaping
        if re.match(r"^\s*\d+[\.)]\s+", stripped):
            html_line = re.sub(r"(\d+[\.)]\s+)", r"<b>\1</b>", html_line)

        # Enhanced heading detection
        is_heading = (
            stripped.endswith(":")
            or stripped.upper().startswith(("КОРОТКО", "TL;DR", "РЕЗЮМЕ", "ЗАКЛЮЧЕНИЕ"))
            or re.match(r"^\s*\d+\.\s+[А-ЯA-Z]", stripped) is not None  # "1. Какие статьи"
        )

        # Special formatting for article references AFTER escaping
        if re.search(r"\b(?:ст\.|Статья)\s*\d+", stripped):
            html_line = re.sub(r"(\b(?:ст\.|Статья)\s*\d+[^\s]*)", r"<b>\1</b>", html_line)

        if is_heading:
            html_line = f"<b>{html_line}</b>"
            out.append(html_line + "<br><br>")
        else:
            out.append(html_line + "<br>")

    # Improved br collapse - better paragraph separation
    html_result = "".join(out)
    html_result = re.sub(r"(?:<br>\s*){4,}", "<br><br><br>", html_result)  # Max 3 <br> tags
    html_result = re.sub(
        r"(?:<br>\s*){3,}", "<br><br>", html_result
    )  # Usually 2 <br> for paragraphs

    return html_result


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


async def _send_html_chunks(message, html_text: str) -> None:
    """Отправляем длинный HTML несколькими сообщениями, без превью ссылок."""
    parts = _split_html_safely(html_text, SAFE_LIMIT)
    logger.info(f"Sending {len(parts)} HTML chunks to user {message.from_user.id}")
    for i, chunk in enumerate(parts):
        logger.debug(f"Chunk {i+1}: {chunk[:100]}...")
        try:
            await message.answer(chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            logger.debug(f"Successfully sent chunk {i+1}")
        except Exception as e:
            logger.warning(f"Error sending chunk {i+1} to user {message.from_user.id}: {e}")
            # если вдруг разорвали тег — санитайз и повтор
            try:
                from html import escape as _esc

                # грубая санация: экранируем всё и восстанавливаем допустимые теги
                safe = _esc(chunk, quote=True)
                # Восстанавливаем все нужные HTML теги
                safe = re.sub(r"&lt;br\s*/?&gt;", "<br>", safe, flags=re.IGNORECASE)
                safe = re.sub(r"&lt;b&gt;", "<b>", safe, flags=re.IGNORECASE)
                safe = re.sub(r"&lt;/b&gt;", "</b>", safe, flags=re.IGNORECASE)
                safe = re.sub(r"&lt;i&gt;", "<i>", safe, flags=re.IGNORECASE)
                safe = re.sub(r"&lt;/i&gt;", "</i>", safe, flags=re.IGNORECASE)
                safe = re.sub(r"&lt;code&gt;", "<code>", safe, flags=re.IGNORECASE)
                safe = re.sub(r"&lt;/code&gt;", "</code>", safe, flags=re.IGNORECASE)
                await message.answer(safe, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            except Exception:
                # финальный фоллбек — голый текст
                await message.answer(re.sub(r"<[^>]+>", "", chunk))
        if i < len(parts) - 1:
            await asyncio.sleep(0.1)


async def _validate_question_or_reply(message: Message, text: str, user_id: int) -> Optional[str]:
    result = InputValidator.validate_question(text, user_id)
    if not result.is_valid:
        bullet = "\n\u0007 "
        error_msg = bullet.join(result.errors)
        if result.severity == ValidationSeverity.CRITICAL:
            await message.answer(
                f"{Emoji.ERROR} <b>Критическая ошибка валидации</b>\n\n\u0007 {error_msg}\n\n<i>Попробуйте переформулировать запрос</i>",
                parse_mode=ParseMode.HTML,
            )
        else:
            await message.answer(
                f"{Emoji.WARNING} <b>Ошибка в запросе</b>\n\n\u0007 {error_msg}",
                parse_mode=ParseMode.HTML,
            )
        return None

    if result.warnings:
        bullet = "\n\u0007 "
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


async def _start_status_indicator(message: Message):
    if not message.bot:
        return None

    if USE_ANIMATION:
        status = AnimatedStatus(message.bot, message.chat.id)
        await status.start()
        return status
    status = ProgressStatus(message.bot, message.chat.id)
    await status.start("Обрабатываю ваш запрос...")  # уже экранировано для HTML
    return status


async def _stop_status_indicator(status) -> None:
    if status is None:
        return
    try:
        if hasattr(status, "complete"):
            await status.complete()
        else:
            await status.stop()
    except Exception:
        pass


# ============ ФУНКЦИИ РЕЙТИНГА И UI ============


def create_rating_keyboard(request_id: int) -> InlineKeyboardMarkup:
    """Создает клавиатуру с кнопками рейтинга для оценки ответа"""
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="👍", callback_data=f"rate_like_{request_id}"),
            InlineKeyboardButton(text="👎", callback_data=f"rate_dislike_{request_id}")
        ]
    ])


async def send_rating_request(message: Message, request_id: int):
    """Отправляет сообщение с запросом на оценку ответа"""
    try:
        rating_keyboard = create_rating_keyboard(request_id)
        await message.answer(
            f"{Emoji.STAR} <b>Оцените качество ответа</b>\n\n"
            "Ваша оценка поможет нам улучшить сервис!",
            parse_mode=ParseMode.HTML,
            reply_markup=rating_keyboard
        )
    except Exception as e:
        logger.error(f"Failed to send rating request: {e}")
        # Не критично, если не удалось отправить запрос на рейтинг


# ============ КОМАНДЫ ============


async def cmd_start(message: Message):
    """Единственная команда - приветствие"""
    if not message.from_user:
        return

    user_session = get_user_session(message.from_user.id)  # noqa: F841 (инициализация)
    # Обеспечим запись в БД
    if db is not None and hasattr(db, "ensure_user"):
        await db.ensure_user(
            message.from_user.id,
            default_trial=TRIAL_REQUESTS,
            is_admin=message.from_user.id in ADMIN_IDS,
        )
    user_name = message.from_user.first_name or "Пользователь"

    # Подробное приветствие
    welcome_raw = f"""
╔═══════════════════════════╗
║  ⚖️ ИИ-Иван ⚖️  ║
╚═══════════════════════════╝

🤖 Привет, {user_name}! Добро пожаловать!

⭐️ Ваш персональный юридический ассистент
━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ Что я умею:
🔍 Анализирую судебную практику РФ
📋 Ищу релевантные дела и решения
💡 Готовлю черновики процессуальных документов  
⚖️ Оцениваю правовые риски и перспективы

🔥 Специализации:
🏠 Гражданское и договорное право
🏢 Корпоративное право и M&A
👨‍💼 Трудовое и административное право
💰 Налоговое право и споры с ФНС

━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Примеры вопросов:

📝 "Можно ли расторгнуть договор поставки за просрочку?"
👨‍💼 "Как правильно уволить сотрудника за нарушения?"
💰 "Какие риски при доначислении НДС?"
🏢 "Порядок увеличения уставного капитала ООО"

━━━━━━━━━━━━━━━━━━━━━━━━━━

🔥 Готов к работе! Отправьте ваш правовой вопрос или выберите действие ниже:
"""
    # Здесь избыточное экранирование не нужно — используем MarkdownV2 c вашим helper'ом
    welcome_text = escape_markdown_v2(welcome_raw)

    # Создаем inline клавиатуру с кнопками (компактное размещение)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🔍 Поиск судебной практики", callback_data="search_practice"),
            InlineKeyboardButton(text="📋 Консультация", callback_data="general_consultation")
        ],
        [
            InlineKeyboardButton(text="📄 Подготовка документов", callback_data="prepare_documents"),
            InlineKeyboardButton(text="🗂️ Работа с документами", callback_data="document_processing")
        ],
        [
            InlineKeyboardButton(text="ℹ️ Помощь", callback_data="help_info")
        ]
    ])

    await message.answer(welcome_text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=keyboard)
    logger.info("User %s started bot", message.from_user.id)


# ============ ОБРАБОТКА ВОПРОСОВ ============


async def process_question(message: Message):
    """Главный обработчик юридических вопросов"""
    if not message.from_user:
        return

    user_id = message.from_user.id
    chat_id = message.chat.id
    message_id = message.message_id

    # Создаем контекст для обработки ошибок
    error_context = ErrorContext(
        user_id=user_id, chat_id=chat_id, message_id=message_id, function_name="process_question"
    )

    user_session = get_user_session(user_id)
    question_text = (message.text or "").strip()
    quota_msg_to_send: Optional[str] = None

    # Проверяем, не ждем ли мы комментарий для рейтинга
    if not hasattr(user_session, "pending_feedback_request_id"):
        user_session.pending_feedback_request_id = None

    if user_session.pending_feedback_request_id is not None:
        await handle_pending_feedback(message, user_session)
        return
    quota_is_trial: bool = False

    # Проверяем, что это не команда
    if question_text.startswith("/"):
        return

    # ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
    if error_handler is None:
        raise SystemException("Error handler not initialized", error_context)
    cleaned = await _validate_question_or_reply(message, question_text, user_id)
    if not cleaned:
        return
    question_text = cleaned

    # Timer
    timer = ResponseTimer()
    timer.start()

    logger.info("Processing question from user %s: %s", user_id, question_text[:100])

    try:
        # Global rate limit per user
        if not await _rate_limit_guard(user_id, message):
            return

        # Контроль доступа через сервис доступа (ООП)
        quota_text = ""
        if access_service is not None:
            decision = await access_service.check_and_consume(user_id)
            if not decision.allowed:
                await message.answer(
                    f"{Emoji.WARNING} <b>Лимит бесплатных запросов исчерпан</b>\n\nВы использовали {TRIAL_REQUESTS} из {TRIAL_REQUESTS}. Оформите подписку за {SUB_PRICE_RUB}₽ в месяц командой /buy",
                    parse_mode=ParseMode.HTML,
                )
                return
            if decision.is_admin:
                quota_text = f"\n\n{Emoji.STATS} <b>Статус: безлимитный доступ</b>"
            elif decision.has_subscription and decision.subscription_until:
                until_dt = datetime.fromtimestamp(decision.subscription_until)
                quota_text = f"\n\n{Emoji.CALENDAR} <b>Подписка активна до:</b> {until_dt:%Y-%m-%d}"
            elif decision.trial_used is not None and decision.trial_remaining is not None:
                quota_is_trial = True
                quota_msg_core = html_escape(
                    f"Бесплатные запросы: {decision.trial_used}/{TRIAL_REQUESTS}. Осталось: {decision.trial_remaining}"
                )
                quota_msg_to_send = f"{Emoji.STATS} <b>{quota_msg_core}</b>"

        # Показываем статус
        status = await _start_status_indicator(message)

        try:
            # Имитация этапов
            if not USE_ANIMATION and hasattr(status, "update_stage"):
                await asyncio.sleep(0.5)
                await status.update_stage(1, f"{Emoji.SEARCH} Анализирую ваш вопрос...")
                await asyncio.sleep(1)
                await status.update_stage(
                    2, f"{Emoji.LOADING} Ищу релевантную судебную практику..."
                )

            # Основной запрос к ИИ
            if openai_service is None:
                raise SystemException("OpenAI service not initialized", error_context)

            request_start_time = time.time()
            stream_manager = None

            # Выбираем промпт в зависимости от режима пользователя
            selected_prompt = LEGAL_SYSTEM_PROMPT
            if hasattr(user_session, "practice_search_mode") and user_session.practice_search_mode:
                selected_prompt = JUDICIAL_PRACTICE_SEARCH_PROMPT
                # Сбрасываем режим после использования
                user_session.practice_search_mode = False

            try:
                if USE_STREAMING and message.bot:
                    # Streaming режим
                    stream_manager = StreamManager(
                        bot=message.bot,
                        chat_id=message.chat.id,
                        update_interval=1.5,
                        buffer_size=100,
                    )

                    # Запускаем streaming
                    await stream_manager.start_streaming(f"{Emoji.ROBOT} Обдумываю ваш вопрос...")

                    # Создаем callback
                    callback = StreamingCallback(stream_manager)

                    # Выполняем streaming запрос
                    result = await openai_service.ask_legal_stream(
                        selected_prompt, question_text, callback=callback
                    )
                else:
                    # Обычный режим
                    if message.bot:
                        async with TypingContext(message.bot, message.chat.id):
                            result = await openai_service.ask_legal(
                                selected_prompt, question_text
                            )
                    else:
                        result = await openai_service.ask_legal(selected_prompt, question_text)

                request_error_type = None
            except Exception as e:
                request_error_type = type(e).__name__

                # Останавливаем streaming в случае ошибки
                if stream_manager:
                    try:
                        await stream_manager.stop()
                    except:
                        pass

                # Специфичная обработка ошибок OpenAI
                if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                    raise OpenAIException(str(e), error_context, is_quota_error=True)
                elif "timeout" in str(e).lower() or "network" in str(e).lower():
                    raise NetworkException(f"OpenAI network error: {str(e)}", error_context)
                else:
                    raise OpenAIException(f"OpenAI API error: {str(e)}", error_context)

            if not USE_STREAMING and not USE_ANIMATION and hasattr(status, "update_stage"):
                await status.update_stage(
                    3, f"{Emoji.DOCUMENT} Формирую структурированный ответ..."
                )
                await asyncio.sleep(0.4)
                await status.update_stage(4, f"{Emoji.MAGIC} Финализирую рекомендации...")

        finally:
            await _stop_status_indicator(status)

        timer.stop()

        # Обрабатываем результат
        if not result.get("ok"):
            error_text = result.get("error", "Неизвестная ошибка")
            logger.error("OpenAI error for user %s: %s", user_id, error_text)

            # Для streaming показываем ошибку в том же сообщении
            if USE_STREAMING and stream_manager:
                await stream_manager.finalize(
                    f"""{Emoji.ERROR} <b>Произошла ошибка</b>

Не удалось получить ответ. Попробуйте ещё раз чуть позже.

{Emoji.HELP} <i>Подсказка</i>: Проверьте формулировку вопроса

<code>{html_escape(error_text[:300])}</code>"""
                )
            else:
                await message.answer(
                    f"""{Emoji.ERROR} <b>Произошла ошибка</b>

Не удалось получить ответ. Попробуйте ещё раз чуть позже.

{Emoji.HELP} <i>Подсказка</i>: Проверьте формулировку вопроса

<code>{html_escape(error_text[:300])}</code>""",
                    parse_mode=ParseMode.HTML,
                )
            return

        # Для streaming ответ уже отправлен через callback
        if not USE_STREAMING:
            # Форматируем ответ для HTML
            response_text = render_legal_html(result.get("text", ""))

            # Отправляем основной текст чанками
            await _send_html_chunks(message, response_text)

        # Информация о времени ответа — отдельным сообщением
        time_info = f"{Emoji.CLOCK} <i>Время ответа: {timer.get_duration_text()}</i>"
        try:
            await message.answer(time_info, parse_mode=ParseMode.HTML)
        except Exception:
            await message.answer(time_info)

        # if non-trial quota footer is present, send it separately
        if "quota_text" in locals() and quota_text and not quota_is_trial:
            try:
                await message.answer(quota_text, parse_mode=ParseMode.HTML)
            except Exception:
                await message.answer(quota_text)

        if quota_msg_to_send:
            try:
                await message.answer(quota_msg_to_send, parse_mode=ParseMode.HTML)
            except Exception:
                await message.answer(quota_msg_to_send)

        # Обновляем статистику в сессии
        user_session.add_question_stats(timer.duration)

        # Записываем статистику в базу данных (если это продвинутая версия БД)
        request_id = None
        if db is not None and hasattr(db, "record_request") and "request_start_time" in locals():
            try:
                request_time_ms = int((time.time() - request_start_time) * 1000)
                request_id = await db.record_request(
                    user_id=user_id,
                    request_type="legal_question",
                    tokens_used=0,  # Пока не подсчитываем токены
                    response_time_ms=request_time_ms,
                    success=result.get("ok", False),
                    error_type=None if result.get("ok", False) else "openai_error",
                )
                logger.debug(f"Recorded request with ID: {request_id}")
            except Exception as db_error:
                logger.warning("Failed to record request statistics: %s", db_error)

        # Отправляем запрос на оценку ответа (только если запрос был успешным)
        if request_id is not None and result.get("ok", False):
            await send_rating_request(message, request_id)

        logger.info("Successfully processed question for user %s in %.2fs", user_id, timer.duration)

    except Exception as e:
        # Обрабатываем все исключения через централизованный обработчик
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

        # Записываем статистику неудачного запроса (если это продвинутая версия БД)
        if hasattr(db, "record_request"):
            try:
                request_time_ms = (
                    int((time.time() - request_start_time) * 1000)
                    if "request_start_time" in locals()
                    else 0
                )
                error_type = (
                    request_error_type if "request_error_type" in locals() else type(e).__name__
                )
                await db.record_request(
                    user_id=user_id,
                    request_type="legal_question",
                    tokens_used=0,
                    response_time_ms=request_time_ms,
                    success=False,
                    error_type=str(error_type),
                )
            except Exception as db_error:
                logger.warning("Failed to record failed request statistics: %s", db_error)

        # Попробуем отправить пользователю понятное сообщение об ошибке
        try:
            await message.answer(
                f"❌ <b>Ошибка обработки запроса</b>\n\n"
                f"{user_message}\n\n"
                f"💡 <b>Рекомендации:</b>\n"
                f"• Переформулируйте вопрос\n"
                f"• Попробуйте через несколько минут\n"
                f"• Обратитесь в поддержку, если проблема повторяется",
                parse_mode=ParseMode.HTML,
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message to user {user_id}: {send_error}")
            try:
                await message.answer("Произошла ошибка. Попробуйте позже.")
            except Exception:
                pass  # Уже ничего не сделать


# ============ ПОДПИСКИ И ПЛАТЕЖИ ============


def _build_payload(method: str, user_id: int) -> str:
    return f"sub:{method}:{user_id}:{int(datetime.now().timestamp())}"


async def send_rub_invoice(message: Message):
    if not message.from_user or not message.bot:
        return

    if not RUB_PROVIDER_TOKEN:
        await message.answer(
            f"{Emoji.WARNING} Оплата картами временно недоступна. Попробуйте Telegram Stars или криптовалюту (/buy)",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return
    prices = [LabeledPrice(label="Подписка на 30 дней", amount=SUB_PRICE_RUB_KOPEKS)]
    payload = _build_payload("rub", message.from_user.id)
    await message.bot.send_invoice(
        chat_id=message.chat.id,
        title="Месячная подписка",
        description="Доступ к ИИ-Иван: анализ практики, документы, рекомендации. Срок: 30 дней.",
        payload=payload,
        provider_token=RUB_PROVIDER_TOKEN,
        currency="RUB",
        prices=prices,
        is_flexible=False,
    )


async def send_stars_invoice(message: Message):
    if not message.from_user or not message.bot:
        return

    if not STARS_PROVIDER_TOKEN:
        raise RuntimeError("Telegram Stars provider token is not configured")
    dynamic_xtr = convert_rub_to_xtr(
        amount_rub=float(SUB_PRICE_RUB),
        rub_per_xtr=getattr(config, "rub_per_xtr", None),
        default_xtr=SUB_PRICE_XTR,
    )
    prices = [LabeledPrice(label="Подписка на 30 дней", amount=dynamic_xtr)]
    payload = _build_payload("xtr", message.from_user.id)
    await message.bot.send_invoice(
        chat_id=message.chat.id,
        title="Месячная подписка (Telegram Stars)",
        description="Оплата в Telegram Stars (XTR). Срок подписки: 30 дней.",
        payload=payload,
        provider_token=STARS_PROVIDER_TOKEN,
        currency="XTR",
        prices=prices,
        is_flexible=False,
    )


async def cmd_buy(message: Message):
    dynamic_xtr = convert_rub_to_xtr(
        amount_rub=float(SUB_PRICE_RUB),
        rub_per_xtr=getattr(config, "rub_per_xtr", None),
        default_xtr=SUB_PRICE_XTR,
    )
    text = (
        f"{Emoji.MAGIC} **Оплата подписки**\n\n"
        f"Стоимость: {SUB_PRICE_RUB} ₽ \\({dynamic_xtr} ⭐\\) за 30 дней\n\n"
        f"Выберите способ оплаты:"
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN_V2)

    # Банковские карты (если настроен токен)
    if RUB_PROVIDER_TOKEN:
        await send_rub_invoice(message)

    # Telegram Stars
    try:
        await send_stars_invoice(message)
    except Exception as e:
        logger.warning("Failed to send stars invoice: %s", e)
        await message.answer(
            f"{Emoji.WARNING} Telegram Stars временно недоступны. Попробуйте другой способ оплаты.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    # Крипта: инвойс через CryptoBot
    payload = _build_payload("crypto", message.from_user.id)
    if crypto_provider is None:
        logger.warning("Crypto provider not initialized; skipping crypto invoice")
        await message.answer(
            f"{Emoji.IDEA} Криптовалюта: временно недоступна (настройте CRYPTO_PAY_TOKEN)",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    try:
        inv = await crypto_provider.create_invoice(
            amount_rub=float(SUB_PRICE_RUB),
            description="Подписка ИИ-Иван на 30 дней",
            payload=payload,
        )
        if inv.get("ok") and "url" in inv:
            await message.answer(
                f"{Emoji.DOWNLOAD} Оплата криптовалютой: перейдите по ссылке\n{inv['url']}",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        else:
            await message.answer(
                f"{Emoji.IDEA} Криптовалюта: временно недоступна (настройте CRYPTO_PAY_TOKEN)",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
    except Exception as e:
        logger.warning("Crypto invoice failed: %s", e)
        await message.answer(
            f"{Emoji.IDEA} Криптовалюта: временно недоступна",
            parse_mode=ParseMode.MARKDOWN_V2,
        )


async def cmd_status(message: Message):
    if db is None:
        await message.answer("Статус временно недоступен")
        return

    user = await db.ensure_user(
        message.from_user.id,
        default_trial=TRIAL_REQUESTS,
        is_admin=message.from_user.id in ADMIN_IDS,
    )
    until = getattr(user, "subscription_until", 0)
    if until and until > 0:
        until_dt = datetime.fromtimestamp(until)
        left_days = max(0, (until_dt - datetime.now()).days)
        sub_text = f"Активна до {until_dt:%Y-%m-%d} (≈{left_days} дн.)"
    else:
        sub_text = "Не активна"

    await message.answer(
        f"{Emoji.STATS} <b>Статус</b>\n\n"
        f"ID: <code>{message.from_user.id}</code>\n"
        f"Роль: {'админ' if getattr(user, 'is_admin', False) else 'пользователь'}\n"
        f"Триал: {getattr(user, 'trial_remaining', 0)} запрос(ов)\n"
        f"Подписка: {sub_text}",
        parse_mode=ParseMode.HTML,
    )


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

        # Форматируем даты
        def format_timestamp(ts):
            if not ts or ts == 0:
                return "Никогда"
            return datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")

        def format_subscription_status(until_ts):
            if not until_ts or until_ts == 0:
                return "❌ Не активна"
            until_dt = datetime.fromtimestamp(until_ts)
            if until_dt < datetime.now():
                return "❌ Истекла"
            days_left = (until_dt - datetime.now()).days
            return f"✅ До {until_dt.strftime('%d.%m.%Y')} ({days_left} дн.)"

        status_text = f"""📊 <b>Моя статистика</b>

👤 <b>Профиль</b>
• ID: <code>{user_id}</code>
• Статус: {'👑 Администратор' if stats.get('is_admin') else '👤 Пользователь'}
• Регистрация: {format_timestamp(getattr(user, 'created_at', 0))}

💰 <b>Баланс и доступ</b>
• Пробные запросы: {stats.get('trial_remaining', 0)} из {TRIAL_REQUESTS}
• Подписка: {format_subscription_status(stats.get('subscription_until', 0))}

📈 <b>Общая статистика</b>
• Всего запросов: {stats.get('total_requests', 0)}
• Успешных: {stats.get('successful_requests', 0)} ✅
• Неудачных: {stats.get('failed_requests', 0)} ❌
• Последний запрос: {format_timestamp(stats.get('last_request_at', 0))}

📅 <b>За последние 30 дней</b>
• Запросов: {stats.get('period_requests', 0)}
• Успешных: {stats.get('period_successful', 0)}
• Потрачено токенов: {stats.get('period_tokens', 0)}
• Среднее время ответа: {stats.get('avg_response_time_ms', 0)} мс"""

        if stats.get("request_types"):
            status_text += f"\n\n📊 <b>Типы запросов (30 дней)</b>\n"
            for req_type, count in stats["request_types"].items():
                emoji = "⚖️" if req_type == "legal_question" else "🤖"
                status_text += f"• {emoji} {req_type}: {count}\n"

        await message.answer(status_text, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Error in cmd_mystats: {e}")
        await message.answer("❌ Ошибка получения статистики. Попробуйте позже.")


# ============ СИСТЕМА РЕЙТИНГА ============


async def handle_pending_feedback(message: Message, user_session: UserSession):
    """Обработка текстового комментария для рейтинга"""
    if not message.text or not user_session.pending_feedback_request_id:
        return

    request_id = user_session.pending_feedback_request_id
    user_id = message.from_user.id
    feedback_text = message.text.strip()

    # Очищаем pending состояние
    user_session.pending_feedback_request_id = None

    try:
        if hasattr(db, "add_rating"):
            success = await db.add_rating(request_id, user_id, -1, feedback_text)
            if success:
                await message.answer(
                    "✅ <b>Спасибо за развернутый отзыв!</b>\n\n"
                    "Ваш комментарий поможет нам улучшить качество ответов.",
                    parse_mode=ParseMode.HTML,
                )
                logger.info(f"Received feedback for request {request_id} from user {user_id}")
            else:
                await message.answer("❌ Ошибка сохранения комментария")
        else:
            await message.answer("❌ Система обратной связи недоступна")

    except Exception as e:
        logger.error(f"Error in handle_pending_feedback: {e}")
        await message.answer("❌ Произошла ошибка при сохранении комментария")


async def handle_rating_callback(callback: CallbackQuery):
    """Обработчик нажатий на кнопки рейтинга"""
    if not callback.data or not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    user_id = callback.from_user.id

    try:
        # Парсим callback_data: "rate_like_123" или "rate_dislike_123"
        parts = callback.data.split("_")
        if len(parts) != 3:
            await callback.answer("❌ Неверный формат данных")
            return

        action = parts[1]  # "like" или "dislike"
        request_id = int(parts[2])

        rating_value = 1 if action == "like" else -1

        if hasattr(db, "add_rating"):
            success = await db.add_rating(request_id, user_id, rating_value)
            if success:
                if action == "like":
                    await callback.answer("✅ Спасибо за оценку! Рады, что ответ был полезен.")
                    await callback.message.edit_text(
                        "💬 <b>Спасибо за оценку!</b> ✅ Отмечено как полезное",
                        parse_mode=ParseMode.HTML,
                    )
                else:
                    await callback.answer("📝 Спасибо за обратную связь!")
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
                        "💬 <b>Что можно улучшить?</b>\n\n"
                        "Ваша обратная связь поможет нам стать лучше:",
                        reply_markup=feedback_keyboard,
                        parse_mode=ParseMode.HTML,
                    )
            else:
                await callback.answer("❌ Ошибка сохранения оценки")
        else:
            await callback.answer("❌ Система рейтинга недоступна")

    except Exception as e:
        logger.error(f"Error in handle_rating_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_feedback_callback(callback: CallbackQuery):
    """Обработчик запроса обратной связи"""
    if not callback.data or not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        parts = callback.data.split("_")
        if len(parts) < 2:
            await callback.answer("❌ Неверный формат данных")
            return

        action = parts[0]  # "feedback" или "skip"
        request_id = int(parts[1])

        if action == "skip":
            await callback.message.edit_text(
                "💬 <b>Спасибо за оценку!</b> 👎 Отмечено для улучшения", parse_mode=ParseMode.HTML
            )
            await callback.answer("✅ Спасибо за обратную связь!")
        elif action == "feedback":
            user_session = get_user_session(callback.from_user.id)
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
            parse_mode=ParseMode.HTML
        )

        # Устанавливаем режим поиска практики для пользователя
        user_session = get_user_session(callback.from_user.id)
        if not hasattr(user_session, "practice_search_mode"):
            user_session.practice_search_mode = False
        user_session.practice_search_mode = True

    except Exception as e:
        logger.error(f"Error in handle_search_practice_callback: {e}")
        await callback.answer("❌ Произошла ошибка")


async def handle_general_consultation_callback(callback: CallbackQuery):
    """Обработчик кнопки 'Общая юридическая консультация'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        await callback.message.answer(
            "📋 <b>Общая юридическая консультация</b>\n\n"
            "💬 Задайте любой юридический вопрос, и я помогу:\n\n"
            "• Анализ правовой ситуации\n"
            "• Поиск релевантных НПА\n"
            "• Рекомендации по действиям\n"
            "• Оценка перспектив дела\n\n"
            "<i>Напишите ваш вопрос следующим сообщением...</i>",
            parse_mode=ParseMode.HTML
        )

        # Обычный режим консультации (по умолчанию)
        user_session = get_user_session(callback.from_user.id)
        if hasattr(user_session, "practice_search_mode"):
            user_session.practice_search_mode = False

    except Exception as e:
        logger.error(f"Error in handle_general_consultation_callback: {e}")
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
            parse_mode=ParseMode.HTML
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
    """Обработчик кнопки 'Помощь'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        # Используем готовый шаблон справки из UI компонентов
        from src.bot.ui_components import MessageTemplates

        help_text = MessageTemplates.HELP

        await callback.message.answer(
            help_text,
            parse_mode=ParseMode.MARKDOWN_V2
        )

        logger.info(f"Help info requested by user {callback.from_user.id}")

    except Exception as e:
        logger.error(f"Error in handle_help_info_callback: {e}")
        await callback.answer("❌ Произошла ошибка при получении справки")


# ============ ОБРАБОТЧИКИ СИСТЕМЫ ДОКУМЕНТООБОРОТА ============

async def handle_document_processing(callback: CallbackQuery):
    """Обработка кнопки работы с документами"""
    try:
        operations = document_manager.get_supported_operations()

        buttons = []
        for op_key, op_info in operations.items():
            emoji = op_info.get("emoji", "📄")
            name = op_info.get("name", op_key)
            buttons.append([InlineKeyboardButton(
                text=f"{emoji} {name}",
                callback_data=f"doc_operation_{op_key}"
            )])

        buttons.append([InlineKeyboardButton(text="◀️ Назад в меню", callback_data="back_to_menu")])

        message_text = """
🗂️ **Работа с документами**

Выберите операцию для работы с документами:

📋 **Саммаризация** - краткая выжимка документа
⚠️ **Анализ рисков** - поиск проблемных мест
💬 **Чат с документом** - задавайте вопросы по тексту
🔒 **Обезличивание** - удаление персональных данных
🌍 **Перевод** - перевод на другие языки
👁️ **OCR** - распознавание сканированных документов

Поддерживаемые форматы: PDF, DOCX, DOC, TXT, изображения
        """

        await callback.message.edit_text(
            message_text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
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

        message_text = f"""
{emoji} **{name}**

{description}

**Поддерживаемые форматы:** {formats}

📎 **Загрузите документ** для обработки или отправьте файл.
        """

        await callback.message.edit_text(
            message_text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="◀️ Назад к операциям", callback_data="document_processing")]
            ])
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
            await message.answer(f"❌ Файл слишком большой. Максимальный размер: {max_size // (1024*1024)} МБ")
            await state.clear()
            return

        # Показываем статус обработки
        operation_info = document_manager.get_operation_info(operation) or {}
        operation_name = operation_info.get("name", operation)

        status_msg = await message.answer(
            f"📄 Обрабатываем документ **{file_name}**...\n\n"
            f"⏳ Операция: {operation_name}\n"
            f"📊 Размер: {file_size // 1024} КБ",
            parse_mode="Markdown"
        )

        try:
            # Скачиваем файл
            file_info = await message.bot.get_file(message.document.file_id)
            file_path = file_info.file_path

            if not file_path:
                raise ProcessingError("Не удалось получить путь к файлу", "FILE_ERROR")

            file_content = await message.bot.download_file(file_path)

            # Обрабатываем документ
            result = await document_manager.process_document(
                user_id=message.from_user.id,
                file_content=file_content.read(),
                original_name=file_name,
                mime_type=mime_type,
                operation=operation,
                **options
            )

            # Удаляем статусное сообщение
            try:
                await status_msg.delete()
            except:
                pass

            if result.success:
                # Форматируем результат для Telegram
                formatted_result = document_manager.format_result_for_telegram(result, operation)

                # Отправляем результат
                await message.answer(
                    formatted_result,
                    parse_mode="Markdown"
                )

                exports = result.data.get("exports") or []
                for export in exports:
                    export_path = export.get("path")
                    if not export_path:
                        continue
                    try:
                        caption = f"{str(export.get('format', 'file')).upper()} — {Path(export_path).name}"
                        await message.answer_document(FSInputFile(export_path), caption=caption)
                    except Exception as send_error:
                        logger.error(f"Не удалось отправить файл {export_path}: {send_error}", exc_info=True)
                        await message.answer(f"⚠️ Не удалось отправить файл {Path(export_path).name}")

                logger.info(f"Successfully processed document {file_name} for user {message.from_user.id}")
            else:
                await message.answer(
                    f"❌ **Ошибка обработки документа**\n\n{result.message}",
                    parse_mode="Markdown"
                )

        except Exception as e:
            # Удаляем статусное сообщение в случае ошибки
            try:
                await status_msg.delete()
            except:
                pass

            await message.answer(
                f"❌ **Ошибка обработки документа**\n\n{str(e)}",
                parse_mode="Markdown"
            )
            logger.error(f"Error processing document {file_name}: {e}", exc_info=True)

        finally:
            # Очищаем состояние
            await state.clear()

    except Exception as e:
        await message.answer(f"❌ Произошла ошибка: {str(e)}")
        logger.error(f"Error in handle_document_upload: {e}", exc_info=True)
        await state.clear()


async def cmd_ratings_stats(message: Message):
    """Команда для просмотра статистики рейтингов (только для админов)"""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("❌ Команда доступна только администраторам")
        return

    if not hasattr(db, "get_ratings_statistics"):
        await message.answer("❌ Статистика рейтингов недоступна")
        return

    try:
        stats_7d = await db.get_ratings_statistics(7)
        stats_30d = await db.get_ratings_statistics(30)
        low_rated = await db.get_low_rated_requests(5)

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
            stats_text += f"\n\n⚠️ <b>Запросы для улучшения:</b>\n"
            for req in low_rated[:3]:
                stats_text += f"• ID {req['request_id']}: рейтинг {req['avg_rating']:.1f} ({req['rating_count']} оценок)\n"

        await message.answer(stats_text, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Error in cmd_ratings_stats: {e}")
        await message.answer("❌ Ошибка получения статистики рейтингов")


async def pre_checkout(pre: PreCheckoutQuery):
    try:
        payload = pre.invoice_payload or ""
        parts = payload.split(":")
        method = parts[1] if len(parts) >= 2 else ""
        if method == "xtr":
            expected_currency = "XTR"
            expected_amount = convert_rub_to_xtr(
                amount_rub=float(SUB_PRICE_RUB),
                rub_per_xtr=getattr(config, "rub_per_xtr", None),
                default_xtr=SUB_PRICE_XTR,
            )
        elif method == "rub":
            expected_currency = "RUB"
            expected_amount = SUB_PRICE_RUB_KOPEKS
        else:
            expected_currency = pre.currency.upper()
            expected_amount = pre.total_amount

        if pre.currency.upper() != expected_currency or int(pre.total_amount) != int(
            expected_amount
        ):
            await pre.answer(ok=False, error_message="Некорректные параметры оплаты")
            return

        await pre.answer(ok=True)
    except Exception:
        await pre.answer(ok=False, error_message="Ошибка проверки оплаты, попробуйте позже")


async def on_successful_payment(message: Message):
    try:
        sp = message.successful_payment
        if sp is None:
            return
        currency_up = sp.currency.upper()
        if currency_up == "RUB":
            provider_name = "telegram_rub"
        elif currency_up == "XTR":
            provider_name = "telegram_stars"
        else:
            provider_name = f"telegram_{currency_up.lower()}"

        if db is not None and sp.telegram_payment_charge_id:
            exists = await db.transaction_exists_by_telegram_charge_id(
                sp.telegram_payment_charge_id
            )
            if exists:
                return
        until_text = ""
        if db is not None:
            await db.record_transaction(
                user_id=message.from_user.id,
                provider=provider_name,
                currency=sp.currency,
                amount=sp.total_amount,
                amount_minor_units=sp.total_amount,
                payload=sp.invoice_payload or "",
                status="success",
                telegram_payment_charge_id=sp.telegram_payment_charge_id,
                provider_payment_charge_id=sp.provider_payment_charge_id,
            )
            await db.extend_subscription_days(message.from_user.id, SUB_DURATION_DAYS)
            user = await db.get_user(message.from_user.id)
            if user and user.subscription_until:
                until_text = datetime.fromtimestamp(user.subscription_until).strftime("%Y-%m-%d")

        await message.answer(
            f"{Emoji.SUCCESS} Оплата получена\\! Подписка активирована на {SUB_DURATION_DAYS} дней.\nДо: {until_text}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
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


async def main():
    """Запуск простого бота"""
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в переменных окружения")

    # Настройка прокси (опционально)
    session = None
    proxy_url = os.getenv("TELEGRAM_PROXY_URL", "").strip()
    if proxy_url:
        logger.info("Using proxy: %s", proxy_url.split("@")[-1])
        proxy_user = os.getenv("TELEGRAM_PROXY_USER", "").strip()
        proxy_pass = os.getenv("TELEGRAM_PROXY_PASS", "").strip()
        if proxy_user and proxy_pass:
            from urllib.parse import urlparse, urlunparse, quote

            if "://" not in proxy_url:
                proxy_url = "http://" + proxy_url
            u = urlparse(proxy_url)
            userinfo = f"{quote(proxy_user, safe='')}:{quote(proxy_pass, safe='')}"
            netloc = f"{userinfo}@{u.hostname}{':' + str(u.port) if u.port else ''}"
            proxy_url = urlunparse((u.scheme, netloc, u.path or "", u.params, u.query, u.fragment))
        session = AiohttpSession(proxy=proxy_url)

    # Создаем бота и диспетчер
    bot = Bot(BOT_TOKEN, session=session)
    dp = Dispatcher()

    # Инициализация системы метрик/кэша/т.п.
    from src.core.metrics import init_metrics, set_system_status
    from src.core.cache import create_cache_backend, ResponseCache
    from src.core.background_tasks import (
        BackgroundTaskManager,
        DatabaseCleanupTask,
        CacheCleanupTask,
        SessionCleanupTask,
        HealthCheckTask,
        MetricsCollectionTask,
    )
    from src.core.health import (
        HealthChecker,
        DatabaseHealthCheck,
        OpenAIHealthCheck,
        SessionStoreHealthCheck,
        RateLimiterHealthCheck,
        SystemResourcesHealthCheck,
    )
    from src.core.scaling import ServiceRegistry, LoadBalancer, SessionAffinity, ScalingManager

    prometheus_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    metrics_collector = init_metrics(
        enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "1") == "1",
        prometheus_port=prometheus_port,
    )
    set_system_status("starting")

    logger.info("🚀 Starting AI-Ivan (simple)")

    # Инициализация глобальных переменных
    global db, openai_service, rate_limiter, access_service, session_store, crypto_provider, error_handler, document_manager

    # Выбираем тип базы данных
    use_advanced_db = os.getenv("USE_ADVANCED_DB", "1") == "1"
    if use_advanced_db:
        from src.core.db_advanced import DatabaseAdvanced

        logger.info("Using advanced database with connection pooling")
        db = DatabaseAdvanced(
            DB_PATH, max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "5")), enable_metrics=True
        )
    else:
        logger.info("Using legacy database")
        db = Database(DB_PATH)

    # Инициализация БД (поддержка sync/async init)
    if hasattr(db, "init"):
        await _maybe_call(db.init)

    # Инициализация кеша
    cache_backend = await create_cache_backend(
        redis_url=config.redis_url,
        fallback_to_memory=True,
        memory_max_size=int(os.getenv("CACHE_MAX_SIZE", "1000")),
    )

    response_cache = ResponseCache(
        backend=cache_backend,
        default_ttl=int(os.getenv("CACHE_TTL", "3600")),
        enable_compression=os.getenv("CACHE_COMPRESSION", "1") == "1",
    )

    # Инициализация rate limiter
    rate_limiter = RateLimiter(
        redis_url=config.redis_url,
        max_requests=config.rate_limit_requests,
        window_seconds=config.rate_limit_window_seconds,
    )
    await rate_limiter.init()

    # Инициализация сервисов
    access_service = AccessService(db=db, trial_limit=TRIAL_REQUESTS, admin_ids=ADMIN_IDS)
    openai_service = OpenAIService(
        cache=response_cache, enable_cache=False  # Временно отключаем кеш для тестирования
    )
    session_store = SessionStore(max_size=USER_SESSIONS_MAX, ttl_seconds=USER_SESSION_TTL_SECONDS)
    crypto_provider = CryptoPayProvider(asset=os.getenv("CRYPTO_ASSET", "USDT"))
    error_handler = ErrorHandler(logger=logger)

    # Инициализация системы документооборота

    document_manager = DocumentManager(openai_service=openai_service)
    logger.info("📄 Document processing system initialized")

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
    if os.getenv("ENABLE_SCALING", "0") == "1":
        try:
            service_registry = ServiceRegistry(
                redis_url=config.redis_url,
                heartbeat_interval=float(os.getenv("HEARTBEAT_INTERVAL", "15.0")),
            )
            await service_registry.initialize()
            await service_registry.start_background_tasks()

            load_balancer = LoadBalancer(service_registry)
            session_affinity = SessionAffinity(
                redis_client=getattr(cache_backend, "_redis", None),
                ttl=int(os.getenv("SESSION_AFFINITY_TTL", "3600")),
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
            logger.info("🔄 Scaling components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize scaling components: {e}")

    # Health checks
    health_checker = HealthChecker(check_interval=float(os.getenv("HEALTH_CHECK_INTERVAL", "30.0")))
    health_checker.register_check(DatabaseHealthCheck(db))
    health_checker.register_check(OpenAIHealthCheck(openai_service))
    health_checker.register_check(SessionStoreHealthCheck(session_store))
    health_checker.register_check(RateLimiterHealthCheck(rate_limiter))
    if os.getenv("ENABLE_SYSTEM_MONITORING", "1") == "1":
        health_checker.register_check(SystemResourcesHealthCheck())
    await health_checker.start_background_checks()

    # Фоновые задачи
    task_manager = BackgroundTaskManager(error_handler)
    if use_advanced_db:
        task_manager.register_task(
            DatabaseCleanupTask(
                db,
                interval_seconds=float(os.getenv("DB_CLEANUP_INTERVAL", "3600")),
                max_old_transactions_days=int(os.getenv("DB_CLEANUP_DAYS", "90")),
            )
        )
    task_manager.register_task(
        CacheCleanupTask(
            [openai_service], interval_seconds=float(os.getenv("CACHE_CLEANUP_INTERVAL", "300"))
        )
    )
    task_manager.register_task(
        SessionCleanupTask(
            session_store, interval_seconds=float(os.getenv("SESSION_CLEANUP_INTERVAL", "600"))
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
            all_components, interval_seconds=float(os.getenv("HEALTH_CHECK_TASK_INTERVAL", "120"))
        )
    )
    if getattr(metrics_collector, "enable_prometheus", False):
        task_manager.register_task(
            MetricsCollectionTask(
                all_components,
                interval_seconds=float(os.getenv("METRICS_COLLECTION_INTERVAL", "30")),
            )
        )
    await task_manager.start_all()
    logger.info(f"🔧 Started {len(task_manager.tasks)} background tasks")

    # Команды
    await bot.set_my_commands(
        [
            BotCommand(command="start", description=f"{Emoji.ROBOT} Начать работу"),
            BotCommand(command="buy", description=f"{Emoji.MAGIC} Купить подписку"),
            BotCommand(command="status", description=f"{Emoji.STATS} Статус подписки"),
            BotCommand(command="mystats", description=f"📊 Моя статистика"),
            BotCommand(command="ratings", description=f"📈 Статистика рейтингов (админ)"),
        ]
    )

    # Роутинг
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_buy, Command("buy"))
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(cmd_mystats, Command("mystats"))
    dp.message.register(cmd_ratings_stats, Command("ratings"))

    dp.callback_query.register(handle_rating_callback, F.data.startswith("rate_"))
    dp.callback_query.register(
        handle_feedback_callback, F.data.startswith(("feedback_", "skip_feedback_"))
    )

    # Обработчики кнопок главного меню
    dp.callback_query.register(handle_search_practice_callback, F.data == "search_practice")
    dp.callback_query.register(handle_general_consultation_callback, F.data == "general_consultation")
    dp.callback_query.register(handle_prepare_documents_callback, F.data == "prepare_documents")
    dp.callback_query.register(handle_help_info_callback, F.data == "help_info")

    # Обработчики системы документооборота
    dp.callback_query.register(handle_document_processing, F.data == "document_processing")
    dp.callback_query.register(handle_document_operation, F.data.startswith("doc_operation_"))
    dp.callback_query.register(handle_back_to_menu, F.data == "back_to_menu")
    dp.message.register(handle_document_upload, DocumentProcessingStates.waiting_for_document, F.document)

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
        f"🗄️ Database: {'advanced' if use_advanced_db else 'legacy'}",
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


if __name__ == "__main__":
    try:
        try:
            import uvloop  # type: ignore

            uvloop.install()
            logger.info("🚀 Включен uvloop для повышенной производительности")
        except ImportError:
            logger.info("⚡ uvloop не доступен, используем стандартный event loop")

        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Бот остановлен")
    except Exception as e:
        logger.exception("💥 Критическая ошибка: %s", e)
        raise
