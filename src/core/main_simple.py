"""
Простая версия Telegram бота ИИ-Иван
Только /start и обработка вопросов, никаких кнопок и лишних команд
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.safe_telegram import send_html_text
from src.documents.document_manager import DocumentManager

if TYPE_CHECKING:
    from src.core.db_advanced import DatabaseAdvanced

import re
from html import escape as html_escape

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
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
)
from dotenv import load_dotenv

from src.bot.logging_setup import setup_logging
from src.bot.promt import JUDICIAL_PRACTICE_SEARCH_PROMPT, LEGAL_SYSTEM_PROMPT
from src.bot.status_manager import ProgressStatus, progress_router

from src.bot.stream_manager import StreamingCallback, StreamManager
from src.bot.ui_components import Emoji, sanitize_telegram_html, render_legal_html
from src.core.audio_service import AudioService
from src.core.access import AccessService
from src.core.db import Database
from src.core.exceptions import (
    ErrorContext,
    ErrorHandler,
    ErrorType,
    NetworkException,
    OpenAIException,
    SystemException,
)
from src.core.openai_service import OpenAIService
from src.core.payments import CryptoPayProvider, convert_rub_to_xtr
from src.core.session_store import SessionStore, UserSession
from src.core.validation import InputValidator, ValidationSeverity
from src.documents.base import ProcessingError
from src.telegram_legal_bot.config import load_config
from src.telegram_legal_bot.ratelimit import RateLimiter

SAFE_LIMIT = 3900  # чуть меньше телеграмного 4096 (запас на теги)
# ============ КОНФИГУРАЦИЯ ============

load_dotenv()
setup_logging()
logger = logging.getLogger("ai-ivan.simple")

config = load_config()

@dataclass(frozen=True)
class WelcomeMedia:
    path: Path
    media_type: str

class ResponseTimer:
    def __init__(self) -> None:
        self._t0 = None
        self.duration = 0.0

    def start(self) -> None:
        self._t0 = time.monotonic()

    def stop(self) -> None:
        if self._t0 is not None:
            self.duration = max(0.0, time.monotonic() - self._t0)

    def get_duration_text(self) -> str:
        s = int(self.duration)
        return f"{s//60:02d}:{s%60:02d}"


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
ANIMATION_EXTENSIONS = {".gif"}
PHOTO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _discover_welcome_media() -> WelcomeMedia | None:
    try:
        images_dir = Path(__file__).resolve().parents[2] / "images"
        if not images_dir.exists():
            return None
        for candidate in sorted(images_dir.iterdir()):
            if not candidate.is_file():
                continue
            suffix = candidate.suffix.lower()
            if suffix in VIDEO_EXTENSIONS:
                return WelcomeMedia(candidate, "video")
            if suffix in ANIMATION_EXTENSIONS:
                return WelcomeMedia(candidate, "animation")
            if suffix in PHOTO_EXTENSIONS:
                return WelcomeMedia(candidate, "photo")
    except Exception as discover_error:
        logger.debug("Welcome media discovery failed: %s", discover_error)
    return None

WELCOME_MEDIA = _discover_welcome_media()
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
SUB_PRICE_RUB_KOPEKS = int(float(SUB_PRICE_RUB) * 100)

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
db: Database | DatabaseAdvanced | None = None
rate_limiter: RateLimiter | None = None
access_service: AccessService | None = None
openai_service: OpenAIService | None = None
audio_service: AudioService | None = None
session_store: SessionStore | None = None
crypto_provider: CryptoPayProvider | None = None
error_handler: ErrorHandler | None = None
document_manager: Any | None = None  # DocumentManager будет инициализирован позже

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
        elif current_chunk:
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


async def send_rating_request(message: Message, request_id: int):
    """Отправляет сообщение с запросом на оценку ответа"""
    try:
        rating_keyboard = create_rating_keyboard(request_id)
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
 
 <b>Добро пожаловать, {user_name}!</b>
    
 Меня зовут <b>ИИ-ИВАН</b>, я ваш виртуальный юридический ассистент.
 
    
 <b>📌 ЧТО Я УМЕЮ ?</b>
  ═══════════════════════════════════════════



    
 <b>📌 ПРИМЕРЫ ОБРАЩЕНИЙ </b>
  ═══════════════════════════════════════════
 
 ▫️  <i>“Администрация отказала по [описание причины], подбери стратегию как ее обойти со ссылками на судебную практику”</i> 
 
 ▫️  <i>“Чем отличатся статья [название] от статьи [название]”</i> 
    
 ▫️  <i>“Подбери судебную практику по [описание дела]”</i> 
    
 ▫️  <i>“Могут ли наследники [описание нюанса]”</i> 
    
 ═══════════════════════════════════════════
    
    <b>ПОПРОБУЙ ПРЯМО СЕЙЧАС 👇</b>
    """




    # Создаем inline клавиатуру с кнопками (компактное размещение)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔍 Поиск судебной практики", callback_data="search_practice"
                ),
                InlineKeyboardButton(text="📋 Консультация", callback_data="general_consultation"),
            ],
            [
                InlineKeyboardButton(
                    text="📄 Подготовка документов", callback_data="prepare_documents"
                ),
                InlineKeyboardButton(
                    text="🗂️ Работа с документами", callback_data="document_processing"
                ),
            ],
            [InlineKeyboardButton(text="ℹ️ Помощь", callback_data="help_info")],
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

    user_id = message.from_user.id
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
                await message.answer(
                    (
                        f"{Emoji.WARNING} <b>Лимит бесплатных запросов исчерпан</b>\n\n"
                        f"Вы использовали {TRIAL_REQUESTS} из {TRIAL_REQUESTS}. "
                        f"Оформите подписку за {SUB_PRICE_RUB}₽ в месяц командой /buy"
                    ),
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

        # Прогресс-бар
        status = await _start_status_indicator(message)

        ok_flag = False
        request_error_type = None
        stream_manager: StreamManager | None = None
        had_stream_content = False
        result: dict[str, Any] = {}
        request_start_time = time.time()

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

                    # 2) Успех, если API вернул ok ИЛИ уже показывали текст пользователю
                    ok_flag = bool(isinstance(result, dict) and result.get("ok")) or had_stream_content

                    # 3) Фолбэк — если стрим не дал результата и текста нет
                    if not ok_flag:
                        result = await openai_service.ask_legal(selected_prompt, question_text)
                        ok_flag = bool(result.get("ok"))

                except Exception as e:
                    # Если что-то упало, но буфер уже есть — считаем успехом и завершаем стрим
                    had_stream_content = bool((stream_manager.pending_text or "").strip())
                    if had_stream_content:
                        logger.warning("Streaming failed, but content exists — using buffered text: %s", e)
                        with suppress(Exception):
                            await stream_manager.finalize(stream_manager.pending_text)
                        result = {"ok": True, "text": stream_manager.pending_text}
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
                    + (f"<br><br><code>{html_escape(error_text[:300])}</code>" if error_text else "")
                ),
                parse_mode=ParseMode.HTML,
            )
            return

        # Если контент уже пришёл стримом — не дублируем
        # Если контент уже пришёл стримом — не дублируем
        if not (USE_STREAMING and had_stream_content):
            text_to_send = (isinstance(result, dict) and (result.get("text") or "")) or ""
            if text_to_send:
                # единый безопасный путь: форматирование → санация → разбиение → отправка
                await send_html_text(
                    bot=message.bot,
                    chat_id=message.chat.id,
                    raw_text=text_to_send,
                    reply_to_message_id=message.message_id,
                )

        # Время ответа
        time_info = f"{Emoji.CLOCK} <i>Время ответа: {timer.get_duration_text()}</i>"
        with suppress(Exception):
            await message.answer(time_info, parse_mode=ParseMode.HTML)

        # Сообщения про квоту/подписку
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
                await db.record_request(
                    user_id=user_id,
                    request_type="legal_question",
                    tokens_used=0,
                    response_time_ms=request_time_ms,
                    success=True,
                    error_type=None,
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
                await db.record_request(
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


def _build_payload(method: str, user_id: int) -> str:
    return f"sub:{method}:{user_id}:{int(datetime.now().timestamp())}"


async def send_rub_invoice(message: Message):
    if not message.from_user or not message.bot:
        return

    if not RUB_PROVIDER_TOKEN:
        await message.answer(
            f"{Emoji.WARNING} Оплата картами временно недоступна. Попробуйте Telegram Stars или криптовалюту (/buy)",
            parse_mode=ParseMode.HTML,
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
        f"{Emoji.MAGIC} <b>Оплата подписки</b>\n\n"
        f"Стоимость: <b>{SUB_PRICE_RUB} ₽</b> (≈{dynamic_xtr} ⭐) за 30 дней\n\n"
        f"Выберите способ оплаты:"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)

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
            parse_mode=ParseMode.HTML,
        )

    # Крипта: инвойс через CryptoBot
    payload = _build_payload("crypto", message.from_user.id)
    if crypto_provider is None:
        logger.warning("Crypto provider not initialized; skipping crypto invoice")
        await message.answer(
            f"{Emoji.IDEA} Криптовалюта: временно недоступна (настройте CRYPTO_PAY_TOKEN)",
            parse_mode=ParseMode.HTML,
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
                parse_mode=ParseMode.HTML,
            )
        else:
            await message.answer(
                f"{Emoji.IDEA} Криптовалюта: временно недоступна (настройте CRYPTO_PAY_TOKEN)",
                parse_mode=ParseMode.HTML,
            )
    except Exception as e:
        logger.warning("Crypto invoice failed: %s", e)
        await message.answer(
            f"{Emoji.IDEA} Криптовалюта: временно недоступна",
            parse_mode=ParseMode.HTML,
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
            status_text += "\n\n📊 <b>Типы запросов (30 дней)</b>\n"
            for req_type, count in stats["request_types"].items():
                emoji = "⚖️" if req_type == "legal_question" else "🤖"
                status_text += f"• {emoji} {req_type}: {count}\n"

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

    if audio_service is None or not config.voice_mode_enabled:
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


async def handle_pending_feedback(message: Message, user_session: UserSession, text_override: str | None = None):
    """Обработка текстового комментария для рейтинга"""
    feedback_source = text_override if text_override is not None else (message.text or "")
    if not feedback_source or not user_session.pending_feedback_request_id:
        return

    request_id = user_session.pending_feedback_request_id
    user_id = message.from_user.id
    feedback_text = feedback_source.strip()

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
            parse_mode=ParseMode.HTML,
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
    """Обработчик кнопки 'Помощь'"""
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных")
        return

    try:
        await callback.answer()

        # Используем готовый шаблон справки из UI компонентов
        from src.bot.ui_components import MessageTemplates

        help_text = MessageTemplates.HELP

        await callback.message.answer(help_text, parse_mode=ParseMode.HTML)

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
            buttons.append(
                [InlineKeyboardButton(text=f"{emoji} {name}", callback_data=f"doc_operation_{op_key}")]
            )

        buttons.append([InlineKeyboardButton(text="◀️ Назад в меню", callback_data="back_to_menu")])

        message_text = (
            "🗂️ <b>Работа с документами</b><br><br>"
            "Выберите операцию для работы с документами:<br><br>"
            "📋 <b>Саммаризация</b> — краткая выжимка документа<br>"
            "⚠️ <b>Анализ рисков</b> — поиск проблемных мест<br>"
            "💬 <b>Чат с документом</b> — задавайте вопросы по тексту<br>"
            "🔒 <b>Обезличивание</b> — удаление персональных данных<br>"
            "🌍 <b>Перевод</b> — перевод на другие языки<br>"
            "👁️ <b>OCR</b> — распознавание сканированных документов<br><br>"
            "Поддерживаемые форматы: PDF, DOCX, DOC, TXT, изображения"
        )

        await callback.message.edit_text(
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

        message_text = (
            f"{emoji} <b>{name}</b><br><br>"
            f"{html_escape(description)}<br><br>"
            f"<b>Поддерживаемые форматы:</b> {html_escape(formats)}<br><br>"
            "📎 <b>Загрузите документ</b> для обработки или отправьте файл."
        )

        await callback.message.edit_text(
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
            await message.answer(
                f"❌ Файл слишком большой. Максимальный размер: {max_size // (1024*1024)} МБ"
            )
            await state.clear()
            return

        # Показываем статус обработки
        operation_info = document_manager.get_operation_info(operation) or {}
        operation_name = operation_info.get("name", operation)

        status_msg = await message.answer(
            f"📄 Обрабатываем документ <b>{html_escape(file_name)}</b>...<br><br>"
            f"⏳ Операция: {html_escape(operation_name)}<br>"
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

            # Обрабатываем документ
            result = await document_manager.process_document(
                user_id=message.from_user.id,
                file_content=file_content.read(),
                original_name=file_name,
                mime_type=mime_type,
                operation=operation,
                **options,
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
                await message.answer(formatted_result, parse_mode=ParseMode.HTML)

                exports = result.data.get("exports") or []
                for export in exports:
                    export_path = export.get("path")
                    if not export_path:
                        continue
                    try:
                        caption = f"{str(export.get('format', 'file')).upper()} — {Path(export_path).name}"
                        await message.answer_document(FSInputFile(export_path), caption=caption)
                    except Exception as send_error:
                        logger.error(
                            f"Не удалось отправить файл {export_path}: {send_error}", exc_info=True
                        )
                        await message.answer(
                            f"⚠️ Не удалось отправить файл {Path(export_path).name}"
                        )

                logger.info(
                    f"Successfully processed document {file_name} for user {message.from_user.id}"
                )
            else:
                await message.answer(
                    f"❌ <b>Ошибка обработки документа</b><br><br>{html_escape(str(result.message))}",
                    parse_mode=ParseMode.HTML,
                )

        except Exception as e:
            # Удаляем статусное сообщение в случае ошибки
            try:
                await status_msg.delete()
            except:
                pass

            await message.answer(
                f"❌ <b>Ошибка обработки документа</b><br><br>{html_escape(str(e))}",
                parse_mode=ParseMode.HTML,
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
            stats_text += "\n\n⚠️ <b>Запросы для улучшения:</b>\n"
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
            f"{Emoji.SUCCESS} <b>Оплата получена!</b> Подписка активирована на {SUB_DURATION_DAYS} дней.\nДо: {until_text}",
            parse_mode=ParseMode.HTML,
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
            from urllib.parse import quote, urlparse, urlunparse

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
    register_progressbar(dp)

    # Инициализация системы метрик/кэша/т.п.
    from src.core.background_tasks import (
        BackgroundTaskManager,
        CacheCleanupTask,
        DatabaseCleanupTask,
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

    prometheus_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    metrics_collector = init_metrics(
        enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "1") == "1",
        prometheus_port=prometheus_port,
    )
    set_system_status("starting")

    logger.info("🚀 Starting AI-Ivan (simple)")

    # Инициализация глобальных переменных
    global db, openai_service, audio_service, rate_limiter, access_service, session_store, crypto_provider, error_handler, document_manager

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
    if config.voice_mode_enabled:
        audio_service = AudioService(
            stt_model=config.voice_stt_model,
            tts_model=config.voice_tts_model,
            tts_voice=config.voice_tts_voice,
            tts_format=config.voice_tts_format,
            max_duration_seconds=config.voice_max_duration_seconds,
        )
        logger.info("Voice mode enabled (stt=%s, tts=%s, voice=%s, format=%s)",
                    config.voice_stt_model,
                    config.voice_tts_model,
                    config.voice_tts_voice,
                    config.voice_tts_format)
    else:
        audio_service = None
        logger.info("Voice mode disabled")
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
            BotCommand(command="mystats", description="📊 Моя статистика"),
            BotCommand(command="ratings", description="📈 Статистика рейтингов (админ)"),
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
    dp.callback_query.register(
        handle_general_consultation_callback, F.data == "general_consultation"
    )
    dp.callback_query.register(handle_prepare_documents_callback, F.data == "prepare_documents")
    dp.callback_query.register(handle_help_info_callback, F.data == "help_info")

    # Обработчики системы документооборота
    dp.callback_query.register(handle_document_processing, F.data == "document_processing")
    dp.callback_query.register(handle_document_operation, F.data.startswith("doc_operation_"))
    dp.callback_query.register(handle_back_to_menu, F.data == "back_to_menu")
    dp.message.register(
        handle_document_upload, DocumentProcessingStates.waiting_for_document, F.document
    )

    if config.voice_mode_enabled:
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
        # Настройка event loop для Windows
        import sys
        if sys.platform == "win32":
            # Для Windows используем WindowsProactorEventLoopPolicy
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            logger.info("Настроен ProactorEventLoop для Windows")
        else:
            # Для других платформ пробуем uvloop
            try:
                import uvloop  # type: ignore
                uvloop.install()
                logger.info("Включен uvloop для повышенной производительности")
            except ImportError:
                logger.info("uvloop не доступен, используем стандартный event loop")

        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
    except Exception as e:
        logger.exception("Критическая ошибка: %s", e)
        raise

