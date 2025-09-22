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
from typing import Optional, Dict, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.db_advanced import DatabaseAdvanced
from html import escape as html_escape

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.types import Message, BotCommand, ErrorEvent, LabeledPrice, PreCheckoutQuery, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command

from src.bot.logging_setup import setup_logging
from src.bot.openai_gateway import ask_legal
from src.bot.promt import LEGAL_SYSTEM_PROMPT
from src.bot.ui_components import Emoji, escape_markdown_v2
from src.bot.status_manager import AnimatedStatus, ProgressStatus, ResponseTimer, QuickStatus, TypingContext
from src.core.db import Database
from src.core.crypto_pay import create_crypto_invoice_async
from src.telegram_legal_bot.config import load_config
from src.telegram_legal_bot.ratelimit import RateLimiter
from src.core.access import AccessService
from src.core.openai_service import OpenAIService
from src.core.session_store import SessionStore, UserSession
from src.core.payments import CryptoPayProvider, convert_rub_to_xtr
from src.core.validation import InputValidator, ValidationError, ValidationSeverity
from src.core.exceptions import (
    ErrorHandler, ErrorContext, ErrorType, ErrorSeverity as ExceptionSeverity,
    ValidationException, DatabaseException, OpenAIException, TelegramException,
    NetworkException, PaymentException, AuthException, RateLimitException,
    SystemException, handle_exceptions, safe_execute
)

# ============ КОНФИГУРАЦИЯ ============

load_dotenv()
setup_logging()
logger = logging.getLogger("ai-ivan.simple")

config = load_config()
BOT_TOKEN = config.telegram_bot_token
USE_ANIMATION = config.use_status_animation
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
    rub_per_xtr=getattr(config, 'rub_per_xtr', None),
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

# Политика сессий
USER_SESSIONS_MAX = int(getattr(config, 'user_sessions_max', 10000) or 10000)
USER_SESSION_TTL_SECONDS = int(getattr(config, 'user_session_ttl_seconds', 3600) or 3600)

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
    
    paragraphs = text.split('\n\n')
    for paragraph in paragraphs:
        if len(current_chunk + paragraph + '\n\n') <= max_length:
            current_chunk += paragraph + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
            else:
                # Параграф слишком длинный, разбиваем принудительно
                while len(paragraph) > max_length:
                    chunks.append(paragraph[:max_length])
                    paragraph = paragraph[max_length:]
                current_chunk = paragraph + '\n\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def _send_html_chunks(message: Message, html_text: str) -> None:
    """Send long HTML-safe text split into Telegram-sized chunks."""
    chunks = chunk_text(html_text)
    for i, chunk in enumerate(chunks):
        try:
            await message.answer(chunk, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.warning("Failed to send with HTML, retrying without formatting: %s", e)
            await message.answer(chunk)
        if i < len(chunks) - 1:
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
    if USE_ANIMATION:
        status = AnimatedStatus(message.bot, message.chat.id)
        await status.start()
        return status
    status = ProgressStatus(message.bot, message.chat.id)
    await status.start("Обрабатываю ваш запрос...")
    return status

async def _stop_status_indicator(status) -> None:
    if status is None:
        return
    try:
        if hasattr(status, 'complete'):
            await status.complete()
        else:
            await status.stop()
    except Exception:
        pass

# ============ КОМАНДЫ ============

async def cmd_start(message: Message):
    """Единственная команда - приветствие"""
    user_session = get_user_session(message.from_user.id)
    # Обеспечим запись в БД
    if db is not None:
        await db.ensure_user(message.from_user.id, default_trial=TRIAL_REQUESTS, is_admin=message.from_user.id in ADMIN_IDS)
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

🔥 Готов к работе! Отправьте ваш правовой вопрос
"""

    welcome_text = escape_markdown_v2(welcome_raw)
    await message.answer(welcome_text, parse_mode=ParseMode.MARKDOWN_V2)
    logger.info("User %s started bot", message.from_user.id)

# ============ ОБРАБОТКА ВОПРОСОВ ============

async def process_question(message: Message):
    """Главный обработчик юридических вопросов"""
    user_id = message.from_user.id
    chat_id = message.chat.id
    message_id = message.message_id
    
    # Создаем контекст для обработки ошибок
    error_context = ErrorContext(
        user_id=user_id,
        chat_id=chat_id,
        message_id=message_id,
        function_name="process_question"
    )
    
    user_session = get_user_session(user_id)
    question_text = (message.text or "").strip()
    quota_msg_to_send: Optional[str] = None
    
    # Проверяем, не ждем ли мы комментарий для рейтинга
    # Добавляем атрибут для старых сессий, если его нет
    if not hasattr(user_session, 'pending_feedback_request_id'):
        user_session.pending_feedback_request_id = None
        
    if user_session.pending_feedback_request_id is not None:
        await handle_pending_feedback(message, user_session)
        return
    quota_is_trial: bool = False
    
    # Проверяем, что это не команда
    if question_text.startswith('/'):
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
        # 'Индикатор печатания во время обработки
        async with TypingContext(message.bot, message.chat.id):
            pass
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
        # Показываем статус + индикатор печатания
        if USE_ANIMATION:
            status = AnimatedStatus(message.bot, message.chat.id)
            await status.start()
        else:
            status = ProgressStatus(message.bot, message.chat.id)
            await status.start("Обрабатываю ваш вопрос\\.\\.\\.")
        
        try:
            # Включаем typing на время этапов и вызова ИИ
            async with TypingContext(message.bot, message.chat.id):
                # Имитируем этапы обработки для лучшего UX
                if not USE_ANIMATION and hasattr(status, 'update_stage'):
                    await asyncio.sleep(0.5)
                    await status.update_stage(1, f"{Emoji.SEARCH} Анализирую ваш вопрос\\.\\.\\.")
                    await asyncio.sleep(1)
                    await status.update_stage(2, f"{Emoji.LOADING} Ищу релевантную судебную практику\\.\\.\\.")
                
                # Основной запрос к ИИ
                # Через сервисный слой, для лёгкого мокинга и замены имплементации
                if openai_service is None:
                    raise SystemException("OpenAI service not initialized", error_context)
                
                request_start_time = time.time()
                try:
                    result = await openai_service.ask_legal(LEGAL_SYSTEM_PROMPT, question_text)
                    request_success = True
                    request_error_type = None
                except Exception as e:
                    request_success = False
                    request_error_type = type(e).__name__
                    # Специфичная обработка ошибок OpenAI
                    if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                        raise OpenAIException(str(e), error_context, is_quota_error=True)
                    elif "timeout" in str(e).lower() or "network" in str(e).lower():
                        raise NetworkException(f"OpenAI network error: {str(e)}", error_context)
                    else:
                        raise OpenAIException(f"OpenAI API error: {str(e)}", error_context)
            
            if not USE_ANIMATION and hasattr(status, 'update_stage'):
                await status.update_stage(3, f"{Emoji.DOCUMENT} Формирую структурированный ответ\\.\\.\\.")
                await asyncio.sleep(0.5)
                await status.update_stage(4, f"{Emoji.MAGIC} Финализирую рекомендации\\.\\.\\.")
        
        finally:
            # Останавливаем статус
            if hasattr(status, 'complete'):
                await status.complete()
            else:
                await status.stop()
        
        timer.stop()
        
        # Обрабатываем результат
        if not result.get("ok"):
            error_text = result.get("error", "Неизвестная ошибка")
            logger.error("OpenAI error for user %s: %s", user_id, error_text)
            
            await message.answer(
                f"""{Emoji.ERROR} <b>Произошла ошибка</b>

Не удалось получить ответ. Попробуйте ещё раз чуть позже.

{Emoji.HELP} <i>Подсказка</i>: Проверьте формулировку вопроса

<code>{error_text[:100]}</code>""",
                parse_mode=ParseMode.HTML
            )
            return
        
        # Форматируем ответ для HTML
        response_text = html_escape(result["text"])  # escape to keep HTML parse_mode safe
        

        
        # Добавляем информацию о времени ответа
        time_info = f"\n\n{Emoji.CLOCK} <i>Время ответа: {timer.get_duration_text()}</i>"
        # response_text += time_info  # send separately below to preserve HTML
        
        # Добавляем информацию о квоте/подписке (кроме случая триала — его отправим отдельным сообщением)
        if 'quota_text' in locals() and quota_text and not quota_is_trial:
            pass  # send separately after chunks
        # Разбиваем на части и отправляем
        await _send_html_chunks(message, response_text)
        
        # Резерв: отправляем без разметки

        # После ответа отправляем отдельное сообщение с квотой триала
        # send time info separately to avoid HTML breakage
        try:
            await message.answer(time_info, parse_mode=ParseMode.HTML)
        except Exception:
            await message.answer(time_info)

        # if non-trial quota footer is present, send it separately
        if 'quota_text' in locals() and quota_text and not quota_is_trial:
            try:
                await message.answer(quota_text, parse_mode=ParseMode.HTML)
            except Exception:
                await message.answer(quota_text)

        if quota_msg_to_send:
            try:
                await message.answer(quota_msg_to_send, parse_mode=ParseMode.HTML)
            except Exception:
                # Резерв без разметки
                await message.answer(quota_msg_to_send)
        
        # Обновляем статистику
        user_session.add_question_stats(timer.duration)
        
        # Записываем статистику в базу данных (если это продвинутая версия БД)
        request_id = None
        if hasattr(db, 'record_request') and 'request_start_time' in locals():
            try:
                request_time_ms = int((time.time() - request_start_time) * 1000)
                request_id = await db.record_request(
                    user_id=user_id,
                    request_type='legal_question',
                    tokens_used=0,  # Пока не подсчитываем токены
                    response_time_ms=request_time_ms,
                    success=result.get("ok", False),
                    error_type=None if result.get("ok", False) else "openai_error"
                )
            except Exception as db_error:
                logger.warning("Failed to record request statistics: %s", db_error)
        
        # Отправляем кнопки для рейтинга (если ответ успешен)
        if result.get("ok", False) and request_id is not None:
            # Используем реальный request_id если есть, иначе генерируем фейковый
            display_request_id = request_id if request_id else int(time.time() * 1000) % 1000000  # Фейковый ID
            logger.info(f"Sending rating buttons with display_request_id={display_request_id} (db_request_id={request_id})")
            
            rating_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="👍 Полезно", callback_data=f"rate_like_{display_request_id}"),
                    InlineKeyboardButton(text="👎 Не помогло", callback_data=f"rate_dislike_{display_request_id}")
                ]
            ])
            
            try:
                await message.answer(
                    "💬 <b>Оцените качество ответа:</b>",
                    reply_markup=rating_keyboard,
                    parse_mode=ParseMode.HTML
                )
            except Exception as rating_error:
                logger.warning("Failed to send rating buttons: %s", rating_error)
        
        logger.info("Successfully processed question for user %s in %.2fs", user_id, timer.duration)
        
    except Exception as e:
        # Обрабатываем все исключения через централизованный обработчик
        if error_handler is not None:
            try:
                custom_exc = await error_handler.handle_exception(e, error_context)
                user_message = custom_exc.user_message
            except Exception:
                # Fallback если обработчик ошибок сам падает
                logger.exception("Error handler failed for user %s", user_id)
                user_message = "Произошла системная ошибка. Попробуйте позже."
        else:
            logger.exception("Error processing question for user %s (no error handler)", user_id)
            user_message = "Произошла ошибка. Попробуйте позже."
        
        # Записываем статистику неудачного запроса (если это продвинутая версия БД)
        if hasattr(db, 'record_request'):
            try:
                request_time_ms = int((time.time() - request_start_time) * 1000) if 'request_start_time' in locals() else 0
                error_type = request_error_type if 'request_error_type' in locals() else type(e).__name__
                # Для неудачных запросов request_id не нужен
                await db.record_request(
                    user_id=user_id,
                    request_type='legal_question',
                    tokens_used=0,
                    response_time_ms=request_time_ms,
                    success=False,
                    error_type=str(error_type)
                )
            except Exception as db_error:
                logger.warning("Failed to record failed request statistics: %s", db_error)
        
        # Очищаем статус в случае ошибки
        try:
            if 'status' in locals():
                if hasattr(status, 'complete'):
                    await status.complete()
                else:
                    await status.stop()
        except:
            pass
        
        # Отправляем пользователю понятное сообщение об ошибке
        try:
            await message.answer(
                f"❌ <b>Ошибка обработки запроса</b>\n\n"
                f"{user_message}\n\n"
                f"💡 <b>Рекомендации:</b>\n"
                f"• Переформулируйте вопрос\n"
                f"• Попробуйте через несколько минут\n"
                f"• Обратитесь в поддержку если проблема повторяется",
                parse_mode=ParseMode.HTML
            )
        except Exception as send_error:
            # Последний резерв - отправляем простое сообщение
            logger.error(f"Failed to send error message to user {user_id}: {send_error}")
            try:
                await message.answer("Произошла ошибка. Попробуйте позже.")
            except:
                pass  # Ничего больше не можем сделать

# ============ ПОДПИСКИ И ПЛАТЕЖИ ============

def _build_payload(method: str, user_id: int) -> str:
    return f"sub:{method}:{user_id}:{int(datetime.now().timestamp())}"

async def send_rub_invoice(message: Message):
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
    dynamic_xtr = convert_rub_to_xtr(
        amount_rub=float(SUB_PRICE_RUB),
        rub_per_xtr=getattr(config, 'rub_per_xtr', None),
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
        rub_per_xtr=getattr(config, 'rub_per_xtr', None),
        default_xtr=SUB_PRICE_XTR,
    )
    text = (
        f"{Emoji.MAGIC} **Оплата подписки**\n\n"
        f"Стоимость: {SUB_PRICE_RUB} ₽ \\({dynamic_xtr} ⭐\\) за 30 дней\n\n"
        f"Выберите способ оплаты:" 
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN_V2)
    # Отправляем доступные варианты
    if RUB_PROVIDER_TOKEN:
        await send_rub_invoice(message)
    try:
        await send_stars_invoice(message)
    except Exception as e:
        logger.warning("Failed to send stars invoice: %s", e)
        await message.answer(
            f"{Emoji.WARNING} Telegram Stars временно недоступны. Попробуйте другой способ оплаты.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    # Крипта: создаем инвойс через CryptoBot, если настроен токен
    payload = _build_payload("crypto", message.from_user.id)
    if crypto_provider is None:
        raise RuntimeError("Crypto provider not initialized")
    inv = await crypto_provider.create_invoice(
        amount_rub=float(SUB_PRICE_RUB),
        description="Подписка ИИ-Иван на 30 дней",
        payload=payload,
    )
    if inv.get("ok"):
        await message.answer(
            f"{Emoji.DOWNLOAD} Оплата криптовалютой: перейдите по ссылке\n{inv['url']}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    else:
        await message.answer(
            f"{Emoji.IDEA} Криптовалюта: временно недоступна (настройте CRYPTO_PAY_TOKEN)",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

async def cmd_status(message: Message):
    if db is None:
        await message.answer("Статус временно недоступен")
        return
    user = await db.ensure_user(message.from_user.id, default_trial=TRIAL_REQUESTS, is_admin=message.from_user.id in ADMIN_IDS)
    until = user.subscription_until
    if until and until > 0:
        until_dt = datetime.fromtimestamp(until)
        left_days = max(0, (until_dt - datetime.now()).days)
        sub_text = f"Активна до {until_dt:%Y-%m-%d} (≈{left_days} дн.)"
    else:
        sub_text = "Не активна"
    
    # Используем HTML для простоты
    await message.answer(
        f"{Emoji.STATS} <b>Статус</b>\n\n"
        f"ID: <code>{message.from_user.id}</code>\n"
        f"Роль: {'админ' if user.is_admin else 'пользователь'}\n"
        f"Триал: {user.trial_remaining} запрос(ов)\n"
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
        user = await db.ensure_user(user_id, default_trial=TRIAL_REQUESTS, is_admin=user_id in ADMIN_IDS)
        
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
        
        # Формируем сообщение с HTML разметкой (проще, чем экранирование символов в MarkdownV2)
        status_text = f"""📊 <b>Моя статистика</b>

👤 <b>Профиль</b>
• ID: <code>{user_id}</code>
• Статус: {'👑 Администратор' if stats['is_admin'] else '👤 Пользователь'}
• Регистрация: {format_timestamp(user.created_at)}

💰 <b>Баланс и доступ</b>
• Пробные запросы: {stats['trial_remaining']} из {TRIAL_REQUESTS}
• Подписка: {format_subscription_status(stats['subscription_until'])}

📈 <b>Общая статистика</b>
• Всего запросов: {stats['total_requests']}
• Успешных: {stats['successful_requests']} ✅
• Неудачных: {stats['failed_requests']} ❌
• Последний запрос: {format_timestamp(stats['last_request_at'])}

📅 <b>За последние 30 дней</b>
• Запросов: {stats['period_requests']}
• Успешных: {stats['period_successful']}
• Потрачено токенов: {stats['period_tokens']}
• Среднее время ответа: {stats['avg_response_time_ms']} мс"""

        if stats['request_types']:
            status_text += f"\n\n📊 <b>Типы запросов (30 дней)</b>\n"
            for req_type, count in stats['request_types'].items():
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
        # Обновляем рейтинг с комментарием
        if hasattr(db, 'add_rating'):
            success = await db.add_rating(request_id, user_id, -1, feedback_text)
            
            if success:
                await message.answer(
                    "✅ <b>Спасибо за развернутый отзыв!</b>\n\n"
                    "Ваш комментарий поможет нам улучшить качество ответов.",
                    parse_mode=ParseMode.HTML
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
        
        # Сохраняем рейтинг в БД
        if hasattr(db, 'add_rating'):
            success = await db.add_rating(request_id, user_id, rating_value)
            
            if success:
                if action == "like":
                    await callback.answer("✅ Спасибо за оценку! Рады, что ответ был полезен.")
                    # Обновляем сообщение
                    await callback.message.edit_text(
                        "💬 <b>Спасибо за оценку!</b> ✅ Отмечено как полезное",
                        parse_mode=ParseMode.HTML
                    )
                else:
                    await callback.answer("📝 Спасибо за обратную связь!")
                    # Предлагаем оставить комментарий
                    feedback_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                        [InlineKeyboardButton(text="📝 Написать комментарий", callback_data=f"feedback_{request_id}")],
                        [InlineKeyboardButton(text="❌ Пропустить", callback_data=f"skip_feedback_{request_id}")]
                    ])
                    
                    await callback.message.edit_text(
                        "💬 <b>Что можно улучшить?</b>\n\n"
                        "Ваша обратная связь поможет нам стать лучше:",
                        reply_markup=feedback_keyboard,
                        parse_mode=ParseMode.HTML
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
                "💬 <b>Спасибо за оценку!</b> 👎 Отмечено для улучшения",
                parse_mode=ParseMode.HTML
            )
            await callback.answer("✅ Спасибо за обратную связь!")
        elif action == "feedback":
            # Сохраняем состояние для получения текстового сообщения
            user_session = get_user_session(callback.from_user.id)
            # Добавляем атрибут для старых сессий, если его нет
            if not hasattr(user_session, 'pending_feedback_request_id'):
                user_session.pending_feedback_request_id = None
            user_session.pending_feedback_request_id = request_id
            
            await callback.message.edit_text(
                "💬 <b>Напишите ваш комментарий:</b>\n\n"
                "<i>Что можно улучшить в ответе? Отправьте текстовое сообщение.</i>",
                parse_mode=ParseMode.HTML
            )
            await callback.answer("✏️ Напишите комментарий следующим сообщением")
            
    except Exception as e:
        logger.error(f"Error in handle_feedback_callback: {e}")
        await callback.answer("❌ Произошла ошибка")

async def cmd_ratings_stats(message: Message):
    """Команда для просмотра статистики рейтингов (только для админов)"""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("❌ Команда доступна только администраторам")
        return
    
    if not hasattr(db, 'get_ratings_statistics'):
        await message.answer("❌ Статистика рейтингов недоступна")
        return
    
    try:
        # Получаем статистику за разные периоды
        stats_7d = await db.get_ratings_statistics(7)
        stats_30d = await db.get_ratings_statistics(30)
        
        # Получаем плохо оцененные запросы
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
                rub_per_xtr=getattr(config, 'rub_per_xtr', None),
                default_xtr=SUB_PRICE_XTR,
            )
        elif method == "rub":
            expected_currency = "RUB"
            expected_amount = SUB_PRICE_RUB_KOPEKS
        else:
            expected_currency = pre.currency.upper()
            expected_amount = pre.total_amount

        if pre.currency.upper() != expected_currency or int(pre.total_amount) != int(expected_amount):
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
        if currency_up == 'RUB':
            method = 'rub'
            provider_name = 'telegram_rub'
            amount_minor = sp.total_amount
        elif currency_up == 'XTR':
            method = 'xtr'
            provider_name = 'telegram_stars'
            amount_minor = sp.total_amount
        else:
            method = currency_up.lower()
            provider_name = f'telegram_{method}'
            amount_minor = sp.total_amount

        if db is not None and sp.telegram_payment_charge_id:
            exists = await db.transaction_exists_by_telegram_charge_id(sp.telegram_payment_charge_id)
            if exists:
                return
        if db is not None:
            # Запись транзакции и продление подписки
            await db.record_transaction(
                user_id=message.from_user.id,
                provider=provider_name,
                currency=sp.currency,
                amount=sp.total_amount,
                amount_minor_units=amount_minor,
                payload=sp.invoice_payload or "",
                status="success",
                telegram_payment_charge_id=sp.telegram_payment_charge_id,
                provider_payment_charge_id=sp.provider_payment_charge_id,
            )
            await db.extend_subscription_days(message.from_user.id, SUB_DURATION_DAYS)
            user = await db.get_user(message.from_user.id)
            until_text = ""
            if user and user.subscription_until:
                until_text = datetime.fromtimestamp(user.subscription_until).strftime("%Y-%m-%d")
        else:
            until_text = ""
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

async def main():
    """Запуск простого бота"""
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в переменных окружения")
    
    # Настройка прокси (опционально)
    session = None
    proxy_url = os.getenv("TELEGRAM_PROXY_URL", "").strip()
    if proxy_url:
        logger.info("Using proxy: %s", proxy_url.split('@')[-1])
        
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

    # Инициализация системы метрик
    from src.core.metrics import init_metrics, set_system_status, get_metrics_collector
    from src.core.cache import create_cache_backend, ResponseCache
    from src.core.background_tasks import (
        BackgroundTaskManager, DatabaseCleanupTask, CacheCleanupTask, 
        SessionCleanupTask, HealthCheckTask, MetricsCollectionTask
    )
    from src.core.health import (
        HealthChecker, DatabaseHealthCheck, OpenAIHealthCheck, 
        SessionStoreHealthCheck, RateLimiterHealthCheck, SystemResourcesHealthCheck
    )
    from src.core.scaling import ServiceRegistry, LoadBalancer, SessionAffinity, ScalingManager
    
    # Инициализируем метрики (если доступны)
    prometheus_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    metrics_collector = init_metrics(
        enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "1") == "1",
        prometheus_port=prometheus_port
    )
    set_system_status("starting")
    
    logger.info("🚀 Starting advanced AI-Ivan with full feature set")
    
    # Инициализация глобальных переменных
    global db, openai_service, rate_limiter, access_service, session_store, crypto_provider, error_handler
    
    # Выбираем тип базы данных
    use_advanced_db = os.getenv("USE_ADVANCED_DB", "1") == "1"
    if use_advanced_db:
        from src.core.db_advanced import DatabaseAdvanced
        logger.info("Using advanced database with connection pooling")
        db = DatabaseAdvanced(
            DB_PATH,
            max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "5")),
            enable_metrics=True
        )
    else:
        logger.info("Using legacy database")
    db = Database(DB_PATH)
    
    await db.init()

    # Инициализация кеша
    cache_backend = await create_cache_backend(
        redis_url=config.redis_url,
        fallback_to_memory=True,
        memory_max_size=int(os.getenv("CACHE_MAX_SIZE", "1000"))
    )
    
    response_cache = ResponseCache(
        backend=cache_backend,
        default_ttl=int(os.getenv("CACHE_TTL", "3600")),
        enable_compression=os.getenv("CACHE_COMPRESSION", "1") == "1"
    )

    # Инициализация rate limiter
    global rate_limiter
    rate_limiter = RateLimiter(
        redis_url=config.redis_url,
        max_requests=config.rate_limit_requests,
        window_seconds=config.rate_limit_window_seconds,
    )
    await rate_limiter.init()

    # Инициализация сервисов
    global access_service
    access_service = AccessService(db=db, trial_limit=TRIAL_REQUESTS, admin_ids=ADMIN_IDS)
    
    global openai_service
    openai_service = OpenAIService(
        cache=response_cache,
        enable_cache=os.getenv("ENABLE_OPENAI_CACHE", "1") == "1"
    )
    
    global session_store
    session_store = SessionStore(max_size=USER_SESSIONS_MAX, ttl_seconds=USER_SESSION_TTL_SECONDS)
    
    global crypto_provider
    crypto_provider = CryptoPayProvider(asset=os.getenv("CRYPTO_ASSET", "USDT"))
    
    # Инициализация обработчика ошибок
    global error_handler
    error_handler = ErrorHandler(logger=logger)
    
    # Регистрируем recovery handlers для критических ошибок
    async def database_recovery_handler(exc):
        """Обработчик восстановления БД"""
        if db is not None:
            try:
                await db.init()
                logger.info("Database recovery completed")
            except Exception as recovery_error:
                logger.error(f"Database recovery failed: {recovery_error}")
    
    error_handler.register_recovery_handler(ErrorType.DATABASE, database_recovery_handler)
    
    # Инициализация компонентов для масштабирования
    scaling_components = None
    if os.getenv("ENABLE_SCALING", "0") == "1":
        try:
            service_registry = ServiceRegistry(
                redis_url=config.redis_url,
                heartbeat_interval=float(os.getenv("HEARTBEAT_INTERVAL", "15.0"))
            )
            await service_registry.initialize()
            await service_registry.start_background_tasks()
            
            load_balancer = LoadBalancer(service_registry)
            
            session_affinity = SessionAffinity(
                redis_client=getattr(cache_backend, '_redis', None),
                ttl=int(os.getenv("SESSION_AFFINITY_TTL", "3600"))
            )
            
            scaling_manager = ScalingManager(
                service_registry=service_registry,
                load_balancer=load_balancer,
                session_affinity=session_affinity
            )
            
            scaling_components = {
                "service_registry": service_registry,
                "load_balancer": load_balancer,
                "session_affinity": session_affinity,
                "scaling_manager": scaling_manager
            }
            
            logger.info("🔄 Scaling components initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize scaling components: {e}")
    
    # Инициализация health checks
    health_checker = HealthChecker(
        check_interval=float(os.getenv("HEALTH_CHECK_INTERVAL", "30.0"))
    )
    
    # Регистрируем health checks
    health_checker.register_check(DatabaseHealthCheck(db))
    health_checker.register_check(OpenAIHealthCheck(openai_service))
    health_checker.register_check(SessionStoreHealthCheck(session_store))
    health_checker.register_check(RateLimiterHealthCheck(rate_limiter))
    
    # Системные ресурсы если доступны
    if os.getenv("ENABLE_SYSTEM_MONITORING", "1") == "1":
        health_checker.register_check(SystemResourcesHealthCheck())
    
    # Запускаем фоновые health checks
    await health_checker.start_background_checks()
    
    # Инициализация фоновых задач
    task_manager = BackgroundTaskManager(error_handler)
    
    # Регистрируем задачи
    if use_advanced_db:
        task_manager.register_task(DatabaseCleanupTask(
            db, 
            interval_seconds=float(os.getenv("DB_CLEANUP_INTERVAL", "3600")),  # 1 час
            max_old_transactions_days=int(os.getenv("DB_CLEANUP_DAYS", "90"))
        ))
    
    task_manager.register_task(CacheCleanupTask(
        [openai_service],
        interval_seconds=float(os.getenv("CACHE_CLEANUP_INTERVAL", "300"))  # 5 минут
    ))
    
    task_manager.register_task(SessionCleanupTask(
        session_store,
        interval_seconds=float(os.getenv("SESSION_CLEANUP_INTERVAL", "600"))  # 10 минут
    ))
    
    # Health check как фоновая задача
    all_components = {
        "database": db,
        "openai_service": openai_service,
        "rate_limiter": rate_limiter,
        "session_store": session_store,
        "error_handler": error_handler,
        "health_checker": health_checker
    }
    
    if scaling_components:
        all_components.update(scaling_components)
    
    task_manager.register_task(HealthCheckTask(
        all_components,
        interval_seconds=float(os.getenv("HEALTH_CHECK_TASK_INTERVAL", "120"))  # 2 минуты
    ))
    
    # Метрики если включены
    if metrics_collector and metrics_collector.enable_prometheus:
        task_manager.register_task(MetricsCollectionTask(
            all_components,
            interval_seconds=float(os.getenv("METRICS_COLLECTION_INTERVAL", "30"))  # 30 секунд
        ))
    
    # Запускаем все фоновые задачи
    await task_manager.start_all()
    
    logger.info(f"🔧 Started {len(task_manager.tasks)} background tasks")
    
    # Устанавливаем команды
    await bot.set_my_commands([
        BotCommand(command="start", description=f"{Emoji.ROBOT} Начать работу"),
        BotCommand(command="buy", description=f"{Emoji.MAGIC} Купить подписку"),
        BotCommand(command="status", description=f"{Emoji.STATS} Статус подписки"),
        BotCommand(command="mystats", description=f"📊 Моя статистика"),
        BotCommand(command="ratings", description=f"📈 Статистика рейтингов (админ)"),
    ])
    
    # Регистрируем обработчики
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_buy, Command("buy"))
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(cmd_mystats, Command("mystats"))
    dp.message.register(cmd_ratings_stats, Command("ratings"))
    
    # Обработчики callback'ов для рейтинга
    dp.callback_query.register(handle_rating_callback, F.data.startswith("rate_"))
    dp.callback_query.register(handle_feedback_callback, F.data.startswith(("feedback_", "skip_feedback_")))
    
    dp.message.register(on_successful_payment, F.successful_payment)
    dp.pre_checkout_query.register(pre_checkout)
    dp.message.register(process_question, F.text & ~F.text.startswith("/"))
    
    # Глобальный обработчик ошибок
    async def telegram_error_handler(event: ErrorEvent):
        """Обработчик ошибок для aiogram с интеграцией ErrorHandler"""
        if error_handler:
            try:
                context = ErrorContext(
                    function_name="telegram_error_handler",
                    additional_data={
                        "update": str(event.update) if event.update else None,
                        "exception_type": type(event.exception).__name__
                    }
                )
                await error_handler.handle_exception(event.exception, context)
            except Exception as handler_error:
                logger.error(f"Error handler failed: {handler_error}")
        
        logger.exception("Critical error in bot: %s", event.exception)
    
    dp.error.register(telegram_error_handler)
    
    # Обновляем статус системы
    set_system_status("running")
    
    # Логируем информацию о запуске
    startup_info = [
        "🤖 Advanced AI-Ivan successfully started!",
        f"📊 Animation: {'enabled' if USE_ANIMATION else 'disabled'}",
        f"🗄️ Database: {'advanced' if use_advanced_db else 'legacy'}",
        f"🔄 Cache: {cache_backend.__class__.__name__}",
        f"📈 Metrics: {'enabled' if metrics_collector and metrics_collector.enable_prometheus else 'disabled'}",
        f"🏥 Health checks: {len(health_checker.checks)} registered",
        f"⚙️ Background tasks: {len(task_manager.tasks)} running",
        f"🔄 Scaling: {'enabled' if scaling_components else 'disabled'}"
    ]
    
    for info in startup_info:
        logger.info(info)
    
    if prometheus_port:
        logger.info(f"📊 Prometheus metrics available at http://localhost:{prometheus_port}/metrics")
    
    try:
        # Запускаем polling
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
        
        # Останавливаем все фоновые задачи
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
        
        # Закрываем основные сервисы
        services_to_close = [
            ("Bot session", lambda: bot.session.close()),
            ("Database", lambda: db.close() if db else None),
            ("Rate limiter", lambda: rate_limiter.close() if rate_limiter else None),
            ("OpenAI service", lambda: openai_service.close() if openai_service else None),
            ("Response cache", lambda: response_cache.close() if response_cache else None)
        ]
        
        for service_name, close_func in services_to_close:
            try:
                result = close_func()
                if result and hasattr(result, '__await__'):
                    await result
                logger.debug(f"✅ {service_name} closed")
            except Exception as e:
                logger.error(f"❌ Error closing {service_name}: {e}")
        
        logger.info("👋 AI-Ivan shutdown complete")

if __name__ == "__main__":
    try:
        # Включаем uvloop для лучшей производительности (Linux/macOS)
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
        exit(1)
