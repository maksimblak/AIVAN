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
from aiogram.types import Message, BotCommand, ErrorEvent, LabeledPrice, PreCheckoutQuery
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
from src.core.session_store import SessionStore
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

class UserSession:
    """Простая сессия пользователя"""
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.questions_count = 0
        self.total_response_time = 0.0
        self.last_question_time: Optional[datetime] = None
        self.created_at = datetime.now()
        
    def add_question_stats(self, response_time: float):
        """Добавить статистику вопроса"""
        self.questions_count += 1
        self.total_response_time += response_time
        self.last_question_time = datetime.now()

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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Примеры вопросов:

📝 "Можно ли расторгнуть договор поставки за просрочку?"
👨‍💼 "Как правильно уволить сотрудника за нарушения?"
💰 "Какие риски при доначислении НДС?"
🏢 "Порядок увеличения уставного капитала ООО"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    quota_is_trial: bool = False
    
    # Проверяем, что это не команда
    if question_text.startswith('/'):
        return
    
    # ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
    if error_handler is None:
        raise SystemException("Error handler not initialized", error_context)
        
    validation_result = InputValidator.validate_question(question_text, user_id)
    
    if not validation_result.is_valid:
        error_msg = "\n• ".join(validation_result.errors)
        if validation_result.severity == ValidationSeverity.CRITICAL:
            await message.answer(
                f"{Emoji.ERROR} <b>Критическая ошибка валидации</b>\n\n• {error_msg}\n\n<i>Обратитесь к администратору</i>",
                parse_mode=ParseMode.HTML
            )
            return
        else:
            await message.answer(
                f"{Emoji.WARNING} <b>Ошибка в запросе</b>\n\n• {error_msg}",
                parse_mode=ParseMode.HTML
            )
            return
    
    # Используем очищенные данные
    question_text = validation_result.cleaned_data
    
    # Показываем предупреждения если есть
    if validation_result.warnings:
        warning_msg = "\n• ".join(validation_result.warnings)
        logger.warning(f"Validation warnings for user {user_id}: {warning_msg}")
    
    if not question_text:
        await message.answer(
            f"{Emoji.WARNING} <b>Пустой запрос</b>\n\nПожалуйста, отправьте текст юридического вопроса.",
            parse_mode=ParseMode.HTML
        )
        return

    # Запускаем таймер
    timer = ResponseTimer()
    timer.start()
    
    logger.info("Processing question from user %s: %s", user_id, question_text[:100])
    
    try:
        # Global rate limit per user
        if rate_limiter is not None:
            allowed = await rate_limiter.allow(user_id)
            if not allowed:
                await message.answer(
                    f"{Emoji.WARNING} <b>Слишком много запросов</b>\n\nПопробуйте позже.",
                    parse_mode=ParseMode.HTML,
                )
                return
        # Индикатор печатания во время обработки
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
                quota_text = escape_markdown_v2(f"\n\n{Emoji.STATS} Админ: безлимитный доступ")
            elif decision.has_subscription and decision.subscription_until:
                until_dt = datetime.fromtimestamp(decision.subscription_until)
                quota_text = escape_markdown_v2(f"\n\n{Emoji.CALENDAR} Подписка активна до: {until_dt:%Y-%m-%d}")
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
        response_text = result["text"]
        
        # Добавляем footer с напоминанием
        footer = f"\n\n{Emoji.WARNING} <i>Данная информация носит консультационный характер и требует проверки практикующим юристом.</i>"
        response_text += footer
        
        # Добавляем информацию о времени ответа
        time_info = f"\n\n{Emoji.CLOCK} <i>Время ответа: {timer.get_duration_text()}</i>"
        response_text += time_info
        
        # Добавляем информацию о квоте/подписке (кроме случая триала — его отправим отдельным сообщением)
        if 'quota_text' in locals() and quota_text and not quota_is_trial:
            response_text += quota_text
        # Разбиваем на части и отправляем
        chunks = chunk_text(response_text)
        
        for i, chunk in enumerate(chunks):
            try:
                await message.answer(chunk, parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.warning("Failed to send with HTML, retrying without formatting: %s", e)
                # Резерв: отправляем без разметки
                await message.answer(chunk)
            
            # Небольшая задержка между сообщениями
            if i < len(chunks) - 1:
                await asyncio.sleep(0.1)

        # После ответа отправляем отдельное сообщение с квотой триала
        if quota_msg_to_send:
            try:
                await message.answer(quota_msg_to_send, parse_mode=ParseMode.HTML)
            except Exception:
                # Резерв без разметки
                await message.answer(quota_msg_to_send)
        
        # Обновляем статистику
        user_session.add_question_stats(timer.duration)
        
        # Записываем статистику в базу данных (если это продвинутая версия БД)
        if hasattr(db, 'record_request') and 'request_start_time' in locals():
            try:
                request_time_ms = int((time.time() - request_start_time) * 1000)
                await db.record_request(
                    user_id=user_id,
                    request_type='legal_question',
                    tokens_used=0,  # Пока не подсчитываем токены
                    response_time_ms=request_time_ms,
                    success=result.get("ok", False),
                    error_type=None if result.get("ok", False) else "openai_error"
                )
            except Exception as db_error:
                logger.warning("Failed to record request statistics: %s", db_error)
        
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
                f"""{Emoji.ERROR} **Ошибка обработки запроса**

{escape_markdown_v2(user_message)}

{Emoji.HELP} *Рекомендации:*
• Переформулируйте вопрос
• Попробуйте через несколько минут
• Обратитесь в поддержку если проблема повторяется""",
                parse_mode=ParseMode.MARKDOWN_V2
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

async def error_handler(event: ErrorEvent):
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
    ])
    
    # Регистрируем обработчики
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_buy, Command("buy"))
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(cmd_mystats, Command("mystats"))
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
