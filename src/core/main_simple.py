"""
Простая версия Telegram бота ИИ-Иван
Только /start и обработка вопросов, никаких кнопок и лишних команд
"""

from __future__ import annotations
import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
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
db: Optional[Database] = None
rate_limiter: Optional[RateLimiter] = None
access_service: Optional[AccessService] = None
openai_service: Optional[OpenAIService] = None
session_store: Optional[SessionStore] = None
crypto_provider: Optional[CryptoPayProvider] = None

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
    user_session = get_user_session(user_id)
    question_text = (message.text or "").strip()
    quota_msg_to_send: Optional[str] = None
    quota_is_trial: bool = False
    
    # Проверяем, что это не команда
    if question_text.startswith('/'):
        return
    
    if not question_text:
        await message.answer(
            f"{Emoji.WARNING} **Пустой запрос**\n\nПожалуйста, отправьте текст юридического вопроса\\.",
            parse_mode=ParseMode.MARKDOWN_V2
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
                    f"{Emoji.WARNING} **Слишком много запросов**\n\nПопробуйте позже.",
                    parse_mode=ParseMode.MARKDOWN_V2,
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
                    f"{Emoji.WARNING} **Лимит бесплатных запросов исчерпан**\n\nВы использовали {TRIAL_REQUESTS} из {TRIAL_REQUESTS}. Оформите подписку за {SUB_PRICE_RUB}₽ в месяц командой /buy",
                    parse_mode=ParseMode.MARKDOWN_V2,
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
                    raise RuntimeError("OpenAI service not initialized")
                result = await openai_service.ask_legal(LEGAL_SYSTEM_PROMPT, question_text)
            
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
                f"""{Emoji.ERROR} **Произошла ошибка**

Не удалось получить ответ\\. Попробуйте ещё раз чуть позже\\.

{Emoji.HELP} *Подсказка*: Проверьте формулировку вопроса

`{error_text[:100]}`""",
                parse_mode=ParseMode.MARKDOWN_V2
            )
            return
        
        # Форматируем ответ
        # Сначала экранируем текст модели для MarkdownV2, затем добавляем служебные части
        safe_model_text = escape_markdown_v2(result["text"])
        response_text = safe_model_text
        
        # Добавляем footer с напоминанием
        footer = f"\n\n{Emoji.WARNING} _Данная информация носит консультационный характер и требует проверки практикующим юристом\\._"
        response_text += footer
        
        # Добавляем информацию о времени ответа
        time_info = f"\n\n{Emoji.CLOCK} _Время ответа: {timer.get_duration_text()}_"
        response_text += time_info
        
        # Добавляем информацию о квоте/подписке (кроме случая триала — его отправим отдельным сообщением)
        if 'quota_text' in locals() and quota_text and not quota_is_trial:
            response_text += quota_text
        # Разбиваем на части и отправляем
        chunks = chunk_text(response_text)
        
        for i, chunk in enumerate(chunks):
            try:
                await message.answer(chunk, parse_mode=ParseMode.MARKDOWN_V2)
            except Exception as e:
                logger.warning("Failed to send with markdown, retrying with escaped text: %s", e)
                try:
                    await message.answer(escape_markdown_v2(chunk), parse_mode=ParseMode.MARKDOWN_V2)
                except Exception as e2:
                    logger.warning("Second markdown attempt failed: %s", e2)
                    # Последний резерв: отправляем без разметки
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
        
        logger.info("Successfully processed question for user %s in %.2fs", user_id, timer.duration)
        
    except Exception as e:
        logger.exception("Error processing question for user %s", user_id)
        
        # Очищаем статус в случае ошибки
        try:
            if 'status' in locals():
                if hasattr(status, 'complete'):
                    await status.complete()
                else:
                    await status.stop()
        except:
            pass
        
        await message.answer(
            f"""{Emoji.ERROR} **Произошла ошибка**

К сожалению, не удалось обработать ваш запрос\\.

{Emoji.HELP} *Попробуйте:*
• Переформулировать вопрос
• Повторить через минуту

`{str(e)[:100]}`""",
            parse_mode=ParseMode.MARKDOWN_V2
        )

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
        f"Стоимость: {SUB_PRICE_RUB}₽ ({dynamic_xtr} Звезд (XTR)) за 30 дней\n\n"
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
    await message.answer(
        f"{Emoji.STATS} **Статус**\n\n"
        f"ID: `{message.from_user.id}`\n"
        f"Роль: {'админ' if user.is_admin else 'пользователь'}\n"
        f"Триал: {user.trial_remaining} запрос(ов)\n"
        f"Подписка: {sub_text}",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

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

    # Инициализация базы данных
    global db
    db = Database(DB_PATH)
    await db.init()

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
    openai_service = OpenAIService()
    global session_store
    session_store = SessionStore(max_size=USER_SESSIONS_MAX, ttl_seconds=USER_SESSION_TTL_SECONDS)
    global crypto_provider
    crypto_provider = CryptoPayProvider(asset=os.getenv("CRYPTO_ASSET", "USDT"))
    
    # Устанавливаем команды
    await bot.set_my_commands([
        BotCommand(command="start", description=f"{Emoji.ROBOT} Начать работу"),
        BotCommand(command="buy", description=f"{Emoji.MAGIC} Купить подписку"),
        BotCommand(command="status", description=f"{Emoji.STATS} Статус подписки"),
    ])
    
    # Регистрируем обработчики
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_buy, Command("buy"))
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(on_successful_payment, F.successful_payment)
    dp.pre_checkout_query.register(pre_checkout)
    dp.message.register(process_question, F.text & ~F.text.startswith("/"))
    
    # Глобальный обработчик ошибок
    dp.error.register(error_handler)
    
    # Запускаем бота
    logger.info("🤖 ИИ-Иван (простая версия) запущен!")
    logger.info("📊 Анимация статусов: %s", "включена" if USE_ANIMATION else "отключена")
    logger.info("💡 Доступные команды: /start")
    
    try:
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("🤖 ИИ-Иван остановлен пользователем")
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        raise
    finally:
        await bot.session.close()
        if db is not None:
            await db.close()
        if rate_limiter is not None:
            await rate_limiter.close()

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
