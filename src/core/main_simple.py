"""
Простая версия Telegram бота ИИ-Иван
Только /start и обработка вопросов, никаких кнопок и лишних команд
"""

from __future__ import annotations
import asyncio
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

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
from src.bot.status_manager import AnimatedStatus, ProgressStatus, ResponseTimer, QuickStatus
from src.core.db import Database
from src.core.crypto_pay import create_crypto_invoice

# ============ КОНФИГУРАЦИЯ ============

load_dotenv()
setup_logging()
logger = logging.getLogger("ai-ivan.simple")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
USE_ANIMATION = os.getenv("USE_STATUS_ANIMATION", "1").lower() in ("1", "true", "yes")
MAX_MESSAGE_LENGTH = 4000

# Подписки и платежи
DB_PATH = os.getenv("DB_PATH", "data/bot.sqlite3")
TRIAL_REQUESTS = int(os.getenv("TRIAL_REQUESTS", "10"))
SUB_DURATION_DAYS = int(os.getenv("SUB_DURATION_DAYS", "30"))

# RUB платеж через Telegram Payments (провайдер-эквайринг)
RUB_PROVIDER_TOKEN = os.getenv("TELEGRAM_PROVIDER_TOKEN_RUB", "").strip()
SUB_PRICE_RUB = int(os.getenv("SUBSCRIPTION_PRICE_RUB", "300"))  # руб.
SUB_PRICE_RUB_KOPEKS = SUB_PRICE_RUB * 100

# Telegram Stars (XTR). В большинстве случаев используется специальный токен "STARS"
STARS_PROVIDER_TOKEN = os.getenv("TELEGRAM_PROVIDER_TOKEN_STARS", "STARS").strip()
SUB_PRICE_XTR = int(os.getenv("SUBSCRIPTION_PRICE_XTR", "3000"))  # XTR

# Админы (через запятую Telegram user_id)
ADMIN_IDS = {int(x) for x in os.getenv("ADMIN_IDS", "").replace(" ", "").split(',') if x}

# Глобальная БД
db: Optional[Database] = None

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

user_sessions: Dict[int, UserSession] = {}

def get_user_session(user_id: int) -> UserSession:
    """Получить или создать сессию пользователя"""
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    return user_sessions[user_id]

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

⚠️ Важно: все ответы — аналитические материалы для внутренней проработки и требуют проверки практикующим юристом.
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
        # Контроль доступа: админ или активная подписка, иначе расходуем триал
        if db is not None:
            user = await db.ensure_user(user_id, default_trial=TRIAL_REQUESTS, is_admin=user_id in ADMIN_IDS)
            has_access = False
            if user.is_admin:
                has_access = True
            else:
                if await db.has_active_subscription(user_id):
                    has_access = True
                else:
                    # Пытаемся списать один бесплатный запрос
                    if await db.decrement_trial(user_id):
                        has_access = True
            if not has_access:
                await message.answer(
                    f"{Emoji.WARNING} **Лимит бесплатных запросов исчерпан**\n\nОформите подписку за {SUB_PRICE_RUB}₽ в месяц. Используйте команду /buy",
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
                return
        # Показываем статус
        if USE_ANIMATION:
            status = AnimatedStatus(message.bot, message.chat.id)
            await status.start()
        else:
            status = ProgressStatus(message.bot, message.chat.id)
            await status.start("Обрабатываю ваш вопрос\\.\\.\\.")
        
        try:
            # Имитируем этапы обработки для лучшего UX
            if not USE_ANIMATION and hasattr(status, 'update_stage'):
                await asyncio.sleep(0.5)
                await status.update_stage(1, f"{Emoji.SEARCH} Анализирую ваш вопрос\\.\\.\\.")
                await asyncio.sleep(1)
                await status.update_stage(2, f"{Emoji.LOADING} Ищу релевантную судебную практику\\.\\.\\.")
            
            # Основной запрос к ИИ
            result = await ask_legal(LEGAL_SYSTEM_PROMPT, question_text)
            
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
    prices = [LabeledPrice(label="Подписка на 30 дней", amount=SUB_PRICE_XTR)]
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
    text = (
        f"{Emoji.MAGIC} **Оплата подписки**\n\n"
        f"Стоимость: {SUB_PRICE_RUB}₽ / 30 дней\n\n"
        f"Выберите способ оплаты:" 
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN_V2)
    # Отправляем доступные варианты
    if RUB_PROVIDER_TOKEN:
        await send_rub_invoice(message)
    await send_stars_invoice(message)
    # Крипта: создаем инвойс через CryptoBot, если настроен токен
    payload = _build_payload("crypto", message.from_user.id)
    inv = create_crypto_invoice(
        amount=float(SUB_PRICE_RUB),  # можно привязать к USDT с пересчетом, для простоты — число
        asset=os.getenv("CRYPTO_ASSET", "USDT"),
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
    await pre.answer(ok=True)

async def on_successful_payment(message: Message):
    try:
        sp = message.successful_payment
        if sp is None:
            return
        method = 'rub' if sp.currency.upper() == 'RUB' else ('xtr' if sp.currency.upper() == 'XTR' else sp.currency)
        if db is not None:
            # Запись транзакции и продление подписки
            await db.record_transaction(
                user_id=message.from_user.id,
                provider=f"telegram_{method}",
                currency=sp.currency,
                amount=sp.total_amount,
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
