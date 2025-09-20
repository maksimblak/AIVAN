"""
Простая версия Telegram бота ИИ-Иван
Только /start и обработка вопросов, никаких кнопок и лишних команд
"""

from __future__ import annotations
import asyncio
import os
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode, ChatAction
from aiogram.types import Message, BotCommand, ErrorEvent
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command

from src.bot.logging_setup import setup_logging
from src.bot.openai_gateway import ask_legal
from src.bot.promt import LEGAL_SYSTEM_PROMPT
from src.bot.ui_components import Emoji, escape_markdown_v2
from src.bot.status_manager import AnimatedStatus, ProgressStatus, ResponseTimer, QuickStatus

# ============ КОНФИГУРАЦИЯ ============

load_dotenv()
setup_logging()
logger = logging.getLogger("ai-ivan.simple")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
USE_ANIMATION = os.getenv("USE_STATUS_ANIMATION", "1").lower() in ("1", "true", "yes")
MAX_MESSAGE_LENGTH = 4000

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

# ============ ФОРМАТИРОВАНИЕ ОТВЕТОВ ============

def format_legal_response(text: str) -> str:
    """Преобразует ответ от OpenAI в красивый маркдаун для Telegram"""
    if not text:
        return text
        
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        
        if not line:
            formatted_lines.append('')
            in_list = False
            continue
            
        # Не экранируем здесь - будем экранировать только нужные части
        # escaped_line = escape_markdown_v2(line)
        
        # Обрабатываем нумерованные пункты (1), 2), 3) и т.д.)
        if re.match(r'^\d+\)\s+', line):
            # Выделяем заголовок жирным и добавляем эмодзи
            parts = line.split(')', 1)
            if len(parts) == 2:
                number = parts[0]
                title = parts[1].strip()
                # Экранируем только специальные символы, но не буквы
                safe_number = escape_markdown_v2(number + ')')
                safe_title = escape_markdown_v2(title)
                formatted_line = f"\n{Emoji.DOCUMENT} **{safe_number}** **{safe_title}**"
                formatted_lines.append(formatted_line)
                in_list = False
                continue
        
        # Обрабатываем маркированные списки (- item)
        if line.startswith('- '):
            content = line[2:].strip()
            
            # Определяем тип пункта по ключевым словам
            emoji = Emoji.SUCCESS
            if any(word in content.lower() for word in ['высок', 'положител', 'благоприят']):
                emoji = Emoji.SUCCESS
            elif any(word in content.lower() for word in ['средн', 'умерен']):
                emoji = Emoji.WARNING  
            elif any(word in content.lower() for word in ['низк', 'отрицател', 'неблагоприят']):
                emoji = Emoji.ERROR
            elif any(word in content.lower() for word in ['шаг', 'этап', 'действи']):
                emoji = Emoji.IDEA
            elif any(word in content.lower() for word in ['документ', 'справк', 'акт']):
                emoji = Emoji.DOCUMENT
            elif any(word in content.lower() for word in ['анализ', 'проверк', 'оценк']):
                emoji = Emoji.SEARCH
            
            # Более аккуратное экранирование для списков
            safe_content = escape_markdown_v2(content)
            formatted_line = f"{emoji} **{safe_content}**"
            formatted_lines.append(formatted_line)
            in_list = True
            continue
        
        # Обрабатываем заголовки разделов (содержат ключевые слова)
        if any(keyword in line.lower() for keyword in [
            'резюме', 'анализ', 'позиция', 'рекомендаци', 'стратеги', 
            'оценка', 'шанс', 'готов подготовить', 'могу:', 'для этого'
        ]):
            # Добавляем эмодзи в зависимости от типа заголовка
            if 'резюме' in line.lower():
                emoji = f"{Emoji.DOCUMENT} "
            elif 'анализ' in line.lower():
                emoji = f"{Emoji.SEARCH} "
            elif 'позиция' in line.lower():
                emoji = f"{Emoji.LAW} "
            elif 'рекомендаци' in line.lower() or 'стратеги' in line.lower():
                emoji = f"{Emoji.IDEA} "
            elif 'оценка' in line.lower() or 'шанс' in line.lower():
                emoji = f"{Emoji.MAGIC} "
            else:
                emoji = f"{Emoji.FIRE} "
                
            formatted_line = f"\n{emoji}**{escape_markdown_v2(line.upper())}**"
            formatted_lines.append(formatted_line)
            in_list = False
            continue
        
        # Обычный текст
        if in_list:
            # Если мы в списке, добавляем небольшой отступ
            formatted_lines.append(f"  _{escape_markdown_v2(line)}_")
        else:
            formatted_lines.append(escape_markdown_v2(line))
    
    return '\n'.join(formatted_lines)

# ============ КОМАНДЫ ============

async def cmd_start(message: Message):
    """Единственная команда - приветствие"""
    user_session = get_user_session(message.from_user.id)
    user_name = message.from_user.first_name or "Пользователь"
    
    # Красивое приветствие с улучшенным дизайном
    welcome_text = f"""╔═══════════════════════════╗
║  {Emoji.LAW} **ИИ\\-Иван** {Emoji.LAW}  ║
╚═══════════════════════════╝

{Emoji.ROBOT} Привет, **{escape_markdown_v2(user_name)}**\\! Добро пожаловать\\!

{Emoji.STAR} **Ваш персональный юридический ассистент**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{Emoji.MAGIC} **Что я умею:**
{Emoji.SEARCH} Анализирую судебную практику РФ
{Emoji.DOCUMENT} Ищу релевантные дела и решения
{Emoji.IDEA} Готовлю черновики процессуальных документов  
{Emoji.LAW} Оцениваю правовые риски и перспективы

{Emoji.FIRE} **Специализации:**
{Emoji.CIVIL} Гражданское и договорное право
{Emoji.CORPORATE} Корпоративное право и M&A
{Emoji.LABOR} Трудовое и административное право
{Emoji.TAX} Налоговое право и споры с ФНС

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{Emoji.IDEA} **Примеры вопросов:**

{Emoji.CONTRACT} *"Можно ли расторгнуть договор поставки за просрочку?"*
{Emoji.LABOR} *"Как правильно уволить сотрудника за нарушения?"*
{Emoji.TAX} *"Какие риски при доначислении НДС?"*
{Emoji.CORPORATE} *"Порядок увеличения уставного капитала ООО"*

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{Emoji.FIRE} **Готов к работе\\! Отправьте ваш правовой вопрос**

{Emoji.WARNING} *Важно: все ответы — аналитические материалы для внутренней проработки и требуют проверки практикующим юристом\\.*"""
    
    await message.answer(welcome_text, parse_mode=ParseMode.MARKDOWN_V2)
    logger.info("User %s started bot", message.from_user.id)

# ============ ОБРАБОТКА ВОПРОСОВ ============

async def _keep_typing(bot: Bot, chat_id: int):
    """Периодически отправляет статус 'печатает' во время обработки запроса"""
    try:
        while True:
            await asyncio.sleep(4)  # Обновляем статус каждые 4 секунды
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
    except asyncio.CancelledError:
        # Задача была отменена - это нормально
        pass
    except Exception as e:
        # Игнорируем ошибки статуса печатания, это не критично
        logger.debug("Error updating typing status: %s", e)

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
            f"""╔═══════════════════════════╗
║  {Emoji.WARNING} **Пустое сообщение** {Emoji.WARNING}  ║  
╚═══════════════════════════╝

Я не получил текст вашего вопроса\\!

{Emoji.FIRE} **Напишите юридический вопрос**, например:

{Emoji.CONTRACT} _"Можно ли расторгнуть договор за просрочку оплаты?"_
{Emoji.LABOR} _"Как правильно оформить увольнение сотрудника?"_  
{Emoji.CORPORATE} _"Какие документы нужны для смены директора ООО?"_
{Emoji.TAX} _"Когда можно применить льготу по налогу на прибыль?"_

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{Emoji.ROBOT} **ИИ\\-Иван** ждёт ваш правовой вопрос\\!""",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    
    # Сразу показываем статус "печатает"
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    
    # Запускаем таймер
    timer = ResponseTimer()
    timer.start()
    
    logger.info("Processing question from user %s: %s", user_id, question_text[:100])
    
    try:
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
            
            # Запускаем задачу периодического обновления статуса "печатает"
            typing_task = asyncio.create_task(_keep_typing(message.bot, message.chat.id))
            
            try:
                # Основной запрос к ИИ
                result = await ask_legal(LEGAL_SYSTEM_PROMPT, question_text)
            finally:
                # Останавливаем периодическое обновление статуса
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
            
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
                f"""╔══════════════════════════╗
║  {Emoji.ERROR} **Ошибка обработки** {Emoji.ERROR}  ║
╚══════════════════════════╝

К сожалению, не удалось обработать ваш запрос\\.

{Emoji.IDEA} **Что можно попробовать:**
{Emoji.SUCCESS} Переформулируйте вопрос более конкретно
{Emoji.SUCCESS} Укажите юрисдикцию и временные рамки  
{Emoji.SUCCESS} Попробуйте через 1\\-2 минуты

{Emoji.HELP} **Пример хорошего вопроса:**
_"Можно ли в Московской области расторгнуть договор аренды коммерческой недвижимости досрочно при задержке платежей на 30 дней?"_

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{Emoji.ROBOT} **ИИ\\-Иван** готов к новому вопросу\\!

_Техническая информация:_ `{error_text[:80]}\\.\\.\\._""",
                parse_mode=ParseMode.MARKDOWN_V2
            )
            return
        
        # Форматируем ответ с красивым маркдауном
        raw_response = result["text"]
        formatted_response = format_legal_response(raw_response)
        
        # Добавляем красивый footer с рамкой
        footer = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{Emoji.ROBOT} **ИИ\\-Иван** \\| {Emoji.CLOCK} _Время ответа: {timer.get_duration_text()}_

{Emoji.WARNING} **Важное напоминание:**
_Данная информация — аналитические материалы для внутренней работы\\. Обязательно требует проверки и анализа практикующим юристом перед применением\\._

{Emoji.STAR} _Для новых вопросов просто отправьте сообщение\\!_"""
        
        response_text = formatted_response + footer
        
        # Показываем статус "печатает" перед отправкой ответа
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        
        # Разбиваем на части и отправляем
        chunks = chunk_text(response_text)
        
        for i, chunk in enumerate(chunks):
            try:
                await message.answer(chunk, parse_mode=ParseMode.MARKDOWN_V2)
            except Exception as e:
                logger.warning("Failed to send with markdown: %s", e)
                # Fallback: отправляем без разметки, но с эмодзи
                clean_chunk = chunk.replace('\\', '').replace('**', '').replace('_', '').replace('`', '')
                await message.answer(clean_chunk)
            
            # Небольшая задержка между сообщениями и статус "печатает" для следующей части
            if i < len(chunks) - 1:
                await asyncio.sleep(0.1)
                await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        
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
            f"""╔══════════════════════════════╗
║  {Emoji.ERROR} **Системная ошибка** {Emoji.ERROR}  ║
╚══════════════════════════════╝

Произошла техническая ошибка при обработке\\. 

{Emoji.IDEA} **Рекомендации:**
{Emoji.SUCCESS} Попробуйте переформулировать вопрос
{Emoji.SUCCESS} Подождите 1\\-2 минуты и повторите
{Emoji.SUCCESS} Упростите формулировку

{Emoji.MAGIC} **Или задайте новый вопрос:**
_"Какие основания для расторжения трудового договора?"_

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{Emoji.ROBOT} **ИИ\\-Иван** остаётся в вашем распоряжении\\!

_Debug:_ `{str(e)[:60]}\\.\\.\\._""",
            parse_mode=ParseMode.MARKDOWN_V2
        )

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
    
    # Устанавливаем ТОЛЬКО команду /start
    await bot.set_my_commands([
        BotCommand(command="start", description=f"{Emoji.ROBOT} Начать работу"),
    ])
    
    # Регистрируем обработчики
    dp.message.register(cmd_start, Command("start"))
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
