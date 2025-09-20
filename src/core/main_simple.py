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

# ============ КОМАНДЫ ============

async def cmd_start(message: Message):
    """Единственная команда - приветствие"""
    user_session = get_user_session(message.from_user.id)
    user_name = message.from_user.first_name or "Пользователь"
    
    welcome_text = f"""{Emoji.ROBOT} Привет, **{escape_markdown_v2(user_name)}**\\!

{Emoji.LAW} **ИИ\\-Иван** — ваш юридический ассистент

{Emoji.ROBOT} Специализируюсь на российском праве и судебной практике
{Emoji.SEARCH} Анализирую дела, нахожу релевантную практику  
{Emoji.DOCUMENT} Готовлю черновики процессуальных документов

{Emoji.WARNING} *Важно*: все ответы требуют проверки юристом

{Emoji.FIRE} **Просто отправьте мне ваш юридический вопрос\\!**"""
    
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
