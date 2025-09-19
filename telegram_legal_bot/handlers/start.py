from __future__ import annotations

from aiogram import Router, types, F
from aiogram.filters import Command

from telegram_legal_bot.utils.message_formatter import md2
from telegram_legal_bot.ui.messages import BotMessages
from telegram_legal_bot.ui.animations import BotAnimations

router = Router(name="start")


@router.message(Command("start"))
async def cmd_start(msg: types.Message) -> None:
    """Приветствие и просьба сразу задать вопрос без кнопок."""
    try:
        # Получаем имя пользователя
        user_name = msg.from_user.first_name if msg.from_user else None
        
        # Показываем анимацию печати
        await BotAnimations.typing_animation(msg, 2.0)
        
        # Отправляем приветственное сообщение и просьбу задать вопрос
        welcome_text = BotMessages.welcome_with_bold(user_name)
        hint = "\n\nНапишите ваш вопрос одним сообщением — я отвечу сразу."
        await msg.answer(welcome_text + hint, parse_mode="HTML")
    except Exception:
        # Fallback без форматирования
        fallback_text = f"""
👋 Добро пожаловать!

Я — AIVAN, ваш юридический ассистент ⚖️

Что я умею:
• Отвечаю на правовые вопросы по РФ  
• Привожу ссылки на нормы
• Даю практические рекомендации

⚠️ Важно: я не заменяю юриста, но помогу сориентироваться!
        """.strip()
        
        await msg.answer(fallback_text, parse_mode=None)


@router.message(Command("help"))  
async def cmd_help(msg: types.Message) -> None:
    """Краткая помощь без кнопок."""
    try:
        help_text = BotMessages.help_main()
        await msg.answer(help_text, parse_mode="HTML")
    except Exception:
        # Fallback
        fallback_text = """
ℹ️ СПРАВКА ПО ИСПОЛЬЗОВАНИЮ

❓ Как задать вопрос:
1. Опишите ситуацию подробно
2. Укажите важные детали
3. Упомяните документы и сроки

🔥 Советы:
• Пишите конкретно
• Указывайте суммы, даты  
• Избегайте общих формулировок

📚 Готов помочь с любыми вопросами права!
        """.strip()
        
        await msg.answer(fallback_text, parse_mode=None)

