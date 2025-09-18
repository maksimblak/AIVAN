from __future__ import annotations
from aiogram import Router, types
from aiogram.filters import Command, CommandStart
from aiogram.enums import ParseMode

from telegram_legal_bot.utils.message_formatter import md2

router = Router()

@router.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    text = (
        f"👋 *{md2('Привет!')}*\n\n"
        f"Я — бот юридических консультаций на базе GPT-5.\n\n"
        f"Что умею:\n"
        f"• {md2('Отвечать на юридические вопросы')} (⚖️)\n"
        f"• {md2('Структурировать ответы')} — кратко, подробно, нормы права\n"
        f"• {md2('Форматировать ответ')} (эмодзи, списки, MarkdownV2)\n\n"
        f"Как пользоваться:\n"
        f"1\\. {md2('Отправьте текстовый вопрос')} — например: _{md2('Как расторгнуть договор аренды?')}_\n"
        f"2\\. Подождите — я покажу статус печати\n"
        f"3\\. Получите оформленный ответ с нормами права 📚\n\n"
        f"⚠️ *{md2('Важно')}*: консультация носит информационный характер и "
        f"не заменяет профессиональную юридическую помощь."
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True)

@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    text = (
        f"🆘 *{md2('Помощь')}:*\n\n"
        f"• Пишите конкретно и по делу — так точнее.\n"
        f"• Есть ограничение запросов в час (антиспам).\n"
        f"• Длинные ответы режу автоматически.\n\n"
        f"Команды:\n"
        f"/start — приветствие\n"
        f"/help — помощь\n"
    )
    await message.answer(text, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True)
