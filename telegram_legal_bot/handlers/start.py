from __future__ import annotations

from aiogram import Router, types
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart

from telegram_legal_bot.utils.message_formatter import md2

router = Router()


def _final_sanitize_md2(text: str) -> str:
    """
    Доп. страховка для MarkdownV2: экранируем дефис и скобки в статике.
    Динамику мы экранируем md2().
    """
    return (
        text.replace("-", r"\-")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


@router.message(CommandStart())
async def cmd_start(message: types.Message) -> None:
    text = (
        f"👋 *{md2('Привет!')}*\n\n"
        f"Я — бот юридических консультаций на базе GPT-5.\n\n"
        f"Что умею:\n"
        f"• {md2('Отвечать на юридические вопросы')} ⚖️\n"
        f"• {md2('Структурировать ответы')} — кратко, подробно, нормы права\n"
        f"• {md2('Форматировать ответ')} — эмодзи, списки, MarkdownV2\n\n"
        f"Как пользоваться:\n"
        f"1\\. {md2('Отправьте текстовый вопрос')} — например: _{md2('Как расторгнуть договор аренды?')}_\n"
        f"2\\. {md2('Подождите — я покажу статус печати')}\n"
        f"3\\. {md2('Получите оформленный ответ с нормами права')} 📚\n\n"
        f"⚠️ *{md2('Важно')}*: {md2('консультация носит информационный характер и не заменяет профессиональную юридическую помощь.')}"
    )

    safe_text = _final_sanitize_md2(text)

    try:
        await message.answer(
            safe_text,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True,
        )
    except TelegramBadRequest:
        # ВАЖНО: переопределяем parse_mode, иначе у бота глобально стоит MARKDOWN_V2
        plain = safe_text.replace("\\", "").replace("*", "").replace("_", "")
        await message.answer(plain, parse_mode=None, disable_web_page_preview=True)


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    text = (
        f"🆘 *{md2('Помощь')}:*\n\n"
        f"• {md2('Пишите конкретно и по делу — так точнее.')}\n"
        f"• {md2('Есть ограничение запросов в час (антиспам).')}\n"
        f"• {md2('Длинные ответы я режу автоматически.')}\n\n"
        f"{md2('Команды')}:\n"
        f"/start — {md2('приветствие')}\n"
        f"/help — {md2('помощь')}\n"
    )
    safe_text = _final_sanitize_md2(text)
    try:
        await message.answer(
            safe_text,
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=True,
        )
    except TelegramBadRequest:
        plain = safe_text.replace("\\", "").replace("*", "").replace("_", "")
        await message.answer(plain, parse_mode=None, disable_web_page_preview=True)
