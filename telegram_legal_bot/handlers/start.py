from __future__ import annotations

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /start — приветствие, краткое описание возможностей и инструкция.
    """
    md2 = lambda t: escape_markdown(t, version=2)  # noqa: E731

    text = (
        f"👋 *{md2('Привет!')}\n*\n"
        f"Я — бот юридических консультаций на базе GPT.\n\n"
        f"Что умею:\n"
        f"• {md2('Отвечать на юридические вопросы')} (⚖️)\n"
        f"• {md2('Структурировать ответы')} — кратко, подробно, нормы права\n"
        f"• {md2('Форматировать ответ')} для удобного чтения (эмодзи, списки)\n\n"
        f"Как пользоваться:\n"
        f"1\\. {md2('Отправьте текстовый вопрос')} — например: _{md2('Как расторгнуть договор аренды?')}_\n"
        f"2\\. Подождите пару секунд — я напишу 🟡 *{md2('печатает')}*\n"
        f"3\\. Получите оформленный ответ с нормами права 📚\n\n"
        f"⚠️ *{md2('Важно')}*: консультация носит информационный характер и "
        f"не заменяет профессиональную юридическую помощь."
    )

    await update.effective_message.reply_text(
        text, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /help — краткая справка по функционалу и ограничениям.
    """
    md2 = lambda t: escape_markdown(t, version=2)  # noqa: E731

    text = (
        f"🆘 *{md2('Помощь')}:*\n\n"
        f"• Спрашивайте одним сообщением — я разберу суть и сформирую ответ.\n"
        f"• Пишите конкретно: так я дам более точную консультацию.\n"
        f"• Лимит: {md2('несколько запросов в час')} per user (антиспам).\n"
        f"• Длина сообщения ограничена Telegram; длинные ответы я разрезаю автоматически.\n\n"
        f"Команды:\n"
        f"/start — приветствие\n"
        f"/help — помощь\n"
    )
    await update.effective_message.reply_text(
        text, parse_mode=ParseMode.MARKDOWN_V2, disable_web_page_preview=True
    )
