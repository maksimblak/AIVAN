"""Обработчики стартовых команд."""
from __future__ import annotations

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

WELCOME_MESSAGE = (
    "⚖️ *Добро пожаловать в LegalBot!*\n\n"
    "Я помогу вам получить *информационные юридические консультации*.\n"
    "Отправьте мне вопрос, описав ситуацию максимально подробно.\n\n"
    "_Как использовать:_\n"
    "1. Сформулируйте вопрос и отправьте его в чат.\n"
    "2. Подождите, пока я подготовлю ответ.\n"
    "3. Ознакомьтесь с консультацией и при необходимости уточните детали._\n\n"
    "Используйте /help, чтобы узнать о дополнительных возможностях."
)

HELP_MESSAGE = (
    "ℹ️ *Справка по LegalBot*\n\n"
    "• Задавайте юридические вопросы текстом.\n"
    "• Я предоставляю краткий вывод, подробное разъяснение и применимые нормы права.\n"
    "• Ограничение: не более 10 запросов в час на пользователя (по умолчанию).\n"
    "• Пожалуйста, не отправляйте персональные данные.\n"
    "• Консультация носит информационный характер и не заменяет работу юриста."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение."""

    await update.effective_message.reply_text(WELCOME_MESSAGE, parse_mode=ParseMode.MARKDOWN_V2)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет справочную информацию."""

    await update.effective_message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.MARKDOWN_V2)
