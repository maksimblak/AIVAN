"""Пакет обработчиков Telegram-бота."""

from telegram_legal_bot.handlers.legal_query import LegalQueryHandler
from telegram_legal_bot.handlers.start import help_command, start

__all__ = ["LegalQueryHandler", "help_command", "start"]
