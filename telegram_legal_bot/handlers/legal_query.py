"""Обработчик юридических запросов."""
from __future__ import annotations

import logging
from typing import Dict, List

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from telegram_legal_bot.config import Settings
from telegram_legal_bot.services.openai_service import LegalAdvice, OpenAIService
from telegram_legal_bot.utils.message_formatter import format_legal_response
from telegram_legal_bot.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class LegalQueryHandler:
    """Инкапсулирует обработку юридических запросов."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._openai_service = OpenAIService(settings.openai)
        self._rate_limiter = RateLimiter(settings.bot.max_requests_per_hour, period_seconds=3600)
        self._history: Dict[int, List[str]] = {}

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Основной обработчик текстовых сообщений."""

        if not update.effective_message or not update.effective_user:
            return

        user_id = update.effective_user.id
        message_text = update.effective_message.text or ""

        if not await self._validate_message(user_id, message_text, update, context):
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        try:
            advice = await self._openai_service.generate_legal_advice(message_text)
            await self._remember_history(user_id, message_text)
            await self._send_response(update, advice)
        except Exception:  # noqa: BLE001 - хотим логировать любую ошибку
            logger.exception("Ошибка при обработке сообщения")
            await update.effective_message.reply_text(
                "⚠️ Произошла ошибка при получении консультации. Попробуйте повторить запрос позже."
            )

    async def _validate_message(
        self,
        user_id: int,
        message_text: str,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> bool:
        """Проверяет корректность запроса."""

        if not message_text.strip():
            await update.effective_message.reply_text("Пожалуйста, отправьте текстовое сообщение с юридическим вопросом.")
            return False

        if len(message_text.strip()) < self._settings.bot.min_question_length:
            await update.effective_message.reply_text(
                "Опишите ситуацию подробнее, чтобы я смог предоставить качественную консультацию."
            )
            return False

        if self._is_spam(message_text):
            await update.effective_message.reply_text("Похоже на спам. Пожалуйста, сформулируйте юридический вопрос.")
            return False

        if not await self._rate_limiter.check(user_id):
            remaining = await self._rate_limiter.remaining(user_id)
            await update.effective_message.reply_text(
                "⌛ Вы исчерпали лимит консультаций. Попробуйте снова позже. "
                f"Остаток доступных запросов: {remaining}."
            )
            return False

        if context.application is not None:
            history = self._history.get(user_id)
            if history:
                logger.debug("История пользователя %s: %s", user_id, history[-3:])

        return True

    @staticmethod
    def _is_spam(message_text: str) -> bool:
        """Простейшая проверка на однообразный текст."""

        stripped = message_text.strip()
        if not stripped:
            return True
        unique_chars = {char for char in stripped.lower() if char.isalpha()}
        return len(unique_chars) <= 2 and len(stripped) > 10

    async def _remember_history(self, user_id: int, message_text: str) -> None:
        """Сохраняет последние вопросы пользователя."""

        history = self._history.setdefault(user_id, [])
        history.append(message_text)
        if len(history) > 5:
            self._history[user_id] = history[-5:]

    async def _send_response(self, update: Update, advice: LegalAdvice) -> None:
        """Отправляет пользователю красиво оформленный ответ."""

        formatted_message = format_legal_response(advice.summary, advice.details, advice.laws)
        await update.effective_message.reply_text(formatted_message, parse_mode=ParseMode.MARKDOWN_V2)
