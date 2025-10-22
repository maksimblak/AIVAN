from __future__ import annotations

import logging

from aiogram import Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.types import CallbackQuery

from core.bot_app.ui_components import Emoji

__all__ = [
    "register_retention_handlers",
]

logger = logging.getLogger("ai-ivan.simple.retention")


async def handle_retention_quick_question(callback: CallbackQuery) -> None:
    """Handle 'quick_question' retention button."""
    try:
        await callback.answer()
        await callback.message.answer(
            f"{Emoji.ROBOT} <b>Отлично!</b>\n\n"
            "Просто напиши свой вопрос, и я отвечу на него.\n\n"
            f"{Emoji.INFO} <i>Пример:</i> Что делать, если нарушили права потребителя?",
            parse_mode=ParseMode.HTML,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in handle_retention_quick_question: %s", exc, exc_info=True)


async def handle_retention_show_features(callback: CallbackQuery) -> None:
    """Handle 'show_features' retention button."""
    try:
        await callback.answer()

        features_text = (
            f"{Emoji.ROBOT} <b>Что я умею:</b>\n\n"
            f"{Emoji.QUESTION} <b>Юридические консультации</b>\n"
            "Отвечаю на вопросы по правовым темам\n\n"
            "📄 <b>Работа с документами</b>\n"
            "• Анализ договоров и документов\n"
            "• Поиск рисков и проблем\n"
            "• Режим «распознавание текста» — извлечение текста из фото\n"
            "• Составление документов\n\n"
            "📚 <b>Судебная практика</b>\n"
            "Поиск релевантных судебных решений\n\n"
            f"{Emoji.MICROPHONE} <b>Голосовые сообщения</b>\n"
            "Отправь голосовое — получишь голосовой ответ\n\n"
            f"{Emoji.INFO} Просто напиши вопрос или выбери действие!"
        )

        await callback.message.answer(features_text, parse_mode=ParseMode.HTML)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in handle_retention_show_features: %s", exc, exc_info=True)


def register_retention_handlers(dp: Dispatcher) -> None:
    """Register retention-related handlers."""
    dp.callback_query.register(handle_retention_quick_question, F.data == "quick_question")
    dp.callback_query.register(handle_retention_show_features, F.data == "show_features")
