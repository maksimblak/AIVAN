"""
Система онбординга пользователей - показ подсказок о возможностях бота
Каждая подсказка показывается только один раз для провождения пользователя по всем функциям
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.db_advanced import DatabaseAdvanced

logger = logging.getLogger(__name__)


# Упорядоченный список подсказок для онбординга (показываются по очереди)
ONBOARDING_HINTS = [
    {
        "key": "search_practice",
        "text": "💡 <i>Кстати, я могу искать судебную практику по вашему вопросу. Просто спросите: \"Найди практику по взысканию неустойки\"</i>",
        "context": ["text_question", "general"],
    },
    {
        "key": "document_analysis",
        "text": "📄 <i>Знаете ли вы, что я могу проанализировать ваши документы? Загрузите договор, иск или решение суда</i>",
        "context": ["text_question", "after_search", "general"],
    },
    {
        "key": "risk_analysis",
        "text": "⚠️ <i>А вы знали? Я могу проверить договор на юридические риски и опасные формулировки</i>",
        "context": ["text_question", "after_doc_analysis", "after_summary", "general"],
    },
    {
        "key": "lawsuit_analysis",
        "text": "⚖️ <i>Знаете ли вы, что я анализирую исковые заявления? Оценю позицию, найду слабые места и дам рекомендации</i>",
        "context": ["text_question", "after_doc_analysis", "after_risk", "general"],
    },
    {
        "key": "doc_generation",
        "text": "✨ <i>Кстати, я могу составить юридический документ за вас. Попробуйте функцию \"Создание документа\"</i>",
        "context": ["text_question", "after_search", "after_risk", "general"],
    },
    {
        "key": "anonymize",
        "text": "🔒 <i>Знаете ли вы, что я могу обезличить документ? Скрою все персональные данные и ФИО</i>",
        "context": ["after_doc_analysis", "general"],
    },
    {
        "key": "ocr",
        "text": "📷 <i>Кстати, я могу распознать текст со сканов и фотографий. Просто отправьте изображение</i>",
        "context": ["after_summary", "general"],
    },
    {
        "key": "help_command",
        "text": "🎯 <i>Посмотрите все мои возможности в разделе \"Работа с документами\" или нажмите /start</i>",
        "context": ["general"],
    },
]


async def get_next_onboarding_hint(
    db: DatabaseAdvanced | None,
    user_id: int,
    context: str = "general",
) -> tuple[str | None, str | None]:
    """
    Получить следующую подсказку онбординга для пользователя

    Args:
        db: База данных
        user_id: ID пользователя
        context: Текущий контекст ('text_question', 'after_search', и т.д.)

    Returns:
        Кортеж (текст подсказки, ключ подсказки) или (None, None) если все подсказки показаны
    """
    if db is None:
        return None, None

    try:
        # Получаем список уже показанных подсказок
        shown_hints = await db.get_shown_hints(user_id)

        # Ищем первую неподказанную подсказку, подходящую по контексту
        for hint in ONBOARDING_HINTS:
            hint_key = hint["key"]

            # Если подсказка уже показана - пропускаем
            if hint_key in shown_hints:
                continue

            # Проверяем, подходит ли подсказка по контексту
            if context in hint["context"]:
                return hint["text"], hint_key

        # Если не нашли подходящую по контексту, ищем любую неподказанную с контекстом "general"
        for hint in ONBOARDING_HINTS:
            hint_key = hint["key"]
            if hint_key not in shown_hints and "general" in hint["context"]:
                return hint["text"], hint_key

        # Все подсказки показаны
        return None, None

    except Exception as e:
        logger.error(f"Error getting next onboarding hint: {e}")
        return None, None


async def mark_hint_as_shown(
    db: DatabaseAdvanced | None,
    user_id: int,
    hint_key: str,
) -> None:
    """
    Пометить подсказку как показанную

    Args:
        db: База данных
        user_id: ID пользователя
        hint_key: Ключ подсказки
    """
    if db is None:
        return

    try:
        await db.mark_hint_shown(user_id, hint_key)
        logger.debug(f"Marked hint '{hint_key}' as shown for user {user_id}")
    except Exception as e:
        logger.error(f"Error marking hint as shown: {e}")


async def should_show_hint(
    db: DatabaseAdvanced | None,
    user_id: int,
) -> bool:
    """
    Определить, нужно ли показать подсказку

    Args:
        db: База данных
        user_id: ID пользователя

    Returns:
        True, если пора показать подсказку
    """
    if db is None:
        return False

    try:
        # Получаем пользователя
        user = await db.get_user(user_id)
        if not user:
            return False

        # Получаем количество запросов
        total_requests = getattr(user, "total_requests", 0)

        # Не показываем подсказки на первом запросе
        if total_requests < 1:
            return False

        # Получаем список показанных подсказок
        shown_hints = await db.get_shown_hints(user_id)

        # Если все подсказки показаны - больше не показываем
        if len(shown_hints) >= len(ONBOARDING_HINTS):
            return False

        # Показываем подсказку примерно каждые 2-3 запроса
        # (чтобы не спамить, но и провести по всем функциям)
        requests_per_hint = 2
        expected_hints_count = total_requests // requests_per_hint

        # Если пользователь сделал достаточно запросов для новой подсказки
        return len(shown_hints) < expected_hints_count

    except Exception as e:
        logger.error(f"Error checking if should show hint: {e}")
        return False


async def get_onboarding_hint(
    db: DatabaseAdvanced | None,
    user_id: int,
    context: str = "general",
) -> str | None:
    """
    Главная функция для получения подсказки онбординга

    Args:
        db: База данных
        user_id: ID пользователя
        context: Текущий контекст

    Returns:
        Текст подсказки или None, если не нужно показывать
    """
    if db is None:
        return None

    try:
        # Проверяем, нужно ли показывать подсказку
        if not await should_show_hint(db, user_id):
            return None

        # Получаем следующую подсказку
        hint_text, hint_key = await get_next_onboarding_hint(db, user_id, context)

        if hint_text and hint_key:
            # Помечаем подсказку как показанную
            await mark_hint_as_shown(db, user_id, hint_key)
            return hint_text

        return None

    except Exception as e:
        logger.error(f"Error in get_onboarding_hint: {e}")
        return None


# Готовые подсказки для разных ситуаций
QUICK_HINTS = {
    "first_message": "💡 <i>Попробуйте описать вашу юридическую ситуацию — я помогу найти решение</i>",
    "document_uploaded": "💡 <i>Я могу сделать краткую выжимку, найти риски или проанализировать документ</i>",
    "long_text": "📄 <i>Длинный текст? Попробуйте загрузить его файлом — я смогу лучше его проанализировать</i>",
}


__all__ = [
    "get_onboarding_hint",
    "get_next_onboarding_hint",
    "mark_hint_as_shown",
    "should_show_hint",
    "ONBOARDING_HINTS",
    "QUICK_HINTS",
]
