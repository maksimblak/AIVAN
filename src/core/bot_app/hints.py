"""
Система контекстных подсказок о возможностях бота
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.db_advanced import DatabaseAdvanced


# Подсказки по категориям
HINTS = {
    "search_practice": [
        "💡 <i>Кстати, я могу искать судебную практику по вашему вопросу. Просто спросите: \"Найди практику по взысканию неустойки\"</i>",
        "🔍 <i>А вы знали? Я анализирую миллионы судебных решений и могу найти релевантную практику для вашего дела</i>",
        "⚖️ <i>Подсказка: Попробуйте спросить меня о судебной практике по любому юридическому вопросу</i>",
    ],
    "document_analysis": [
        "📄 <i>Знаете ли вы, что я могу проанализировать ваши документы? Загрузите договор, иск или решение суда</i>",
        "📋 <i>Кстати, я умею делать краткую выжимку из больших документов. Загрузите файл и выберите нужную операцию</i>",
        "🔍 <i>Подсказка: Я могу найти риски в договоре или проанализировать исковое заявление</i>",
    ],
    "risk_analysis": [
        "⚠️ <i>А вы знали? Я могу проверить договор на юридические риски и опасные формулировки</i>",
        "🛡️ <i>Кстати, перед подписанием договора я могу найти в нём скрытые риски и проблемные условия</i>",
    ],
    "lawsuit_analysis": [
        "⚖️ <i>Знаете ли вы, что я анализирую исковые заявления? Оценю позицию, найду слабые места и дам рекомендации</i>",
        "📊 <i>Подсказка: Загрузите исковое заявление, и я помогу оценить шансы на успех</i>",
    ],
    "doc_generation": [
        "✨ <i>Кстати, я могу составить юридический документ за вас. Попробуйте функцию \"Создание документа\"</i>",
        "📝 <i>А вы знали? Я могу подготовить иск, жалобу или договор по вашему описанию ситуации</i>",
    ],
    "anonymize": [
        "🔒 <i>Знаете ли вы, что я могу обезличить документ? Скрою все персональные данные и ФИО</i>",
        "🕶️ <i>Подсказка: Перед отправкой документа третьим лицам используйте функцию обезличивания</i>",
    ],
    "ocr": [
        "📷 <i>Кстати, я могу распознать текст со сканов и фотографий. Просто отправьте изображение</i>",
        "🔍 <i>А вы знали? Я извлекаю текст даже из плохих фотографий документов</i>",
    ],
    "general": [
        "🎯 <i>Посмотрите все мои возможности в разделе \"Работа с документами\" или нажмите /start</i>",
        "💬 <i>Нужна помощь? Просто опишите вашу задачу, и я подскажу, как лучше её решить</i>",
    ],
}


# Карта контекстов: когда какую подсказку показывать
CONTEXT_MAP = {
    "text_question": ["search_practice", "document_analysis", "general"],
    "after_search": ["document_analysis", "doc_generation"],
    "after_doc_analysis": ["risk_analysis", "lawsuit_analysis", "anonymize"],
    "after_summary": ["risk_analysis", "lawsuit_analysis", "ocr"],
    "after_risk": ["doc_generation", "lawsuit_analysis"],
    "general": ["search_practice", "document_analysis", "doc_generation", "general"],
}


def get_hint(context: str = "general", exclude_features: list[str] | None = None) -> str | None:
    """
    Получить контекстную подсказку

    Args:
        context: Контекст использования ('text_question', 'after_search', и т.д.)
        exclude_features: Список уже использованных функций (не показываем подсказки о них)

    Returns:
        Текст подсказки или None
    """
    exclude_features = exclude_features or []

    # Получаем список категорий для данного контекста
    categories = CONTEXT_MAP.get(context, CONTEXT_MAP["general"])

    # Фильтруем уже использованные функции
    available_categories = [cat for cat in categories if cat not in exclude_features]

    if not available_categories:
        # Если все уже использовали, показываем общую подсказку
        available_categories = ["general"]

    # Выбираем случайную категорию
    category = random.choice(available_categories)

    # Выбираем случайную подсказку из категории
    hints = HINTS.get(category, HINTS["general"])
    return random.choice(hints)


def should_show_hint(user_message_count: int, last_hint_count: int) -> bool:
    """
    Определить, нужно ли показать подсказку

    Args:
        user_message_count: Общее количество сообщений пользователя
        last_hint_count: Номер сообщения, когда была показана последняя подсказка

    Returns:
        True, если пора показать подсказку
    """
    # Не показываем подсказку на первых 2 сообщениях
    if user_message_count < 2:
        return False

    # Показываем подсказку каждые 3-5 сообщений
    messages_since_hint = user_message_count - last_hint_count

    if messages_since_hint >= 5:
        return True

    if messages_since_hint >= 3:
        # Случайно с вероятностью 40%
        return random.random() < 0.4

    return False


async def get_user_used_features(db: DatabaseAdvanced, user_id: int) -> list[str]:
    """
    Получить список использованных пользователем функций

    Args:
        db: База данных
        user_id: ID пользователя

    Returns:
        Список использованных функций
    """
    try:
        # Получаем статистику использования функций
        stats = await db.get_user_statistics(user_id, days=365)

        used_features = []

        # Проверяем, какие функции использовались
        feature_mapping = {
            "summarize": "document_analysis",
            "analyze_risks": "risk_analysis",
            "lawsuit_analysis": "lawsuit_analysis",
            "anonymize": "anonymize",
            "ocr": "ocr",
            "translate": "document_analysis",
            "chat": "document_analysis",
        }

        for feature_key, category in feature_mapping.items():
            feature_count = stats.get(f"{feature_key}_count", 0)
            if feature_count > 0 and category not in used_features:
                used_features.append(category)

        # Проверяем, использовался ли поиск практики
        # (можно добавить проверку через ключевые слова в истории запросов)

        return used_features
    except Exception:
        return []


async def get_contextual_hint(
    db: DatabaseAdvanced | None,
    user_id: int,
    context: str = "general",
) -> str | None:
    """
    Получить контекстную подсказку с учётом истории пользователя

    Args:
        db: База данных
        user_id: ID пользователя
        context: Текущий контекст

    Returns:
        Текст подсказки или None, если не нужно показывать
    """
    if db is None:
        # Без БД показываем случайные подсказки
        if random.random() < 0.3:  # 30% вероятность
            return get_hint(context)
        return None

    try:
        # Получаем пользователя
        user = await db.get_user(user_id)
        if not user:
            return None

        # Получаем количество сообщений
        user_message_count = getattr(user, "total_requests", 0)
        # Используем updated_at как метку последней подсказки для простоты
        last_hint_count = 0  # Упрощённая версия

        # Проверяем, нужно ли показывать подсказку
        if not should_show_hint(user_message_count, last_hint_count):
            return None

        # Получаем использованные функции
        used_features = await get_user_used_features(db, user_id)

        # Получаем подсказку
        hint = get_hint(context, used_features)

        return hint
    except Exception:
        # В случае ошибки не показываем подсказку
        return None


# Готовые подсказки для разных ситуаций
QUICK_HINTS = {
    "first_message": "💡 <i>Попробуйте описать вашу юридическую ситуацию — я помогу найти решение</i>",
    "document_uploaded": "💡 <i>Я могу сделать краткую выжимку, найти риски или проанализировать документ</i>",
    "long_text": "📄 <i>Длинный текст? Попробуйте загрузить его файлом — я смогу лучше его проанализировать</i>",
}


__all__ = [
    "get_hint",
    "should_show_hint",
    "get_contextual_hint",
    "get_user_used_features",
    "QUICK_HINTS",
    "HINTS",
    "CONTEXT_MAP",
]
