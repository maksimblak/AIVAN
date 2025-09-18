# Минимальный __init__, чтобы не создавать циклические импорты.
# Если хочешь "удобный" импорт — делаем реэкспорт только нужной фабрики.

from .legal_query import build_legal_message_handler

__all__ = ["build_legal_message_handler"]
