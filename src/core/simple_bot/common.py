from __future__ import annotations

from typing import Any

from src.core.simple_bot import context as ctx
from src.core.session_store import UserSession
from src.core.validation import InputValidator
from src.core.exceptions import ValidationException, ErrorContext


__all__ = [
    "ensure_valid_user_id",
    "get_user_session",
    "get_safe_db_method",
]


def ensure_valid_user_id(raw_user_id: int | None, *, context: str) -> int:
    """Validate Telegram user id and raise ValidationException on failure."""
    result = InputValidator.validate_user_id(raw_user_id)
    if result.is_valid and result.cleaned_data:
        return int(result.cleaned_data)

    errors = ", ".join(result.errors or ["Недопустимый идентификатор пользователя"])
    try:
        normalized_user_id = int(raw_user_id) if raw_user_id is not None else None
    except (TypeError, ValueError):
        normalized_user_id = None

    raise ValidationException(
        errors,
        ErrorContext(user_id=normalized_user_id, function_name=context),
    )


def get_user_session(user_id: int) -> UserSession:
    store = ctx.session_store
    if store is None:
        raise RuntimeError("Session store not initialized")
    return store.get_or_create(user_id)


def get_safe_db_method(method_name: str, default_return=None):
    """Return DB coroutine when available."""
    _ = default_return  # backward compatibility with previous signature
    db = ctx.db
    if db is None or not hasattr(db, method_name):
        return None
    return getattr(db, method_name)
