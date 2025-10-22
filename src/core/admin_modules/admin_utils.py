"""
Shared utilities для admin commands
"""

import inspect
import contextlib
import logging

from functools import wraps
from typing import Any, Callable
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup


logger = logging.getLogger(__name__)

_GLOBAL_ADMIN_IDS: set[int] | None = None


FEATURE_KEYS = [
    "legal_question",
    "document_upload",
    "voice_message",
    "document_summary",
    "contract_analysis",
]

def require_admin(func):
    """
    Decorator для проверки admin доступа

    Использование:
        @require_admin
        async def cmd_example(message: Message, db, admin_ids: list[int]):
            ...
    """
    signature = inspect.signature(func)
    try:
        target_param = next(iter(signature.parameters.values()))
    except StopIteration as exc:
        raise RuntimeError("require_admin expects a handler with at least one positional argument") from exc

    target_name = target_param.name

    @wraps(func)
    async def wrapper(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        target = bound.arguments.get(target_name)
        admin_ids_arg = bound.arguments.get('admin_ids')
        global _GLOBAL_ADMIN_IDS

        if target is None:
            raise RuntimeError("require_admin decorator requires a target argument (Message or CallbackQuery)")

        if admin_ids_arg is None:
            admin_ids_arg = _GLOBAL_ADMIN_IDS

        if admin_ids_arg is None:
            raise RuntimeError("require_admin decorator requires an 'admin_ids' argument")

        user = getattr(target, 'from_user', None)
        user_id = getattr(user, 'id', None)

        if user_id is None:
            if isinstance(target, CallbackQuery):
                await target.answer("⚠️ Доступ запрещен", show_alert=True)
            else:
                await target.answer("⚠️ Доступ запрещен")
            return None

        container = admin_ids_arg if hasattr(admin_ids_arg, '__contains__') else set(admin_ids_arg)

        if user_id not in container:
            if isinstance(target, CallbackQuery):
                await target.answer("⚠️ Доступ запрещен", show_alert=True)
            else:
                await target.answer("⚠️ Доступ запрещен")
            return None

        return await func(*args, **kwargs)

    return wrapper


async def render_dashboard(
    dashboard_builder: Callable,
    target: Message | CallbackQuery,
    **kwargs
) -> None:
    """
    Универсальная функция для рендеринга dashboard

    Args:
        dashboard_builder: async функция возвращающая (text, keyboard)
        target: Message или CallbackQuery
        **kwargs: параметры для builder

    Использование:
        async def build_my_dashboard(db):
            text = "..."
            keyboard = InlineKeyboardMarkup(...)
            return text, keyboard

        # В command handler:
        await render_dashboard(build_my_dashboard, message, db=db)

        # В callback handler:
        await render_dashboard(build_my_dashboard, callback, db=db)
    """
    text, keyboard = await dashboard_builder(**kwargs)

    await edit_or_answer(target, text, keyboard)



async def edit_or_answer(
    target: Message | CallbackQuery,
    text: str,
    keyboard: InlineKeyboardMarkup | None = None,
    parse_mode: str | None = "HTML",
) -> None:
    """Send a response or edit the original message depending on the target."""
    if isinstance(target, Message):
        kwargs: dict[str, Any] = {"reply_markup": keyboard}
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        await target.answer(text, **kwargs)
    else:
        if target.message:
            kwargs = {"reply_markup": keyboard}
            if parse_mode:
                kwargs["parse_mode"] = parse_mode
            try:
                await target.message.edit_text(text, **kwargs)
            except Exception as exc:
                # Телеграм возвращает ошибку, если контент и разметка не изменились.
                # Игнорируем этот кейс, чтобы не падать хендлером.
                low = str(exc).lower()
                if "message is not modified" in low:
                    # Сигнализируем пользователю, что изменений нет (без алерта)
                    with contextlib.suppress(Exception):
                        await target.answer("Без изменений")
                    return
                raise
        else:
            await target.answer(text, show_alert=True)


async def parse_user_id(message: Message, command_name: str) -> int | None:
    """Extract user id argument from command message; report errors to the user."""
    parts = (message.text or "").split()
    if len(parts) < 2:
        await message.answer(f"Использование: /{command_name} <user_id>")
        return None
    try:
        return int(parts[1])
    except ValueError:
        await message.answer("Неверный формат user_id")
        return None


def handle_errors(error_message: str = "Ошибка выполнения"):
    """
    Decorator для обработки ошибок

    Использование:
        @handle_errors("Error in analytics")
        async def get_analytics(db):
            # код который может упасть
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception:
                logger.exception("%s (handler=%s)", error_message, func.__name__)
                return None
        return wrapper
    return decorator


def back_keyboard(callback_data: str = "admin_refresh") -> InlineKeyboardMarkup:
    """Стандартная клавиатура с кнопкой возврата."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="« Назад", callback_data=callback_data)]]
    )




def set_admin_ids(admin_ids: set[int]) -> None:
    global _GLOBAL_ADMIN_IDS
    _GLOBAL_ADMIN_IDS = set(admin_ids)

__all__ = (
    "back_keyboard",
    "edit_or_answer",
    "require_admin",
    "set_admin_ids",
    "handle_errors",
)
