"""
Shared utilities для admin commands
"""

import inspect

from functools import wraps
from typing import Any, Callable
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup


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

        if target is None:
            raise RuntimeError("require_admin decorator requires a target argument (Message or CallbackQuery)")
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

    if isinstance(target, Message):
        await target.answer(text, parse_mode="HTML", reply_markup=keyboard)
    else:  # CallbackQuery
        await target.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)


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
            except Exception as e:
                print(f"{error_message}: {e}")
                return None
        return wrapper
    return decorator


def back_keyboard(callback_data: str = "admin_refresh") -> InlineKeyboardMarkup:
    """Стандартная клавиатура с кнопкой возврата."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="« Назад", callback_data=callback_data)]]
    )

