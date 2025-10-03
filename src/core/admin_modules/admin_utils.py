"""
Shared utilities для admin commands
"""

from functools import wraps
from typing import Callable, Tuple, Any
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup


def require_admin(func):
    """
    Decorator для проверки admin доступа

    Использование:
        @require_admin
        async def cmd_example(message: Message, db, admin_ids: list[int]):
            # Не нужна проверка - decorator делает это
            ...
    """
    @wraps(func)
    async def wrapper(target, db, admin_ids: list[int], *args, **kwargs):
        user_id = target.from_user.id

        if user_id not in admin_ids:
            if isinstance(target, CallbackQuery):
                await target.answer("⛔️ Доступ запрещен", show_alert=True)
            else:
                await target.answer("⛔️ Доступ запрещен")
            return

        return await func(target, db, admin_ids, *args, **kwargs)

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
