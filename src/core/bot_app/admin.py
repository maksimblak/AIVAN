from __future__ import annotations

import logging

from aiogram import Dispatcher
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from src.core.bot_app import context as simple_context
from src.core.bot_app.common import ensure_valid_user_id
from src.core.exceptions import ValidationException

__all__ = ["register_admin_handlers"]

logger = logging.getLogger("ai-ivan.simple.admin")


async def cmd_error_stats(message: Message) -> None:
    """Provide aggregated error statistics to administrators."""
    if not message.from_user:
        await message.answer("❌ Команда доступна только в диалоге с ботом")
        return

    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="cmd_error_stats")
    except ValidationException as exc:
        logger.warning("Некорректный пользователь id in cmd_error_stats: %s", exc)
        await message.answer("❌ Ошибка идентификатора пользователя")
        return

    if user_id not in simple_context.ADMIN_IDS:
        await message.answer("❌ Команда доступна только администраторам")
        return

    error_handler = simple_context.error_handler
    if not error_handler:
        await message.answer("❌ Система мониторинга ошибок не инициализирована")
        return

    stats = error_handler.get_error_stats()
    if not stats:
        await message.answer("✅ Критических ошибок не зафиксировано")
        return

    lines = ["🚨 <b>Статистика ошибок</b>"]
    for error_type, count in sorted(stats.items(), key=lambda item: item[0]):
        lines.append(f"• {error_type}: {count}")

    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


def register_admin_handlers(dp: Dispatcher) -> None:
    """Register administrative commands."""
    dp.message.register(cmd_error_stats, Command("errors"))
