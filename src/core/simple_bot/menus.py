from __future__ import annotations

import logging
from datetime import datetime

from aiogram import Dispatcher
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.ui_components import Emoji
from src.core.simple_bot import context as ctx
from src.core.simple_bot.common import ensure_valid_user_id
from src.core.simple_bot.payments import get_plan_pricing
from src.core.simple_bot.stats import generate_user_stats_response, normalize_stats_period
from src.core.exceptions import ErrorContext, ValidationException

logger = logging.getLogger("ai-ivan.simple.menus")

__all__ = ["register_menu_handlers", "cmd_status", "cmd_mystats"]

SECTION_DIVIDER = "<code>────────────────────</code>"


async def cmd_status(message: Message) -> None:
    db = ctx.db
    if db is None:
        await message.answer("Статус временно недоступен")
        return

    if not message.from_user:
        await message.answer("Статус доступен только для авторизованных пользователей")
        return

    try:
        user_id = ensure_valid_user_id(message.from_user.id, context="cmd_status")
    except ValidationException as exc:
        error_handler = ctx.error_handler
        context = ErrorContext(function_name="cmd_status", chat_id=message.chat.id if message.chat else None)
        if error_handler:
            await error_handler.handle_exception(exc, context)
        else:
            logger.warning("Validation error in cmd_status: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} <b>Не удалось получить статус.</b>\nПопробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    user = await db.ensure_user(
        user_id,
        default_trial=ctx.TRIAL_REQUESTS,
        is_admin=user_id in ctx.ADMIN_IDS,
    )

    until_ts = int(getattr(user, "subscription_until", 0) or 0)
    now_ts = int(datetime.now().timestamp())
    has_active = until_ts > now_ts
    plan_id = getattr(user, "subscription_plan", None)
    plan_info = get_plan_pricing(plan_id) if plan_id else None
    if plan_info:
        plan_label = plan_info.plan.name
    elif plan_id:
        plan_label = plan_id
    elif has_active:
        plan_label = "Безлимит"
    else:
        plan_label = "нет"

    if until_ts > 0:
        until_dt = datetime.fromtimestamp(until_ts)
        if has_active:
            left_days = max(0, (until_dt - datetime.now()).days)
            until_text = f"{until_dt:%Y-%m-%d} (≈{left_days} дн.)"
        else:
            until_text = f"Истекла {until_dt:%Y-%m-%d}"
    else:
        until_text = "Не активна"

    quota_balance_raw = getattr(user, "subscription_requests_balance", None)
    quota_balance = int(quota_balance_raw) if quota_balance_raw is not None else None

    lines = [
        f"{Emoji.STATS} <b>Статус</b>",
        "",
        f"ID: <code>{user_id}</code>",
        f"Роль: {'админ' if getattr(user, 'is_admin', False) else 'пользователь'}",
        f"Триал: {getattr(user, 'trial_remaining', 0)} запрос(ов)",
        "Подписка:",
    ]
    if plan_info or plan_id or until_ts:
        lines.append(f"• План: {plan_label}")
        lines.append(f"• Доступ до: {until_text}")
        if plan_info and quota_balance is not None:
            lines.append(f"• Остаток запросов: {max(0, quota_balance)}")
        elif plan_id and quota_balance is not None:
            lines.append(f"• Остаток запросов: {max(0, quota_balance)}")
        elif has_active and not plan_id:
            lines.append("• Лимит: без ограничений")
    else:
        lines.append("• Не активна")

    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_mystats(message: Message) -> None:
    db = ctx.db
    if db is None:
        await message.answer("Статистика временно недоступна")
        return

    if not message.from_user:
        await message.answer("Статистика доступна только авторизованным пользователям")
        return

    days = 30
    if message.text:
        parts = message.text.strip().split()
        if len(parts) >= 2:
            try:
                days = int(parts[1])
            except ValueError:
                days = 30

    days = normalize_stats_period(days)

    try:
        stats_text, keyboard = await generate_user_stats_response(
            message.from_user.id,
            days,
            divider=SECTION_DIVIDER,
        )
        await message.answer(stats_text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error in cmd_mystats: %s", exc)
        await message.answer("❌ Ошибка получения статистики. Попробуйте позже.")


def register_menu_handlers(dp: Dispatcher) -> None:
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(cmd_mystats, Command("mystats"))
