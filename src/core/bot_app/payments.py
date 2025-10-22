from __future__ import annotations

import logging
from datetime import datetime
from html import escape as html_escape

from aiogram import Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    PreCheckoutQuery,
)

from core.bot_app.ui_components import Emoji
try:
    from src.core.db_advanced import TransactionStatus
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    from enum import Enum

    class TransactionStatus(str, Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
from src.core.payments import convert_rub_to_xtr
from src.core.bot_app import context as ctx
from src.core.subscription_payments import (
    SubscriptionPayloadError,
    build_subscription_payload,
    parse_subscription_payload,
)
from src.core.runtime import SubscriptionPlanPricing

logger = logging.getLogger("ai-ivan.simple.payments")

__all__ = [
    "register_payment_handlers",
    "get_plan_pricing",
    "send_plan_catalog",
    "plan_catalog_text",
    "plan_stars_amount",
]


def format_rub(amount_rub: int) -> str:
    return f"{amount_rub:,}".replace(",", " ")


def plan_stars_amount(plan_info: SubscriptionPlanPricing) -> int:
    amount = int(plan_info.price_stars or 0)
    if amount <= 0:
        cfg = ctx.settings()
        amount = convert_rub_to_xtr(
            amount_rub=float(plan_info.plan.price_rub),
            rub_per_xtr=cfg.rub_per_xtr,
            default_xtr=cfg.subscription_price_xtr,
        )
    return max(int(amount), 0)


def get_plan_pricing(plan_id: str | None) -> SubscriptionPlanPricing | None:
    if not plan_id:
        return ctx.DEFAULT_SUBSCRIPTION_PLAN
    return (ctx.SUBSCRIPTION_PLAN_MAP or {}).get(plan_id)


_CATALOG_HEADER_LINES = [
    "✨ <b>Каталог подписок AIVAN</b>",
    "━━━━━━━━━━━━",
    "",
    "💡 <b>Выберите идеальный тариф для себя</b>",
    "🎯 Доступ ко всем функциям AI-юриста",
    "⚡ Мгновенные ответы на юридические вопросы",
    "📄 Анализ и составление документов",
    "",
]

_DEF_NO_PLANS_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[[InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_main")]]
)


def plan_catalog_text() -> str:
    subscription_plans = ctx.SUBSCRIPTION_PLANS or ()
    if not subscription_plans:
        return f"{Emoji.WARNING} Подписки временно недоступны. Попробуйте позже."

    lines: list[str] = list(_CATALOG_HEADER_LINES)
    for idx, plan_info in enumerate(subscription_plans, 1):
        plan = plan_info.plan
        stars_amount = plan_stars_amount(plan_info)

        lines.append("╔═══════════════════════╗")
        plan_emoji = "💎" if idx == 1 else "👑" if idx == 2 else "✨"
        lines.append(f"║ {plan_emoji} <b>{html_escape(plan.name).upper()}</b>")
        lines.append("╠═══════════════════════╣")
        lines.append(f"║ ⏰ <b>Срок:</b> {plan.duration_days} дней")
        lines.append(f"║ 📊 <b>Запросов:</b> {plan.request_quota}")

        if plan.description:
            lines.append(f"║ 💬 {html_escape(plan.description)}")

        price_line = f"║ 💰 <b>Цена:</b> {format_rub(plan.price_rub)} ₽"
        if stars_amount > 0:
            price_line += f" / {stars_amount} ⭐"
        lines.append(price_line)
        lines.append("╚═══════════════════════╝")
        lines.append("")

    lines.append("👇 <b>Выберите тариф для оплаты</b>")
    return "\n".join(lines)


def _build_plan_catalog_keyboard() -> InlineKeyboardMarkup:
    subscription_plans = ctx.SUBSCRIPTION_PLANS or ()
    if not subscription_plans:
        return _DEF_NO_PLANS_KEYBOARD

    rows: list[list[InlineKeyboardButton]] = []
    for idx, plan_info in enumerate(subscription_plans, 1):
        stars_amount = plan_stars_amount(plan_info)
        plan_emoji = "💎" if idx == 1 else "👑" if idx == 2 else "✨"
        price_label = f"{format_rub(plan_info.plan.price_rub)} ₽"
        if stars_amount > 0:
            price_label += f" • {stars_amount} ⭐"
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{plan_emoji} {plan_info.plan.name} — {price_label}",
                    callback_data=f"select_plan:{plan_info.plan.plan_id}",
                )
            ]
        )
    rows.append([InlineKeyboardButton(text="⬅️ Вернуться в меню", callback_data="back_to_main")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def send_plan_catalog(message: Message, *, edit: bool = False) -> None:
    text = plan_catalog_text()
    keyboard = _build_plan_catalog_keyboard()
    kwargs = dict(parse_mode=ParseMode.HTML, reply_markup=keyboard)
    if edit:
        try:
            await message.edit_text(text, **kwargs)
        except Exception:
            await message.answer(text, **kwargs)
    else:
        await message.answer(text, **kwargs)


async def cmd_buy(message: Message) -> None:
    await send_plan_catalog(message, edit=False)


async def handle_buy_catalog_callback(callback: CallbackQuery) -> None:
    if not callback.message:
        await callback.answer("Ошибка: нет данных сообщения", show_alert=True)
        return
    await callback.answer()
    await send_plan_catalog(callback.message, edit=True)


async def handle_get_subscription_callback(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return
    await callback.answer()
    try:
        await send_plan_catalog(callback.message, edit=False)
    except Exception:
        await callback.message.answer(
            f"{Emoji.WARNING} Не удалось показать каталог подписок. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )


async def handle_cancel_subscription_callback(callback: CallbackQuery) -> None:
    if not callback.from_user or callback.message is None:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return

    db = ctx.db
    try:
        await callback.answer()
        if db is None:
            message_text = (
                f"{Emoji.DIAMOND} <b>Отмена подписки</b>\n\n"
                "Сервис управления подписками временно недоступен. Напишите в поддержку — команда /help."
            )
        else:
            user_id = callback.from_user.id
            user_record = await db.ensure_user(
                user_id,
                default_trial=ctx.TRIAL_REQUESTS,
                is_admin=user_id in ctx.ADMIN_IDS,
            )
            has_subscription = await db.has_active_subscription(user_id)
            if has_subscription:
                cancellation_applied = await db.cancel_subscription(user_id)
                updated_record = await db.get_user(user_id)
                if updated_record is not None:
                    user_record = updated_record
                until_ts = int(getattr(user_record, "subscription_until", 0) or 0)
                until_text = (
                    datetime.fromtimestamp(until_ts).strftime("%d.%m.%Y") if until_ts else "—"
                )
                if cancellation_applied:
                    message_text = (
                        f"{Emoji.DIAMOND} <b>Отмена подписки</b>\n\n"
                        f"Отмена оформлена. Доступ сохранится до {until_text}, после чего подписка отключится.\n"
                        "Если передумали, выберите «🔄 Сменить тариф», чтобы продлить доступ."
                    )
                else:
                    message_text = (
                        f"{Emoji.DIAMOND} <b>Отмена подписки</b>\n\n"
                        f"Отмена активна. Доступ сохранится до {until_text}."
                    )
            else:
                message_text = (
                    f"{Emoji.DIAMOND} <b>Отмена подписки</b>\n\n"
                    "Подписка не активна. Оформите новый тариф через /buy."
                )

        await callback.message.answer(message_text, parse_mode=ParseMode.HTML)
    except Exception as exc:
        logger.error("Error in handle_cancel_subscription_callback: %s", exc)
        await callback.answer("❌ Произошла ошибка", show_alert=True)


def _plan_details_keyboard(plan_info: SubscriptionPlanPricing) -> tuple[InlineKeyboardMarkup, list[str]]:
    rows: list[list[InlineKeyboardButton]] = []
    unavailable: list[str] = []

    if ctx.RUB_PROVIDER_TOKEN:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"💳 Карта • {format_rub(plan_info.plan.price_rub)} ₽",
                    callback_data=f"pay_plan:{plan_info.plan.plan_id}:rub",
                )
            ]
        )
    else:
        unavailable.append("💳 Оплата картой — временно недоступна")

    stars_amount = plan_stars_amount(plan_info)
    if stars_amount > 0 and ctx.STARS_PROVIDER_TOKEN:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"⭐ Telegram Stars • {stars_amount}",
                    callback_data=f"pay_plan:{plan_info.plan.plan_id}:stars",
                )
            ]
        )
    else:
        unavailable.append("⭐ Telegram Stars — временно недоступно")

    if ctx.crypto_provider is not None:
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🪙 Криптовалюта • {format_rub(plan_info.plan.price_rub)} ₽",
                    callback_data=f"pay_plan:{plan_info.plan.plan_id}:crypto",
                )
            ]
        )
    else:
        unavailable.append("🪙 Криптовалюта — временно недоступна")

    if ctx.robokassa_provider is not None and getattr(ctx.robokassa_provider, "is_available", False):
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🏦 RoboKassa • {format_rub(plan_info.plan.price_rub)} ₽",
                    callback_data=f"pay_plan:{plan_info.plan.plan_id}:robokassa",
                )
            ]
        )

    if ctx.yookassa_provider is not None and getattr(ctx.yookassa_provider, "is_available", False):
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"💳 YooKassa • {format_rub(plan_info.plan.price_rub)} ₽",
                    callback_data=f"pay_plan:{plan_info.plan.plan_id}:yookassa",
                )
            ]
        )

    rows.append(
        [InlineKeyboardButton(text=f"{Emoji.BACK} Назад к тарифам", callback_data="buy_catalog")]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows), unavailable


async def handle_select_plan_callback(callback: CallbackQuery) -> None:
    data = callback.data or ""
    parts = data.split(":", 1)
    if len(parts) != 2:
        await callback.answer("❌ Некорректный тариф", show_alert=True)
        return

    plan_id = parts[1]
    plan_info = get_plan_pricing(plan_id)
    if not plan_info or not callback.message:
        await callback.answer("❌ Тариф недоступен", show_alert=True)
        return

    await callback.answer()
    plan = plan_info.plan
    stars_amount = plan_stars_amount(plan_info)
    lines = [f"{Emoji.DIAMOND} <b>{html_escape(plan.name)}</b>"]
    if plan.description:
        lines.append(f"<i>{html_escape(plan.description)}</i>")

    lines.extend(
        [
            "",
            f"{Emoji.CALENDAR} Период доступа: {plan.duration_days} дней",
            f"{Emoji.DOCUMENT} Лимит запросов: {plan.request_quota}",
        ]
    )
    price_line = f"💳 {format_rub(plan.price_rub)} ₽"
    if stars_amount > 0:
        price_line += f" • {stars_amount} ⭐"
    lines.append(price_line)
    lines.append("")
    lines.append(f"{Emoji.MAGIC} Выберите удобный способ оплаты ниже.")

    keyboard, unavailable = _plan_details_keyboard(plan_info)
    if unavailable:
        lines.append("")
        lines.append(f"{Emoji.WARNING} Временно недоступно:")
        lines.extend(f"• {item}" for item in unavailable)

    text = "\n".join(lines)
    try:
        await callback.message.edit_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)
    except Exception:
        await callback.message.answer(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


async def handle_pay_plan_callback(callback: CallbackQuery) -> None:
    if not callback.from_user or not callback.message:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return

    data = callback.data or ""
    parts = data.split(":")
    if len(parts) != 3:
        await callback.answer("❌ Некорректные параметры оплаты", show_alert=True)
        return

    _, plan_id, method = parts
    plan_info = get_plan_pricing(plan_id)
    if not plan_info:
        await callback.answer("❌ Тариф недоступен", show_alert=True)
        return

    await callback.answer()
    user_id = callback.from_user.id
    if method == "rub":
        await _send_rub_invoice(callback.message, plan_info, user_id)
    elif method in {"stars", "xtr"}:
        await _send_stars_invoice(callback.message, plan_info, user_id)
    elif method == "crypto":
        await _send_crypto_invoice(callback.message, plan_info, user_id)
    elif method == "robokassa":
        await _send_robokassa_invoice(callback.message, plan_info, user_id)
    elif method == "yookassa":
        await _send_yookassa_invoice(callback.message, plan_info, user_id)
    else:
        await callback.message.answer("❌ Этот способ оплаты не поддерживается")


async def handle_verify_payment_callback(callback: CallbackQuery) -> None:
    if not callback.from_user:
        await callback.answer("❌ Ошибка данных", show_alert=True)
        return

    data = callback.data or ""
    parts = data.split(":")
    if len(parts) != 3:
        await callback.answer("❌ Некорректные параметры", show_alert=True)
        return

    _, provider_code, payment_id = parts
    provider_code = provider_code.lower()
    payment_id = payment_id.strip()

    await callback.answer()
    if not payment_id:
        await callback.message.answer(
            f"{Emoji.WARNING} Платеж не найден. Попробуйте ещё раз.",
            parse_mode=ParseMode.HTML,
        )
        return

    db = ctx.db
    provider_obj = None
    if provider_code == "robokassa":
        provider_obj = ctx.robokassa_provider
    elif provider_code == "yookassa":
        provider_obj = ctx.yookassa_provider

    if provider_obj is None:
        await callback.message.answer(
            f"{Emoji.WARNING} Этот способ оплаты недоступен.",
            parse_mode=ParseMode.HTML,
        )
        return

    if db is None:
        await callback.message.answer(
            f"{Emoji.WARNING} Сервис временно недоступен. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        transaction = await db.get_transaction_by_provider_charge_id(provider_code, payment_id)
    except Exception as exc:
        logger.error("Failed to load transaction for provider %s: %s", provider_code, exc)
        transaction = None

    if transaction is None:
        await callback.message.answer(
            f"{Emoji.WARNING} Платеж не найден. Убедитесь, что вы использовали последнюю ссылку оплаты.",
            parse_mode=ParseMode.HTML,
        )
        return

    if transaction.user_id != callback.from_user.id:
        await callback.message.answer(
            f"{Emoji.WARNING} Проверка доступна только владельцу платежа.",
            parse_mode=ParseMode.HTML,
        )
        return

    current_status = TransactionStatus.from_value(transaction.status)
    if current_status == TransactionStatus.COMPLETED:
        await callback.message.answer(
            f"{Emoji.SUCCESS} Платёж уже подтвержден. Статус подписки можно посмотреть через /status.",
            parse_mode=ParseMode.HTML,
        )
        return

    poll_method = getattr(provider_obj, "poll_payment", None)
    if poll_method is None:
        await callback.message.answer(
            f"{Emoji.WARNING} Проверка оплаты для этого провайдера пока не реализована.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        result = await poll_method(payment_id)
    except Exception as exc:
        logger.error("Payment polling failed (%s): %s", provider_code, exc)
        await callback.message.answer(
            f"{Emoji.WARNING} Не удалось проверить оплату. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    if result.status == TransactionStatus.PENDING:
        await callback.message.answer(
            f"{Emoji.WARNING} Платёж ещё обрабатывается. Попробуйте снова через минуту.",
            parse_mode=ParseMode.HTML,
        )
        return

    if result.status in {TransactionStatus.CANCELLED, TransactionStatus.FAILED}:
        await db.update_transaction(transaction.id, status=result.status)
        reason = result.description or "Провайдер сообщил об отмене"
        await callback.message.answer(
            f"{Emoji.ERROR} Оплата не прошла: {html_escape(reason)}",
            parse_mode=ParseMode.HTML,
        )
        return

    payload_raw = transaction.payload or ""
    try:
        payload = parse_subscription_payload(payload_raw)
    except SubscriptionPayloadError as exc:
        logger.error("Failed to parse payload for transaction %s: %s", transaction.id, exc)
        await callback.message.answer(
            f"{Emoji.ERROR} Ошибка обработки платежа. Свяжитесь с поддержкой.",
            parse_mode=ParseMode.HTML,
        )
        return

    plan_info = get_plan_pricing(payload.plan_id) if payload.plan_id else ctx.DEFAULT_SUBSCRIPTION_PLAN
    if plan_info is None:
        await callback.message.answer(
            f"{Emoji.ERROR} Не удалось определить тариф. Свяжитесь с поддержкой.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        await db.update_transaction(transaction.id, status=TransactionStatus.COMPLETED)
        new_until, new_balance = await db.apply_subscription_purchase(
            user_id=transaction.user_id,
            plan_id=plan_info.plan.plan_id,
            duration_days=plan_info.plan.duration_days,
            request_quota=plan_info.plan.request_quota,
        )
    except Exception as exc:
        logger.error("Failed to finalize subscription for transaction %s: %s", transaction.id, exc)
        await callback.message.answer(
            f"{Emoji.ERROR} Не удалось активировать подписку, напишите поддержке.",
            parse_mode=ParseMode.HTML,
        )
        return

    until_dt = datetime.fromtimestamp(new_until)
    balance_text = (
        f"Остаток запросов: {max(0, new_balance)}"
        if plan_info.plan.request_quota
        else "Безлимит"
    )
    success_text = (
        f"{Emoji.SUCCESS} Оплата подтверждена!\n\n"
        f"План: {plan_info.plan.name}\n"
        f"Доступ до: {until_dt:%d.%m.%Y %H:%M}\n"
        f"{balance_text}"
    )
    await callback.message.answer(success_text, parse_mode=ParseMode.HTML)


async def _send_rub_invoice(message: Message, plan_info: SubscriptionPlanPricing, user_id: int) -> None:
    from aiogram.types import LabeledPrice  # local import to avoid top-level circular

    cfg = ctx.settings()
    if not ctx.RUB_PROVIDER_TOKEN:
        await message.answer(f"{Emoji.WARNING} Оплата картой временно недоступна.", parse_mode=ParseMode.HTML)
        return

    payload = build_subscription_payload(plan_info.plan.plan_id, "rub", user_id)
    prices = [
        LabeledPrice(
            label=plan_info.plan.name,
            amount=plan_info.price_rub_kopeks,
        )
    ]
    description = (
        f"Доступ к ИИ-Иван на {plan_info.plan.duration_days} дн.\n"
        f"Квота: {plan_info.plan.request_quota} запросов."
    )
    await message.bot.send_invoice(
        chat_id=message.chat.id,
        title=f"Подписка • {plan_info.plan.name}",
        description=description,
        payload=payload,
        provider_token=ctx.RUB_PROVIDER_TOKEN,
        currency="RUB",
        prices=prices,
        is_flexible=False,
        need_email=cfg.yookassa_require_email,
        need_phone_number=cfg.yookassa_require_phone,
        send_email_to_provider=cfg.yookassa_require_email,
        send_phone_number_to_provider=cfg.yookassa_require_phone,
    )


async def _send_stars_invoice(message: Message, plan_info: SubscriptionPlanPricing, user_id: int) -> None:
    from aiogram.types import LabeledPrice  # local import

    if not ctx.STARS_PROVIDER_TOKEN:
        await message.answer(f"{Emoji.WARNING} Telegram Stars временно недоступны.", parse_mode=ParseMode.HTML)
        return

    stars_amount = plan_stars_amount(plan_info)
    if stars_amount <= 0:
        await message.answer(
            f"{Emoji.WARNING} Не удалось рассчитать стоимость в Stars, попробуйте другой способ.",
            parse_mode=ParseMode.HTML,
        )
        return

    payload = build_subscription_payload(plan_info.plan.plan_id, "stars", user_id)
    prices = [LabeledPrice(label=plan_info.plan.name, amount=stars_amount)]
    description = (
        f"Оплата в Telegram Stars. Срок: {plan_info.plan.duration_days} дн.\n"
        f"Квота: {plan_info.plan.request_quota} запросов."
    )
    await message.bot.send_invoice(
        chat_id=message.chat.id,
        title=f"Подписка • {plan_info.plan.name} (Stars)",
        description=description,
        payload=payload,
        provider_token=ctx.STARS_PROVIDER_TOKEN,
        currency="XTR",
        prices=prices,
        is_flexible=False,
    )


async def _record_pending_transaction(
    *,
    user_id: int,
    provider: str,
    amount_minor_units: int,
    payload: str,
    provider_payment_charge_id: str | None = None,
) -> int:
    db = ctx.db
    if db is None:
        raise RuntimeError("Database not initialized")
    return await db.record_transaction(
        user_id=user_id,
        provider=provider,
        currency="RUB",
        amount=amount_minor_units,
        amount_minor_units=amount_minor_units,
        payload=payload,
        status=TransactionStatus.PENDING,
        provider_payment_charge_id=provider_payment_charge_id,
    )


def _external_payment_keyboard(provider: str, payment_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Проверить оплату",
                    callback_data=f"verify_payment:{provider}:{payment_id}",
                )
            ],
            [InlineKeyboardButton(text=f"{Emoji.BACK} Назад к тарифам", callback_data="buy_catalog")],
        ]
    )


async def _send_robokassa_invoice(message: Message, plan_info: SubscriptionPlanPricing, user_id: int) -> None:
    provider = ctx.robokassa_provider
    if provider is None or not getattr(provider, "is_available", False):
        await message.answer(
            f"{Emoji.WARNING} Оплата через RoboKassa временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        return

    amount_minor = plan_info.price_rub_kopeks
    payload = build_subscription_payload(plan_info.plan.plan_id, "robokassa", user_id)
    try:
        transaction_id = await _record_pending_transaction(
            user_id=user_id,
            provider="robokassa",
            amount_minor_units=amount_minor,
            payload=payload,
        )
    except Exception as exc:
        logger.error("Failed to record RoboKassa transaction: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} Не удалось начать оплату. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        creation = await provider.create_invoice(
            amount_rub=float(plan_info.plan.price_rub),
            description=f"Подписка {plan_info.plan.name} на {plan_info.plan.duration_days} дн.",
            payload=payload,
            invoice_id=transaction_id,
        )
    except Exception as exc:
        logger.error("RoboKassa invoice error: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} Не удалось создать счет RoboKassa.",
            parse_mode=ParseMode.HTML,
        )
        from contextlib import suppress

        with suppress(Exception):
            await ctx.db.update_transaction(transaction_id, status=TransactionStatus.FAILED)
        return

    if not creation.ok or not creation.url or not creation.payment_id:
        logger.warning("RoboKassa invoice creation failed: %s", creation.error or creation.raw)
        await message.answer(
            f"{Emoji.WARNING} Оплата через RoboKassa временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        from contextlib import suppress

        with suppress(Exception):
            await ctx.db.update_transaction(transaction_id, status=TransactionStatus.FAILED)
        return

    from contextlib import suppress

    with suppress(Exception):
        await ctx.db.update_transaction(
            transaction_id,
            provider_payment_charge_id=str(creation.payment_id),
        )

    payment_text = (
        f"🏦 <b>RoboKassa</b>\n\n"
        f"1. Нажмите на ссылку и оплатите счет картой или через СБП.\n"
        f"2. После оплаты вернитесь и нажмите кнопку \"Проверить оплату\".\n\n"
        f"{creation.url}"
    )
    await message.answer(
        payment_text,
        parse_mode=ParseMode.HTML,
        reply_markup=_external_payment_keyboard("robokassa", str(creation.payment_id)),
        disable_web_page_preview=True,
    )


async def _send_yookassa_invoice(message: Message, plan_info: SubscriptionPlanPricing, user_id: int) -> None:
    provider = ctx.yookassa_provider
    if provider is None or not getattr(provider, "is_available", False):
        await message.answer(
            f"{Emoji.WARNING} Оплата через YooKassa временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        return

    amount_minor = plan_info.price_rub_kopeks
    payload = build_subscription_payload(plan_info.plan.plan_id, "yookassa", user_id)
    try:
        transaction_id = await _record_pending_transaction(
            user_id=user_id,
            provider="yookassa",
            amount_minor_units=amount_minor,
            payload=payload,
        )
    except Exception as exc:
        logger.error("Failed to record YooKassa transaction: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} Не удалось начать оплату. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        creation = await provider.create_invoice(
            amount_rub=float(plan_info.plan.price_rub),
            description=f"Подписка {plan_info.plan.name} на {plan_info.plan.duration_days} дн.",
            payload=payload,
            metadata={"transaction_id": transaction_id, "plan_id": plan_info.plan.plan_id},
        )
    except Exception as exc:
        from contextlib import suppress

        logger.error("YooKassa invoice error: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} Не удалось создать счет YooKassa.",
            parse_mode=ParseMode.HTML,
        )
        with suppress(Exception):
            await ctx.db.update_transaction(transaction_id, status=TransactionStatus.FAILED)
        return

    if not creation.ok or not creation.url or not creation.payment_id:
        from contextlib import suppress

        logger.warning("YooKassa invoice creation failed: %s", creation.error or creation.raw)
        await message.answer(
            f"{Emoji.WARNING} Оплата через YooKassa временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        with suppress(Exception):
            await ctx.db.update_transaction(transaction_id, status=TransactionStatus.FAILED)
        return

    from contextlib import suppress

    with suppress(Exception):
        await ctx.db.update_transaction(
            transaction_id,
            provider_payment_charge_id=str(creation.payment_id),
        )

    payment_text = (
        f"💳 <b>YooKassa</b>\n\n"
        f"1. Перейдите по ссылке и оплатите счет.\n"
        f"2. После оплаты вернитесь и нажмите кнопку \"Проверить оплату\".\n\n"
        f"{creation.url}"
    )
    await message.answer(
        payment_text,
        parse_mode=ParseMode.HTML,
        reply_markup=_external_payment_keyboard("yookassa", str(creation.payment_id)),
        disable_web_page_preview=True,
    )


async def _send_crypto_invoice(message: Message, plan_info: SubscriptionPlanPricing, user_id: int) -> None:
    provider = ctx.crypto_provider
    if provider is None:
        await message.answer(
            f"{Emoji.WARNING} Криптовалюта временно недоступна.",
            parse_mode=ParseMode.HTML,
        )
        return

    payload = build_subscription_payload(plan_info.plan.plan_id, "crypto", user_id)
    try:
        invoice = await provider.create_invoice(
            amount_rub=float(plan_info.plan.price_rub),
            description=f"Подписка {plan_info.plan.name} на {plan_info.plan.duration_days} дн.",
            payload=payload,
        )
    except Exception as exc:
        logger.warning("Crypto invoice failed: %s", exc)
        await message.answer(
            f"{Emoji.WARNING} Не удалось создать крипто-счет. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    url = invoice.get("url") if isinstance(invoice, dict) else None
    if invoice and invoice.get("ok") and url:
        await message.answer(
            f"{Emoji.DOWNLOAD} Оплата криптовалютой: перейдите по ссылке\n{url}",
            parse_mode=ParseMode.HTML,
        )
    else:
        await message.answer(
            f"{Emoji.IDEA} Криптовалюта временно недоступна.",
            parse_mode=ParseMode.HTML,
        )


async def handle_ignore_callback(callback: CallbackQuery) -> None:
    await callback.answer()


async def pre_checkout(pre: PreCheckoutQuery) -> None:
    from src.core.validation import InputValidator  # local import to avoid circularity

    try:
        payload_raw = pre.invoice_payload or ""
        parsed = None
        try:
            parsed = parse_subscription_payload(payload_raw)
        except SubscriptionPayloadError:
            parsed = None

        plan_info = get_plan_pricing(parsed.plan_id if parsed else None)
        if plan_info is None:
            plan_info = ctx.DEFAULT_SUBSCRIPTION_PLAN
        if plan_info is None:
            await pre.answer(ok=False, error_message="Подписка недоступна")
            return

        method = (parsed.method if parsed else "").lower()
        if method == "xtr":
            method = "stars"

        if parsed and pre.from_user and parsed.user_id and parsed.user_id != pre.from_user.id:
            await pre.answer(ok=False, error_message="Счёт предназначен для другого пользователя")
            return

        if method == "rub":
            expected_currency = "RUB"
            expected_amount = plan_info.price_rub_kopeks
        elif method == "stars":
            expected_currency = "XTR"
            expected_amount = plan_stars_amount(plan_info)
        else:
            expected_currency = pre.currency.upper()
            expected_amount = pre.total_amount

        if expected_amount <= 0:
            await pre.answer(ok=False, error_message="Некорректная сумма оплаты")
            return

        if pre.currency.upper() != expected_currency or int(pre.total_amount) != int(expected_amount):
            await pre.answer(ok=False, error_message="Некорректные параметры оплаты")
            return

        amount_major = pre.total_amount / 100 if expected_currency == "RUB" else pre.total_amount
        amount_check = InputValidator.validate_payment_amount(amount_major, expected_currency)
        if not amount_check.is_valid:
            await pre.answer(ok=False, error_message="Сумма оплаты вне допустимого диапазона")
            return

        await pre.answer(ok=True)
    except Exception:
        await pre.answer(ok=False, error_message="Ошибка проверки оплаты, попробуйте позже")


async def on_successful_payment(message: Message) -> None:
    try:
        sp = message.successful_payment
        if sp is None or message.from_user is None:
            return

        currency_up = sp.currency.upper()
        if currency_up == "RUB":
            provider_name = "telegram_rub"
        elif currency_up == "XTR":
            provider_name = "telegram_stars"
        else:
            provider_name = f"telegram_{currency_up.lower()}"

        payload_raw = sp.invoice_payload or ""
        parsed_payload = None
        try:
            parsed_payload = parse_subscription_payload(payload_raw)
        except SubscriptionPayloadError:
            parsed_payload = None

        plan_info = get_plan_pricing(parsed_payload.plan_id if parsed_payload else None)
        if plan_info is None:
            plan_info = ctx.DEFAULT_SUBSCRIPTION_PLAN

        cfg = ctx.settings()
        duration_days = plan_info.plan.duration_days if plan_info else max(1, int(cfg.sub_duration_days or 30))
        quota = plan_info.plan.request_quota if plan_info else 0
        plan_id = (
            plan_info.plan.plan_id
            if plan_info
            else (parsed_payload.plan_id if parsed_payload and parsed_payload.plan_id else "legacy")
        )

        user_id = message.from_user.id
        new_until = None
        new_balance: int | None = None

        db = ctx.db
        if db is not None and sp.telegram_payment_charge_id:
            exists = await db.transaction_exists_by_telegram_charge_id(sp.telegram_payment_charge_id)
            if exists:
                return

        if db is not None:
            await db.record_transaction(
                user_id=user_id,
                provider=provider_name,
                currency=sp.currency,
                amount=sp.total_amount,
                amount_minor_units=sp.total_amount,
                payload=payload_raw,
                status=TransactionStatus.COMPLETED.value,
                telegram_payment_charge_id=sp.telegram_payment_charge_id,
                provider_payment_charge_id=sp.provider_payment_charge_id,
            )

            if plan_info is not None:
                new_until, new_balance = await db.apply_subscription_purchase(
                    user_id,
                    plan_id=plan_id,
                    duration_days=duration_days,
                    request_quota=quota,
                )
            else:
                await db.extend_subscription_days(user_id, duration_days)
                user = await db.get_user(user_id)
                if user and user.subscription_until:
                    new_until = int(user.subscription_until)
                if user and getattr(user, "subscription_requests_balance", None) is not None:
                    new_balance = int(getattr(user, "subscription_requests_balance"))

        response_lines = [f"{Emoji.SUCCESS} <b>Оплата получена!</b>"]
        if plan_info is not None:
            response_lines.append(f"Тариф: <b>{plan_info.plan.name}</b>")
            response_lines.append(f"Срок действия: {duration_days} дней")
            response_lines.append(f"Квота: {plan_info.plan.request_quota} запросов")
        elif parsed_payload and parsed_payload.plan_id:
            response_lines.append(f"Тариф: {parsed_payload.plan_id}")
            response_lines.append(f"Срок действия: {duration_days} дней")

        if new_until:
            until_text = datetime.fromtimestamp(new_until).strftime("%Y-%m-%d")
            response_lines.append(f"Доступ до: {until_text}")

        if plan_info is not None and new_balance is not None:
            response_lines.append(f"Остаток запросов: {new_balance}")

        response_lines.append("Проверить подписку — команда /status.")

        await message.answer("\n".join(response_lines), parse_mode=ParseMode.HTML)
    except Exception:
        logger.exception("Failed to handle successful payment")


def register_payment_handlers(dp: Dispatcher) -> None:
    dp.message.register(cmd_buy, Command("buy"))
    dp.callback_query.register(handle_buy_catalog_callback, F.data == "buy_catalog")
    dp.callback_query.register(handle_get_subscription_callback, F.data == "get_subscription")
    dp.callback_query.register(handle_cancel_subscription_callback, F.data == "cancel_subscription")
    dp.callback_query.register(handle_select_plan_callback, F.data.startswith("select_plan:"))
    dp.callback_query.register(handle_pay_plan_callback, F.data.startswith("pay_plan:"))
    dp.callback_query.register(handle_verify_payment_callback, F.data.startswith("verify_payment:"))
    dp.callback_query.register(handle_ignore_callback, F.data == "ignore")
    dp.pre_checkout_query.register(pre_checkout)
    dp.message.register(on_successful_payment, F.successful_payment)

