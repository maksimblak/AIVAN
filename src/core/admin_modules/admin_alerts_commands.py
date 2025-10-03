
"""
Admin commands для Automated Alerts
"""

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from src.core.admin_modules.automated_alerts import AutomatedAlerts, AlertConfig


alerts_router = Router(name="alerts_admin")


@alerts_router.message(Command("alerts"))
async def cmd_alerts(message: Message, db, bot, admin_ids: list[int]):
    """Проверить текущие alerts"""
    if message.from_user.id not in admin_ids:
        await message.answer("⛔️ Доступ запрещен")
        return

    alert_system = AutomatedAlerts(db, bot, admin_ids)

    await message.answer("🔍 Проверяю все метрики...")

    # Проверить alerts
    alerts = await alert_system.check_all_alerts()

    if not alerts:
        text = "✅ <b>Все метрики в норме!</b>\n\n"
        text += "Нет критических или warning alerts."

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Проверить снова", callback_data="alerts:refresh")],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="alerts:config")]
        ])

        await message.answer(text, parse_mode="HTML", reply_markup=keyboard)
        return

    # Group alerts by severity
    critical = [a for a in alerts if a.severity == "critical"]
    warnings = [a for a in alerts if a.severity == "warning"]
    info = [a for a in alerts if a.severity == "info"]

    text = "🔔 <b>Active Alerts</b>\n\n"
    text += f"🔴 Critical: {len(critical)}\n"
    text += f"⚠️ Warnings: {len(warnings)}\n"
    text += f"ℹ️ Info: {len(info)}\n\n"

    # Show critical alerts
    if critical:
        text += "<b>🚨 CRITICAL:</b>\n\n"
        for alert in critical[:5]:  # Limit to 5
            text += f"• <b>{alert.title}</b>\n"
            text += f"  {alert.message}\n"
            text += f"  <i>Action: {alert.action_required}</i>\n\n"

    # Show warnings
    if warnings:
        text += "<b>⚠️ WARNINGS:</b>\n\n"
        for alert in warnings[:3]:
            text += f"• <b>{alert.title}</b>\n"
            text += f"  {alert.message}\n\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📋 View by Category", callback_data="alerts:by_category")],
        [InlineKeyboardButton(text="🔄 Refresh", callback_data="alerts:refresh")],
        [InlineKeyboardButton(text="⚙️ Settings", callback_data="alerts:config")]
    ])

    await message.answer(text, parse_mode="HTML", reply_markup=keyboard)


@alerts_router.callback_query(F.data == "alerts:refresh")
async def handle_alerts_refresh(callback: CallbackQuery, db, bot, admin_ids: list[int]):
    """Обновить alerts"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    await callback.answer("🔍 Проверяю...")

    alert_system = AutomatedAlerts(db, bot, admin_ids)
    alerts = await alert_system.check_all_alerts()

    if not alerts:
        text = "✅ <b>Все метрики в норме!</b>\n\n"
        text += "Нет критических или warning alerts."

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Проверить снова", callback_data="alerts:refresh")],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="alerts:config")]
        ])

        await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
        return

    critical = [a for a in alerts if a.severity == "critical"]
    warnings = [a for a in alerts if a.severity == "warning"]
    info = [a for a in alerts if a.severity == "info"]

    text = "🔔 <b>Active Alerts</b>\n\n"
    text += f"🔴 Critical: {len(critical)}\n"
    text += f"⚠️ Warnings: {len(warnings)}\n"
    text += f"ℹ️ Info: {len(info)}\n\n"

    if critical:
        text += "<b>🚨 CRITICAL:</b>\n\n"
        for alert in critical[:5]:
            text += f"• <b>{alert.title}</b>\n"
            text += f"  {alert.message}\n"
            text += f"  <i>Action: {alert.action_required}</i>\n\n"

    if warnings:
        text += "<b>⚠️ WARNINGS:</b>\n\n"
        for alert in warnings[:3]:
            text += f"• <b>{alert.title}</b>\n"
            text += f"  {alert.message}\n\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📋 View by Category", callback_data="alerts:by_category")],
        [InlineKeyboardButton(text="🔄 Refresh", callback_data="alerts:refresh")],
        [InlineKeyboardButton(text="⚙️ Settings", callback_data="alerts:config")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)


@alerts_router.callback_query(F.data == "alerts:by_category")
async def handle_alerts_by_category(callback: CallbackQuery, db, bot, admin_ids: list[int]):
    """Показать alerts по категориям"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    text = "📋 <b>Select Alert Category:</b>\n\n"
    text += "Choose category to view detailed alerts"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💰 Revenue", callback_data="alerts:cat:revenue")],
        [InlineKeyboardButton(text="🎯 Retention", callback_data="alerts:cat:retention")],
        [InlineKeyboardButton(text="📊 PMF", callback_data="alerts:cat:pmf")],
        [InlineKeyboardButton(text="⚙️ Technical", callback_data="alerts:cat:technical")],
        [InlineKeyboardButton(text="◀️ Back", callback_data="alerts:back")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
    await callback.answer()


@alerts_router.callback_query(F.data.startswith("alerts:cat:"))
async def handle_category_alerts(callback: CallbackQuery, db, bot, admin_ids: list[int]):
    """Показать alerts конкретной категории"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    category = callback.data.split(":")[-1]

    alert_system = AutomatedAlerts(db, bot, admin_ids)
    all_alerts = await alert_system.check_all_alerts()

    # Filter by category
    category_alerts = [a for a in all_alerts if a.category == category]

    emoji_map = {
        "revenue": "💰",
        "retention": "🎯",
        "pmf": "📊",
        "technical": "⚙️"
    }

    if not category_alerts:
        text = f"{emoji_map.get(category, '📋')} <b>{category.title()} Alerts</b>\n\n"
        text += "✅ No alerts in this category"

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Back", callback_data="alerts:by_category")]
        ])

        await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
        await callback.answer()
        return

    text = f"{emoji_map.get(category, '📋')} <b>{category.title()} Alerts</b>\n\n"

    for alert in category_alerts:
        severity_emoji = {
            "critical": "🔴",
            "warning": "⚠️",
            "info": "ℹ️"
        }.get(alert.severity, "")

        text += f"{severity_emoji} <b>{alert.title}</b>\n"
        text += f"  {alert.message}\n"
        text += f"  <i>Action: {alert.action_required}</i>\n\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back to categories", callback_data="alerts:by_category")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
    await callback.answer()


@alerts_router.callback_query(F.data == "alerts:config")
async def handle_alerts_config(callback: CallbackQuery, db, admin_ids: list[int]):
    """Настройки alerts"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    config = AlertConfig()  # Default config

    text = "⚙️ <b>Alert Configuration</b>\n\n"

    text += "<b>💰 Revenue Thresholds:</b>\n"
    text += f"  MRR drop: >{config.mrr_drop_threshold}%\n"
    text += f"  Churn spike: >{config.churn_spike_threshold}%\n"
    text += f"  Quick Ratio min: {config.quick_ratio_min}\n\n"

    text += "<b>🎯 Retention Thresholds:</b>\n"
    text += f"  Day-30 retention min: {config.day_30_retention_min}%\n"
    text += f"  Power user churn: {config.power_user_churn_threshold} users\n\n"

    text += "<b>📊 PMF Thresholds:</b>\n"
    text += f"  NPS min: {config.nps_min}\n"
    text += f"  NPS drop: >{config.nps_drop_threshold}\n"
    text += f"  DAU/MAU min: {config.dau_mau_min}%\n\n"

    text += "<b>⚙️ Technical Thresholds:</b>\n"
    text += f"  Error rate: >{config.error_rate_threshold}%\n"
    text += f"  Success rate min: {config.feature_success_rate_min}%\n\n"

    text += "<i>Для изменения настроек используйте /alert_config</i>"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back", callback_data="alerts:back")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
    await callback.answer()


@alerts_router.callback_query(F.data == "alerts:back")
async def handle_back_to_alerts(callback: CallbackQuery, db, bot, admin_ids: list[int]):
    """Вернуться к главному меню alerts"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    alert_system = AutomatedAlerts(db, bot, admin_ids)
    alerts = await alert_system.check_all_alerts()

    if not alerts:
        text = "✅ <b>Все метрики в норме!</b>\n\n"
        text += "Нет критических или warning alerts."

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Проверить снова", callback_data="alerts:refresh")],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="alerts:config")]
        ])

        await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
        await callback.answer()
        return

    critical = [a for a in alerts if a.severity == "critical"]
    warnings = [a for a in alerts if a.severity == "warning"]
    info = [a for a in alerts if a.severity == "info"]

    text = "🔔 <b>Active Alerts</b>\n\n"
    text += f"🔴 Critical: {len(critical)}\n"
    text += f"⚠️ Warnings: {len(warnings)}\n"
    text += f"ℹ️ Info: {len(info)}\n\n"

    if critical:
        text += "<b>🚨 CRITICAL:</b>\n\n"
        for alert in critical[:5]:
            text += f"• <b>{alert.title}</b>\n"
            text += f"  {alert.message}\n"
            text += f"  <i>Action: {alert.action_required}</i>\n\n"

    if warnings:
        text += "<b>⚠️ WARNINGS:</b>\n\n"
        for alert in warnings[:3]:
            text += f"• <b>{alert.title}</b>\n"
            text += f"  {alert.message}\n\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📋 View by Category", callback_data="alerts:by_category")],
        [InlineKeyboardButton(text="🔄 Refresh", callback_data="alerts:refresh")],
        [InlineKeyboardButton(text="⚙️ Settings", callback_data="alerts:config")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
    await callback.answer()


@alerts_router.message(Command("digest"))
async def cmd_daily_digest(message: Message, db, bot, admin_ids: list[int]):
    """Отправить daily digest принудительно"""
    if message.from_user.id not in admin_ids:
        await message.answer("⛔️ Доступ запрещен")
        return

    alert_system = AutomatedAlerts(db, bot, admin_ids)

    await message.answer("📊 Генерирую daily digest...")

    await alert_system.send_daily_digest()
