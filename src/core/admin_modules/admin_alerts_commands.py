"""
Admin commands для Automated Alerts
"""

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from src.core.admin_modules.admin_utils import edit_or_answer, require_admin
from src.core.admin_modules.automated_alerts import AlertConfig, AutomatedAlerts, group_alerts_by_severity


alerts_router = Router(name="alerts_admin")


def _build_no_alerts_view() -> tuple[str, InlineKeyboardMarkup]:
    text = """✅ <b>Все метрики в норме!</b>

Нет критических или warning alerts."""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Проверить снова", callback_data="alerts:refresh")],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="alerts:config")],
        ]
    )
    return text, keyboard

def _build_overview_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📋 View by Category", callback_data="alerts:by_category")],
            [InlineKeyboardButton(text="🔄 Refresh", callback_data="alerts:refresh")],
            [InlineKeyboardButton(text="⚙️ Settings", callback_data="alerts:config")],
        ]
    )


def _build_alerts_overview(alerts: list) -> tuple[str, InlineKeyboardMarkup]:
    grouped = group_alerts_by_severity(alerts)
    critical = grouped.get("critical", [])
    warnings = grouped.get("warning", [])
    info = grouped.get("info", [])

    lines: list[str] = [
        "🔔 <b>Active Alerts</b>",
        "",
        f"🚨 Critical: {len(critical)}",
        f"⚠️ Warnings: {len(warnings)}",
        f"ℹ️ Info: {len(info)}",
        "",
    ]

    if critical:
        lines.append("<b>🚨 CRITICAL:</b>")
        lines.append("")
        for alert in critical[:5]:
            lines.append(f"• <b>{alert.title}</b>")
            lines.append(f"  {alert.message}")
            lines.append(f"  <i>Action: {alert.action_required}</i>")
            lines.append("")

    if warnings:
        lines.append("<b>⚠️ WARNINGS:</b>")
        lines.append("")
        for alert in warnings[:3]:
            lines.append(f"• <b>{alert.title}</b>")
            lines.append(f"  {alert.message}")
            lines.append("")

    joiner = chr(10)
    text = joiner.join(lines).rstrip()
    return text, _build_overview_keyboard()


def _build_category_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="💰 Revenue", callback_data="alerts:cat:revenue")],
            [InlineKeyboardButton(text="🎯 Retention", callback_data="alerts:cat:retention")],
            [InlineKeyboardButton(text="📊 PMF", callback_data="alerts:cat:pmf")],
            [InlineKeyboardButton(text="⚙️ Technical", callback_data="alerts:cat:technical")],
            [InlineKeyboardButton(text="◀️ Back", callback_data="alerts:back")],
        ]
    )


def _build_category_text(category: str, alerts: list) -> str:
    emoji_map = {
        "revenue": "💰",
        "retention": "🎯",
        "pmf": "📊",
        "technical": "⚙️",
    }
    severity_map = {
        "critical": "🚨",
        "warning": "⚠️",
        "info": "ℹ️",
    }

    header = f"{emoji_map.get(category, '📋')} <b>{category.title()} Alerts</b>"
    if not alerts:
        return f"""{header}

✅ No alerts in this category"""

    lines = [header, ""]
    for alert in alerts:
        severity = severity_map.get(alert.severity, "")
        lines.append(f"{severity} <b>{alert.title}</b>")
        lines.append(f"  {alert.message}")
        lines.append(f"  <i>Action: {alert.action_required}</i>")
        lines.append("")

    joiner = chr(10)
    return joiner.join(lines).rstrip()


@alerts_router.message(Command("alerts"))
@require_admin
async def cmd_alerts(message: Message, db, bot, admin_ids: list[int]):
    """Показать текущее состояние алертов."""
    alert_system = AutomatedAlerts(db, bot, admin_ids)

    await message.answer("🔍 Проверяю все метрики...")
    alerts = await alert_system.check_all_alerts()

    if not alerts:
        text, keyboard = _build_no_alerts_view()
    else:
        text, keyboard = _build_alerts_overview(alerts)

    await message.answer(text, parse_mode="HTML", reply_markup=keyboard)


@alerts_router.callback_query(F.data == "alerts:refresh")
@require_admin
async def handle_alerts_refresh(callback: CallbackQuery, db, bot, admin_ids: list[int]):
    """Обновить список алертов."""
    await callback.answer("🔍 Проверяю...")

    alert_system = AutomatedAlerts(db, bot, admin_ids)
    alerts = await alert_system.check_all_alerts()

    if not alerts:
        text, keyboard = _build_no_alerts_view()
    else:
        text, keyboard = _build_alerts_overview(alerts)

    await edit_or_answer(callback, text, keyboard)


@alerts_router.callback_query(F.data == "alerts:by_category")
@require_admin
async def handle_alerts_by_category(callback: CallbackQuery, db, bot, admin_ids: list[int]):
    """Показать меню выбора категории алертов."""
    text = """📋 <b>Select Alert Category:</b>

Choose category to view detailed alerts"""
    await edit_or_answer(callback, text, _build_category_keyboard())
    await callback.answer()


@alerts_router.callback_query(F.data.startswith("alerts:cat:"))
@require_admin
async def handle_category_alerts(callback: CallbackQuery, db, bot, admin_ids: list[int]):
    """Показать алерты конкретной категории."""
    category = callback.data.split(":")[-1]

    alert_system = AutomatedAlerts(db, bot, admin_ids)
    all_alerts = await alert_system.check_all_alerts()
    category_alerts = [a for a in all_alerts if a.category == category]

    text = _build_category_text(category, category_alerts)
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Back", callback_data="alerts:by_category")]]
    )
    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@alerts_router.callback_query(F.data == "alerts:config")
@require_admin
async def handle_alerts_config(callback: CallbackQuery, db, admin_ids: list[int]):
    """Показать текущие пороги для алертов."""
    config = AlertConfig()

    text = f"""⚙️ <b>Alert Configuration</b>

"""
    text += f"""<b>💰 Revenue Thresholds:</b>
  MRR drop: >{config.mrr_drop_threshold}%
  Churn spike: >{config.churn_spike_threshold}%
  Quick Ratio min: {config.quick_ratio_min}

"""
    text += f"""<b>🎯 Retention Thresholds:</b>
  Day-30 retention min: {config.day_30_retention_min}%
  Power user churn: {config.power_user_churn_threshold} users

"""
    text += f"""<b>📊 PMF Thresholds:</b>
  NPS min: {config.nps_min}
  NPS drop: >{config.nps_drop_threshold}
  DAU/MAU min: {config.dau_mau_min}%

"""
    text += f"""<b>⚙️ Technical Thresholds:</b>
  Error rate: >{config.error_rate_threshold}%
  Success rate min: {config.feature_success_rate_min}%

"""
    text += "<i>Для изменения настроек используйте /alert_config</i>"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Back", callback_data="alerts:back")]]
    )

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@alerts_router.callback_query(F.data == "alerts:back")
@require_admin
async def handle_back_to_alerts(callback: CallbackQuery, db, bot, admin_ids: list[int]):
    """Вернуться к обзору алертов."""
    alert_system = AutomatedAlerts(db, bot, admin_ids)
    alerts = await alert_system.check_all_alerts()

    if not alerts:
        text, keyboard = _build_no_alerts_view()
    else:
        text, keyboard = _build_alerts_overview(alerts)

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@alerts_router.message(Command("digest"))
@require_admin
async def cmd_daily_digest(message: Message, db, bot, admin_ids: list[int]):
    """Отправить принудительный daily digest."""
    alert_system = AutomatedAlerts(db, bot, admin_ids)
    await message.answer("📊 Генерирую daily digest...")
    await alert_system.send_daily_digest()
