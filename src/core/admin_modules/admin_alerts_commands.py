"""
Admin commands для Automated Alerts
"""

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from html import escape as html_escape

from src.core.admin_modules.admin_utils import edit_or_answer, require_admin
from src.core.admin_modules.automated_alerts import AlertConfig, AutomatedAlerts, group_alerts_by_severity


alerts_router = Router(name="alerts_admin")


def _build_no_alerts_view() -> tuple[str, InlineKeyboardMarkup]:
    text = """✅ <b>Все метрики в норме!</b>

Нет критических или предупредительных алертов."""
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
            [InlineKeyboardButton(text="📋 По категориям", callback_data="alerts:by_category")],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="alerts:refresh")],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="alerts:config")],
        ]
    )


def _build_alerts_overview(alerts: list) -> tuple[str, InlineKeyboardMarkup]:
    grouped = group_alerts_by_severity(alerts)
    critical = grouped.get("critical", [])
    warnings = grouped.get("warning", [])
    info = grouped.get("info", [])

    lines: list[str] = [
        "🔔 <b>Активные алерты</b>",
        "",
        f"🚨 Критические: {len(critical)}",
        f"⚠️ Предупреждения: {len(warnings)}",
        f"ℹ️ Информация: {len(info)}",
        "",
    ]

    if critical:
        lines.append("<b>🚨 КРИТИЧЕСКИЕ:</b>")
        lines.append("")
        for alert in critical[:5]:
            title = html_escape(alert.title or "")
            message = html_escape(alert.message or "")
            action_required = html_escape(alert.action_required or "")
            lines.append(f"• <b>{title}</b>")
            lines.append(f"  {message}")
            lines.append(f"  <i>Действие: {action_required}</i>")
            lines.append("")

    if warnings:
        lines.append("<b>⚠️ ПРЕДУПРЕЖДЕНИЯ:</b>")
        lines.append("")
        for alert in warnings[:3]:
            title = html_escape(alert.title or "")
            message = html_escape(alert.message or "")
            lines.append(f"• <b>{title}</b>")
            lines.append(f"  {message}")
            lines.append("")

    joiner = chr(10)
    text = joiner.join(lines).rstrip()
    return text, _build_overview_keyboard()


def _build_category_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="💰 Выручка", callback_data="alerts:cat:revenue")],
            [InlineKeyboardButton(text="🎯 Удержание", callback_data="alerts:cat:retention")],
            [InlineKeyboardButton(text="📊 PMF (рынок)", callback_data="alerts:cat:pmf")],
            [InlineKeyboardButton(text="⚙️ Технические", callback_data="alerts:cat:technical")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="alerts:back")],
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

    header = {
        'revenue': '💰 <b>Алерты по выручке</b>',
        'retention': '🎯 <b>Алерты по удержанию</b>',
        'pmf': '📊 <b>Алерты PMF</b>',
        'technical': '⚙️ <b>Технические алерты</b>'
    }.get(category, '📋 <b>Алерты</b>')
    if not alerts:
        return f"{header} ✅ В этой категории сейчас нет алертов"

    lines = [header, ""]
    for alert in alerts:
        severity = severity_map.get(alert.severity, "")
        title = html_escape(alert.title or "")
        message = html_escape(alert.message or "")
        action_required = html_escape(alert.action_required or "")
        lines.append(f"{severity} <b>{title}</b>")
        lines.append(f"  {message}")
        lines.append(f"  <i>Действие: {action_required}</i>")
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
    alerts = await alert_system.check_all_alerts(force_refresh=True)

    if not alerts:
        text, keyboard = _build_no_alerts_view()
    else:
        text, keyboard = _build_alerts_overview(alerts)

    await edit_or_answer(callback, text, keyboard)


@alerts_router.callback_query(F.data == "alerts:by_category")
@require_admin
async def handle_alerts_by_category(callback: CallbackQuery, db, bot, admin_ids: list[int]):
    """Показать меню выбора категории алертов."""
    text = """📋 <b>Выберите категорию алертов</b>

Укажите, что нужно показать подробнее"""
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
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Назад", callback_data="alerts:by_category")]]
    )
    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@alerts_router.callback_query(F.data == "alerts:config")
@require_admin
async def handle_alerts_config(callback: CallbackQuery, db, admin_ids: list[int]):
    """Показать текущие пороги для алертов."""
    config = AlertConfig()

    text = f"""⚙️ <b>Конфигурация алертов</b>

<b>💰 Пороговые значения по выручке:</b>
  Падение MRR: >{config.mrr_drop_threshold}%
  Рост оттока: >{config.churn_spike_threshold}%
  Минимальный Quick Ratio: {config.quick_ratio_min}

<b>🎯 Пороговые значения по удержанию:</b>
  Retention на 30-й день: >{config.day_30_retention_min}%
  Отток power-пользователей: >{config.power_user_churn_threshold}

<b>📊 Пороговые значения PMF:</b>
  Минимальный NPS: {config.nps_min}
  Падение NPS: >{config.nps_drop_threshold}
  Минимальный DAU/MAU: {config.dau_mau_min}%

<b>⚙️ Пороговые значения технических метрик:</b>
  Доля ошибок: >{config.error_rate_threshold}%
  Минимальный процент успешных запросов: {config.feature_success_rate_min}%

<b>🕒 Кэширование:</b>
  TTL проверки алертов: {config.alerts_cache_ttl_seconds} с

<i>Изменить значения можно командой /alert_config</i>"""

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Назад", callback_data="alerts:back")]]
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
    await message.answer("📊 Генерирую ежедневный дайджест...")
    await alert_system.send_daily_digest()
