"""
Админ-команды для управления и аналитики бота
"""

from __future__ import annotations

import logging
from html import escape as html_escape
from typing import TYPE_CHECKING

from aiogram import F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from src.bot.ui_components import Emoji
from src.core.admin_modules.admin_analytics import (
    AdminAnalytics,
    PLAN_SEGMENT_DEFS,
    PLAN_SEGMENT_ORDER,
)
from src.core.admin_modules.admin_utils import back_keyboard, edit_or_answer, require_admin, set_admin_ids
from src.core.admin_modules.admin_alerts_commands import alerts_router
from src.core.admin_modules.admin_behavior_commands import behavior_router
from src.core.admin_modules.admin_cohort_commands import cohort_router
from src.core.admin_modules.admin_pmf_commands import pmf_router
from src.core.admin_modules.admin_retention_commands import retention_router
from src.core.admin_modules.admin_revenue_commands import revenue_router
from src.core.safe_telegram import send_html_text

if TYPE_CHECKING:
    from src.core.db_advanced import DatabaseAdvanced

logger = logging.getLogger(__name__)


_GLOBAL_DB: DatabaseAdvanced | None = None



def _resolve_db(db: DatabaseAdvanced | None) -> DatabaseAdvanced:
    global _GLOBAL_DB
    candidate = db or _GLOBAL_DB
    if candidate is None:
        raise RuntimeError("Database is not configured for admin commands")
    return candidate


def create_main_menu() -> InlineKeyboardMarkup:
    """Главное меню админ-панели"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Аналитика", callback_data="admin_menu:analytics")],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="admin_menu:refresh")]
        ]
    )


async def _build_admin_summary(db: DatabaseAdvanced | None = None) -> str:
    analytics = AdminAnalytics(_resolve_db(db))
    segments = await analytics.get_user_segments()
    conversion_metrics = await analytics.get_conversion_metrics()
    feature_usage = await analytics.get_feature_usage_stats(days=30)

    # Форматируем планы
    plan_lines = []
    total_paid = 0
    for plan_id in PLAN_SEGMENT_ORDER:
        segment = segments.get(f'plan_{plan_id}')
        if segment:
            total_paid += segment.user_count
            plan_lines.append(f"  {PLAN_SEGMENT_DEFS[plan_id]['button']} <b>{segment.user_count}</b>")

    plan_block = ""
    if plan_lines:
        plan_block = "\n\n<b>💎 Платные подписки:</b>\n" + "\n".join(plan_lines)

    # Форматируем статистику использования функций
    feature_icons = {
        "summarize": "📄",
        "analyze_risks": "⚠️",
        "lawsuit_analysis": "⚖️",
        "anonymize": "🕶️",
        "ocr": "📷",
        "translate": "🌐",
        "chat": "💬",
    }

    feature_names = {
        "summarize": "Краткая выжимка",
        "analyze_risks": "Риск-анализ",
        "lawsuit_analysis": "Анализ искового",
        "anonymize": "Обезличивание",
        "ocr": "Распознавание текста",
        "translate": "Перевод",
        "chat": "Чат с документом",
    }

    # Если данных нет, показываем все функции с нулями
    if not feature_usage:
        feature_usage = {key: 0 for key in feature_names.keys()}

    feature_lines = []
    sorted_features = sorted(feature_usage.items(), key=lambda x: x[1], reverse=True)
    for feature_key, count in sorted_features[:5]:  # Топ-5
        icon = feature_icons.get(feature_key, "•")
        name = feature_names.get(feature_key, feature_key)
        feature_lines.append(f"  {icon} {name}: <b>{count}</b>")

    feature_block = "\n\n<b>🔧 Популярные функции (30 дн.):</b>\n" + "\n".join(feature_lines)

    # Форматируем конверсию с индикатором
    conversion_rate = conversion_metrics.conversion_rate
    if conversion_rate >= 15:
        conv_indicator = "🟢"
    elif conversion_rate >= 8:
        conv_indicator = "🟡"
    else:
        conv_indicator = "🔴"

    return f"""
╔═══════════════════════════════════╗
       <b>🎛 АДМИН-ПАНЕЛЬ</b>
╚═══════════════════════════════════╝

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  <b>📊 ПОЛЬЗОВАТЕЛИ</b>
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

🆕 Новые (7 дн.)         <b>{segments['new_users'].user_count}</b>
⚡️ Суперактивные          <b>{segments['power_users'].user_count}</b>
🚫 Только бесплатные      <b>{segments['freeloaders'].user_count}</b>

<b>⚠️ Требуют внимания:</b>
  ⏰ Группа риска         <b>{segments['at_risk'].user_count}</b>
  📉 Отток                <b>{segments['churned'].user_count}</b>{plan_block}{feature_block}

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  <b>📈 КОНВЕРСИЯ</b>
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

💰 Переходы из триала     <b>{segments['trial_converters'].user_count}</b>
👥 Всего на триале        <b>{conversion_metrics.total_trial_users}</b>
✅ Перешли на оплату      <b>{conversion_metrics.converted_to_paid}</b>

{conv_indicator} <b>Конверсия: {conversion_metrics.conversion_rate}%</b>
⏱ Среднее время: <b>{conversion_metrics.avg_time_to_conversion_days}</b> дн.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<i>📱 Выберите раздел для детального анализа</i>
"""


# Создаем router для admin команд
admin_router = Router()



def create_analytics_menu() -> InlineKeyboardMarkup:
    """Создание главного меню аналитики"""
    rows = [
        [
            InlineKeyboardButton(text="⚡ Суперактивные", callback_data="admin_segment:power_users"),
            InlineKeyboardButton(text="⚠️ Группа риска", callback_data="admin_segment:at_risk"),
        ],
        [
            InlineKeyboardButton(text="📉 Отток", callback_data="admin_segment:churned"),
            InlineKeyboardButton(text="💰 Переходы в оплату", callback_data="admin_segment:trial_converters"),
        ],
        [
            InlineKeyboardButton(text="🚫 Только бесплатные", callback_data="admin_segment:freeloaders"),
            InlineKeyboardButton(text="🆕 Новые пользователи", callback_data="admin_segment:new_users"),
        ],
    ]

    plan_buttons = [
        InlineKeyboardButton(
            text=PLAN_SEGMENT_DEFS[plan_id]['button'],
            callback_data=f"admin_segment:plan_{plan_id}",
        )
        for plan_id in PLAN_SEGMENT_ORDER
    ]
    if plan_buttons:
        rows.append(plan_buttons)

    rows.extend(
        [
            [
                InlineKeyboardButton(text="📊 Аналитика конверсии", callback_data="admin_stats:conversion"),
                InlineKeyboardButton(text="📈 Ежедневная статистика", callback_data="admin_stats:daily"),
            ],
            [
                InlineKeyboardButton(text="🔄 Обновить", callback_data="admin_refresh"),
            ],
            [
                InlineKeyboardButton(text="◀️ Назад", callback_data="admin_menu:back"),
            ],
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=rows)


@admin_router.message(Command("admin"))
@require_admin
async def cmd_admin(message: Message, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Главная команда админ-панели"""

    summary = await _build_admin_summary(db)
    await message.answer(summary, parse_mode=ParseMode.HTML, reply_markup=create_main_menu())


@admin_router.callback_query(F.data == "admin_menu:analytics")
@require_admin
async def handle_admin_menu_analytics(callback: CallbackQuery, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Показать раздел аналитики из главного меню."""

    summary = await _build_admin_summary(db)
    await edit_or_answer(callback, summary, create_analytics_menu())
    await callback.answer()


@admin_router.callback_query(F.data == "admin_menu:refresh")
@require_admin
async def handle_admin_menu_refresh(callback: CallbackQuery, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Обновить данные на главном экране."""

    summary = await _build_admin_summary(db)

    if callback.message:
        await edit_or_answer(callback, summary, create_main_menu())
    await callback.answer("✅ Обновлено")


@admin_router.callback_query(F.data == "admin_menu:back")
@require_admin
async def handle_admin_menu_back(callback: CallbackQuery, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Вернуться в главное меню админ-панели."""

    summary = await _build_admin_summary(db)
    await edit_or_answer(callback, summary, create_main_menu())
    await callback.answer()


@admin_router.callback_query(F.data.startswith("admin_segment:"))
@require_admin
async def handle_segment_view(callback: CallbackQuery, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Просмотр детальной информации по сегменту"""

    if not callback.data:
        await callback.answer("❌ Ошибка данных")
        return

    segment_id = callback.data.replace("admin_segment:", "")

    analytics = AdminAnalytics(_resolve_db(db))
    segments = await analytics.get_user_segments()

    if segment_id not in segments:
        await callback.answer("❌ Сегмент не найден")
        return

    segment = segments[segment_id]

    # Форматируем вывод
    output = analytics.format_segment_summary(segment, max_users=10)

    # Добавляем кнопку возврата
    keyboard = back_keyboard("admin_refresh")

    await edit_or_answer(callback, output, keyboard)
    await callback.answer()


@admin_router.callback_query(F.data == "admin_stats:conversion")
@require_admin
async def handle_conversion_stats(callback: CallbackQuery, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Детальная статистика конверсии"""

    analytics = AdminAnalytics(_resolve_db(db))
    conversion = await analytics.get_conversion_metrics()
    churn = await analytics.get_churn_metrics(period_days=30)

    output = f"""
<b>💹 ДЕТАЛЬНАЯ СТАТИСТИКА КОНВЕРСИИ</b>

<b>📊 Триал → Оплата:</b>
• Всего пользователей на триале: {conversion.total_trial_users}
• Перешли на оплату: {conversion.converted_to_paid}
• Конверсия: <b>{conversion.conversion_rate}%</b>
• Среднее число запросов до покупки: {conversion.avg_trial_requests_before_conversion}
• Среднее время до конверсии: {conversion.avg_time_to_conversion_days} дней

<b>📉 Отток (30 дней):</b>
• Истекло подписок: {churn.total_expired}
• Продлили: {churn.renewed_count}
• Ушли (отток): {churn.churned_count}
• Уровень удержания: <b>{churn.retention_rate}%</b>

<b>💡 Рекомендации:</b>
"""

    # Добавляем рекомендации на основе данных
    if conversion.conversion_rate < 10:
        output += "⚠️ Низкая конверсия триала — пересмотреть лимиты и онбординг\n"
    if churn.retention_rate < 50:
        output += "⚠️ Высокий отток — пересмотреть стратегию удержания\n"
    if conversion.avg_time_to_conversion_days > 7:
        output += "⚠️ Долгая конверсия — добавить стимулирующие акции для быстрой покупки\n"

    if not any([conversion.conversion_rate < 10, churn.retention_rate < 50, conversion.avg_time_to_conversion_days > 7]):
        output += "✅ Показатели в норме!\n"

    keyboard = back_keyboard("admin_refresh")

    await edit_or_answer(callback, output, keyboard)
    await callback.answer()


@admin_router.callback_query(F.data == "admin_stats:daily")
@require_admin
async def handle_daily_stats(callback: CallbackQuery, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Ежедневная статистика"""

    analytics = AdminAnalytics(_resolve_db(db))
    daily_stats = await analytics.get_daily_stats(days=7)

    output = "<b>📈 ЕЖЕДНЕВНАЯ СТАТИСТИКА (7 дней)</b>\n\n"

    for day in daily_stats:
        output += f"<b>{day['date']}</b>\n"
        output += f"  • Запросов: {day['requests']}\n"
        output += f"  • Активных пользователей: {day['active_users']}\n"
        output += f"  • Токенов: {day['total_tokens']:,}\n"
        output += f"  • Среднее время ответа: {day['avg_response_time_ms']} мс\n\n"

    if daily_stats:
        # Добавляем тренды
        latest = daily_stats[0]
        prev = daily_stats[1] if len(daily_stats) > 1 else latest

        requests_change = ((latest['requests'] - prev['requests']) / max(prev['requests'], 1)) * 100
        users_change = ((latest['active_users'] - prev['active_users']) / max(prev['active_users'], 1)) * 100

        output += "<b>📊 Тренды (день к дню):</b>\n"
        output += f"  • Запросы: {requests_change:+.1f}%\n"
        output += f"  • Активные пользователи: {users_change:+.1f}%\n"

    keyboard = back_keyboard("admin_refresh")

    await edit_or_answer(callback, output, keyboard)
    await callback.answer()


@admin_router.callback_query(F.data == "admin_refresh")
@require_admin
async def handle_refresh(callback: CallbackQuery, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Обновление раздела аналитики"""

    summary = await _build_admin_summary(db)

    if callback.message:
        await edit_or_answer(callback, summary, create_analytics_menu())
    await callback.answer("✅ Обновлено")


@admin_router.message(Command("export_users"))
@require_admin
async def cmd_export_users(message: Message, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Экспорт списка пользователей определенного сегмента"""

    # Парсим аргументы команды: /export_users <segment>
    args = (message.text or "").split(maxsplit=1)
    segment_id = args[1] if len(args) > 1 else "power_users"

    analytics = AdminAnalytics(_resolve_db(db))
    segments = await analytics.get_user_segments()

    if segment_id not in segments:
        await message.answer(
            f"{Emoji.ERROR} Неизвестный сегмент: {segment_id}\n\n"
            f"Доступные: {', '.join(segments.keys())}"
        )
        return

    segment = segments[segment_id]

    # Формируем CSV
    csv_lines = ["user_id,всего_запросов,последняя_активность,доп_инфо"]

    for user in segment.users:
        user_id = user.get('user_id', 'н/д')
        total_requests = user.get('total_requests', 0)
        last_active = user.get('last_active', user.get('registered_at', 'н/д'))

        # Дополнительная информация зависит от сегмента
        if segment_id == 'power_users':
            additional = f"{user.get('avg_requests_per_day', 0)} запр./день"
        elif segment_id == 'at_risk':
            additional = f"истекает через {user.get('days_until_expiry', 0)} дн."
        elif segment_id == 'churned':
            additional = f"LTV: {user.get('ltv', 0)} ₽"
        else:
            additional = ""

        csv_lines.append(f"{user_id},{total_requests},{last_active},{additional}")

    csv_content = "\n".join(csv_lines)

    # Отправляем как файл
    from io import BytesIO
    from aiogram.types import BufferedInputFile

    file_bytes = BytesIO(csv_content.encode('utf-8'))
    file = BufferedInputFile(file_bytes.getvalue(), filename=f"{segment_id}_export.csv")

    await message.answer_document(
        file,
        caption=f"📊 Экспорт сегмента: <b>{segment.name}</b>\n"
                f"Пользователей: {segment.user_count}",
        parse_mode=ParseMode.HTML
    )


@admin_router.message(Command("broadcast"))
@require_admin
async def cmd_broadcast(message: Message, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """
    Отправка сообщения группе пользователей
    Использование: /broadcast <segment> <сообщение>
    """

    args = (message.text or "").split(maxsplit=2)

    if len(args) < 3:
        await message.answer(
            f"{Emoji.INFO} <b>Использование:</b>\n"
            f"/broadcast &lt;segment&gt; &lt;сообщение&gt;\n\n"
            f"<b>Доступные сегменты:</b>\n"
            f"• power_users\n"
            f"• at_risk\n"
            f"• churned\n"
            f"• trial_converters\n"
            f"• freeloaders\n"
            f"• new_users\n"
            f"• vip",
            parse_mode=ParseMode.HTML
        )
        return

    segment_id = args[1]
    broadcast_message = args[2]

    analytics = AdminAnalytics(_resolve_db(db))
    segments = await analytics.get_user_segments()

    if segment_id not in segments:
        await message.answer(f"{Emoji.ERROR} Неизвестный сегмент: {segment_id}")
        return

    segment = segments[segment_id]
    user_ids = [user['user_id'] for user in segment.users]

    # Подтверждение
    confirm_keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"✅ Отправить {len(user_ids)} польз.",
                    callback_data=f"broadcast_confirm:{segment_id}"
                ),
                InlineKeyboardButton(text="❌ Отмена", callback_data="broadcast_cancel"),
            ]
        ]
    )

    # Сохраняем сообщение в сессии (в реальности нужна БД или кеш)
    # Для простоты используем message.bot.data
    if not hasattr(message.bot, '_broadcast_cache'):
        message.bot._broadcast_cache = {}  # type: ignore

    cache_key = f"{message.from_user.id}:{segment_id}"
    message.bot._broadcast_cache[cache_key] = {  # type: ignore
        'user_ids': user_ids,
        'message': broadcast_message,
        'segment_name': segment.name,
    }

    await message.answer(
        f"<b>⚠️ Подтверждение рассылки</b>\n\n"
        f"Сегмент: <b>{segment.name}</b>\n"
        f"Получателей: <b>{len(user_ids)}</b>\n\n"
        f"<b>Сообщение:</b>\n{html_escape(broadcast_message)}\n\n"
        f"Подтвердите отправку:",
        parse_mode=ParseMode.HTML,
        reply_markup=confirm_keyboard
    )



@admin_router.callback_query(F.data.startswith("broadcast_confirm:"))
@require_admin
async def handle_broadcast_confirm(callback: CallbackQuery, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Trigger broadcast delivery after admin confirmation."""

    data = callback.data or ""
    _, _, segment_id = data.partition(":")
    cache = getattr(callback.bot, "_broadcast_cache", {})
    cache_key = f"{callback.from_user.id}:{segment_id}"
    payload = cache.pop(cache_key, None) if segment_id else None

    if not payload:
        await callback.answer("Broadcast payload not found", show_alert=True)
        if callback.message:
            await edit_or_answer(callback, "<b>Broadcast payload not found.</b>", None)
        return

    user_ids = payload.get('user_ids') or []
    message_text = payload.get('message') or ""
    segment_label = payload.get('segment_name') or segment_id or 'unknown'

    sent = 0
    failed = []

    for user_id in user_ids:
        try:
            await send_html_text(callback.bot, user_id, message_text)
            sent += 1
        except Exception as exc:
            failed.append(user_id)
            logger.warning("Failed to deliver broadcast to %s: %s", user_id, exc)

    summary_lines = [
        "<b>Broadcast completed</b>",
        "",
        f"Segment: <b>{segment_label}</b>",
        f"Recipients: {sent}",
    ]
    if failed:
        summary_lines.append(f"Failures: {len(failed)}")

    summary_text = "\n".join(summary_lines)

    if callback.message:
        await edit_or_answer(callback, summary_text, None)

    await callback.answer("Done")
    logger.info(
        "Admin %s broadcasted to %s: sent=%s failed=%s",
        callback.from_user.id if callback.from_user else 'unknown',
        segment_id,
        sent,
        len(failed),
    )


@admin_router.callback_query(F.data.startswith("broadcast_cancel:"))
@require_admin
async def handle_broadcast_cancel(callback: CallbackQuery, db: DatabaseAdvanced | None = None, admin_ids: set[int] | None = None):
    """Cancel a prepared broadcast."""

    data = callback.data or ""
    _, _, segment_id = data.partition(":")
    cache = getattr(callback.bot, "_broadcast_cache", {})
    if segment_id:
        cache.pop(f"{callback.from_user.id}:{segment_id}", None)

    if callback.message:
        await edit_or_answer(callback, "<b>Broadcast cancelled.</b>", None)

    await callback.answer("Cancelled")
    logger.info(
        "Admin %s cancelled broadcast for %s",
        callback.from_user.id if callback.from_user else 'unknown',
        segment_id,
    )


def setup_admin_commands(dp, db: DatabaseAdvanced, admin_ids: set[int]):
    """
    ����������� �����-������ � dispatcher

    �������������:
        setup_admin_commands(dp, db, {123456, 789012})
    """
    global _GLOBAL_DB
    _GLOBAL_DB = db
    set_admin_ids(admin_ids)

    routers = [
        admin_router,
        alerts_router,
        behavior_router,
        cohort_router,
        pmf_router,
        retention_router,
        revenue_router,
    ]

    for router in routers:
        router.message.filter(lambda msg, _admins=admin_ids: msg.from_user and msg.from_user.id in _admins)

        message_observer = router.observers.get('message')
        if message_observer is not None:
            for handler in getattr(message_observer, 'handlers', []):
                handler.callback.__globals__['db'] = db
                handler.callback.__globals__['admin_ids'] = admin_ids

        callback_observer = router.observers.get('callback_query')
        if callback_observer is not None:
            for handler in getattr(callback_observer, 'handlers', []):
                handler.callback.__globals__['db'] = db
                handler.callback.__globals__['admin_ids'] = admin_ids

        dp.include_router(router)

        logger.info(f"Супер-админы авторизованы для {len(admin_ids)} команд")


__all__ = (
    "admin_router",
    "cmd_admin",
    "handle_admin_menu_analytics",
    "handle_admin_menu_refresh",
    "handle_admin_menu_back",
    "handle_segment_view",
    "handle_conversion_stats",
    "handle_daily_stats",
    "handle_refresh",
    "cmd_export_users",
    "cmd_broadcast",
    "handle_broadcast_confirm",
    "handle_broadcast_cancel",
    "setup_admin_commands",
)
