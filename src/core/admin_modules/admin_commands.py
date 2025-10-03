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
from src.core.admin_modules.admin_analytics import AdminAnalytics
from src.core.admin_modules.admin_utils import back_keyboard, edit_or_answer
from src.core.safe_telegram import send_html_text

if TYPE_CHECKING:
    from src.core.db_advanced import DatabaseAdvanced

logger = logging.getLogger(__name__)

# Создаем router для admin команд
admin_router = Router()


def is_admin(user_id: int, admin_ids: set[int]) -> bool:
    """Проверка, является ли пользователь админом"""
    return user_id in admin_ids


def create_analytics_menu() -> InlineKeyboardMarkup:
    """Создание главного меню аналитики"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
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
            [
                InlineKeyboardButton(text="👑 VIP", callback_data="admin_segment:vip"),
            ],
            [
                InlineKeyboardButton(text="📊 Аналитика конверсии", callback_data="admin_stats:conversion"),
                InlineKeyboardButton(text="📈 Ежедневная статистика", callback_data="admin_stats:daily"),
            ],
            [
                InlineKeyboardButton(text="🔄 Обновить", callback_data="admin_refresh"),
            ],
        ]
    )


@admin_router.message(Command("admin"))
async def cmd_admin(message: Message, db: DatabaseAdvanced, admin_ids: set[int]):
    """Главная команда админ-панели"""
    if not message.from_user or not is_admin(message.from_user.id, admin_ids):
        await message.answer(f"{Emoji.ERROR} У вас нет доступа к админ-панели")
        return

    analytics = AdminAnalytics(db)

    # Получаем краткую сводку
    segments = await analytics.get_user_segments()
    conversion_metrics = await analytics.get_conversion_metrics()

    summary = f"""
<b>🎛 АДМИН-ПАНЕЛЬ</b>

<b>📊 Сводка по пользователям:</b>

⚡ Суперактивные: <b>{segments['power_users'].user_count}</b>
⚠️ Группа риска: <b>{segments['at_risk'].user_count}</b>
📉 Отток: <b>{segments['churned'].user_count}</b>
💰 Переходы из триала: <b>{segments['trial_converters'].user_count}</b>
🚫 Только бесплатные: <b>{segments['freeloaders'].user_count}</b>
🆕 Новые пользователи (7 дн.): <b>{segments['new_users'].user_count}</b>
👑 VIP-пользователи: <b>{segments['vip'].user_count}</b>

<b>💹 Конверсия Триал → Оплата:</b>
• Всего триал-пользователей: {conversion_metrics.total_trial_users}
• Перешли на оплату: {conversion_metrics.converted_to_paid}
• Конверсия: <b>{conversion_metrics.conversion_rate}%</b>
• Среднее время до покупки: {conversion_metrics.avg_time_to_conversion_days} дней

<i>Выберите раздел для детального анализа:</i>
"""

    await message.answer(summary, parse_mode=ParseMode.HTML, reply_markup=create_analytics_menu())


@admin_router.callback_query(F.data.startswith("admin_segment:"))
async def handle_segment_view(callback: CallbackQuery, db: DatabaseAdvanced, admin_ids: set[int]):
    """Просмотр детальной информации по сегменту"""
    if not callback.from_user or not is_admin(callback.from_user.id, admin_ids):
        await callback.answer("❌ Нет доступа", show_alert=True)
        return

    if not callback.data:
        await callback.answer("❌ Ошибка данных")
        return

    segment_id = callback.data.replace("admin_segment:", "")

    analytics = AdminAnalytics(db)
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
async def handle_conversion_stats(callback: CallbackQuery, db: DatabaseAdvanced, admin_ids: set[int]):
    """Детальная статистика конверсии"""
    if not callback.from_user or not is_admin(callback.from_user.id, admin_ids):
        await callback.answer("❌ Нет доступа", show_alert=True)
        return

    analytics = AdminAnalytics(db)
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
async def handle_daily_stats(callback: CallbackQuery, db: DatabaseAdvanced, admin_ids: set[int]):
    """Ежедневная статистика"""
    if not callback.from_user or not is_admin(callback.from_user.id, admin_ids):
        await callback.answer("❌ Нет доступа", show_alert=True)
        return

    analytics = AdminAnalytics(db)
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
async def handle_refresh(callback: CallbackQuery, db: DatabaseAdvanced, admin_ids: set[int]):
    """Обновление главного меню"""
    if not callback.from_user or not is_admin(callback.from_user.id, admin_ids):
        await callback.answer("❌ Нет доступа", show_alert=True)
        return

    analytics = AdminAnalytics(db)
    segments = await analytics.get_user_segments()
    conversion_metrics = await analytics.get_conversion_metrics()

    summary = f"""
<b>🎛 АДМИН-ПАНЕЛЬ</b>

<b>📊 Сводка по пользователям:</b>

⚡ Суперактивные: <b>{segments['power_users'].user_count}</b>
⚠️ Группа риска: <b>{segments['at_risk'].user_count}</b>
📉 Отток: <b>{segments['churned'].user_count}</b>
💰 Переходы из триала: <b>{segments['trial_converters'].user_count}</b>
🚫 Только бесплатные: <b>{segments['freeloaders'].user_count}</b>
🆕 Новые пользователи (7 дн.): <b>{segments['new_users'].user_count}</b>
👑 VIP-пользователи: <b>{segments['vip'].user_count}</b>

<b>💹 Конверсия Триал → Оплата:</b>
• Всего триал-пользователей: {conversion_metrics.total_trial_users}
• Перешли на оплату: {conversion_metrics.converted_to_paid}
• Конверсия: <b>{conversion_metrics.conversion_rate}%</b>
• Среднее время до покупки: {conversion_metrics.avg_time_to_conversion_days} дней

<i>Выберите раздел для детального анализа:</i>
"""

    if callback.message:
        await edit_or_answer(callback, summary, create_analytics_menu())
    await callback.answer("✅ Обновлено")


@admin_router.message(Command("export_users"))
async def cmd_export_users(message: Message, db: DatabaseAdvanced, admin_ids: set[int]):
    """Экспорт списка пользователей определенного сегмента"""
    if not message.from_user or not is_admin(message.from_user.id, admin_ids):
        await message.answer(f"{Emoji.ERROR} У вас нет доступа к админ-панели")
        return

    # Парсим аргументы команды: /export_users <segment>
    args = (message.text or "").split(maxsplit=1)
    segment_id = args[1] if len(args) > 1 else "power_users"

    analytics = AdminAnalytics(db)
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
async def cmd_broadcast(message: Message, db: DatabaseAdvanced, admin_ids: set[int]):
    """
    Отправка сообщения группе пользователей
    Использование: /broadcast <segment> <сообщение>
    """
    if not message.from_user or not is_admin(message.from_user.id, admin_ids):
        await message.answer(f"{Emoji.ERROR} У вас нет доступа к админ-панели")
        return

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

    analytics = AdminAnalytics(db)
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
        'message': broadcast_message
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


# TODO: Добавить обработчик broadcast_confirm для фактической отправки


def setup_admin_commands(dp, db: DatabaseAdvanced, admin_ids: set[int]):
    """
    Регистрация админ-команд в dispatcher

    Использование:
        setup_admin_commands(dp, db, {123456, 789012})
    """
    # Регистрируем router с передачей зависимостей
    admin_router.message.filter(lambda msg: msg.from_user and is_admin(msg.from_user.id, admin_ids))

    # Передаем db и admin_ids как зависимости
    for handler in admin_router.observers['message']:
        handler.callback.__globals__['db'] = db
        handler.callback.__globals__['admin_ids'] = admin_ids

    for handler in admin_router.observers['callback_query']:
        handler.callback.__globals__['db'] = db
        handler.callback.__globals__['admin_ids'] = admin_ids

    dp.include_router(admin_router)

    logger.info(f"✅ Админ-команды зарегистрированы для {len(admin_ids)} админов")
