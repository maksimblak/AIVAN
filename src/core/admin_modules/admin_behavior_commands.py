"""
Admin команды для поведенческой аналитики
Что нравится пользователям, что не нравится, где отваливаются
"""

from __future__ import annotations

import logging
from html import escape as html_escape

from aiogram import F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from src.bot.ui_components import Emoji
from src.core.admin_modules.admin_utils import back_keyboard, render_dashboard, require_admin
from src.core.user_behavior_tracker import UserBehaviorTracker

logger = logging.getLogger(__name__)

behavior_router = Router()


def create_behavior_menu() -> InlineKeyboardMarkup:
    """Меню поведенческой аналитики"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📊 Популярные фичи", callback_data="behavior:popular"),
                InlineKeyboardButton(text="💔 Заброшенные фичи", callback_data="behavior:abandoned"),
            ],
            [
                InlineKeyboardButton(text="🔥 Точки трения", callback_data="behavior:friction"),
                InlineKeyboardButton(text="😊 Feedback по фичам", callback_data="behavior:feedback"),
            ],
            [
                InlineKeyboardButton(text="🛣️ User Journey", callback_data="behavior:journey"),
                InlineKeyboardButton(text="⏰ Пик активности", callback_data="behavior:peak_hours"),
            ],
            [
                InlineKeyboardButton(text="🎯 Вовлеченность", callback_data="behavior:engagement"),
                InlineKeyboardButton(text="📉 Underutilized", callback_data="behavior:underutilized"),
            ],
            [
                InlineKeyboardButton(text="« Назад", callback_data="admin_refresh"),
            ],
        ]
    )


@behavior_router.message(Command("behavior"))
@require_admin
async def cmd_behavior(message: Message, db, admin_ids: set[int]):
    """Главная команда поведенческой аналитики"""
    tracker = UserBehaviorTracker(db)

    # Краткая сводка
    top_features = await tracker.get_top_features(days=7, limit=5)
    frictions = await tracker.identify_friction_points(days=7)

    summary = "<b>🎯 ПОВЕДЕНЧЕСКАЯ АНАЛИТИКА</b>\n\n"

    summary += "<b>📊 Топ-5 фичей за неделю:</b>\n"
    for i, feat in enumerate(top_features, 1):
        emoji = '🔥' if i == 1 else '⭐' if i <= 3 else '✅'
        summary += f"{emoji} {feat['feature']}: {feat['uses']} использований ({feat['unique_users']} польз.)\n"

    summary += f"\n<b>🔥 Точек трения найдено:</b> {len(frictions)}\n"

    if frictions:
        top_friction = frictions[0]
        summary += f"Самая критичная: <b>{top_friction.location}</b> (impact: {top_friction.impact_score:.0f}/100)\n"

    summary += "\n<i>Выберите раздел для детального анализа:</i>"

    async def build_dashboard():
        return summary, create_behavior_menu()

    await render_dashboard(build_dashboard, message)


@behavior_router.callback_query(F.data == "behavior:popular")
@require_admin
async def handle_popular_features(callback: CallbackQuery, db, admin_ids: set[int]):
    """Популярные фичи с детальной статистикой"""
    tracker = UserBehaviorTracker(db)
    top_features = await tracker.get_top_features(days=30, limit=10)

    output = "<b>📊 ПОПУЛЯРНЫЕ ФИЧИ (30 дней)</b>\n\n"

    for i, feat in enumerate(top_features, 1):
        medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f"{i}."

        output += f"{medal} <b>{feat['feature']}</b>\n"
        output += f"   • Использований: {feat['uses']}\n"
        output += f"   • Уникальных пользователей: {feat['unique_users']}\n"
        output += f"   • Успешность: {feat['success_rate']:.1f}%\n"

        if feat['avg_duration_ms'] > 0:
            output += f"   • Среднее время: {feat['avg_duration_ms'] / 1000:.1f}с\n"

        output += "\n"

    # Insights
    if top_features:
        total_uses = sum(f['uses'] for f in top_features)
        top_3_uses = sum(f['uses'] for f in top_features[:3])
        concentration = (top_3_uses / total_uses) * 100

        output += "<b>💡 Инсайты:</b>\n"
        output += f"• Топ-3 фичи составляют {concentration:.0f}% всего использования\n"

        if concentration > 80:
            output += "⚠️ Высокая концентрация - пользователи игнорируют другие фичи\n"

    async def build_dashboard():
        return output, back_keyboard("behavior:menu")

    await render_dashboard(build_dashboard, callback)
    await callback.answer()


@behavior_router.callback_query(F.data == "behavior:friction")
@require_admin
async def handle_friction_points(callback: CallbackQuery, db, admin_ids: set[int]):
    """Точки трения где пользователи застревают"""
    tracker = UserBehaviorTracker(db)
    frictions = await tracker.identify_friction_points(days=14)

    output = tracker.format_friction_report(frictions)

    if frictions:
        output += "\n<b>🔧 Рекомендуемые действия:</b>\n"

        for friction in frictions[:3]:
            if friction.friction_type == 'error':
                output += f"• {friction.location}: исправить баги, улучшить error handling\n"
            elif friction.friction_type == 'abandon':
                output += f"• {friction.location}: упростить UX, добавить подсказки\n"
            elif friction.friction_type == 'timeout':
                output += f"• {friction.location}: оптимизировать производительность\n"
            elif friction.friction_type == 'confusion':
                output += f"• {friction.location}: улучшить онбординг, добавить туториал\n"

    async def build_dashboard():
        return output, back_keyboard("behavior:menu")

    await render_dashboard(build_dashboard, callback)
    await callback.answer()


@behavior_router.callback_query(F.data == "behavior:engagement")
@require_admin
async def handle_engagement(callback: CallbackQuery, db, admin_ids: set[int]):
    """Детальные метрики вовлеченности"""
    tracker = UserBehaviorTracker(db)
    engagements = await tracker.get_feature_engagement(days=30)

    output = tracker.format_engagement_report(engagements)

    # Добавляем общую статистику
    if engagements:
        avg_repeat = sum(e.repeat_usage_rate for e in engagements) / len(engagements)
        avg_satisfaction = sum(e.satisfaction_score for e in engagements) / len(engagements)

        output += "<b>📈 Общая вовлеченность:</b>\n"
        output += f"• Средний repeat usage: {avg_repeat:.1f}%\n"
        output += f"• Средняя удовлетворенность: {avg_satisfaction:.0f}/100\n\n"

        # Классификация фич
        rising = [e for e in engagements if e.trend == 'rising']
        declining = [e for e in engagements if e.trend == 'declining']

        if rising:
            output += f"📈 Растущие фичи: {', '.join(e.feature_name for e in rising)}\n"
        if declining:
            output += f"📉 Падающие фичи: {', '.join(e.feature_name for e in declining)}\n"

    async def build_dashboard():
        return output, back_keyboard("behavior:menu")

    await render_dashboard(build_dashboard, callback)
    await callback.answer()


@behavior_router.callback_query(F.data == "behavior:underutilized")
@require_admin
async def handle_underutilized(callback: CallbackQuery, db, admin_ids: set[int]):
    """Фичи которые не используются"""
    tracker = UserBehaviorTracker(db)
    underutilized = await tracker.get_underutilized_features(days=30)

    output = "<b>💔 НЕДОИСПОЛЬЗУЕМЫЕ ФИЧИ</b>\n\n"

    if not underutilized:
        output += "✅ Все фичи активно используются!\n"
    else:
        unused = [f for f in underutilized if f['status'] == 'unused']
        low_use = [f for f in underutilized if f['status'] == 'underutilized']

        if unused:
            output += "<b>🚫 Совсем не используются (0 использований):</b>\n"
            for feat in unused:
                output += f"• {feat['feature']}\n"
            output += "\n"

        if low_use:
            output += "<b>⚠️ Используются редко (<10 раз за месяц):</b>\n"
            for feat in low_use:
                output += f"• {feat['feature']}: {feat['uses']} раз\n"
            output += "\n"

        output += "<b>💡 Возможные причины:</b>\n"
        output += "• Пользователи не знают о фиче (плохая discovery)\n"
        output += "• Фича не нужна целевой аудитории\n"
        output += "• Сложный UX или барьер входа\n"
        output += "• Фича не отвечает ожиданиям\n\n"

        output += "<b>🔧 Что делать:</b>\n"
        output += "1. Улучшить onboarding и tutorials\n"
        output += "2. Добавить промо фичи в bot flow\n"
        output += "3. Провести опрос - нужна ли фича?\n"
        output += "4. Рассмотреть удаление если совсем не востребована\n"

    async def build_dashboard():
        return output, back_keyboard("behavior:menu")

    await render_dashboard(build_dashboard, callback)
    await callback.answer()


@behavior_router.callback_query(F.data == "behavior:peak_hours")
@require_admin
async def handle_peak_hours(callback: CallbackQuery, db, admin_ids: set[int]):
    """Пиковые часы активности"""
    tracker = UserBehaviorTracker(db)
    hourly_stats = await tracker.get_usage_by_hour(days=7)

    output = "<b>⏰ АКТИВНОСТЬ ПО ЧАСАМ (7 дней)</b>\n\n"

    # Находим пиковые часы
    if hourly_stats:
        sorted_hours = sorted(hourly_stats.items(), key=lambda x: x[1]['total_events'], reverse=True)

        output += "<b>🔥 Топ-5 пиковых часов:</b>\n"
        for hour, stats in sorted_hours[:5]:
            output += f"{hour:02d}:00 - {stats['total_events']} событий ({stats['unique_users']} польз.)\n"

        output += "\n<b>📊 Распределение по времени суток:</b>\n"

        morning = sum(stats['total_events'] for h, stats in hourly_stats.items() if 6 <= h < 12)
        day = sum(stats['total_events'] for h, stats in hourly_stats.items() if 12 <= h < 18)
        evening = sum(stats['total_events'] for h, stats in hourly_stats.items() if 18 <= h < 24)
        night = sum(stats['total_events'] for h, stats in hourly_stats.items() if h < 6)

        total = morning + day + evening + night

        if total > 0:
            output += f"🌅 Утро (6-12): {morning} ({morning/total*100:.1f}%)\n"
            output += f"☀️ День (12-18): {day} ({day/total*100:.1f}%)\n"
            output += f"🌆 Вечер (18-24): {evening} ({evening/total*100:.1f}%)\n"
            output += f"🌙 Ночь (0-6): {night} ({night/total*100:.1f}%)\n\n"

        # Рекомендации
        peak_hour = sorted_hours[0][0]
        output += "<b>💡 Рекомендации:</b>\n"
        output += f"• Планируйте обновления вне пика ({peak_hour:02d}:00)\n"
        output += f"• Делайте анонсы в пиковые часы\n"
        output += f"• Усиливайте support в пиковое время\n"

    async def build_dashboard():
        return output, back_keyboard("behavior:menu")

    await render_dashboard(build_dashboard, callback)
    await callback.answer()


@behavior_router.callback_query(F.data == "behavior:feedback")
@require_admin
async def handle_feature_feedback(callback: CallbackQuery, db, admin_ids: set[int]):
    """Feedback по отдельным фичам"""
    tracker = UserBehaviorTracker(db)

    # Получаем feedback по топ фичам
    top_features = await tracker.get_top_features(days=30, limit=5)

    output = "<b>😊 FEEDBACK ПО ФИЧАМ</b>\n\n"

    for feat in top_features:
        feature_name = feat['feature']
        feedback = await tracker.get_feature_feedback(feature_name, days=30)

        sentiment_emoji = '😍' if feedback.net_sentiment > 50 else '😊' if feedback.net_sentiment > 0 else '😐' if feedback.net_sentiment > -50 else '😞'

        output += f"<b>{feature_name}</b> {sentiment_emoji}\n"
        output += f"  👍 Положительных: {feedback.positive_signals}\n"
        output += f"  👎 Негативных: {feedback.negative_signals}\n"
        output += f"  📊 Net sentiment: {feedback.net_sentiment:+.0f}\n"

        if feedback.explicit_feedback:
            output += f"  💬 Комментариев: {len(feedback.explicit_feedback)}\n"
            # Показываем 1-2 последних комментария
            for comment in feedback.explicit_feedback[:2]:
                truncated = comment[:80] + "..." if len(comment) > 80 else comment
                output += f"     • <i>{html_escape(truncated)}</i>\n"

        output += "\n"

    async def build_dashboard():
        return output, back_keyboard("behavior:menu")

    await render_dashboard(build_dashboard, callback)
    await callback.answer()


@behavior_router.callback_query(F.data == "behavior:journey")
@require_admin
async def handle_user_journey(callback: CallbackQuery, db, admin_ids: set[int]):
    """Анализ типичного пути пользователя"""
    # Запрашиваем user_id для детального анализа
    output = "<b>🛣️ АНАЛИЗ USER JOURNEY</b>\n\n"
    output += "Для детального анализа используйте:\n"
    output += "<code>/journey &lt;user_id&gt;</code>\n\n"

    output += "<b>📊 Типичные паттерны:</b>\n"
    output += "• Успешный путь: регистрация → trial → первый вопрос → voice → payment\n"
    output += "• Проблемный путь: регистрация → trial → ошибка → abandonment\n\n"

    output += "<i>Используйте команду /journey с конкретным user_id для деталей</i>"

    async def build_dashboard():
        return output, back_keyboard("behavior:menu")

    await render_dashboard(build_dashboard, callback)
    await callback.answer()


@behavior_router.message(Command("journey"))
@require_admin
async def cmd_user_journey(message: Message, db, admin_ids: set[int]):
    """Детальный путь конкретного пользователя"""
    args = (message.text or "").split()
    if len(args) < 2:
        await message.answer(
            "Использование: /journey <user_id>\n"
            "Пример: /journey 123456789"
        )
        return

    try:
        user_id = int(args[1])
    except ValueError:
        await message.answer(f"{Emoji.ERROR} Неверный формат user_id")
        return

    tracker = UserBehaviorTracker(db)
    journey = await tracker.get_user_journey(user_id)

    output = f"<b>🛣️ USER JOURNEY #{user_id}</b>\n\n"

    if not journey.journey_steps:
        output += "Нет данных о активности этого пользователя"
    else:
        output += f"<b>Статус:</b> {'✅ Завершен' if journey.completed else '⏳ В процессе'}\n"
        output += f"<b>Всего времени:</b> {journey.total_time_seconds // 60} минут\n"

        if journey.drop_off_point:
            output += f"<b>Drop-off point:</b> ⚠️ {journey.drop_off_point}\n"

        if journey.friction_points:
            output += f"<b>Friction points:</b> {', '.join(journey.friction_points)}\n"

        output += f"\n<b>📍 Путь ({len(journey.journey_steps)} шагов):</b>\n"

        for step in journey.journey_steps[:15]:  # первые 15 шагов
            emoji = '✅' if step['success'] else '❌'
            output += f"{step['step_number']}. {emoji} {step['feature']}\n"

        if len(journey.journey_steps) > 15:
            output += f"... и еще {len(journey.journey_steps) - 15} шагов\n"

    await message.answer(output, parse_mode=ParseMode.HTML)


@behavior_router.callback_query(F.data == "behavior:menu")
@require_admin
async def back_to_behavior_menu(callback: CallbackQuery, db, admin_ids: set[int]):
    """Возврат в главное меню поведенческой аналитики"""
    tracker = UserBehaviorTracker(db)
    top_features = await tracker.get_top_features(days=7, limit=5)
    frictions = await tracker.identify_friction_points(days=7)

    summary = "<b>🎯 ПОВЕДЕНЧЕСКАЯ АНАЛИТИКА</b>\n\n"

    summary += "<b>📊 Топ-5 фичей за неделю:</b>\n"
    for i, feat in enumerate(top_features, 1):
        emoji = '🔥' if i == 1 else '⭐' if i <= 3 else '✅'
        summary += f"{emoji} {feat['feature']}: {feat['uses']} использований\n"

    summary += f"\n<b>🔥 Точек трения:</b> {len(frictions)}\n"

    summary += "\n<i>Выберите раздел для детального анализа:</i>"

    async def build_dashboard():
        return summary, create_behavior_menu()

    await render_dashboard(build_dashboard, callback)
    await callback.answer()
