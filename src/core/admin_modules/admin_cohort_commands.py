"""
Админ-команды для когортного анализа
"""

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from src.core.admin_modules.admin_formatters import format_trend
from src.core.admin_modules.admin_utils import FEATURE_KEYS, edit_or_answer, require_admin
from src.core.admin_modules.cohort_analytics import CohortAnalytics

cohort_router = Router(name="cohort_admin")


@cohort_router.message(Command("cohort"))
@require_admin
async def cmd_cohort(message: Message, db, admin_ids: list[int]):
    """Главное меню когортного анализа"""
    analytics = CohortAnalytics(db)

    # Получить сравнение когорт
    comparison = await analytics.compare_cohorts(months_back=6)

    # Формирование сообщения
    text = "📊 <b>Когортный анализ — динамика удержания</b>\n\n"

    text += f"🏆 <b>Лучшая когорта:</b> {comparison.best_cohort}\n"
    text += f"📉 <b>Худшая когорта:</b> {comparison.worst_cohort}\n\n"

    text += f"📈 <b>Тренд удержания:</b> {format_trend(comparison.retention_trend)}\n"
    text += f"💰 <b>Тренд конверсии:</b> {format_trend(comparison.conversion_trend)}\n\n"

    text += "<b>🔍 Ключевые выводы:</b>\n"
    for insight in comparison.key_insights[:5]:
        text += f"• {insight}\n"

    text += "\n<b>📅 Когорты (последние 6 месяцев):</b>\n\n"

    for cohort in comparison.cohorts_data[:6]:
        text += f"<b>{cohort.cohort_month}</b> ({cohort.cohort_size} пользователей)\n"
        text += f"  День 1: {cohort.day_1_retention:.1f}% | День 7: {cohort.day_7_retention:.1f}%\n"
        text += (
            f"  День 30: {cohort.day_30_retention:.1f}% | День 90: {cohort.day_90_retention:.1f}%\n"
        )
        text += f"  Конверсия: {cohort.conversion_rate:.1f}% | ARPU: {cohort.arpu:.0f}₽\n\n"

    # Кнопки
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Детали когорты", callback_data="cohort:select_month")],
            [
                InlineKeyboardButton(
                    text="🎯 Использование функций", callback_data="cohort:feature_adoption"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📈 Кривые удержания", callback_data="cohort:retention_curves"
                )
            ],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="cohort:refresh")],
        ]
    )

    await message.answer(text, parse_mode="HTML", reply_markup=keyboard)


@cohort_router.callback_query(F.data == "cohort:refresh")
@require_admin
async def handle_cohort_refresh(callback: CallbackQuery, db, admin_ids: list[int]):
    """Обновить данные когортного анализа"""
    await callback.answer("🔄 Обновляю...")

    # Повторно вызвать главное меню
    analytics = CohortAnalytics(db)
    comparison = await analytics.compare_cohorts(months_back=6)

    text = "📊 <b>Когортный анализ — динамика удержания</b>\n\n"
    text += f"🏆 <b>Лучшая когорта:</b> {comparison.best_cohort}\n"
    text += f"📉 <b>Худшая когорта:</b> {comparison.worst_cohort}\n\n"
    text += f"📈 <b>Тренд удержания:</b> {format_trend(comparison.retention_trend)}\n"
    text += f"💰 <b>Тренд конверсии:</b> {format_trend(comparison.conversion_trend)}\n\n"

    text += "<b>🔍 Ключевые выводы:</b>\n"
    for insight in comparison.key_insights[:5]:
        text += f"• {insight}\n"

    text += "\n<b>📅 Когорты (последние 6 месяцев):</b>\n\n"
    for cohort in comparison.cohorts_data[:6]:
        text += f"<b>{cohort.cohort_month}</b> ({cohort.cohort_size} пользователей)\n"
        text += f"  День 1: {cohort.day_1_retention:.1f}% | День 7: {cohort.day_7_retention:.1f}%\n"
        text += (
            f"  День 30: {cohort.day_30_retention:.1f}% | День 90: {cohort.day_90_retention:.1f}%\n"
        )
        text += f"  Конверсия: {cohort.conversion_rate:.1f}% | ARPU: {cohort.arpu:.0f}₽\n\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Детали когорты", callback_data="cohort:select_month")],
            [
                InlineKeyboardButton(
                    text="🎯 Использование функций", callback_data="cohort:feature_adoption"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📈 Кривые удержания", callback_data="cohort:retention_curves"
                )
            ],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="cohort:refresh")],
        ]
    )

    await edit_or_answer(callback, text, keyboard)


@cohort_router.callback_query(F.data == "cohort:select_month")
@require_admin
async def handle_select_month(callback: CallbackQuery, db, admin_ids: list[int]):
    """Выбрать месяц для детального просмотра"""
    analytics = CohortAnalytics(db)
    comparison = await analytics.compare_cohorts(months_back=6)

    # Создать кнопки для выбора когорты
    buttons = []
    for cohort in comparison.cohorts_data[:6]:
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"{cohort.cohort_month} ({cohort.cohort_size} пользователей)",
                    callback_data=f"cohort:details:{cohort.cohort_month}",
                )
            ]
        )

    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="cohort:back")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)

    await edit_or_answer(callback, "📅 <b>Выберите когорту для детального анализа:</b>", keyboard)
    await callback.answer()


@cohort_router.callback_query(F.data.startswith("cohort:details:"))
@require_admin
async def handle_cohort_details(callback: CallbackQuery, db, admin_ids: list[int]):
    """Детальный просмотр когорты"""
    cohort_month = callback.data.split(":")[-1]

    analytics = CohortAnalytics(db)
    cohort = await analytics.get_cohort_metrics(cohort_month)

    text = f"📊 <b>Детали когорты {cohort.cohort_month}</b>\n\n"

    text += f"👥 <b>Размер когорты:</b> {cohort.cohort_size} пользователей\n\n"

    text += "<b>📈 Показатели удержания:</b>\n"
    text += f"  День 1:  {cohort.day_1_retention:.1f}%\n"
    text += f"  День 7:  {cohort.day_7_retention:.1f}%\n"
    text += f"  День 30: {cohort.day_30_retention:.1f}%\n"
    text += f"  День 90: {cohort.day_90_retention:.1f}%\n\n"

    text += "<b>💰 Показатели выручки:</b>\n"
    text += f"  Платящих пользователей: {cohort.paid_users}\n"
    text += f"  Конверсия: {cohort.conversion_rate:.1f}%\n"
    text += f"  Совокупная выручка: {cohort.total_revenue:,}₽\n"
    text += f"  ARPU: {cohort.arpu:.0f}₽\n\n"

    text += "<b>🎯 Вовлечённость:</b>\n"
    text += f"  Среднее число запросов на пользователя: {cohort.avg_requests_per_user:.1f}\n"
    text += f"  Средний жизненный цикл: {cohort.avg_lifetime_days:.1f} дней\n"
    text += f"  Суперактивных пользователей: {cohort.power_users_count}\n"
    text += f"  Среднее число используемых функций: {cohort.avg_features_used:.1f}\n\n"

    text += "<b>🔥 Лучшие функции:</b>\n"
    for feature, adoption in cohort.top_features[:5]:
        text += f"  • {feature}: {adoption:.1f}% пользователей\n"

    text += "\n<b>📉 Отток:</b>\n"
    text += f"  Пользователей в оттоке: {cohort.churned_count}\n"
    text += f"  Доля оттока: {cohort.churn_rate:.1f}%\n"
    if cohort.avg_days_to_churn:
        text += f"  Среднее число дней до оттока: {cohort.avg_days_to_churn:.1f}\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Назад к списку", callback_data="cohort:select_month")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="cohort:back")],
        ]
    )

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@cohort_router.callback_query(F.data == "cohort:feature_adoption")
@require_admin
async def handle_feature_adoption(callback: CallbackQuery, db, admin_ids: list[int]):
    """Использование функций по когортам"""
    # Список популярных фич для анализа
    features = FEATURE_KEYS

    buttons = []
    for feature in features:
        buttons.append(
            [
                InlineKeyboardButton(
                    text=feature.replace("_", " ").title(),
                    callback_data=f"cohort:feature:{feature}",
                )
            ]
        )

    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="cohort:back")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)

    await edit_or_answer(
        callback, "🎯 <b>Выберите фичу для анализа использования по когортам:</b>", keyboard
    )
    await callback.answer()


@cohort_router.callback_query(F.data.startswith("cohort:feature:"))
@require_admin
async def handle_feature_details(callback: CallbackQuery, db, admin_ids: list[int]):
    """Детальный разбор использования фичи"""
    feature_name = callback.data.split(":")[-1]

    analytics = CohortAnalytics(db)
    adoption = await analytics.get_feature_adoption_by_cohort(feature_name, months_back=6)

    text = f"🎯 <b>Использование функции: {feature_name.replace('_', ' ').title()}</b>\n\n"

    text += "<b>📊 Уровень использования по когортам:</b>\n"
    for cohort_month, rate in sorted(adoption.cohort_adoption.items(), reverse=True):
        text += f"  {cohort_month}: {rate:.1f}%\n"

    text += "\n<b>⏱ Среднее время до первого использования:</b>\n"
    for cohort_month, days in sorted(adoption.avg_days_to_first_use.items(), reverse=True):
        text += f"  {cohort_month}: {days:.1f} дней\n"

    text += "\n<b>🔗 Влияние на удержание:</b>\n"
    text += f"  С фичей: {adoption.users_with_feature_retention:.1f}%\n"
    text += f"  Без фичи: {adoption.users_without_feature_retention:.1f}%\n"
    text += f"  Прирост удержания: <b>{adoption.retention_lift:+.1f}%</b>\n\n"

    if adoption.retention_lift > 10:
        text += "✅ <b>Эта фича значительно улучшает удержание!</b>\n"
    elif adoption.retention_lift > 0:
        text += "ℹ️ Фича положительно влияет на удержание\n"
    else:
        text += "⚠️ Фича не улучшает удержание — возможно стоит переработать\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Назад к фичам", callback_data="cohort:feature_adoption")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="cohort:back")],
        ]
    )

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@cohort_router.callback_query(F.data == "cohort:retention_curves")
@require_admin
async def handle_retention_curves(callback: CallbackQuery, db, admin_ids: list[int]):
    """Визуализация кривых удержания"""
    analytics = CohortAnalytics(db)
    comparison = await analytics.compare_cohorts(months_back=6)

    text = "📈 <b>Кривые удержания</b>\n\n"
    text += "Сравнение удержания по дням для всех когорт:\n\n"

    for cohort in comparison.cohorts_data[:6]:
        text += f"<b>{cohort.cohort_month}</b>\n"

        # ASCII visualization (ограничиваем значения до 100%)
        day_1_filled = int(min(100, cohort.day_1_retention) / 10)
        day_7_filled = int(min(100, cohort.day_7_retention) / 10)
        day_30_filled = int(min(100, cohort.day_30_retention) / 10)
        day_90_filled = int(min(100, cohort.day_90_retention) / 10)

        day_1_bar = "█" * day_1_filled + "░" * (10 - day_1_filled)
        day_7_bar = "█" * day_7_filled + "░" * (10 - day_7_filled)
        day_30_bar = "█" * day_30_filled + "░" * (10 - day_30_filled)
        day_90_bar = "█" * day_90_filled + "░" * (10 - day_90_filled)

        text += f"  Д1:  {day_1_bar} {cohort.day_1_retention:.0f}%\n"
        text += f"  Д7:  {day_7_bar} {cohort.day_7_retention:.0f}%\n"
        text += f"  Д30: {day_30_bar} {cohort.day_30_retention:.0f}%\n"
        text += f"  Д90: {day_90_bar} {cohort.day_90_retention:.0f}%\n\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Назад", callback_data="cohort:back")]]
    )

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@cohort_router.callback_query(F.data == "cohort:back")
@require_admin
async def handle_back_to_main(callback: CallbackQuery, db, admin_ids: list[int]):
    """Вернуться в главное меню когортного анализа"""
    analytics = CohortAnalytics(db)
    comparison = await analytics.compare_cohorts(months_back=6)

    text = "📊 <b>Когортный анализ — динамика удержания</b>\n\n"
    text += f"🏆 <b>Лучшая когорта:</b> {comparison.best_cohort}\n"
    text += f"📉 <b>Худшая когорта:</b> {comparison.worst_cohort}\n\n"
    text += f"📈 <b>Тренд удержания:</b> {format_trend(comparison.retention_trend)}\n"
    text += f"💰 <b>Тренд конверсии:</b> {format_trend(comparison.conversion_trend)}\n\n"

    text += "<b>🔍 Ключевые выводы:</b>\n"
    for insight in comparison.key_insights[:5]:
        text += f"• {insight}\n"

    text += "\n<b>📅 Когорты (последние 6 месяцев):</b>\n\n"
    for cohort in comparison.cohorts_data[:6]:
        text += f"<b>{cohort.cohort_month}</b> ({cohort.cohort_size} пользователей)\n"
        text += f"  День 1: {cohort.day_1_retention:.1f}% | День 7: {cohort.day_7_retention:.1f}%\n"
        text += (
            f"  День 30: {cohort.day_30_retention:.1f}% | День 90: {cohort.day_90_retention:.1f}%\n"
        )
        text += f"  Конверсия: {cohort.conversion_rate:.1f}% | ARPU: {cohort.arpu:.0f}₽\n\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Детали когорты", callback_data="cohort:select_month")],
            [
                InlineKeyboardButton(
                    text="🎯 Использование функций", callback_data="cohort:feature_adoption"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📈 Кривые удержания", callback_data="cohort:retention_curves"
                )
            ],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="cohort:refresh")],
        ]
    )

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


__all__ = (
    "cohort_router",
    "cmd_cohort",
    "handle_cohort_refresh",
    "handle_select_month",
    "handle_cohort_details",
    "handle_feature_adoption",
    "handle_feature_details",
    "handle_retention_curves",
    "handle_back_to_main",
)
