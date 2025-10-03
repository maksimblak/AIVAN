"""
Admin commands для Cohort Analysis
"""

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from src.core.admin_modules.admin_utils import FEATURE_KEYS, edit_or_answer, require_admin
from src.core.admin_modules.cohort_analytics import CohortAnalytics
from src.core.admin_modules.admin_formatters import format_trend


cohort_router = Router(name="cohort_admin")


@cohort_router.message(Command("cohort"))
@require_admin
async def cmd_cohort(message: Message, db, admin_ids: list[int]):
    """Главное меню cohort analysis"""
    analytics = CohortAnalytics(db)

    # Получить сравнение когорт
    comparison = await analytics.compare_cohorts(months_back=6)

    # Формирование сообщения
    text = "📊 <b>Cohort Analysis - Retention Dynamics</b>\n\n"

    text += f"🏆 <b>Лучшая когорта:</b> {comparison.best_cohort}\n"
    text += f"📉 <b>Худшая когорта:</b> {comparison.worst_cohort}\n\n"

    text += f"📈 <b>Тренд retention:</b> {format_trend(comparison.retention_trend)}\n"
    text += f"💰 <b>Тренд conversion:</b> {format_trend(comparison.conversion_trend)}\n\n"

    text += "<b>🔍 Ключевые инсайты:</b>\n"
    for insight in comparison.key_insights[:5]:
        text += f"• {insight}\n"

    text += "\n<b>📅 Когорты (последние 6 месяцев):</b>\n\n"

    for cohort in comparison.cohorts_data[:6]:
        text += f"<b>{cohort.cohort_month}</b> ({cohort.cohort_size} users)\n"
        text += f"  Day 1: {cohort.day_1_retention:.1f}% | Day 7: {cohort.day_7_retention:.1f}%\n"
        text += f"  Day 30: {cohort.day_30_retention:.1f}% | Day 90: {cohort.day_90_retention:.1f}%\n"
        text += f"  Conversion: {cohort.conversion_rate:.1f}% | ARPU: {cohort.arpu:.0f}₽\n\n"

    # Кнопки
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Детали когорты", callback_data="cohort:select_month")],
        [InlineKeyboardButton(text="🎯 Feature Adoption", callback_data="cohort:feature_adoption")],
        [InlineKeyboardButton(text="📈 Retention Curves", callback_data="cohort:retention_curves")],
        [InlineKeyboardButton(text="🔄 Обновить", callback_data="cohort:refresh")]
    ])

    await message.answer(text, parse_mode="HTML", reply_markup=keyboard)



@cohort_router.callback_query(F.data == "cohort:refresh")
@require_admin
async def handle_cohort_refresh(callback: CallbackQuery, db, admin_ids: list[int]):
    """Обновить cohort analysis"""
    await callback.answer("🔄 Обновляю...")

    # Повторно вызвать главное меню
    analytics = CohortAnalytics(db)
    comparison = await analytics.compare_cohorts(months_back=6)

    text = "📊 <b>Cohort Analysis - Retention Dynamics</b>\n\n"
    text += f"🏆 <b>Лучшая когорта:</b> {comparison.best_cohort}\n"
    text += f"📉 <b>Худшая когорта:</b> {comparison.worst_cohort}\n\n"
    text += f"📈 <b>Тренд retention:</b> {format_trend(comparison.retention_trend)}\n"
    text += f"💰 <b>Тренд conversion:</b> {format_trend(comparison.conversion_trend)}\n\n"

    text += "<b>🔍 Ключевые инсайты:</b>\n"
    for insight in comparison.key_insights[:5]:
        text += f"• {insight}\n"

    text += "\n<b>📅 Когорты (последние 6 месяцев):</b>\n\n"
    for cohort in comparison.cohorts_data[:6]:
        text += f"<b>{cohort.cohort_month}</b> ({cohort.cohort_size} users)\n"
        text += f"  Day 1: {cohort.day_1_retention:.1f}% | Day 7: {cohort.day_7_retention:.1f}%\n"
        text += f"  Day 30: {cohort.day_30_retention:.1f}% | Day 90: {cohort.day_90_retention:.1f}%\n"
        text += f"  Conversion: {cohort.conversion_rate:.1f}% | ARPU: {cohort.arpu:.0f}₽\n\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Детали когорты", callback_data="cohort:select_month")],
        [InlineKeyboardButton(text="🎯 Feature Adoption", callback_data="cohort:feature_adoption")],
        [InlineKeyboardButton(text="📈 Retention Curves", callback_data="cohort:retention_curves")],
        [InlineKeyboardButton(text="🔄 Обновить", callback_data="cohort:refresh")]
    ])

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
        buttons.append([
            InlineKeyboardButton(
                text=f"{cohort.cohort_month} ({cohort.cohort_size} users)",
                callback_data=f"cohort:details:{cohort.cohort_month}"
            )
        ])

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

    text += "<b>📈 Retention Rates:</b>\n"
    text += f"  Day 1:  {cohort.day_1_retention:.1f}%\n"
    text += f"  Day 7:  {cohort.day_7_retention:.1f}%\n"
    text += f"  Day 30: {cohort.day_30_retention:.1f}%\n"
    text += f"  Day 90: {cohort.day_90_retention:.1f}%\n\n"

    text += "<b>💰 Revenue Metrics:</b>\n"
    text += f"  Paid users: {cohort.paid_users}\n"
    text += f"  Conversion rate: {cohort.conversion_rate:.1f}%\n"
    text += f"  Total revenue: {cohort.total_revenue:,}₽\n"
    text += f"  ARPU: {cohort.arpu:.0f}₽\n\n"

    text += "<b>🎯 Engagement:</b>\n"
    text += f"  Avg requests/user: {cohort.avg_requests_per_user:.1f}\n"
    text += f"  Avg lifetime: {cohort.avg_lifetime_days:.1f} дней\n"
    text += f"  Power users: {cohort.power_users_count}\n"
    text += f"  Avg features used: {cohort.avg_features_used:.1f}\n\n"

    text += "<b>🔥 Top Features:</b>\n"
    for feature, adoption in cohort.top_features[:5]:
        text += f"  • {feature}: {adoption:.1f}% adoption\n"

    text += f"\n<b>📉 Churn:</b>\n"
    text += f"  Churned users: {cohort.churned_count}\n"
    text += f"  Churn rate: {cohort.churn_rate:.1f}%\n"
    if cohort.avg_days_to_churn:
        text += f"  Avg days to churn: {cohort.avg_days_to_churn:.1f}\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Назад к списку", callback_data="cohort:select_month")],
        [InlineKeyboardButton(text="🏠 Главное меню", callback_data="cohort:back")]
    ])

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@cohort_router.callback_query(F.data == "cohort:feature_adoption")
@require_admin
async def handle_feature_adoption(callback: CallbackQuery, db, admin_ids: list[int]):
    """Feature adoption по когортам"""
    # Список популярных фич для анализа
    features = FEATURE_KEYS

    buttons = []
    for feature in features:
        buttons.append([
            InlineKeyboardButton(
                text=feature.replace("_", " ").title(),
                callback_data=f"cohort:feature:{feature}"
            )
        ])

    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="cohort:back")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)

    await edit_or_answer(callback, "🎯 <b>Выберите фичу для анализа adoption по когортам:</b>", keyboard)
    await callback.answer()


@cohort_router.callback_query(F.data.startswith("cohort:feature:"))
@require_admin
async def handle_feature_details(callback: CallbackQuery, db, admin_ids: list[int]):
    """Детали adoption фичи"""
    feature_name = callback.data.split(":")[-1]

    analytics = CohortAnalytics(db)
    adoption = await analytics.get_feature_adoption_by_cohort(feature_name, months_back=6)

    text = f"🎯 <b>Feature Adoption: {feature_name.replace('_', ' ').title()}</b>\n\n"

    text += "<b>📊 Adoption Rate по когортам:</b>\n"
    for cohort_month, rate in sorted(adoption.cohort_adoption.items(), reverse=True):
        text += f"  {cohort_month}: {rate:.1f}%\n"

    text += "\n<b>⏱ Среднее время до первого использования:</b>\n"
    for cohort_month, days in sorted(adoption.avg_days_to_first_use.items(), reverse=True):
        text += f"  {cohort_month}: {days:.1f} дней\n"

    text += f"\n<b>🔗 Влияние на Retention:</b>\n"
    text += f"  С фичей: {adoption.users_with_feature_retention:.1f}%\n"
    text += f"  Без фичи: {adoption.users_without_feature_retention:.1f}%\n"
    text += f"  Retention Lift: <b>{adoption.retention_lift:+.1f}%</b>\n\n"

    if adoption.retention_lift > 10:
        text += "✅ <b>Эта фича значительно улучшает retention!</b>\n"
    elif adoption.retention_lift > 0:
        text += "ℹ️ Фича положительно влияет на retention\n"
    else:
        text += "⚠️ Фича не улучшает retention - возможно стоит переработать\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Назад к фичам", callback_data="cohort:feature_adoption")],
        [InlineKeyboardButton(text="🏠 Главное меню", callback_data="cohort:back")]
    ])

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@cohort_router.callback_query(F.data == "cohort:retention_curves")
@require_admin
async def handle_retention_curves(callback: CallbackQuery, db, admin_ids: list[int]):
    """Retention curves визуализация"""
    analytics = CohortAnalytics(db)
    comparison = await analytics.compare_cohorts(months_back=6)

    text = "📈 <b>Retention Curves</b>\n\n"
    text += "Сравнение retention по дням для всех когорт:\n\n"

    for cohort in comparison.cohorts_data[:6]:
        text += f"<b>{cohort.cohort_month}</b>\n"

        # ASCII visualization
        day_1_bar = "█" * int(cohort.day_1_retention / 10) + "░" * (10 - int(cohort.day_1_retention / 10))
        day_7_bar = "█" * int(cohort.day_7_retention / 10) + "░" * (10 - int(cohort.day_7_retention / 10))
        day_30_bar = "█" * int(cohort.day_30_retention / 10) + "░" * (10 - int(cohort.day_30_retention / 10))
        day_90_bar = "█" * int(cohort.day_90_retention / 10) + "░" * (10 - int(cohort.day_90_retention / 10))

        text += f"  D1:  {day_1_bar} {cohort.day_1_retention:.0f}%\n"
        text += f"  D7:  {day_7_bar} {cohort.day_7_retention:.0f}%\n"
        text += f"  D30: {day_30_bar} {cohort.day_30_retention:.0f}%\n"
        text += f"  D90: {day_90_bar} {cohort.day_90_retention:.0f}%\n\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Назад", callback_data="cohort:back")]
    ])

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@cohort_router.callback_query(F.data == "cohort:back")
@require_admin
async def handle_back_to_main(callback: CallbackQuery, db, admin_ids: list[int]):
    """Вернуться в главное меню cohort"""
    analytics = CohortAnalytics(db)
    comparison = await analytics.compare_cohorts(months_back=6)

    text = "📊 <b>Cohort Analysis - Retention Dynamics</b>\n\n"
    text += f"🏆 <b>Лучшая когорта:</b> {comparison.best_cohort}\n"
    text += f"📉 <b>Худшая когорта:</b> {comparison.worst_cohort}\n\n"
    text += f"📈 <b>Тренд retention:</b> {format_trend(comparison.retention_trend)}\n"
    text += f"💰 <b>Тренд conversion:</b> {format_trend(comparison.conversion_trend)}\n\n"

    text += "<b>🔍 Ключевые инсайты:</b>\n"
    for insight in comparison.key_insights[:5]:
        text += f"• {insight}\n"

    text += "\n<b>📅 Когорты (последние 6 месяцев):</b>\n\n"
    for cohort in comparison.cohorts_data[:6]:
        text += f"<b>{cohort.cohort_month}</b> ({cohort.cohort_size} users)\n"
        text += f"  Day 1: {cohort.day_1_retention:.1f}% | Day 7: {cohort.day_7_retention:.1f}%\n"
        text += f"  Day 30: {cohort.day_30_retention:.1f}% | Day 90: {cohort.day_90_retention:.1f}%\n"
        text += f"  Conversion: {cohort.conversion_rate:.1f}% | ARPU: {cohort.arpu:.0f}₽\n\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Детали когорты", callback_data="cohort:select_month")],
        [InlineKeyboardButton(text="🎯 Feature Adoption", callback_data="cohort:feature_adoption")],
        [InlineKeyboardButton(text="📈 Retention Curves", callback_data="cohort:retention_curves")],
        [InlineKeyboardButton(text="🔄 Обновить", callback_data="cohort:refresh")]
    ])

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()
