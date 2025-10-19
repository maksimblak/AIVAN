"""
Админ-команды для аналитики удержания
Детальный анализ кто остается и кто уходит
"""

from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from src.core.admin_modules.retention_analytics import RetentionAnalytics
from src.core.admin_modules.admin_utils import back_keyboard, edit_or_answer, parse_user_id, require_admin

logger = logging.getLogger(__name__)

retention_router = Router()

INDICATOR_LABELS = {
    'low_usage': '📊 Низкая активность',
    'had_errors': '🐛 Технические проблемы',
    'limited_exploration': '🎯 Не изучили продукт',
    'poor_experience': '😞 Негативный опыт',
    'immediate_abandonment': '⚡ Бросили сразу после оплаты',
    'price_sensitive': '💰 Цена против ценности'
}


def create_retention_menu() -> InlineKeyboardMarkup:
    """Меню retention аналитики"""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="💎 Оставшиеся пользователи", callback_data="retention:retained"),
                InlineKeyboardButton(text="📉 Пользователи в оттоке", callback_data="retention:churned"),
            ],
            [
                InlineKeyboardButton(text="⚖️ Сравнить группы", callback_data="retention:compare"),
            ],
            [
                InlineKeyboardButton(text="🔍 Анализ пользователя", callback_data="retention:deep_dive"),
            ],
            [
                InlineKeyboardButton(text="« Назад", callback_data="admin_refresh"),
            ],
        ]
    )


@retention_router.message(Command("retention"))
@require_admin
async def cmd_retention(message: Message, db, admin_ids: set[int]):
    """Главная команда retention аналитики"""
    analytics = RetentionAnalytics(db)

    # Краткая сводка
    retained = await analytics.get_retained_users(min_payments=2)
    churned = await analytics.get_churned_users(days_since_expiry=30)

    summary = "<b>💎 Аналитика удержания</b>\n\n"

    summary += f"<b>✅ Продлили (2+ оплат):</b> {len(retained)} пользователей\n"
    if retained:
        avg_payments = sum(u.payment_count for u in retained) / len(retained)
        avg_power_score = sum(u.power_user_score for u in retained) / len(retained)
        summary += f"   • Среднее платежей: {avg_payments:.1f}\n"
        summary += f"   • Средний Power Score: {avg_power_score:.1f}/100\n"

    summary += f"\n<b>❌ Отток (1 оплата):</b> {len(churned)} пользователей\n"
    if churned:
        high_winback = sum(1 for u in churned if u.winback_probability > 60)
        summary += f"   • Высокий потенциал возврата: {high_winback}\n"

    summary += "\n<i>Выберите раздел для детального анализа:</i>"

    await message.answer(summary, parse_mode=ParseMode.HTML, reply_markup=create_retention_menu())


@retention_router.callback_query(F.data == "retention:retained")
@require_admin
async def handle_retained_users(callback: CallbackQuery, db, admin_ids: set[int]):
    """Детальный анализ продлевающих пользователей"""
    analytics = RetentionAnalytics(db)
    retained = await analytics.get_retained_users(min_payments=2)

    if not retained:
        await callback.answer("Нет продлевающих пользователей", show_alert=True)
        return

    output = "<b>💎 Продлевающие пользователи</b>\n\n"

    # Общая статистика
    avg_power = sum(u.power_user_score for u in retained) / len(retained)
    avg_diversity = sum(u.feature_diversity for u in retained) / len(retained)
    avg_requests_day = sum(u.avg_requests_per_day for u in retained) / len(retained)

    output += "<b>📊 Средние показатели:</b>\n"
    output += f"• Индекс Power: {avg_power:.1f}/100\n"
    output += f"• Разнообразие функций: {avg_diversity:.1f}%\n"
    output += f"• Активность: {avg_requests_day:.1f} запросов/день\n\n"

    # Топ-3 power users
    top_3 = sorted(retained, key=lambda u: u.power_user_score, reverse=True)[:3]

    output += "<b>👑 Топ-3 суперактивных:</b>\n\n"
    for i, user in enumerate(top_3, 1):
        medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉'

        output += f"{medal} Пользователь #{user.user_id}\n"
        output += f"   • Платежей: {user.payment_count}\n"
        output += f"   • Индекс Power: {user.power_user_score}/100\n"
        output += f"   • Активность: {user.avg_requests_per_day:.1f} запр./день\n"
        output += "   • Любимые фичи:\n"

        for feature, count in user.favorite_features[:3]:
            output += f"      - {feature}: {count} раз\n"

        output += "\n"

    # Общие паттерны
    output += "<b>🔍 Что их объединяет:</b>\n"

    # Самые популярные фичи
    all_favorites = {}
    for user in retained:
        for feature, count in user.favorite_features[:3]:
            all_favorites[feature] = all_favorites.get(feature, 0) + count

    top_features = sorted(all_favorites.items(), key=lambda x: x[1], reverse=True)[:3]

    output += "\n<b>Любимые фичи:</b>\n"
    for feature, total_count in top_features:
        output += f"• {feature}: используют {total_count} раз\n"

    # Паттерны времени
    weekday_count = sum(1 for u in retained if u.usage_patterns.get('is_weekday_user', False))
    daytime_count = sum(1 for u in retained if u.usage_patterns.get('is_daytime_user', False))

    output += "\n<b>Паттерны использования:</b>\n"
    output += f"• Будние дни: {weekday_count}/{len(retained)} ({weekday_count/len(retained)*100:.0f}%)\n"
    output += f"• Дневное время: {daytime_count}/{len(retained)} ({daytime_count/len(retained)*100:.0f}%)\n"

    # Insights
    output += "\n<b>💡 Ключевые инсайты:</b>\n"

    if avg_diversity > 50:
        output += "✅ Используют разнообразные фичи - продукт полезен целиком\n"
    else:
        output += "⚠️ Фокусируются на 1-2 фичах - остальное не нужно?\n"

    if avg_requests_day > 3:
        output += "✅ Очень высокая активность - продукт критичен для работы\n"
    else:
        output += "📊 Умеренная активность - дополнительный инструмент\n"

    keyboard = back_keyboard("retention:menu")

    await edit_or_answer(callback, output, keyboard)
    await callback.answer()


@retention_router.callback_query(F.data == "retention:churned")
@require_admin
async def handle_churned_users(callback: CallbackQuery, db, admin_ids: set[int]):
    """Детальный анализ пользователей в оттоке"""
    analytics = RetentionAnalytics(db)
    churned = await analytics.get_churned_users(days_since_expiry=90)

    if not churned:
        await callback.answer("Нет пользователей в оттоке", show_alert=True)
        return

    output = "<b>📉 Пользователи в оттоке (не продлили)</b>\n\n"

    output += f"<b>Всего не продлили:</b> {len(churned)}\n\n"

    # Распределение по причинам
    all_indicators = {}
    for user in churned:
        for indicator in user.churn_indicators:
            all_indicators[indicator] = all_indicators.get(indicator, 0) + 1

    sorted_indicators = sorted(all_indicators.items(), key=lambda x: x[1], reverse=True)

    output += "<b>🔍 Причины оттока:</b>\n"
    indicator_names = INDICATOR_LABELS

    for indicator, count in sorted_indicators:
        name = indicator_names.get(indicator, indicator)
        pct = (count / len(churned)) * 100
        output += f"{name}: {count} ({pct:.0f}%)\n"

    # Неиспользованные фичи
    output += "\n<b>❌ Что НЕ попробовали:</b>\n"

    all_unused = {}
    for user in churned:
        for feature in user.unused_features:
            all_unused[feature] = all_unused.get(feature, 0) + 1

    top_unused = sorted(all_unused.items(), key=lambda x: x[1], reverse=True)[:5]

    for feature, count in top_unused:
        pct = (count / len(churned)) * 100
        output += f"• {feature}: {pct:.0f}% не использовали\n"

    # Win-back потенциал
    output += "\n<b>🎯 Потенциал возврата:</b>\n"

    high_prob = sum(1 for u in churned if u.winback_probability > 60)
    medium_prob = sum(1 for u in churned if 30 < u.winback_probability <= 60)
    low_prob = sum(1 for u in churned if u.winback_probability <= 30)

    output += f"🟢 Высокий (>60%): {high_prob} пользователей\n"
    output += f"🟡 Средний (30-60%): {medium_prob} пользователей\n"
    output += f"🔴 Низкий (<30%): {low_prob} пользователей\n"

    # Рекомендации
    output += "\n<b>💡 Рекомендации:</b>\n"

    if 'low_usage' in dict(sorted_indicators):
        if all_indicators['low_usage'] > len(churned) * 0.3:
            output += "1️⃣ ПРИОРИТЕТ: Улучшить онбординге - 30%+ не поняли ценность\n"

    if 'had_errors' in dict(sorted_indicators):
        output += "2️⃣ Срочно исправить технические проблемы\n"

    if 'limited_exploration' in dict(sorted_indicators):
        output += "3️⃣ Добавить пошаговое руководство по функциям\n"

    if 'price_sensitive' in dict(sorted_indicators):
        output += "4️⃣ A/B тест цены или добавить доступный тариф\n"

    # Показываем 2-3 примера
    output += "\n<b>📋 Примеры пользователей в оттоке:</b>\n\n"

    examples = sorted(churned, key=lambda u: u.winback_probability, reverse=True)[:3]

    for user in examples:
        output += f"Пользователь #{user.user_id}\n"
        output += f"   • Запросов: {user.total_requests}\n"
        issues = ', '.join(INDICATOR_LABELS.get(code, code) for code in user.churn_indicators[:2])
        output += f"   • Проблемы: {issues}\n"
        output += f"   • Потенциал возврата: {user.winback_probability:.0f}%\n"
        output += f"   • Действие: <i>{user.recommended_action}</i>\n\n"

    keyboard = back_keyboard("retention:menu")

    await edit_or_answer(callback, output, keyboard)
    await callback.answer()


@retention_router.callback_query(F.data == "retention:compare")
@require_admin
async def handle_compare_groups(callback: CallbackQuery, db, admin_ids: set[int]):
    """Сравнение удержанных и ушедших"""
    analytics = RetentionAnalytics(db)
    comparison = await analytics.compare_retained_vs_churned()

    if "error" in comparison:
        await callback.answer(comparison["error"], show_alert=True)
        return

    output = "<b>⚖️ Продлевающие vs отток</b>\n\n"

    retained_data = comparison["retained"]
    churned_data = comparison["churned"]

    # Сравнение по активности
    output += "<b>📊 Активность:</b>\n"
    output += f"Продлевающие: {retained_data['avg_requests']:.1f} запросов\n"
    output += f"Отток: {churned_data['avg_requests']:.1f} запросов\n"

    requests_diff = retained_data['avg_requests'] - churned_data['avg_requests']
    churn_avg_requests = churned_data['avg_requests']
    requests_pct = (requests_diff / churn_avg_requests * 100) if churn_avg_requests else 0
    output += f"→ Разница: {requests_diff:+.1f} ({requests_pct:+.0f}%)\n\n"

    # Lifetime
    output += "<b>⏱ Время жизни:</b>\n"
    output += f"Продлевающие: {retained_data['avg_lifetime_days']:.0f} дней\n"
    output += f"Отток: {churned_data['avg_lifetime_days']:.0f} дней\n\n"

    # Power Score (только у retained)
    output += "<b>⚡ Индекс Power:</b>\n"
    output += f"Продлевающие: {retained_data['avg_power_score']:.1f}/100\n"
    output += "Отток: нет данных (не продлили)\n\n"

    # Feature Diversity
    output += "<b>🎯 Разнообразие функций:</b>\n"
    output += f"Продлевающие: {retained_data['avg_feature_diversity']:.1f}%\n"
    output += "Отток: изучают меньше функций\n\n"

    # Топ фичи для retained
    output += "<b>💎 Что ценят продлевающие:</b>\n"
    for feature, count in list(retained_data['top_features'].items())[:3]:
        output += f"• {feature}: {count} использований\n"

    # Что НЕ используют churned
    output += "\n<b>❌ Что не пробуют ушедшие:</b>\n"
    for feature, count in list(churned_data['unused_features_common'].items())[:3]:
        output += f"• {feature}: {count} пользователей пропустили\n"

    # Ключевые выводы
    output += "\n<b>🎯 КЛЮЧЕВЫЕ ВЫВОДЫ:</b>\n\n"

    if requests_diff > 50:
        requests_ratio = (retained_data['avg_requests'] / churn_avg_requests) if churn_avg_requests else 0
        output += f"1️⃣ Продлевающие в {requests_ratio:.1f}x активнее!\n"
        output += "   → Нужно мотивировать новых к активности\n\n"

    if retained_data['avg_feature_diversity'] > 50:
        output += "2️⃣ Продлевающие изучают больше фич\n"
        output += "   → Показывать все возможности в онбординге\n\n"

    top_unused = list(churned_data['unused_features_common'].keys())[0] if churned_data['unused_features_common'] else None
    if top_unused:
        output += f"3️⃣ Ушедшие пропускают {top_unused}\n"
        output += "   → Либо промо этой фичи, либо она не нужна\n\n"

    # Процент weekday/daytime users
    if 'common_patterns' in retained_data:
        weekday_pct = retained_data['common_patterns']['weekday_preference_pct']
        if weekday_pct > 70:
            output += f"4️⃣ {weekday_pct:.0f}% продлевающих - B2B пользователи\n"
            output += "   → Фокус на профессиональное использование\n"

    keyboard = back_keyboard("retention:menu")

    await edit_or_answer(callback, output, keyboard)
    await callback.answer()


@retention_router.callback_query(F.data == "retention:deep_dive")
@require_admin
async def handle_deep_dive(callback: CallbackQuery, db, admin_ids: set[int]):
    """Запрос на глубокий анализ конкретного пользователя"""
    output = "<b>🔍 Глубокий анализ</b>\n\n"
    output += "Для детального анализа конкретного пользователя используйте:\n\n"
    output += "<code>/deepdive &lt;ID пользователя&gt;</code>\n\n"
    output += "Вы получите:\n"
    output += "• Полный профиль (продлевает/отток)\n"
    output += "• Любимые фичи и паттерны\n"
    output += "• Показатели оттока (если применимо)\n"
    output += "• Персональные рекомендации\n"

    keyboard = back_keyboard("retention:menu")

    await edit_or_answer(callback, output, keyboard)
    await callback.answer()


@retention_router.message(Command("deepdive"))
@require_admin
async def cmd_deep_dive_user(message: Message, db, admin_ids: set[int]):
    """Детальный анализ конкретного пользователя"""
    user_id = await parse_user_id(message, "deepdive")
    if user_id is None:
        return


    analytics = RetentionAnalytics(db)

    # Проверяем в retained
    retained_list = await analytics.get_retained_users(min_payments=2)
    retained_user = next((u for u in retained_list if u.user_id == user_id), None)

    if retained_user:
        output = f"<b>💎 Пользователь (продлевает) #{user_id}</b>\n\n"
        output += "<b>💰 Монетизация:</b>\n"
        output += f"• Платежей: {retained_user.payment_count}\n"
        output += f"• Потрачено: {retained_user.total_spent} руб\n"
        output += f"• Время жизни: {retained_user.lifetime_days} дней\n"
        output += f"• Время до первой оплаты: {retained_user.time_to_first_payment_days} дней\n\n"

        output += "<b>📊 Активность:</b>\n"
        output += f"• Всего запросов: {retained_user.total_requests}\n"
        output += f"• В день: {retained_user.avg_requests_per_day:.1f}\n"
        output += f"• Дней активности в неделю: {retained_user.days_active_per_week:.1f}\n"
        output += f"• Максимальная серия: {retained_user.streak_max_days} дней подряд\n\n"

        output += "<b>🎯 Вовлеченность:</b>\n"
        output += f"• Индекс Power: {retained_user.power_user_score:.1f}/100\n"
        output += f"• Разнообразие функций: {retained_user.feature_diversity:.1f}%\n"
        output += f"• Вероятность удержания: {retained_user.retention_probability:.1f}%\n\n"

        output += "<b>💝 Любимые фичи:</b>\n"
        for feature, count in retained_user.favorite_features[:5]:
            output += f"• {feature}: {count} раз\n"

        output += "\n<b>⏰ Паттерны:</b>\n"
        if retained_user.usage_patterns.get('peak_hour') is not None:
            output += f"• Пик активности: {retained_user.usage_patterns['peak_hour']:02d}:00\n"
        if retained_user.usage_patterns.get('is_weekday_user'):
            output += "• Тип: Будние дни (B2B?)\n"
        else:
            output += "• Тип: Выходные (B2C?)\n"

        output += "\n<b>✅ Что делает его ценным:</b>\n"

        if retained_user.power_user_score > 70:
            output += "• 🌟 Power user - критически зависит от продукта\n"
        if retained_user.payment_count >= 3:
            output += "• 💎 Лояльный клиент - 3+ продления\n"
        if retained_user.feature_diversity > 60:
            output += "• 🎯 Использует продукт целиком\n"
        if retained_user.avg_requests_per_day > 5:
            output += "• ⚡ Экстремально активен\n"

    else:
        # Проверяем в churned
        churned_list = await analytics.get_churned_users(days_since_expiry=90)
        churned_user = next((u for u in churned_list if u.user_id == user_id), None)

        if churned_user:
            output = f"<b>📉 Пользователь в оттоке #{user_id}</b>\n\n"
            output += "<b>💰 История:</b>\n"
            output += f"• Платежей: {churned_user.payment_count}\n"
            output += f"• Потрачено: {churned_user.total_spent} руб\n"
            output += f"• Был активен: {churned_user.lifetime_days} дней\n"
            output += f"• Последняя активность: {churned_user.last_active_days_ago} дней назад\n\n"

            output += "<b>📊 Активность:</b>\n"
            output += f"• Всего запросов: {churned_user.total_requests}\n\n"

            output += "<b>❌ Причины оттока:</b>\n"
            for indicator in churned_user.churn_indicators:
                output += f"• {indicator}\n"

            output += "\n<b>🎯 Что НЕ попробовал:</b>\n"
            for feature in churned_user.unused_features[:5]:
                output += f"• {feature}\n"

            if churned_user.drop_off_feature:
                output += f"\n<b>📍 Последняя активность:</b>\n{churned_user.drop_off_feature}\n"

            output += "\n<b>🔧 Проблемы:</b>\n"
            if churned_user.had_technical_issues:
                output += "⚠️ Были технические ошибки\n"
            if churned_user.received_poor_responses:
                output += "😞 Получал плохие ответы\n"

            output += "\n<b>🎯 План возврата:</b>\n"
            output += f"• Вероятность: {churned_user.winback_probability:.0f}%\n"
            output += f"• Действие: <i>{churned_user.recommended_action}</i>\n"

        else:
            output = f"<b>Пользователь #{user_id}</b>\n\n"
            output += "Пользователь не найден в базе или не делал оплат."

    await message.answer(output, parse_mode=ParseMode.HTML)


@retention_router.callback_query(F.data == "retention:menu")
@require_admin
async def back_to_retention_menu(callback: CallbackQuery, db, admin_ids: set[int]):
    """Возврат в меню retention"""
    analytics = RetentionAnalytics(db)
    retained = await analytics.get_retained_users(min_payments=2)
    churned = await analytics.get_churned_users(days_since_expiry=30)

    summary = "<b>💎 Аналитика удержания</b>\n\n"

    summary += f"<b>✅ Продлили (2+ оплат):</b> {len(retained)} пользователей\n"
    if retained:
        avg_payments = sum(u.payment_count for u in retained) / len(retained)
        avg_power_score = sum(u.power_user_score for u in retained) / len(retained)
        summary += f"   • Среднее платежей: {avg_payments:.1f}\n"
        summary += f"   • Средний Power Score: {avg_power_score:.1f}/100\n"

    summary += f"\n<b>❌ Отток (1 оплата):</b> {len(churned)} пользователей\n"
    if churned:
        high_winback = sum(1 for u in churned if u.winback_probability > 60)
        summary += f"   • Высокий потенциал возврата: {high_winback}\n"

    summary += "\n<i>Выберите раздел для детального анализа:</i>"

    if callback.message:
        await edit_or_answer(callback, summary, create_retention_menu())
    await callback.answer()


__all__ = (
    "retention_router",
    "cmd_retention",
    "handle_retained_users",
    "handle_churned_users",
    "handle_compare_groups",
    "handle_deep_dive",
    "cmd_deep_dive_user",
    "back_to_retention_menu",
)
