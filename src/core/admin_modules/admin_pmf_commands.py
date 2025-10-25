"""
Админ-команды для метрик PMF/NPS
"""

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from src.core.admin_modules.admin_utils import FEATURE_KEYS, edit_or_answer, require_admin
from src.core.admin_modules.pmf_metrics import PMFMetrics

pmf_router = Router(name="pmf_admin")


@pmf_router.message(Command("pmf"))
@require_admin
async def cmd_pmf(message: Message, db, admin_ids: list[int]):
    """Главное меню метрик PMF"""
    metrics = PMFMetrics(db)

    # Получить все метрики
    nps = await metrics.get_nps(days=30)
    sean_ellis = await metrics.get_sean_ellis_score(days=30)
    usage = await metrics.get_usage_intensity()

    text = "📊 <b>Дашборд Product-Market Fit</b>\n\n"

    # NPS Section
    text += "<b>🎯 Индекс лояльности (NPS)</b>\n"
    text += f"  Индекс: <b>{nps.nps_score:+.0f}</b> {_nps_emoji(nps.nps_score)}\n"
    text += f"  Промоутеры: {nps.promoters} ({nps.promoter_rate:.1f}%)\n"
    text += f"  Нейтралы: {nps.passives}\n"
    text += f"  Критики: {nps.detractors} ({nps.detractor_rate:.1f}%)\n"
    text += f"  Средний балл: {nps.average_score:.1f}/10\n"
    text += f"  Доля ответов: {nps.response_rate:.1f}%\n"
    text += f"  Тренд: {_format_trend(nps.trend)}\n\n"

    # Sean Ellis Test
    text += "<b>💎 Тест Шона Эллиса (PMF)</b>\n"
    text += f"  Очень расстроены: <b>{sean_ellis.pmf_score:.1f}%</b>\n"
    text += f"  Статус PMF: {_pmf_status(sean_ellis.pmf_achieved)}\n"
    text += f"  Ответов: {sean_ellis.total_responses}\n\n"

    # Usage Intensity
    text += "<b>📈 Интенсивность использования</b>\n"
    text += f"  DAU (день): {usage.dau}\n"
    text += f"  WAU (неделя): {usage.wau}\n"
    text += f"  MAU (месяц): {usage.mau}\n"
    text += f"  DAU/MAU: {usage.dau_mau_ratio:.1f}% {_stickiness_emoji(usage.dau_mau_ratio)}\n"
    text += f"  Суперактивные: {usage.power_user_percentage:.1f}%\n"
    text += f"  Удержание L28: {usage.l28_retention:.1f}%\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Детализация NPS", callback_data="pmf:nps_details")],
            [InlineKeyboardButton(text="🎯 PMF по функциям", callback_data="pmf:feature_pmf")],
            [InlineKeyboardButton(text="📤 Отправить NPS-опрос", callback_data="pmf:send_survey")],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="pmf:refresh")],
        ]
    )

    await message.answer(text, parse_mode="HTML", reply_markup=keyboard)


def _nps_emoji(nps: float) -> str:
    """Emoji для NPS score"""
    if nps >= 50:
        return "🌟 Отлично"
    elif nps >= 30:
        return "✅ Хорошо"
    elif nps >= 0:
        return "⚠️ Средне"
    else:
        return "🔴 Плохо"


def _pmf_status(achieved: bool) -> str:
    """Статус PMF"""
    if achieved:
        return "✅ <b>PMF достигнут!</b>"
    else:
        return "⚠️ PMF пока не достигнут"


def _stickiness_emoji(ratio: float) -> str:
    """Emoji для stickiness"""
    if ratio >= 20:
        return "✅"
    elif ratio >= 10:
        return "⚠️"
    else:
        return "🔴"


def _format_trend(trend: str) -> str:
    """Форматирование тренда"""
    emoji_map = {
        "improving": "📈 Улучшается",
        "stable": "➡️ Стабильно",
        "declining": "📉 Ухудшается",
        "insufficient_data": "❓ Недостаточно данных",
    }
    return emoji_map.get(trend, trend)


@pmf_router.callback_query(F.data == "pmf:refresh")
@require_admin
async def handle_pmf_refresh(callback: CallbackQuery, db, admin_ids: list[int]):
    """Обновить дашборд PMF"""
    await callback.answer("🔄 Обновляю...")

    metrics = PMFMetrics(db)
    nps = await metrics.get_nps(days=30)
    sean_ellis = await metrics.get_sean_ellis_score(days=30)
    usage = await metrics.get_usage_intensity()

    text = "📊 <b>Дашборд Product-Market Fit</b>\n\n"
    text += "<b>🎯 Индекс лояльности (NPS)</b>\n"
    text += f"  Индекс: <b>{nps.nps_score:+.0f}</b> {_nps_emoji(nps.nps_score)}\n"
    text += f"  Промоутеры: {nps.promoters} ({nps.promoter_rate:.1f}%)\n"
    text += f"  Нейтралы: {nps.passives}\n"
    text += f"  Критики: {nps.detractors} ({nps.detractor_rate:.1f}%)\n"
    text += f"  Средний балл: {nps.average_score:.1f}/10\n"
    text += f"  Доля ответов: {nps.response_rate:.1f}%\n"
    text += f"  Тренд: {_format_trend(nps.trend)}\n\n"

    text += "<b>💎 Тест Шона Эллиса (PMF)</b>\n"
    text += f"  Очень расстроены: <b>{sean_ellis.pmf_score:.1f}%</b>\n"
    text += f"  Статус PMF: {_pmf_status(sean_ellis.pmf_achieved)}\n"
    text += f"  Ответов: {sean_ellis.total_responses}\n\n"

    text += "<b>📈 Интенсивность использования</b>\n"
    text += f"  DAU (день): {usage.dau}\n"
    text += f"  WAU (неделя): {usage.wau}\n"
    text += f"  MAU (месяц): {usage.mau}\n"
    text += f"  DAU/MAU: {usage.dau_mau_ratio:.1f}% {_stickiness_emoji(usage.dau_mau_ratio)}\n"
    text += f"  Суперактивные: {usage.power_user_percentage:.1f}%\n"
    text += f"  Удержание L28: {usage.l28_retention:.1f}%\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Детализация NPS", callback_data="pmf:nps_details")],
            [InlineKeyboardButton(text="🎯 PMF по функциям", callback_data="pmf:feature_pmf")],
            [InlineKeyboardButton(text="📤 Отправить NPS-опрос", callback_data="pmf:send_survey")],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="pmf:refresh")],
        ]
    )

    await edit_or_answer(callback, text, keyboard)


@pmf_router.callback_query(F.data == "pmf:nps_details")
@require_admin
async def handle_nps_details(callback: CallbackQuery, db, admin_ids: list[int]):
    """Детальный NPS breakdown"""
    metrics = PMFMetrics(db)
    nps = await metrics.get_nps(days=30)

    text = "🎯 <b>Детальный разбор NPS</b>\n\n"

    text += f"<b>Итоговый NPS: {nps.nps_score:+.0f}</b> {_nps_emoji(nps.nps_score)}\n\n"

    text += "<b>📊 Распределение:</b>\n"
    text += f"  Промоутеры (9-10): {nps.promoters} ({nps.promoter_rate:.1f}%)\n"
    text += f"  Нейтралы (7-8): {nps.passives}\n"
    text += f"  Критики (0-6): {nps.detractors} ({nps.detractor_rate:.1f}%)\n\n"

    text += f"<b>📈 Средний балл:</b> {nps.average_score:.1f}/10\n\n"

    if nps.previous_nps is not None:
        change = nps.nps_score - nps.previous_nps
        text += f"<b>📊 Прошлый период:</b> {nps.previous_nps:+.0f}\n"
        text += f"<b>Изменение:</b> {change:+.1f} {_format_trend(nps.trend)}\n\n"

    if nps.nps_by_segment:
        text += "<b>🎯 NPS по сегментам:</b>\n"
        for segment, score in sorted(nps.nps_by_segment.items(), key=lambda x: x[1], reverse=True):
            text += f"  • {segment}: {score:+.0f}\n"

    text += f"\n<b>Доля ответов:</b> {nps.response_rate:.1f}%\n"

    # Recommendations
    text += "\n<b>💡 Рекомендации:</b>\n"
    if nps.nps_score < 0:
        text += "  🔴 Критично: сосредоточьтесь на устранении проблем критиков\n"
    elif nps.nps_score < 30:
        text += "  ⚠️ Улучшить: опросите критиков, чтобы выяснить причины\n"
    else:
        text += "  ✅ Хорошо: используйте промоутеров для рекомендаций\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Назад", callback_data="pmf:back")]]
    )

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@pmf_router.callback_query(F.data == "pmf:feature_pmf")
@require_admin
async def handle_feature_pmf(callback: CallbackQuery, db, admin_ids: list[int]):
    """Список фич для PMF анализа"""
    features = FEATURE_KEYS

    buttons = []
    for feature in features:
        buttons.append(
            [
                InlineKeyboardButton(
                    text=feature.replace("_", " ").title(),
                    callback_data=f"pmf:feature_details:{feature}",
                )
            ]
        )

    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="pmf:back")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)

    await edit_or_answer(callback, "🎯 <b>Выберите функцию для анализа PMF:</b>", keyboard)
    await callback.answer()


@pmf_router.callback_query(F.data.startswith("pmf:feature_details:"))
@require_admin
async def handle_feature_details(callback: CallbackQuery, db, admin_ids: list[int]):
    """Детальный PMF для фичи"""
    feature_name = callback.data.split(":")[-1]

    metrics = PMFMetrics(db)
    pmf = await metrics.get_feature_pmf(feature_name, days=30)

    text = f"🎯 <b>PMF функции: {feature_name.replace('_', ' ').title()}</b>\n\n"

    text += f"<b>Оценка PMF: {pmf.pmf_score:.0f}/100</b> {_pmf_rating_emoji(pmf.pmf_rating)}\n"
    text += f"<b>Категория:</b> {pmf.pmf_rating.upper()}\n\n"

    text += "<b>📊 Метрики использования:</b>\n"
    text += f"  Всего пользователей: {pmf.total_users}\n"
    text += f"  Активных пользователей: {pmf.active_users}\n"
    text += f"  Частота использования: {pmf.usage_frequency:.1f} раз в неделю\n\n"

    text += f"<b>😊 Удовлетворённость:</b> {pmf.satisfaction_score:.0f}/100\n\n"

    text += f"<b>💡 Вывод:</b>\n{pmf.key_insight}\n\n"

    # Action items based on PMF rating
    text += "<b>🎯 Следующие шаги:</b>\n"
    if pmf.pmf_rating == "strong":
        text += "  • Расширить инвестиции\n"
        text += "  • Добавить премиальные возможности\n"
        text += "  • Использовать как основной аргумент продаж\n"
    elif pmf.pmf_rating == "moderate":
        text += "  • Улучшить UX\n"
        text += "  • Проводить опрос пользователей\n"
        text += "  • Провести A/B‑тест улучшений\n"
    elif pmf.pmf_rating == "weak":
        text += "  • Нужен серьёзный редизайн\n"
        text += "  • Подумать о смене фокуса\n"
        text += "  • Провести глубинные интервью\n"
    else:  # kill
        text += "  • Рассмотреть отключение функции\n"
        text += "  • Освободить ресурсы для более перспективных функций\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="◀️ Назад к списку функций", callback_data="pmf:feature_pmf"
                )
            ],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="pmf:back")],
        ]
    )

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


def _pmf_rating_emoji(rating: str) -> str:
    """Emoji для PMF rating"""
    emoji_map = {"strong": "🌟", "moderate": "✅", "weak": "⚠️", "kill": "🗑️"}
    return emoji_map.get(rating, "")


@pmf_router.callback_query(F.data == "pmf:send_survey")
@require_admin
async def handle_send_survey(callback: CallbackQuery, db, admin_ids: list[int]):
    """Отправить NPS опрос"""
    text = "📤 <b>Отправить NPS-опрос</b>\n\n"
    text += "Выберите сегмент пользователей для отправки опроса:\n\n"
    text += "• Суперактивные — активные платящие пользователи\n"
    text += "• Конвертировавшиеся из триала — недавно оплатили\n"
    text += "• Группа риска — могут уйти\n"
    text += "• Все платящие — все пользователи с оплатой\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🌟 Суперактивные", callback_data="pmf:survey:power_users")],
            [
                InlineKeyboardButton(
                    text="💎 Конвертировавшиеся из триала",
                    callback_data="pmf:survey:trial_converters",
                )
            ],
            [InlineKeyboardButton(text="⚠️ Группа риска", callback_data="pmf:survey:at_risk")],
            [InlineKeyboardButton(text="💰 Все платящие", callback_data="pmf:survey:all_paid")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="pmf:back")],
        ]
    )

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@pmf_router.callback_query(F.data.startswith("pmf:survey:"))
@require_admin
async def handle_survey_segment(callback: CallbackQuery, db, admin_ids: list[int]):
    """Отправить опросы выбранному сегменту"""
    segment = callback.data.split(":")[-1]

    await callback.answer("📤 Отправляю опросы...", show_alert=False)

    # Получить пользователей сегмента
    async with db.pool.acquire() as conn:
        if segment == "all_paid":
            cursor = await conn.execute(
                """
                SELECT DISTINCT user_id
                FROM payments
                WHERE status = 'completed'
            """
            )
        elif segment == "power_users":
            cursor = await conn.execute(
                """
                SELECT user_id
                FROM users
                WHERE total_requests > 50
                  AND (
                      SELECT COUNT(*) FROM payments
                      WHERE payments.user_id = users.user_id AND status = 'completed'
                  ) >= 2
            """
            )
        elif segment == "trial_converters":
            cursor = await conn.execute(
                """
                SELECT DISTINCT user_id
                FROM payments
                WHERE status = 'completed'
                  AND created_at > strftime('%s', 'now', '-7 days')
            """
            )
        elif segment == "at_risk":
            cursor = await conn.execute(
                """
                SELECT user_id
                FROM users
                WHERE total_requests > 20
                  AND (strftime('%s', 'now') - last_active) > 604800
                  AND (
                      SELECT COUNT(*) FROM payments
                      WHERE payments.user_id = users.user_id AND status = 'completed'
                  ) >= 1
            """
            )
        else:
            await edit_or_answer(callback, "❌ Неизвестный сегмент", parse_mode=None)
            return

        rows = await cursor.fetchall()
        await cursor.close()

    # Отправить опросы
    metrics = PMFMetrics(db)
    sent_count = 0

    for row in rows:
        user_id = row[0]
        success = await metrics.send_nps_survey(user_id, trigger=f"admin_bulk_{segment}")
        if success:
            sent_count += 1

    text = "✅ <b>NPS-опросы отправлены</b>\n\n"
    text += f"Сегмент: {segment}\n"
    text += f"Отправлено: {sent_count} пользователям\n\n"
    text += "Пользователи получат опрос при следующем взаимодействии с ботом."

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="◀️ Назад", callback_data="pmf:back")]]
    )

    await edit_or_answer(callback, text, keyboard)


@pmf_router.callback_query(F.data == "pmf:back")
@require_admin
async def handle_back_to_main(callback: CallbackQuery, db, admin_ids: list[int]):
    """Вернуться в главное меню PMF"""
    metrics = PMFMetrics(db)
    nps = await metrics.get_nps(days=30)
    sean_ellis = await metrics.get_sean_ellis_score(days=30)
    usage = await metrics.get_usage_intensity()

    text = "📊 <b>Дашборд Product-Market Fit</b>\n\n"
    text += "<b>🎯 Индекс лояльности (NPS)</b>\n"
    text += f"  Индекс: <b>{nps.nps_score:+.0f}</b> {_nps_emoji(nps.nps_score)}\n"
    text += f"  Промоутеры: {nps.promoters} ({nps.promoter_rate:.1f}%)\n"
    text += f"  Нейтралы: {nps.passives}\n"
    text += f"  Критики: {nps.detractors} ({nps.detractor_rate:.1f}%)\n"
    text += f"  Средний балл: {nps.average_score:.1f}/10\n"
    text += f"  Доля ответов: {nps.response_rate:.1f}%\n"
    text += f"  Тренд: {_format_trend(nps.trend)}\n\n"

    text += "<b>💎 Тест Шона Эллиса (PMF)</b>\n"
    text += f"  Очень расстроены: <b>{sean_ellis.pmf_score:.1f}%</b>\n"
    text += f"  Статус PMF: {_pmf_status(sean_ellis.pmf_achieved)}\n"
    text += f"  Ответов: {sean_ellis.total_responses}\n\n"

    text += "<b>📈 Интенсивность использования</b>\n"
    text += f"  DAU (день): {usage.dau}\n"
    text += f"  WAU (неделя): {usage.wau}\n"
    text += f"  MAU (месяц): {usage.mau}\n"
    text += f"  DAU/MAU: {usage.dau_mau_ratio:.1f}% {_stickiness_emoji(usage.dau_mau_ratio)}\n"
    text += f"  Суперактивные: {usage.power_user_percentage:.1f}%\n"
    text += f"  Удержание L28: {usage.l28_retention:.1f}%\n"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Детализация NPS", callback_data="pmf:nps_details")],
            [InlineKeyboardButton(text="🎯 PMF по функциям", callback_data="pmf:feature_pmf")],
            [InlineKeyboardButton(text="📤 Отправить NPS-опрос", callback_data="pmf:send_survey")],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="pmf:refresh")],
        ]
    )

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


__all__ = (
    "pmf_router",
    "cmd_pmf",
    "handle_pmf_refresh",
    "handle_nps_details",
    "handle_feature_pmf",
    "handle_feature_details",
    "handle_send_survey",
    "handle_survey_segment",
    "handle_back_to_main",
)
