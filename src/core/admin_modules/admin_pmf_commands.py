"""
Admin commands для PMF/NPS metrics
"""

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from src.core.admin_modules.admin_utils import FEATURE_KEYS, edit_or_answer, require_admin
from src.core.admin_modules.pmf_metrics import PMFMetrics


pmf_router = Router(name="pmf_admin")


@pmf_router.message(Command("pmf"))
@require_admin
async def cmd_pmf(message: Message, db, admin_ids: list[int]):
    """Главное меню PMF metrics"""
    metrics = PMFMetrics(db)

    # Получить все метрики
    nps = await metrics.get_nps(days=30)
    sean_ellis = await metrics.get_sean_ellis_score(days=30)
    usage = await metrics.get_usage_intensity()

    text = "📊 <b>Product-Market Fit Dashboard</b>\n\n"

    # NPS Section
    text += "<b>🎯 Net Promoter Score (NPS)</b>\n"
    text += f"  Score: <b>{nps.nps_score:+.0f}</b> {_nps_emoji(nps.nps_score)}\n"
    text += f"  Promoters: {nps.promoters} ({nps.promoter_rate:.1f}%)\n"
    text += f"  Passives: {nps.passives}\n"
    text += f"  Detractors: {nps.detractors} ({nps.detractor_rate:.1f}%)\n"
    text += f"  Avg Score: {nps.average_score:.1f}/10\n"
    text += f"  Response Rate: {nps.response_rate:.1f}%\n"
    text += f"  Trend: {_format_trend(nps.trend)}\n\n"

    # Sean Ellis Test
    text += "<b>💎 Sean Ellis Test (PMF)</b>\n"
    text += f"  Very Disappointed: <b>{sean_ellis.pmf_score:.1f}%</b>\n"
    text += f"  PMF Status: {_pmf_status(sean_ellis.pmf_achieved)}\n"
    text += f"  Responses: {sean_ellis.total_responses}\n\n"

    # Usage Intensity
    text += "<b>📈 Usage Intensity</b>\n"
    text += f"  DAU: {usage.dau}\n"
    text += f"  WAU: {usage.wau}\n"
    text += f"  MAU: {usage.mau}\n"
    text += f"  DAU/MAU: {usage.dau_mau_ratio:.1f}% {_stickiness_emoji(usage.dau_mau_ratio)}\n"
    text += f"  Power Users: {usage.power_user_percentage:.1f}%\n"
    text += f"  L28 Retention: {usage.l28_retention:.1f}%\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 NPS Details", callback_data="pmf:nps_details")],
        [InlineKeyboardButton(text="🎯 Feature PMF", callback_data="pmf:feature_pmf")],
        [InlineKeyboardButton(text="📤 Send NPS Survey", callback_data="pmf:send_survey")],
        [InlineKeyboardButton(text="🔄 Refresh", callback_data="pmf:refresh")]
    ])

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
        return "✅ <b>PMF Achieved!</b>"
    else:
        return "⚠️ PMF not yet achieved"


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
        "insufficient_data": "❓ Недостаточно данных"
    }
    return emoji_map.get(trend, trend)


@pmf_router.callback_query(F.data == "pmf:refresh")
@require_admin
async def handle_pmf_refresh(callback: CallbackQuery, db, admin_ids: list[int]):
    """Обновить PMF dashboard"""
    await callback.answer("🔄 Обновляю...")

    metrics = PMFMetrics(db)
    nps = await metrics.get_nps(days=30)
    sean_ellis = await metrics.get_sean_ellis_score(days=30)
    usage = await metrics.get_usage_intensity()

    text = "📊 <b>Product-Market Fit Dashboard</b>\n\n"
    text += "<b>🎯 Net Promoter Score (NPS)</b>\n"
    text += f"  Score: <b>{nps.nps_score:+.0f}</b> {_nps_emoji(nps.nps_score)}\n"
    text += f"  Promoters: {nps.promoters} ({nps.promoter_rate:.1f}%)\n"
    text += f"  Passives: {nps.passives}\n"
    text += f"  Detractors: {nps.detractors} ({nps.detractor_rate:.1f}%)\n"
    text += f"  Avg Score: {nps.average_score:.1f}/10\n"
    text += f"  Response Rate: {nps.response_rate:.1f}%\n"
    text += f"  Trend: {_format_trend(nps.trend)}\n\n"

    text += "<b>💎 Sean Ellis Test (PMF)</b>\n"
    text += f"  Very Disappointed: <b>{sean_ellis.pmf_score:.1f}%</b>\n"
    text += f"  PMF Status: {_pmf_status(sean_ellis.pmf_achieved)}\n"
    text += f"  Responses: {sean_ellis.total_responses}\n\n"

    text += "<b>📈 Usage Intensity</b>\n"
    text += f"  DAU: {usage.dau}\n"
    text += f"  WAU: {usage.wau}\n"
    text += f"  MAU: {usage.mau}\n"
    text += f"  DAU/MAU: {usage.dau_mau_ratio:.1f}% {_stickiness_emoji(usage.dau_mau_ratio)}\n"
    text += f"  Power Users: {usage.power_user_percentage:.1f}%\n"
    text += f"  L28 Retention: {usage.l28_retention:.1f}%\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 NPS Details", callback_data="pmf:nps_details")],
        [InlineKeyboardButton(text="🎯 Feature PMF", callback_data="pmf:feature_pmf")],
        [InlineKeyboardButton(text="📤 Send NPS Survey", callback_data="pmf:send_survey")],
        [InlineKeyboardButton(text="🔄 Refresh", callback_data="pmf:refresh")]
    ])

    await edit_or_answer(callback, text, keyboard)


@pmf_router.callback_query(F.data == "pmf:nps_details")
@require_admin
async def handle_nps_details(callback: CallbackQuery, db, admin_ids: list[int]):
    """Детальный NPS breakdown"""
    metrics = PMFMetrics(db)
    nps = await metrics.get_nps(days=30)

    text = "🎯 <b>NPS Detailed Breakdown</b>\n\n"

    text += f"<b>Overall NPS: {nps.nps_score:+.0f}</b> {_nps_emoji(nps.nps_score)}\n\n"

    text += "<b>📊 Distribution:</b>\n"
    text += f"  Promoters (9-10): {nps.promoters} ({nps.promoter_rate:.1f}%)\n"
    text += f"  Passives (7-8): {nps.passives}\n"
    text += f"  Detractors (0-6): {nps.detractors} ({nps.detractor_rate:.1f}%)\n\n"

    text += f"<b>📈 Average Score:</b> {nps.average_score:.1f}/10\n\n"

    if nps.previous_nps is not None:
        change = nps.nps_score - nps.previous_nps
        text += f"<b>📊 Previous Period:</b> {nps.previous_nps:+.0f}\n"
        text += f"<b>Change:</b> {change:+.1f} {_format_trend(nps.trend)}\n\n"

    if nps.nps_by_segment:
        text += "<b>🎯 NPS by Segment:</b>\n"
        for segment, score in sorted(nps.nps_by_segment.items(), key=lambda x: x[1], reverse=True):
            text += f"  • {segment}: {score:+.0f}\n"

    text += f"\n<b>Response Rate:</b> {nps.response_rate:.1f}%\n"

    # Recommendations
    text += "\n<b>💡 Recommendations:</b>\n"
    if nps.nps_score < 0:
        text += "  🔴 Критично: фокус на fixing detractor issues\n"
    elif nps.nps_score < 30:
        text += "  ⚠️ Улучшать: survey detractors для выявления проблем\n"
    else:
        text += "  ✅ Хорошо: leverage promoters для referrals\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back", callback_data="pmf:back")]
    ])

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


@pmf_router.callback_query(F.data == "pmf:feature_pmf")
@require_admin
async def handle_feature_pmf(callback: CallbackQuery, db, admin_ids: list[int]):
    """Список фич для PMF анализа"""
    features = FEATURE_KEYS

    buttons = []
    for feature in features:
        buttons.append([
            InlineKeyboardButton(
                text=feature.replace("_", " ").title(),
                callback_data=f"pmf:feature_details:{feature}"
            )
        ])

    buttons.append([InlineKeyboardButton(text="◀️ Back", callback_data="pmf:back")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)

    await edit_or_answer(callback, "🎯 <b>Select feature for PMF analysis:</b>", keyboard)
    await callback.answer()


@pmf_router.callback_query(F.data.startswith("pmf:feature_details:"))
@require_admin
async def handle_feature_details(callback: CallbackQuery, db, admin_ids: list[int]):
    """Детальный PMF для фичи"""
    feature_name = callback.data.split(":")[-1]

    metrics = PMFMetrics(db)
    pmf = await metrics.get_feature_pmf(feature_name, days=30)

    text = f"🎯 <b>Feature PMF: {feature_name.replace('_', ' ').title()}</b>\n\n"

    text += f"<b>PMF Score: {pmf.pmf_score:.0f}/100</b> {_pmf_rating_emoji(pmf.pmf_rating)}\n"
    text += f"<b>Rating:</b> {pmf.pmf_rating.upper()}\n\n"

    text += "<b>📊 Usage Metrics:</b>\n"
    text += f"  Total users: {pmf.total_users}\n"
    text += f"  Active users: {pmf.active_users}\n"
    text += f"  Usage frequency: {pmf.usage_frequency:.1f} uses/week\n\n"

    text += f"<b>😊 Satisfaction:</b> {pmf.satisfaction_score:.0f}/100\n\n"

    text += f"<b>💡 Insight:</b>\n{pmf.key_insight}\n\n"

    # Action items based on PMF rating
    text += "<b>🎯 Next Steps:</b>\n"
    if pmf.pmf_rating == "strong":
        text += "  • Invest more resources\n"
        text += "  • Add premium features\n"
        text += "  • Use as primary selling point\n"
    elif pmf.pmf_rating == "moderate":
        text += "  • Improve UX\n"
        text += "  • Survey users for feedback\n"
        text += "  • A/B test improvements\n"
    elif pmf.pmf_rating == "weak":
        text += "  • Major redesign needed\n"
        text += "  • Consider pivot\n"
        text += "  • Deep user interviews\n"
    else:  # kill
        text += "  • Consider removing feature\n"
        text += "  • Free up resources for better features\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back to features", callback_data="pmf:feature_pmf")],
        [InlineKeyboardButton(text="🏠 Main menu", callback_data="pmf:back")]
    ])

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()


def _pmf_rating_emoji(rating: str) -> str:
    """Emoji для PMF rating"""
    emoji_map = {
        "strong": "🌟",
        "moderate": "✅",
        "weak": "⚠️",
        "kill": "🗑️"
    }
    return emoji_map.get(rating, "")


@pmf_router.callback_query(F.data == "pmf:send_survey")
@require_admin
async def handle_send_survey(callback: CallbackQuery, db, admin_ids: list[int]):
    """Отправить NPS опрос"""
    text = "📤 <b>Send NPS Survey</b>\n\n"
    text += "Выберите сегмент пользователей для отправки опроса:\n\n"
    text += "• Power Users - активные платящие пользователи\n"
    text += "• Trial Converters - недавно оплатили\n"
    text += "• At Risk - могут уйти\n"
    text += "• All Paid Users - все платящие\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🌟 Power Users", callback_data="pmf:survey:power_users")],
        [InlineKeyboardButton(text="💎 Trial Converters", callback_data="pmf:survey:trial_converters")],
        [InlineKeyboardButton(text="⚠️ At Risk", callback_data="pmf:survey:at_risk")],
        [InlineKeyboardButton(text="💰 All Paid", callback_data="pmf:survey:all_paid")],
        [InlineKeyboardButton(text="◀️ Back", callback_data="pmf:back")]
    ])

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
            cursor = await conn.execute("""
                SELECT DISTINCT user_id
                FROM payments
                WHERE status = 'completed'
            """)
        elif segment == "power_users":
            cursor = await conn.execute("""
                SELECT user_id
                FROM users
                WHERE total_requests > 50
                  AND (
                      SELECT COUNT(*) FROM payments
                      WHERE payments.user_id = users.user_id AND status = 'completed'
                  ) >= 2
            """)
        elif segment == "trial_converters":
            cursor = await conn.execute("""
                SELECT DISTINCT user_id
                FROM payments
                WHERE status = 'completed'
                  AND created_at > strftime('%s', 'now', '-7 days')
            """)
        elif segment == "at_risk":
            cursor = await conn.execute("""
                SELECT user_id
                FROM users
                WHERE total_requests > 20
                  AND (strftime('%s', 'now') - last_active) > 604800
                  AND (
                      SELECT COUNT(*) FROM payments
                      WHERE payments.user_id = users.user_id AND status = 'completed'
                  ) >= 1
            """)
        else:
            await edit_or_answer(callback, "❌ Unknown segment", parse_mode=None)
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

    text = f"✅ <b>NPS Surveys Sent</b>\n\n"
    text += f"Segment: {segment}\n"
    text += f"Sent to: {sent_count} users\n\n"
    text += "Пользователи получат опрос при следующем взаимодействии с ботом."

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back", callback_data="pmf:back")]
    ])

    await edit_or_answer(callback, text, keyboard)


@pmf_router.callback_query(F.data == "pmf:back")
@require_admin
async def handle_back_to_main(callback: CallbackQuery, db, admin_ids: list[int]):
    """Вернуться в главное меню PMF"""
    metrics = PMFMetrics(db)
    nps = await metrics.get_nps(days=30)
    sean_ellis = await metrics.get_sean_ellis_score(days=30)
    usage = await metrics.get_usage_intensity()

    text = "📊 <b>Product-Market Fit Dashboard</b>\n\n"
    text += "<b>🎯 Net Promoter Score (NPS)</b>\n"
    text += f"  Score: <b>{nps.nps_score:+.0f}</b> {_nps_emoji(nps.nps_score)}\n"
    text += f"  Promoters: {nps.promoters} ({nps.promoter_rate:.1f}%)\n"
    text += f"  Passives: {nps.passives}\n"
    text += f"  Detractors: {nps.detractors} ({nps.detractor_rate:.1f}%)\n"
    text += f"  Avg Score: {nps.average_score:.1f}/10\n"
    text += f"  Response Rate: {nps.response_rate:.1f}%\n"
    text += f"  Trend: {_format_trend(nps.trend)}\n\n"

    text += "<b>💎 Sean Ellis Test (PMF)</b>\n"
    text += f"  Very Disappointed: <b>{sean_ellis.pmf_score:.1f}%</b>\n"
    text += f"  PMF Status: {_pmf_status(sean_ellis.pmf_achieved)}\n"
    text += f"  Responses: {sean_ellis.total_responses}\n\n"

    text += "<b>📈 Usage Intensity</b>\n"
    text += f"  DAU: {usage.dau}\n"
    text += f"  WAU: {usage.wau}\n"
    text += f"  MAU: {usage.mau}\n"
    text += f"  DAU/MAU: {usage.dau_mau_ratio:.1f}% {_stickiness_emoji(usage.dau_mau_ratio)}\n"
    text += f"  Power Users: {usage.power_user_percentage:.1f}%\n"
    text += f"  L28 Retention: {usage.l28_retention:.1f}%\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 NPS Details", callback_data="pmf:nps_details")],
        [InlineKeyboardButton(text="🎯 Feature PMF", callback_data="pmf:feature_pmf")],
        [InlineKeyboardButton(text="📤 Send NPS Survey", callback_data="pmf:send_survey")],
        [InlineKeyboardButton(text="🔄 Refresh", callback_data="pmf:refresh")]
    ])

    await edit_or_answer(callback, text, keyboard)
    await callback.answer()
