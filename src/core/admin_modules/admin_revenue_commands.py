"""
Admin commands для Revenue Analytics
"""

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from src.core.admin_modules.revenue_analytics import RevenueAnalytics


revenue_router = Router(name="revenue_admin")


@revenue_router.message(Command("revenue"))
async def cmd_revenue(message: Message, db, admin_ids: list[int]):
    """Главное меню revenue analytics"""
    if message.from_user.id not in admin_ids:
        await message.answer("⛔️ Доступ запрещен")
        return

    analytics = RevenueAnalytics(db)

    # Текущий месяц MRR
    mrr = await analytics.get_mrr_breakdown()
    arr_metrics = await analytics.get_arr_metrics()
    unit_econ = await analytics.get_unit_economics()

    text = "💰 <b>Revenue Analytics Dashboard</b>\n\n"

    # MRR Overview
    text += f"<b>📊 MRR ({mrr.month}):</b> {mrr.total_mrr:,}₽\n"
    text += f"  Growth: {mrr.mrr_growth_rate:+.1f}% {_growth_emoji(mrr.mrr_growth_rate)}\n"
    text += f"  Net New MRR: {mrr.net_new_mrr:+,}₽\n\n"

    # MRR Breakdown
    text += "<b>🔍 MRR Breakdown:</b>\n"
    text += f"  New: +{mrr.new_mrr:,}₽ ({mrr.new_customers} customers)\n"
    text += f"  Expansion: +{mrr.expansion_mrr:,}₽\n"
    text += f"  Churn: -{mrr.churn_mrr:,}₽ ({mrr.churned_customers} lost)\n"
    text += f"  Contraction: -{mrr.contraction_mrr:,}₽\n\n"

    # ARR
    text += f"<b>📈 ARR:</b> {arr_metrics.arr:,}₽\n"
    text += f"  Projected ARR (12mo): {arr_metrics.projected_arr:,}₽\n"
    text += f"  Quick Ratio: {arr_metrics.quick_ratio:.2f} {_quick_ratio_status(arr_metrics.quick_ratio)}\n\n"

    # Customers
    text += f"<b>👥 Customers:</b> {mrr.total_paying_customers}\n"
    text += f"  ARPU: {mrr.arpu:,.0f}₽\n"
    text += f"  Churn Rate: {mrr.customer_churn_rate:.1f}%\n\n"

    # Unit Economics
    text += "<b>💎 Unit Economics:</b>\n"
    text += f"  LTV: {unit_econ.ltv:,.0f}₽\n"
    text += f"  CAC: {unit_econ.cac:,.0f}₽\n"
    text += f"  LTV/CAC: {unit_econ.ltv_cac_ratio:.2f}x {_ltv_cac_status(unit_econ.ltv_cac_ratio)}\n"
    text += f"  Payback: {unit_econ.payback_period:.1f} months\n"
    text += f"  Gross Margin: {unit_econ.gross_margin*100:.0f}%\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 MRR History", callback_data="revenue:mrr_history")],
        [InlineKeyboardButton(text="🔮 Revenue Forecast", callback_data="revenue:forecast")],
        [InlineKeyboardButton(text="🛤️ Runway Calculator", callback_data="revenue:runway")],
        [InlineKeyboardButton(text="📈 Unit Economics", callback_data="revenue:unit_econ")],
        [InlineKeyboardButton(text="🔄 Refresh", callback_data="revenue:refresh")]
    ])

    await message.answer(text, parse_mode="HTML", reply_markup=keyboard)


def _growth_emoji(rate: float) -> str:
    """Emoji для growth rate"""
    if rate > 10:
        return "🚀"
    elif rate > 0:
        return "✅"
    elif rate > -10:
        return "⚠️"
    else:
        return "🔴"


def _quick_ratio_status(ratio: float) -> str:
    """Статус Quick Ratio"""
    if ratio > 4:
        return "🌟 Excellent"
    elif ratio > 2:
        return "✅ Good"
    elif ratio > 1:
        return "⚠️ OK"
    else:
        return "🔴 Poor"


def _ltv_cac_status(ratio: float) -> str:
    """Статус LTV/CAC"""
    if ratio > 3:
        return "✅"
    elif ratio > 1:
        return "⚠️"
    else:
        return "🔴"


@revenue_router.callback_query(F.data == "revenue:refresh")
async def handle_revenue_refresh(callback: CallbackQuery, db, admin_ids: list[int]):
    """Обновить revenue dashboard"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    await callback.answer("🔄 Обновляю...")

    analytics = RevenueAnalytics(db)
    mrr = await analytics.get_mrr_breakdown()
    arr_metrics = await analytics.get_arr_metrics()
    unit_econ = await analytics.get_unit_economics()

    text = "💰 <b>Revenue Analytics Dashboard</b>\n\n"
    text += f"<b>📊 MRR ({mrr.month}):</b> {mrr.total_mrr:,}₽\n"
    text += f"  Growth: {mrr.mrr_growth_rate:+.1f}% {_growth_emoji(mrr.mrr_growth_rate)}\n"
    text += f"  Net New MRR: {mrr.net_new_mrr:+,}₽\n\n"

    text += "<b>🔍 MRR Breakdown:</b>\n"
    text += f"  New: +{mrr.new_mrr:,}₽ ({mrr.new_customers} customers)\n"
    text += f"  Expansion: +{mrr.expansion_mrr:,}₽\n"
    text += f"  Churn: -{mrr.churn_mrr:,}₽ ({mrr.churned_customers} lost)\n"
    text += f"  Contraction: -{mrr.contraction_mrr:,}₽\n\n"

    text += f"<b>📈 ARR:</b> {arr_metrics.arr:,}₽\n"
    text += f"  Projected ARR (12mo): {arr_metrics.projected_arr:,}₽\n"
    text += f"  Quick Ratio: {arr_metrics.quick_ratio:.2f} {_quick_ratio_status(arr_metrics.quick_ratio)}\n\n"

    text += f"<b>👥 Customers:</b> {mrr.total_paying_customers}\n"
    text += f"  ARPU: {mrr.arpu:,.0f}₽\n"
    text += f"  Churn Rate: {mrr.customer_churn_rate:.1f}%\n\n"

    text += "<b>💎 Unit Economics:</b>\n"
    text += f"  LTV: {unit_econ.ltv:,.0f}₽\n"
    text += f"  CAC: {unit_econ.cac:,.0f}₽\n"
    text += f"  LTV/CAC: {unit_econ.ltv_cac_ratio:.2f}x {_ltv_cac_status(unit_econ.ltv_cac_ratio)}\n"
    text += f"  Payback: {unit_econ.payback_period:.1f} months\n"
    text += f"  Gross Margin: {unit_econ.gross_margin*100:.0f}%\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 MRR History", callback_data="revenue:mrr_history")],
        [InlineKeyboardButton(text="🔮 Revenue Forecast", callback_data="revenue:forecast")],
        [InlineKeyboardButton(text="🛤️ Runway Calculator", callback_data="revenue:runway")],
        [InlineKeyboardButton(text="📈 Unit Economics", callback_data="revenue:unit_econ")],
        [InlineKeyboardButton(text="🔄 Refresh", callback_data="revenue:refresh")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)


@revenue_router.callback_query(F.data == "revenue:mrr_history")
async def handle_mrr_history(callback: CallbackQuery, db, admin_ids: list[int]):
    """MRR History за последние месяцы"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    analytics = RevenueAnalytics(db)
    history = await analytics.get_mrr_history(months=12)

    text = "📊 <b>MRR History (12 months)</b>\n\n"

    for mrr in history:
        text += f"<b>{mrr.month}</b>\n"
        text += f"  MRR: {mrr.total_mrr:,}₽ ({mrr.mrr_growth_rate:+.1f}%)\n"
        text += f"  New: +{mrr.new_mrr:,} | Exp: +{mrr.expansion_mrr:,}\n"
        text += f"  Churn: -{mrr.churn_mrr:,} | Customers: {mrr.total_paying_customers}\n\n"

    # ASCII chart
    if history:
        text += "<b>📈 MRR Trend:</b>\n"
        max_mrr = max(m.total_mrr for m in history) if history else 1

        for mrr in history[-6:]:  # Последние 6 месяцев
            bar_length = int((mrr.total_mrr / max_mrr) * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            text += f"{mrr.month}: {bar} {mrr.total_mrr:,}₽\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back", callback_data="revenue:back")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
    await callback.answer()


@revenue_router.callback_query(F.data == "revenue:forecast")
async def handle_revenue_forecast(callback: CallbackQuery, db, admin_ids: list[int]):
    """Revenue forecast"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    analytics = RevenueAnalytics(db)
    forecasts = await analytics.get_revenue_forecast(months_ahead=6)

    if not forecasts:
        await callback.answer("❌ Недостаточно данных для прогноза", show_alert=True)
        return

    text = "🔮 <b>Revenue Forecast (6 months)</b>\n\n"

    text += f"<b>Assumptions:</b>\n"
    text += f"  Growth Rate: {forecasts[0].assumed_growth_rate*100:+.1f}%/month\n"
    text += f"  Churn Rate: {forecasts[0].assumed_churn_rate*100:.1f}%/month\n\n"

    text += "<b>📊 Projections:</b>\n\n"

    for fc in forecasts[:6]:
        text += f"<b>{fc.month}</b> (confidence: {fc.confidence*100:.0f}%)\n"
        text += f"  Conservative: {fc.mrr_forecast_low:,}₽\n"
        text += f"  Expected: {fc.mrr_forecast_mid:,}₽\n"
        text += f"  Optimistic: {fc.mrr_forecast_high:,}₽\n\n"

    # Визуализация expected forecast
    max_mrr = max(f.mrr_forecast_high for f in forecasts[:6])
    text += "<b>📈 Expected Trajectory:</b>\n"

    for fc in forecasts[:6]:
        bar_length = int((fc.mrr_forecast_mid / max_mrr) * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        text += f"{fc.month}: {bar}\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back", callback_data="revenue:back")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
    await callback.answer()


@revenue_router.callback_query(F.data == "revenue:runway")
async def handle_runway_calculator(callback: CallbackQuery, db, admin_ids: list[int]):
    """Runway calculator"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    text = "🛤️ <b>Runway Calculator</b>\n\n"

    text += "Для расчета runway введите команду:\n\n"
    text += "<code>/runway [cash] [monthly_burn]</code>\n\n"

    text += "<b>Пример:</b>\n"
    text += "<code>/runway 500000 -50000</code>\n\n"

    text += "Где:\n"
    text += "• cash - текущий баланс в рублях\n"
    text += "• monthly_burn - ежемесячный расход (negative)\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back", callback_data="revenue:back")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
    await callback.answer()


@revenue_router.message(Command("runway"))
async def cmd_runway(message: Message, db, admin_ids: list[int]):
    """Runway calculation"""
    if message.from_user.id not in admin_ids:
        await message.answer("⛔️ Доступ запрещен")
        return

    args = message.text.split()[1:]

    if len(args) < 2:
        await message.answer(
            "❌ Неверный формат\n\n"
            "Используйте: <code>/runway [cash] [monthly_burn]</code>\n"
            "Пример: <code>/runway 500000 -50000</code>",
            parse_mode="HTML"
        )
        return

    try:
        current_cash = int(args[0])
        monthly_burn = int(args[1])
    except ValueError:
        await message.answer("❌ Неверные числа")
        return

    analytics = RevenueAnalytics(db)
    runway = await analytics.calculate_runway(current_cash, monthly_burn)

    text = "🛤️ <b>Runway Analysis</b>\n\n"

    text += f"<b>💰 Current Cash:</b> {current_cash:,}₽\n"
    text += f"<b>🔥 Monthly Burn:</b> {monthly_burn:,}₽\n\n"

    text += f"<b>⏱ Runway:</b> {runway['runway_months']} months\n"
    text += f"<b>📅 Cash out date:</b> {runway['runway_end_date']}\n\n"

    text += f"<b>💎 Current MRR:</b> {runway['current_mrr']:,}₽\n"
    text += f"<b>🎯 Breakeven MRR:</b> {runway['breakeven_mrr']:,}₽\n"
    text += f"<b>📈 MRR Growth:</b> {runway['mrr_growth_rate']:+.1f}%/month\n\n"

    if runway['months_to_breakeven']:
        text += f"<b>⏳ Months to Breakeven:</b> {runway['months_to_breakeven']}\n\n"

        if runway['months_to_breakeven'] < runway['runway_months']:
            text += "✅ <b>You'll reach breakeven before running out of cash!</b>\n"
        else:
            text += "🔴 <b>Warning: You'll run out of cash before breakeven</b>\n"
            text += f"Need {runway['months_to_breakeven'] - runway['runway_months']} more months of runway\n"
    else:
        text += "⚠️ At current growth rate, won't reach breakeven\n"

    await message.answer(text, parse_mode="HTML")


@revenue_router.callback_query(F.data == "revenue:unit_econ")
async def handle_unit_economics(callback: CallbackQuery, db, admin_ids: list[int]):
    """Детальный Unit Economics"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    analytics = RevenueAnalytics(db)
    unit_econ = await analytics.get_unit_economics()

    text = "💎 <b>Unit Economics Deep Dive</b>\n\n"

    text += "<b>💰 Customer Lifetime Value (LTV):</b>\n"
    text += f"  {unit_econ.ltv:,.0f}₽\n\n"

    text += "<b>📊 Calculation:</b>\n"
    text += f"  Monthly Churn: {unit_econ.monthly_churn*100:.2f}%\n"
    text += f"  Avg Lifetime: {unit_econ.avg_customer_lifetime_months:.1f} months\n"
    text += f"  ARPU: {unit_econ.ltv / unit_econ.avg_customer_lifetime_months:,.0f}₽/month\n"
    text += f"  LTV = ARPU × Lifetime\n\n"

    text += "<b>💸 Customer Acquisition Cost (CAC):</b>\n"
    text += f"  {unit_econ.cac:,.0f}₽\n"
    text += f"  <i>Note: Estimated based on LTV (30% ratio)</i>\n\n"

    text += "<b>🎯 Key Metrics:</b>\n"
    text += f"  LTV/CAC Ratio: {unit_econ.ltv_cac_ratio:.2f}x {_ltv_cac_status(unit_econ.ltv_cac_ratio)}\n"
    text += f"  Payback Period: {unit_econ.payback_period:.1f} months\n"
    text += f"  Gross Margin: {unit_econ.gross_margin*100:.0f}%\n\n"

    text += "<b>💡 Benchmarks:</b>\n"
    text += "  LTV/CAC > 3 = ✅ Excellent\n"
    text += "  Payback < 12 months = ✅ Good\n"
    text += "  Gross Margin > 70% = ✅ Healthy\n\n"

    # Recommendations
    if unit_econ.ltv_cac_ratio < 3:
        text += "⚠️ <b>Action:</b> Improve retention or reduce CAC\n"
    elif unit_econ.payback_period > 12:
        text += "⚠️ <b>Action:</b> Increase ARPU or reduce CAC\n"
    else:
        text += "✅ <b>Unit economics look healthy!</b>\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back", callback_data="revenue:back")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
    await callback.answer()


@revenue_router.callback_query(F.data == "revenue:back")
async def handle_back_to_main(callback: CallbackQuery, db, admin_ids: list[int]):
    """Вернуться в главное меню revenue"""
    if callback.from_user.id not in admin_ids:
        await callback.answer("⛔️ Доступ запрещен", show_alert=True)
        return

    analytics = RevenueAnalytics(db)
    mrr = await analytics.get_mrr_breakdown()
    arr_metrics = await analytics.get_arr_metrics()
    unit_econ = await analytics.get_unit_economics()

    text = "💰 <b>Revenue Analytics Dashboard</b>\n\n"
    text += f"<b>📊 MRR ({mrr.month}):</b> {mrr.total_mrr:,}₽\n"
    text += f"  Growth: {mrr.mrr_growth_rate:+.1f}% {_growth_emoji(mrr.mrr_growth_rate)}\n"
    text += f"  Net New MRR: {mrr.net_new_mrr:+,}₽\n\n"

    text += "<b>🔍 MRR Breakdown:</b>\n"
    text += f"  New: +{mrr.new_mrr:,}₽ ({mrr.new_customers} customers)\n"
    text += f"  Expansion: +{mrr.expansion_mrr:,}₽\n"
    text += f"  Churn: -{mrr.churn_mrr:,}₽ ({mrr.churned_customers} lost)\n"
    text += f"  Contraction: -{mrr.contraction_mrr:,}₽\n\n"

    text += f"<b>📈 ARR:</b> {arr_metrics.arr:,}₽\n"
    text += f"  Projected ARR (12mo): {arr_metrics.projected_arr:,}₽\n"
    text += f"  Quick Ratio: {arr_metrics.quick_ratio:.2f} {_quick_ratio_status(arr_metrics.quick_ratio)}\n\n"

    text += f"<b>👥 Customers:</b> {mrr.total_paying_customers}\n"
    text += f"  ARPU: {mrr.arpu:,.0f}₽\n"
    text += f"  Churn Rate: {mrr.customer_churn_rate:.1f}%\n\n"

    text += "<b>💎 Unit Economics:</b>\n"
    text += f"  LTV: {unit_econ.ltv:,.0f}₽\n"
    text += f"  CAC: {unit_econ.cac:,.0f}₽\n"
    text += f"  LTV/CAC: {unit_econ.ltv_cac_ratio:.2f}x {_ltv_cac_status(unit_econ.ltv_cac_ratio)}\n"
    text += f"  Payback: {unit_econ.payback_period:.1f} months\n"
    text += f"  Gross Margin: {unit_econ.gross_margin*100:.0f}%\n"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 MRR History", callback_data="revenue:mrr_history")],
        [InlineKeyboardButton(text="🔮 Revenue Forecast", callback_data="revenue:forecast")],
        [InlineKeyboardButton(text="🛤️ Runway Calculator", callback_data="revenue:runway")],
        [InlineKeyboardButton(text="📈 Unit Economics", callback_data="revenue:unit_econ")],
        [InlineKeyboardButton(text="🔄 Refresh", callback_data="revenue:refresh")]
    ])

    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)
    await callback.answer()
