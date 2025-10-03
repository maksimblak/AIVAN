"""
Admin commands для Revenue Analytics
"""

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from src.core.admin_modules.admin_formatters import growth_emoji, ltv_cac_status, quick_ratio_status
from src.core.admin_modules.admin_utils import back_keyboard, edit_or_answer, render_dashboard, require_admin
from src.core.admin_modules.revenue_analytics import RevenueAnalytics


revenue_router = Router(name="revenue_admin")


async def _build_revenue_dashboard(db) -> tuple[str, InlineKeyboardMarkup]:
    analytics = RevenueAnalytics(db)

    mrr = await analytics.get_mrr_breakdown()
    arr_metrics = await analytics.get_arr_metrics()
    unit_econ = await analytics.get_unit_economics()

    lines = [
        "💰 <b>Revenue Analytics Dashboard</b>",
        "",
        f"<b>📊 MRR ({mrr.month}):</b> {mrr.total_mrr:,}₽",
        f"  Growth: {mrr.mrr_growth_rate:+.1f}% {growth_emoji(mrr.mrr_growth_rate)}",
        f"  Net New MRR: {mrr.net_new_mrr:+,}₽",
        "",
        "<b>🔍 MRR Breakdown:</b>",
        f"  New: +{mrr.new_mrr:,}₽ ({mrr.new_customers} customers)",
        f"  Expansion: +{mrr.expansion_mrr:,}₽",
        f"  Churn: -{mrr.churn_mrr:,}₽ ({mrr.churned_customers} lost)",
        f"  Contraction: -{mrr.contraction_mrr:,}₽",
        "",
        f"<b>📈 ARR:</b> {arr_metrics.arr:,}₽",
        f"  Projected ARR (12mo): {arr_metrics.projected_arr:,}₽",
        f"  Quick Ratio: {arr_metrics.quick_ratio:.2f} {quick_ratio_status(arr_metrics.quick_ratio)}",
        "",
        f"<b>👥 Customers:</b> {mrr.total_paying_customers}",
        f"  ARPU: {mrr.arpu:,.0f}₽",
        f"  Churn Rate: {mrr.customer_churn_rate:.1f}%",
        "",
        "<b>💎 Unit Economics:</b>",
        f"  LTV: {unit_econ.ltv:,.0f}₽",
        f"  CAC: {unit_econ.cac:,.0f}₽",
        f"  LTV/CAC: {unit_econ.ltv_cac_ratio:.2f}x {ltv_cac_status(unit_econ.ltv_cac_ratio)}",
        f"  Payback: {unit_econ.payback_period:.1f} months",
        f"  Gross Margin: {unit_econ.gross_margin*100:.0f}%",
    ]

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 MRR History", callback_data="revenue:mrr_history")],
            [InlineKeyboardButton(text="🔮 Revenue Forecast", callback_data="revenue:forecast")],
            [InlineKeyboardButton(text="🛤️ Runway Calculator", callback_data="revenue:runway")],
            [InlineKeyboardButton(text="📈 Unit Economics", callback_data="revenue:unit_econ")],
            [InlineKeyboardButton(text="🔄 Refresh", callback_data="revenue:refresh")],
        ]
    )

    joiner = chr(10)
    text = joiner.join(lines)
    return text, keyboard


@revenue_router.message(Command("revenue"))
@require_admin
async def cmd_revenue(message: Message, db, admin_ids: list[int]):
    """Главное меню revenue analytics"""

    async def build_dashboard():
        return await _build_revenue_dashboard(db)

    await render_dashboard(build_dashboard, message)


@revenue_router.callback_query(F.data == "revenue:refresh")
@require_admin
async def handle_revenue_refresh(callback: CallbackQuery, db, admin_ids: list[int]):
    """Обновить revenue dashboard"""
    await callback.answer("🔄 Обновляю...")

    async def build_dashboard():
        return await _build_revenue_dashboard(db)

    await render_dashboard(build_dashboard, callback)


@revenue_router.callback_query(F.data == "revenue:mrr_history")
@require_admin
async def handle_mrr_history(callback: CallbackQuery, db, admin_ids: list[int]):
    """MRR History за последние месяцы"""
    analytics = RevenueAnalytics(db)
    history = await analytics.get_mrr_history(months=12)

    lines = ["📊 <b>MRR History (12 months)</b>", ""]

    for entry in history:
        lines.append(f"<b>{entry.month}</b>")
        lines.append(f"  MRR: {entry.total_mrr:,}₽ ({entry.mrr_growth_rate:+.1f}%)")
        lines.append(f"  New: +{entry.new_mrr:,} | Exp: +{entry.expansion_mrr:,}")
        lines.append(f"  Churn: -{entry.churn_mrr:,} | Customers: {entry.total_paying_customers}")
        lines.append("")

    if history:
        lines.append("<b>📈 MRR Trend:</b>")
        max_mrr = max(entry.total_mrr for entry in history)
        scale = max_mrr or 1
        for entry in history[-6:]:
            ratio = entry.total_mrr / scale if scale else 0
            bar_length = max(0, min(20, int(round(ratio * 20))))
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"{entry.month}: {bar} {entry.total_mrr:,}₽")

    joiner = chr(10)
    text = joiner.join(lines).rstrip()
    await edit_or_answer(callback, text, back_keyboard("revenue:back"))
    await callback.answer()


@revenue_router.callback_query(F.data == "revenue:forecast")
@require_admin
async def handle_revenue_forecast(callback: CallbackQuery, db, admin_ids: list[int]):
    """Revenue forecast"""
    analytics = RevenueAnalytics(db)
    forecasts = await analytics.get_revenue_forecast(months_ahead=6)

    if not forecasts:
        await callback.answer("❌ Недостаточно данных для прогноза", show_alert=True)
        return

    baseline = forecasts[0]
    lines = [
        "🔮 <b>Revenue Forecast (6 months)</b>",
        "",
        "<b>Assumptions:</b>",
        f"  Growth Rate: {baseline.assumed_growth_rate*100:+.1f}%/month",
        f"  Churn Rate: {baseline.assumed_churn_rate*100:.1f}%/month",
        "",
        "<b>📊 Projections:</b>",
    ]

    for fc in forecasts[:6]:
        lines.append(f"<b>{fc.month}</b> (confidence: {fc.confidence*100:.0f}%)")
        lines.append(f"  Conservative: {fc.mrr_forecast_low:,}₽")
        lines.append(f"  Expected: {fc.mrr_forecast_mid:,}₽")
        lines.append(f"  Optimistic: {fc.mrr_forecast_high:,}₽")
        lines.append("")

    recent = forecasts[:6]
    if recent:
        max_mrr = max((f.mrr_forecast_high for f in recent), default=0) or 1
        lines.append("<b>📈 Expected Trajectory:</b>")
        for fc in recent:
            ratio = fc.mrr_forecast_mid / max_mrr if max_mrr else 0
            bar_length = max(0, min(20, int(round(ratio * 20))))
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"{fc.month}: {bar} {fc.mrr_forecast_mid:,}₽")

    joiner = chr(10)
    text = joiner.join(lines).rstrip()
    await edit_or_answer(callback, text, back_keyboard("revenue:back"))
    await callback.answer()


@revenue_router.callback_query(F.data == "revenue:runway")
@require_admin
async def handle_runway_calculator(callback: CallbackQuery, db, admin_ids: list[int]):
    """Инструкция по расчету runway"""
    lines = [
        "🛤️ <b>Runway Calculator</b>",
        "",
        "Для расчета runway введите команду:",
        "",
        "<code>/runway [cash] [monthly_burn]</code>",
        "",
        "<b>Пример:</b>",
        "<code>/runway 500000 -50000</code>",
        "",
        "Где:",
        "• cash - текущий баланс в рублях",
        "• monthly_burn - ежемесячный расход (negative)",
    ]

    joiner = chr(10)
    text = joiner.join(lines)
    await edit_or_answer(callback, text, back_keyboard("revenue:back"))
    await callback.answer()


@revenue_router.message(Command("runway"))
@require_admin
async def cmd_runway(message: Message, db, admin_ids: list[int]):
    """Рассчет runway по введенным параметрам"""
    args = (message.text or "").split()[1:]

    if len(args) < 2:
        error_text = """❌ Неверный формат

Используйте: <code>/runway [cash] [monthly_burn]</code>
Пример: <code>/runway 500000 -50000</code>"""
        await message.answer(error_text, parse_mode="HTML")
        return

    try:
        current_cash = int(args[0])
        monthly_burn = int(args[1])
    except ValueError:
        await message.answer("❌ Неверные числа")
        return

    analytics = RevenueAnalytics(db)
    runway = await analytics.calculate_runway(current_cash, monthly_burn)

    lines = [
        "🛤️ <b>Runway Analysis</b>",
        "",
        f"<b>💰 Current Cash:</b> {current_cash:,}₽",
        f"<b>🔥 Monthly Burn:</b> {monthly_burn:,}₽",
        "",
        f"<b>⏱ Runway:</b> {runway['runway_months']} months",
        f"<b>📅 Cash out date:</b> {runway['runway_end_date']}",
        "",
        f"<b>💎 Current MRR:</b> {runway['current_mrr']:,}₽",
        f"<b>🎯 Breakeven MRR:</b> {runway['breakeven_mrr']:,}₽",
        f"<b>📈 MRR Growth:</b> {runway['mrr_growth_rate']:+.1f}%/month",
        "",
    ]

    if runway.get('months_to_breakeven'):
        lines.append(f"<b>⏳ Months to Breakeven:</b> {runway['months_to_breakeven']}")
        lines.append("")
        if runway['months_to_breakeven'] < runway['runway_months']:
            lines.append("✅ <b>You'll reach breakeven before running out of cash!</b>")
        else:
            lines.append("🔴 <b>Warning: You'll run out of cash before breakeven</b>")
            deficit = runway['months_to_breakeven'] - runway['runway_months']
            lines.append(f"Need {deficit} more months of runway")
    else:
        lines.append("⚠️ At current growth rate, won't reach breakeven")

    joiner = chr(10)
    await message.answer(joiner.join(lines), parse_mode="HTML")


@revenue_router.callback_query(F.data == "revenue:unit_econ")
@require_admin
async def handle_unit_economics(callback: CallbackQuery, db, admin_ids: list[int]):
    """Детальный Unit Economics"""
    analytics = RevenueAnalytics(db)
    unit_econ = await analytics.get_unit_economics()

    lines = [
        "💎 <b>Unit Economics Deep Dive</b>",
        "",
        "<b>💰 Customer Lifetime Value (LTV):</b>",
        f"  {unit_econ.ltv:,.0f}₽",
        "",
        "<b>📊 Calculation:</b>",
        f"  Monthly Churn: {unit_econ.monthly_churn*100:.2f}%",
        f"  Avg Lifetime: {unit_econ.avg_customer_lifetime_months:.1f} months",
        f"  ARPU: {unit_econ.ltv / unit_econ.avg_customer_lifetime_months:,.0f}₽/month",
        "  LTV = ARPU × Lifetime",
        "",
        "<b>💸 Customer Acquisition Cost (CAC):</b>",
        f"  {unit_econ.cac:,.0f}₽",
        "  <i>Note: Estimated based on LTV (30% ratio)</i>",
        "",
        "<b>🎯 Key Metrics:</b>",
        f"  LTV/CAC Ratio: {unit_econ.ltv_cac_ratio:.2f}x {ltv_cac_status(unit_econ.ltv_cac_ratio)}",
        f"  Payback Period: {unit_econ.payback_period:.1f} months",
        f"  Gross Margin: {unit_econ.gross_margin*100:.0f}%",
        "",
        "<b>💡 Benchmarks:</b>",
        "  LTV/CAC > 3 = ✅ Excellent",
        "  Payback < 12 months = ✅ Good",
        "  Gross Margin > 70% = ✅ Healthy",
        "",
    ]

    if unit_econ.ltv_cac_ratio < 3:
        lines.append("⚠️ <b>Action:</b> Improve retention or reduce CAC")
    elif unit_econ.payback_period > 12:
        lines.append("⚠️ <b>Action:</b> Increase ARPU or reduce CAC")
    else:
        lines.append("✅ <b>Unit economics look healthy!</b>")

    joiner = chr(10)
    text = joiner.join(lines)
    await edit_or_answer(callback, text, back_keyboard("revenue:back"))
    await callback.answer()


@revenue_router.callback_query(F.data == "revenue:back")
@require_admin
async def handle_back_to_main(callback: CallbackQuery, db, admin_ids: list[int]):
    """Вернуться в главное меню revenue"""

    async def build_dashboard():
        return await _build_revenue_dashboard(db)

    await render_dashboard(build_dashboard, callback)
    await callback.answer()
