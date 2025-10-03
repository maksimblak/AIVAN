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
        "💰 <b>Дашборд выручки</b>",
        "",
        f"<b>📊 MRR ({mrr.month}):</b> {mrr.total_mrr:,}₽",
        f"  Рост: {mrr.mrr_growth_rate:+.1f}% {growth_emoji(mrr.mrr_growth_rate)}",
        f"  Чистый новый MRR: {mrr.net_new_mrr:+,}₽",
        "",
        "<b>🔍 Структура MRR:</b>",
        f"  Новые: +{mrr.new_mrr:,}₽ ({mrr.new_customers} клиентов)",
        f"  Расширение: +{mrr.expansion_mrr:,}₽",
        f"  Отток: -{mrr.churn_mrr:,}₽ ({mrr.churned_customers} потеряно)",
        f"  Сокращение: -{mrr.contraction_mrr:,}₽",
        "",
        f"<b>📈 ARR:</b> {arr_metrics.arr:,}₽",
        f"  Прогнозный ARR (12 мес): {arr_metrics.projected_arr:,}₽",
        f"  Quick Ratio: {arr_metrics.quick_ratio:.2f} {quick_ratio_status(arr_metrics.quick_ratio)}",
        "",
        f"<b>👥 Клиенты:</b> {mrr.total_paying_customers}",
        f"  ARPU: {mrr.arpu:,.0f}₽",
        f"  Отток клиентов: {mrr.customer_churn_rate:.1f}%",
        "",
        "<b>💎 Юнит-экономика:</b>",
        f"  LTV: {unit_econ.ltv:,.0f}₽",
        f"  CAC: {unit_econ.cac:,.0f}₽",
        f"  LTV/CAC: {unit_econ.ltv_cac_ratio:.2f}x {ltv_cac_status(unit_econ.ltv_cac_ratio)}",
        f"  Окупаемость: {unit_econ.payback_period:.1f} мес.",
        f"  Валовая маржа: {unit_econ.gross_margin*100:.0f}%",
    ]

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 История MRR", callback_data="revenue:mrr_history")],
            [InlineKeyboardButton(text="🔮 Прогноз выручки", callback_data="revenue:forecast")],
            [InlineKeyboardButton(text="🛤️ Калькулятор runway", callback_data="revenue:runway")],
            [InlineKeyboardButton(text="📈 Юнит-экономика", callback_data="revenue:unit_econ")],
            [InlineKeyboardButton(text="🔄 Обновить", callback_data="revenue:refresh")],
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

    lines = ["📊 <b>История MRR (12 месяцев)</b>", ""]

    for entry in history:
        lines.append(f"<b>{entry.month}</b>")
        lines.append(f"  MRR: {entry.total_mrr:,}₽ ({entry.mrr_growth_rate:+.1f}%)")
        lines.append(f"  Новые: +{entry.new_mrr:,} | Расширение: +{entry.expansion_mrr:,}")
        lines.append(f"  Отток: -{entry.churn_mrr:,} | Клиенты: {entry.total_paying_customers}")
        lines.append("")

    if history:
        lines.append("<b>📈 Тренд MRR:</b>")
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
        "🔮 <b>Прогноз выручки (6 месяцев)</b>",
        "",
        "<b>Предпосылки:</b>",
        f"  Темп роста: {baseline.assumed_growth_rate*100:+.1f}%/month",
        f"  Отток: {baseline.assumed_churn_rate*100:.1f}%/month",
        "",
        "<b>📊 Прогнозные значения:</b>",
    ]

    for fc in forecasts[:6]:
        lines.append(f"<b>{fc.month}</b> (уверенность: {fc.confidence*100:.0f}%)")
        lines.append(f"  Консервативный: {fc.mrr_forecast_low:,}₽")
        lines.append(f"  Базовый: {fc.mrr_forecast_mid:,}₽")
        lines.append(f"  Оптимистичный: {fc.mrr_forecast_high:,}₽")
        lines.append("")

    recent = forecasts[:6]
    if recent:
        max_mrr = max((f.mrr_forecast_high for f in recent), default=0) or 1
        lines.append("<b>📈 Ожидаемая динамика:</b>")
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
        "🛤️ <b>Анализ runway</b>",
        "",
        f"<b>💰 Текущий кеш:</b> {current_cash:,}₽",
        f"<b>🔥 Месячный расход:</b> {monthly_burn:,}₽",
        "",
        f"<b>⏱ Runway:</b> {runway['runway_months']} мес.",
        f"<b>📅 Дата окончания средств:</b> {runway['runway_end_date']}",
        "",
        f"<b>💎 Текущий MRR:</b> {runway['current_mrr']:,}₽",
        f"<b>🎯 MRR точки безубыточности:</b> {runway['breakeven_mrr']:,}₽",
        f"<b>📈 Рост MRR:</b> {runway['mrr_growth_rate']:+.1f}%/мес",
        "",
    ]

    if runway.get('months_to_breakeven'):
        lines.append(f"<b>⏳ Месяцев до безубыточности:</b> {runway['months_to_breakeven']}")
        lines.append("")
        if runway['months_to_breakeven'] < runway['runway_months']:
            lines.append("✅ <b>Достигнете безубыточности раньше, чем закончится кеш!</b>")
        else:
            lines.append("🔴 <b>Внимание: средства закончатся до безубыточности</b>")
            deficit = runway['months_to_breakeven'] - runway['runway_months']
            lines.append(f"Нужно ещё {deficit} мес. runway")
    else:
        lines.append("⚠️ При текущем темпе роста до безубыточности не дойдём")

    joiner = chr(10)
    await message.answer(joiner.join(lines), parse_mode="HTML")


@revenue_router.callback_query(F.data == "revenue:unit_econ")
@require_admin
async def handle_unit_economics(callback: CallbackQuery, db, admin_ids: list[int]):
    """Детальный Unit Economics"""
    analytics = RevenueAnalytics(db)
    unit_econ = await analytics.get_unit_economics()

    lines = [
        "💎 <b>Глубокий анализ юнит-экономики</b>",
        "",
        "<b>💰 Пожизненная ценность клиента (LTV):</b>",
        f"  {unit_econ.ltv:,.0f}₽",
        "",
        "<b>📊 Расчёт:</b>",
        f"  Месячный отток: {unit_econ.monthly_churn*100:.2f}%",
        f"  Средний срок жизни: {unit_econ.avg_customer_lifetime_months:.1f} мес.",
        f"  ARPU: {unit_econ.ltv / unit_econ.avg_customer_lifetime_months:,.0f}₽/мес",
        "  LTV = ARPU × Срок жизни",
        "",
        "<b>💸 Стоимость привлечения клиента (CAC):</b>",
        f"  {unit_econ.cac:,.0f}₽",
        "  <i>Примечание: оценка по LTV (доля 30%)</i>",
        "",
        "<b>🎯 Ключевые показатели:</b>",
        f"  Соотношение LTV/CAC: {unit_econ.ltv_cac_ratio:.2f}x {ltv_cac_status(unit_econ.ltv_cac_ratio)}",
        f"  Срок окупаемости: {unit_econ.payback_period:.1f} мес.",
        f"  Валовая маржа: {unit_econ.gross_margin*100:.0f}%",
        "",
        "<b>💡 Бенчмарки:</b>",
        "  LTV/CAC > 3 = ✅ Отлично",
        "  Окупаемость < 12 мес. = ✅ Хорошо",
        "  Валовая маржа > 70% = ✅ Хорошо",
        "",
    ]

    if unit_econ.ltv_cac_ratio < 3:
        lines.append("⚠️ <b>Действие:</b> улучшить удержание или снизить CAC")
    elif unit_econ.payback_period > 12:
        lines.append("⚠️ <b>Действие:</b> увеличить ARPU или снизить CAC")
    else:
        lines.append("✅ <b>Юнит-экономика в порядке!</b>")

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
