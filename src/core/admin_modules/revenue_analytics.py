"""
💰 Revenue Analytics - MRR/ARR Tracking

Отслеживание здоровья бизнеса через revenue metrics:
- MRR (Monthly Recurring Revenue)
- ARR (Annual Recurring Revenue)
- Churn MRR
- Expansion MRR
- Net MRR Growth
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class MRRBreakdown:
    """MRR разбивка по компонентам"""

    month: str  # "2025-01"

    # Core MRR components
    new_mrr: int  # MRR от новых платящих пользователей
    expansion_mrr: int  # MRR от апгрейдов/дополнительных платежей
    contraction_mrr: int  # MRR потерянный из-за даунгрейдов
    churn_mrr: int  # MRR потерянный из-за ухода пользователей

    # Calculated
    total_mrr: int  # Текущий MRR
    net_new_mrr: int  # new + expansion - contraction - churn
    mrr_growth_rate: float  # % рост MRR

    # User counts
    new_customers: int
    churned_customers: int
    total_paying_customers: int

    # Derived metrics
    arpu: float  # Average Revenue Per User (MRR / paying customers)
    customer_churn_rate: float  # % ушедших пользователей


@dataclass
class ARRMetrics:
    """Annual Recurring Revenue метрики"""

    arr: int  # MRR × 12
    projected_arr: int  # С учетом текущих трендов

    # Quick Ratio: (New MRR + Expansion) / (Churn + Contraction)
    # >4 = отлично, 2-4 = хорошо, <1 = плохо
    quick_ratio: float

    # CAC Payback Period (months)
    # Сколько месяцев нужно чтобы окупить привлечение клиента
    cac_payback_months: float | None

    # Burn Multiple: Net Burn / Net New ARR
    # <1 = эффективный рост, >3 = расточительный
    burn_multiple: float | None


@dataclass
class RevenueForecasts:
    """Прогнозы revenue"""

    month: str

    # Conservative forecast (10th percentile)
    mrr_forecast_low: int
    # Expected forecast (50th percentile)
    mrr_forecast_mid: int
    # Optimistic forecast (90th percentile)
    mrr_forecast_high: int

    # Confidence interval
    confidence: float  # 0-1

    # Assumptions
    assumed_churn_rate: float
    assumed_growth_rate: float


@dataclass
class UnitEconomics:
    """Unit economics расчеты"""

    # Customer Acquisition Cost
    cac: float  # Средняя стоимость привлечения клиента

    # Lifetime Value
    ltv: float  # Средний LTV клиента

    # LTV/CAC ratio (target: >3)
    ltv_cac_ratio: float

    # Payback period (months)
    payback_period: float

    # Gross margin %
    gross_margin: float

    # Monthly churn rate
    monthly_churn: float

    # Customer lifetime (months)
    avg_customer_lifetime_months: float


class RevenueAnalytics:
    """
    Revenue Analytics для tracking MRR/ARR
    """

    def __init__(self, db):
        self.db = db

    async def get_mrr_breakdown(self, month: str | None = None) -> MRRBreakdown:
        """
        Получить MRR breakdown за месяц

        Args:
            month: "2025-01" или None для текущего месяца
        """
        if month is None:
            month = datetime.now().strftime("%Y-%m")

        async with self.db.pool.acquire() as conn:
            # New MRR - первые платежи в этом месяце
            cursor = await conn.execute(
                """
                SELECT COALESCE(SUM(amount), 0) as new_mrr, COUNT(DISTINCT user_id) as new_customers
                FROM payments
                WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
                  AND status = 'completed'
                  AND user_id NOT IN (
                      SELECT user_id FROM payments
                      WHERE strftime('%Y-%m', created_at, 'unixepoch') < ?
                        AND status = 'completed'
                  )
            """,
                (month, month),
            )
            row = await cursor.fetchone()
            await cursor.close()

            new_mrr = row[0] if row else 0
            new_customers = row[1] if row else 0

            # Expansion MRR - повторные платежи от существующих клиентов
            cursor = await conn.execute(
                """
                SELECT COALESCE(SUM(amount), 0) as expansion_mrr
                FROM payments
                WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
                  AND status = 'completed'
                  AND user_id IN (
                      SELECT user_id FROM payments
                      WHERE strftime('%Y-%m', created_at, 'unixepoch') < ?
                        AND status = 'completed'
                  )
            """,
                (month, month),
            )
            row = await cursor.fetchone()
            await cursor.close()

            expansion_mrr = row[0] if row else 0

            # Churn MRR - пользователи кто платил в предыдущем месяце но не в текущем
            prev_month = self._get_previous_month(month)

            cursor = await conn.execute(
                """
                SELECT COALESCE(SUM(prev_amount), 0) as churn_mrr, COUNT(*) as churned
                FROM (
                    SELECT DISTINCT p1.user_id, p1.amount as prev_amount
                    FROM payments p1
                    WHERE strftime('%Y-%m', p1.created_at, 'unixepoch') = ?
                      AND p1.status = 'completed'
                      AND p1.user_id NOT IN (
                          SELECT user_id FROM payments
                          WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
                            AND status = 'completed'
                      )
                )
            """,
                (prev_month, month),
            )
            row = await cursor.fetchone()
            await cursor.close()

            churn_mrr = row[0] if row else 0
            churned_customers = row[1] if row else 0

            # Contraction MRR (если есть разные тарифы - пока 0)
            contraction_mrr = 0

            # Total MRR - все активные подписки в этом месяце
            cursor = await conn.execute(
                """
                SELECT COALESCE(SUM(amount), 0) as total, COUNT(DISTINCT user_id) as customers
                FROM payments
                WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
                  AND status = 'completed'
            """,
                (month,),
            )
            row = await cursor.fetchone()
            await cursor.close()

            total_mrr = row[0] if row else 0
            total_paying = row[1] if row else 0

            # Net New MRR
            net_new_mrr = new_mrr + expansion_mrr - contraction_mrr - churn_mrr

            # MRR Growth Rate
            cursor = await conn.execute(
                """
                SELECT COALESCE(SUM(amount), 0) as prev_mrr
                FROM payments
                WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
                  AND status = 'completed'
            """,
                (prev_month,),
            )
            row = await cursor.fetchone()
            await cursor.close()

            prev_mrr = row[0] if row else 0
            mrr_growth_rate = ((total_mrr - prev_mrr) / prev_mrr * 100) if prev_mrr > 0 else 0

            # ARPU
            arpu = total_mrr / total_paying if total_paying > 0 else 0

            # Customer churn rate
            cursor = await conn.execute(
                """
                SELECT COUNT(DISTINCT user_id) as prev_customers
                FROM payments
                WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
                  AND status = 'completed'
            """,
                (prev_month,),
            )
            row = await cursor.fetchone()
            await cursor.close()

            prev_customers = row[0] if row else 0
            customer_churn_rate = (
                (churned_customers / prev_customers * 100) if prev_customers > 0 else 0
            )

            return MRRBreakdown(
                month=month,
                new_mrr=new_mrr,
                expansion_mrr=expansion_mrr,
                contraction_mrr=contraction_mrr,
                churn_mrr=churn_mrr,
                total_mrr=total_mrr,
                net_new_mrr=net_new_mrr,
                mrr_growth_rate=mrr_growth_rate,
                new_customers=new_customers,
                churned_customers=churned_customers,
                total_paying_customers=total_paying,
                arpu=arpu,
                customer_churn_rate=customer_churn_rate,
            )

    def _get_previous_month(self, month: str) -> str:
        """Получить предыдущий месяц в формате YYYY-MM"""
        year, mon = map(int, month.split("-"))
        if mon == 1:
            return f"{year-1}-12"
        else:
            return f"{year}-{mon-1:02d}"

    async def get_arr_metrics(self) -> ARRMetrics:
        """Получить ARR метрики"""
        # Текущий MRR
        current_month = datetime.now().strftime("%Y-%m")
        mrr = await self.get_mrr_breakdown(current_month)

        arr = mrr.total_mrr * 12

        # Projected ARR (с учетом текущего роста)
        # ARR × (1 + monthly_growth_rate)^12
        monthly_growth = mrr.mrr_growth_rate / 100
        projected_arr = int(arr * ((1 + monthly_growth) ** 12))

        # Quick Ratio
        new_expansion = mrr.new_mrr + mrr.expansion_mrr
        churn_contraction = mrr.churn_mrr + mrr.contraction_mrr
        # Используем 999.0 вместо inf для корректного отображения и сравнений
        quick_ratio = (new_expansion / churn_contraction) if churn_contraction > 0 else 999.0

        # CAC Payback (нужны данные о marketing spend - пока None)
        cac_payback_months = None

        # Burn Multiple (нужны данные о burn rate - пока None)
        burn_multiple = None

        return ARRMetrics(
            arr=arr,
            projected_arr=projected_arr,
            quick_ratio=quick_ratio,
            cac_payback_months=cac_payback_months,
            burn_multiple=burn_multiple,
        )

    async def get_revenue_forecast(self, months_ahead: int = 12) -> list[RevenueForecasts]:
        """
        Прогноз revenue на N месяцев вперед

        Использует historical data для экстраполяции
        """
        # Получить historical MRR за последние 6 месяцев
        historical = []
        current = datetime.now()

        for i in range(6, 0, -1):
            month_dt = current - timedelta(days=30 * i)
            month_str = month_dt.strftime("%Y-%m")
            try:
                mrr = await self.get_mrr_breakdown(month_str)
                historical.append(mrr)
            except Exception:
                pass

        if len(historical) < 2:
            # Недостаточно данных для прогноза
            return []

        # Средний growth rate
        growth_rates = [h.mrr_growth_rate / 100 for h in historical if h.mrr_growth_rate != 0]
        avg_growth_rate = sum(growth_rates) / len(growth_rates) if growth_rates else 0

        # Средний churn rate
        churn_rates = [h.customer_churn_rate / 100 for h in historical]
        avg_churn_rate = sum(churn_rates) / len(churn_rates) if churn_rates else 0

        # Текущий MRR
        latest = historical[-1]
        base_mrr = latest.total_mrr

        # Генерация прогнозов
        forecasts = []

        for month_offset in range(1, months_ahead + 1):
            future_month = (current + timedelta(days=30 * month_offset)).strftime("%Y-%m")

            # Conservative: growth_rate - 1 std dev
            conservative_growth = avg_growth_rate * 0.7
            mrr_low = int(base_mrr * ((1 + conservative_growth) ** month_offset))

            # Expected: avg growth rate
            mrr_mid = int(base_mrr * ((1 + avg_growth_rate) ** month_offset))

            # Optimistic: growth_rate + 1 std dev
            optimistic_growth = avg_growth_rate * 1.3
            mrr_high = int(base_mrr * ((1 + optimistic_growth) ** month_offset))

            # Confidence уменьшается с горизонтом прогноза
            confidence = max(0.3, 1.0 - (month_offset / months_ahead) * 0.7)

            forecasts.append(
                RevenueForecasts(
                    month=future_month,
                    mrr_forecast_low=mrr_low,
                    mrr_forecast_mid=mrr_mid,
                    mrr_forecast_high=mrr_high,
                    confidence=confidence,
                    assumed_churn_rate=avg_churn_rate,
                    assumed_growth_rate=avg_growth_rate,
                )
            )

        return forecasts

    async def get_unit_economics(self) -> UnitEconomics:
        """
        Рассчитать Unit Economics

        Note: CAC требует данных о marketing spend (пока используем placeholder)
        """
        async with self.db.pool.acquire() as conn:
            # Average monthly churn rate (за последние 6 месяцев)
            cursor = await conn.execute(
                """
                SELECT AVG(monthly_churn) as avg_churn
                FROM (
                    SELECT
                        strftime('%Y-%m', created_at, 'unixepoch') as month,
                        COUNT(DISTINCT user_id) as total_customers,
                        (
                            SELECT COUNT(DISTINCT p1.user_id)
                            FROM payments p1
                            WHERE strftime('%Y-%m', p1.created_at, 'unixepoch') =
                                  strftime('%Y-%m', datetime(payments.created_at, 'unixepoch', '-1 month'))
                              AND p1.status = 'completed'
                              AND p1.user_id NOT IN (
                                  SELECT user_id FROM payments
                                  WHERE strftime('%Y-%m', created_at, 'unixepoch') = month
                                    AND status = 'completed'
                              )
                        ) * 100.0 / total_customers as monthly_churn
                    FROM payments
                    WHERE created_at > strftime('%s', 'now', '-180 day')
                      AND status = 'completed'
                    GROUP BY month
                )
            """
            )
            row = await cursor.fetchone()
            await cursor.close()

            monthly_churn = (row[0] / 100) if row and row[0] else 0.05  # Default 5%

            # Customer lifetime = 1 / monthly_churn_rate
            avg_lifetime_months = 1 / monthly_churn if monthly_churn > 0 else 20

            # ARPU
            current_month = datetime.now().strftime("%Y-%m")
            mrr_data = await self.get_mrr_breakdown(current_month)
            arpu = mrr_data.arpu

            # LTV = ARPU × Customer Lifetime
            ltv = arpu * avg_lifetime_months

            # CAC - placeholder (нужны данные о marketing spend)
            # Допустим средний CAC = 30% от LTV (industry benchmark)
            cac = ltv * 0.3

            # LTV/CAC ratio
            ltv_cac_ratio = ltv / cac if cac > 0 else 0

            # Payback period = CAC / ARPU (months)
            payback_period = cac / arpu if arpu > 0 else 0

            # Gross Margin (для SaaS обычно ~80%)
            # Для AI бота: учитываем OpenAI API costs
            # Допустим OpenAI costs = 20% от revenue → gross margin = 80%
            gross_margin = 0.80

            return UnitEconomics(
                cac=cac,
                ltv=ltv,
                ltv_cac_ratio=ltv_cac_ratio,
                payback_period=payback_period,
                gross_margin=gross_margin,
                monthly_churn=monthly_churn,
                avg_customer_lifetime_months=avg_lifetime_months,
            )

    async def get_mrr_history(self, months: int = 12) -> list[MRRBreakdown]:
        """Получить historical MRR за последние N месяцев"""
        history = []
        current = datetime.now()

        for i in range(months, 0, -1):
            month_dt = current - timedelta(days=30 * i)
            month_str = month_dt.strftime("%Y-%m")

            try:
                mrr = await self.get_mrr_breakdown(month_str)
                history.append(mrr)
            except Exception:
                pass

        return history

    async def calculate_runway(self, current_cash: int, monthly_burn: int) -> dict[str, Any]:
        """
        Рассчитать runway (сколько месяцев до нулевого баланса)

        Args:
            current_cash: Текущий cash в рублях
            monthly_burn: Ежемесячный расход (negative number)

        Returns:
            {
                'runway_months': int,
                'runway_end_date': str,
                'breakeven_mrr': int,  # MRR нужный для breakeven
                'months_to_breakeven': int | None
            }
        """
        # Runway = cash / abs(monthly_burn)
        runway_months = current_cash / abs(monthly_burn) if monthly_burn < 0 else float("inf")

        end_date = datetime.now() + timedelta(days=30 * runway_months)

        # Breakeven MRR = monthly costs (assuming monthly_burn includes costs)
        breakeven_mrr = abs(monthly_burn)

        # Текущий MRR
        current_month = datetime.now().strftime("%Y-%m")
        mrr_data = await self.get_mrr_breakdown(current_month)
        current_mrr = mrr_data.total_mrr

        # Months to breakeven (с учетом текущего роста)
        if mrr_data.mrr_growth_rate > 0:
            # MRR растет - можем рассчитать когда достигнем breakeven
            monthly_growth = mrr_data.mrr_growth_rate / 100

            # current_mrr × (1 + growth)^N = breakeven_mrr
            # N = log(breakeven_mrr / current_mrr) / log(1 + growth)
            if current_mrr > 0 and current_mrr < breakeven_mrr:
                import math

                months_to_breakeven = math.log(breakeven_mrr / current_mrr) / math.log(
                    1 + monthly_growth
                )
                months_to_breakeven = int(months_to_breakeven)
            else:
                months_to_breakeven = None  # Уже breakeven или не растет
        else:
            months_to_breakeven = None

        return {
            "runway_months": int(runway_months),
            "runway_end_date": end_date.strftime("%Y-%m-%d"),
            "breakeven_mrr": breakeven_mrr,
            "months_to_breakeven": months_to_breakeven,
            "current_mrr": current_mrr,
            "mrr_growth_rate": mrr_data.mrr_growth_rate,
        }
