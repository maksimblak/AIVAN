"""
🎯 Cohort Analysis - Retention Dynamics

Анализ retention по когортам регистрации для понимания:
- Как меняется retention со временем
- Какие когорты самые sticky
- Влияние изменений продукта на retention
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import json


@dataclass
class CohortMetrics:
    """Метрики для одной когорты"""
    cohort_month: str  # "2025-01"
    cohort_size: int  # Количество пользователей в когорте

    # Retention rates по дням
    day_1_retention: float  # % вернувшихся на день 1
    day_7_retention: float
    day_30_retention: float
    day_90_retention: float

    # Revenue metrics
    paid_users: int  # Сколько заплатили
    conversion_rate: float  # % конвертировавшихся в paid
    total_revenue: int  # Общий revenue от когорты
    arpu: float  # Average Revenue Per User

    # Feature adoption
    avg_features_used: float  # Среднее количество использованных фич
    top_features: list[tuple[str, float]]  # [(feature, adoption_rate%)]

    # Engagement
    avg_requests_per_user: float
    avg_lifetime_days: float
    power_users_count: int  # С power_user_score > 70

    # Churn
    churned_count: int
    churn_rate: float  # %
    avg_days_to_churn: float | None


@dataclass
class CohortComparison:
    """Сравнение когорт между собой"""
    best_cohort: str  # Какая когорта лучшая по retention
    worst_cohort: str

    retention_trend: str  # "improving", "stable", "declining"
    conversion_trend: str

    key_insights: list[str]  # Список текстовых инсайтов

    # Data for visualization
    cohorts_data: list[CohortMetrics]


@dataclass
class FeatureAdoptionCohort:
    """Adoption конкретной фичи по когортам"""
    feature_name: str

    # Adoption timeline для каждой когорты
    cohort_adoption: dict[str, float]  # {cohort_month: adoption_rate%}

    # Time to adoption
    avg_days_to_first_use: dict[str, float]  # {cohort_month: days}

    # Retention correlation
    # Пользователи использовавшие эту фичу vs не использовавшие
    users_with_feature_retention: float
    users_without_feature_retention: float
    retention_lift: float  # % улучшение retention


class CohortAnalytics:
    """
    Cohort Analysis для понимания retention dynamics
    """

    def __init__(self, db):
        self.db = db

    async def get_cohort_metrics(self, cohort_month: str) -> CohortMetrics:
        """
        Получить метрики для конкретной когорты

        Args:
            cohort_month: "2025-01" формат
        """
        async with self.db.pool.acquire() as conn:
            # Cohort size
            cursor = await conn.execute("""
                SELECT COUNT(*) as size
                FROM users
                WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
            """, (cohort_month,))
            row = await cursor.fetchone()
            await cursor.close()
            cohort_size = row[0] if row else 0

            if cohort_size == 0:
                raise ValueError(f"Cohort {cohort_month} не найдена")

            # Retention rates
            day_1 = await self._calculate_retention(conn, cohort_month, 1)
            day_7 = await self._calculate_retention(conn, cohort_month, 7)
            day_30 = await self._calculate_retention(conn, cohort_month, 30)
            day_90 = await self._calculate_retention(conn, cohort_month, 90)

            # Revenue metrics
            cursor = await conn.execute("""
                SELECT
                    COUNT(DISTINCT user_id) as paid_users,
                    COALESCE(SUM(amount), 0) as total_revenue
                FROM payments
                WHERE user_id IN (
                    SELECT user_id FROM users
                    WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
                )
                AND status = 'completed'
            """, (cohort_month,))
            row = await cursor.fetchone()
            await cursor.close()

            paid_users = row[0] if row and row[0] else 0
            total_revenue = row[1] if row and row[1] else 0
            conversion_rate = (paid_users / cohort_size * 100) if cohort_size > 0 else 0
            arpu = total_revenue / cohort_size if cohort_size > 0 else 0

            # Feature adoption
            top_features = await self._get_cohort_feature_adoption(conn, cohort_month)

            cursor = await conn.execute("""
                SELECT AVG(feature_count) as avg_features
                FROM (
                    SELECT user_id, COUNT(DISTINCT feature) as feature_count
                    FROM behavior_events
                    WHERE user_id IN (
                        SELECT user_id FROM users
                        WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
                    )
                    GROUP BY user_id
                )
            """, (cohort_month,))
            row = await cursor.fetchone()
            await cursor.close()
            avg_features_used = row[0] if row and row[0] else 0

            # Engagement
            cursor = await conn.execute("""
                SELECT
                    AVG(request_count) as avg_requests,
                    AVG(lifetime_days) as avg_lifetime
                FROM (
                    SELECT
                        u.user_id,
                        u.total_requests as request_count,
                        (strftime('%s', 'now') - u.created_at) / 86400.0 as lifetime_days
                    FROM users u
                    WHERE strftime('%Y-%m', u.created_at, 'unixepoch') = ?
                )
            """, (cohort_month,))
            row = await cursor.fetchone()
            await cursor.close()

            avg_requests = row[0] if row and row[0] else 0
            avg_lifetime = row[1] if row and row[1] else 0

            # Power users (approximation - users with high activity)
            cursor = await conn.execute("""
                SELECT COUNT(*) as power_users
                FROM users
                WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
                  AND total_requests > 50
                  AND (
                      SELECT COUNT(*) FROM payments
                      WHERE payments.user_id = users.user_id
                        AND status = 'completed'
                  ) >= 2
            """, (cohort_month,))
            row = await cursor.fetchone()
            await cursor.close()
            power_users_count = row[0] if row else 0

            # Churn metrics
            cursor = await conn.execute("""
                SELECT
                    COUNT(*) as churned,
                    AVG(days_to_churn) as avg_days_to_churn
                FROM (
                    SELECT
                        u.user_id,
                        (u.last_active - u.created_at) / 86400.0 as days_to_churn
                    FROM users u
                    WHERE strftime('%Y-%m', u.created_at, 'unixepoch') = ?
                      AND (strftime('%s', 'now') - u.last_active) > 2592000
                      AND (
                          SELECT COUNT(*) FROM payments
                          WHERE payments.user_id = u.user_id
                            AND status = 'completed'
                      ) = 1
                )
            """, (cohort_month,))
            row = await cursor.fetchone()
            await cursor.close()

            churned_count = row[0] if row and row[0] else 0
            avg_days_to_churn = row[1] if row and row[1] else None
            churn_rate = (churned_count / paid_users * 100) if paid_users > 0 else 0

            return CohortMetrics(
                cohort_month=cohort_month,
                cohort_size=cohort_size,
                day_1_retention=day_1,
                day_7_retention=day_7,
                day_30_retention=day_30,
                day_90_retention=day_90,
                paid_users=paid_users,
                conversion_rate=conversion_rate,
                total_revenue=total_revenue,
                arpu=arpu,
                avg_features_used=avg_features_used,
                top_features=top_features,
                avg_requests_per_user=avg_requests,
                avg_lifetime_days=avg_lifetime,
                power_users_count=power_users_count,
                churned_count=churned_count,
                churn_rate=churn_rate,
                avg_days_to_churn=avg_days_to_churn
            )

    async def _calculate_retention(self, conn, cohort_month: str, day_offset: int) -> float:
        """
        Рассчитать retention rate для когорты на day_offset

        Retention = % пользователей активных на день N после регистрации
        """
        cursor = await conn.execute("""
            WITH cohort_users AS (
                SELECT user_id, created_at
                FROM users
                WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
            )
            SELECT
                COUNT(DISTINCT cu.user_id) as total_users,
                COUNT(DISTINCT CASE
                    WHEN EXISTS (
                        SELECT 1 FROM behavior_events be
                        WHERE be.user_id = cu.user_id
                          AND be.timestamp >= cu.created_at + (? * 86400)
                          AND be.timestamp < cu.created_at + ((? + 1) * 86400)
                    ) THEN cu.user_id
                END) as retained_users
            FROM cohort_users cu
        """, (cohort_month, day_offset, day_offset))

        row = await cursor.fetchone()
        await cursor.close()

        if not row or row[0] == 0:
            return 0.0

        total, retained = row[0], row[1]
        return (retained / total * 100) if total > 0 else 0.0

    async def _get_cohort_feature_adoption(self, conn, cohort_month: str) -> list[tuple[str, float]]:
        """Топ фичи и их adoption rate в когорте"""
        cursor = await conn.execute("""
            WITH cohort_size AS (
                SELECT COUNT(*) as size
                FROM users
                WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
            )
            SELECT
                be.feature,
                COUNT(DISTINCT be.user_id) * 100.0 / cohort_size.size as adoption_rate
            FROM behavior_events be
            CROSS JOIN cohort_size
            WHERE be.user_id IN (
                SELECT user_id FROM users
                WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
            )
            GROUP BY be.feature, cohort_size.size
            ORDER BY adoption_rate DESC
            LIMIT 5
        """, (cohort_month, cohort_month))

        rows = await cursor.fetchall()
        await cursor.close()

        return [(row[0], row[1]) for row in rows]

    async def compare_cohorts(self, months_back: int = 6) -> CohortComparison:
        """
        Сравнить последние N месяцев когорт

        Args:
            months_back: Сколько месяцев назад анализировать
        """
        # Получить список когорт
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT DISTINCT strftime('%Y-%m', created_at, 'unixepoch') as cohort_month
                FROM users
                WHERE created_at > strftime('%s', 'now', ? || ' month')
                ORDER BY cohort_month DESC
            """, (f'-{months_back}',))
            rows = await cursor.fetchall()
            await cursor.close()

        cohort_months = [row[0] for row in rows if row[0]]

        if len(cohort_months) < 2:
            return CohortComparison(
                best_cohort="N/A",
                worst_cohort="N/A",
                retention_trend="insufficient_data",
                conversion_trend="insufficient_data",
                key_insights=["Недостаточно данных для сравнения когорт"],
                cohorts_data=[]
            )

        # Получить метрики для всех когорт
        cohorts_data = []
        for month in cohort_months:
            try:
                metrics = await self.get_cohort_metrics(month)
                cohorts_data.append(metrics)
            except ValueError:
                continue

        if not cohorts_data:
            return CohortComparison(
                best_cohort="N/A",
                worst_cohort="N/A",
                retention_trend="no_data",
                conversion_trend="no_data",
                key_insights=["Нет данных о когортах"],
                cohorts_data=[]
            )

        # Найти лучшую/худшую когорту по day_30_retention
        best = max(cohorts_data, key=lambda c: c.day_30_retention)
        worst = min(cohorts_data, key=lambda c: c.day_30_retention)

        # Определить тренды
        retention_trend = self._calculate_trend([c.day_30_retention for c in cohorts_data])
        conversion_trend = self._calculate_trend([c.conversion_rate for c in cohorts_data])

        # Генерация инсайтов
        insights = self._generate_insights(cohorts_data, best, worst)

        return CohortComparison(
            best_cohort=best.cohort_month,
            worst_cohort=worst.cohort_month,
            retention_trend=retention_trend,
            conversion_trend=conversion_trend,
            key_insights=insights,
            cohorts_data=cohorts_data
        )

    def _calculate_trend(self, values: list[float]) -> str:
        """Определить тренд по списку значений (от старых к новым)"""
        if len(values) < 3:
            return "insufficient_data"

        # Linear regression slope approximation
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Classify trend
        if slope > 2:
            return "improving"
        elif slope < -2:
            return "declining"
        else:
            return "stable"

    def _generate_insights(self, cohorts: list[CohortMetrics], best: CohortMetrics, worst: CohortMetrics) -> list[str]:
        """Генерация текстовых инсайтов"""
        insights = []

        # 1. Best vs Worst cohort
        retention_diff = best.day_30_retention - worst.day_30_retention
        insights.append(
            f"📊 Лучшая когорта ({best.cohort_month}) имеет retention на {retention_diff:.1f}% "
            f"выше чем худшая ({worst.cohort_month})"
        )

        # 2. Conversion trend
        avg_conversion = sum(c.conversion_rate for c in cohorts) / len(cohorts)
        latest_conversion = cohorts[0].conversion_rate if cohorts else 0

        if latest_conversion > avg_conversion * 1.1:
            insights.append(f"✅ Последняя когорта конвертится лучше среднего ({latest_conversion:.1f}% vs {avg_conversion:.1f}%)")
        elif latest_conversion < avg_conversion * 0.9:
            insights.append(f"⚠️ Последняя когорта конвертится хуже среднего ({latest_conversion:.1f}% vs {avg_conversion:.1f}%)")

        # 3. Feature adoption comparison
        best_features = set(f[0] for f in best.top_features[:3])
        worst_features = set(f[0] for f in worst.top_features[:3])

        unique_to_best = best_features - worst_features
        if unique_to_best:
            insights.append(f"💎 Лучшая когорта активно использует: {', '.join(unique_to_best)}")

        # 4. Power users
        avg_power_users = sum(c.power_users_count for c in cohorts) / len(cohorts)
        if best.power_users_count > avg_power_users * 1.5:
            insights.append(f"🌟 Лучшая когорта производит в 1.5x больше power users")

        # 5. Churn warning
        high_churn_cohorts = [c for c in cohorts if c.churn_rate > 50]
        if high_churn_cohorts:
            insights.append(f"🔴 {len(high_churn_cohorts)} когорт с churn rate > 50%")

        return insights

    async def get_feature_adoption_by_cohort(self, feature_name: str, months_back: int = 6) -> FeatureAdoptionCohort:
        """
        Анализ adoption конкретной фичи по когортам

        Показывает:
        - Как быстро когорты находят эту фичу
        - Влияет ли эта фича на retention
        """
        async with self.db.pool.acquire() as conn:
            # Получить когорты
            cursor = await conn.execute("""
                SELECT DISTINCT strftime('%Y-%m', created_at, 'unixepoch') as cohort_month
                FROM users
                WHERE created_at > strftime('%s', 'now', ? || ' month')
                ORDER BY cohort_month DESC
            """, (f'-{months_back}',))
            rows = await cursor.fetchall()
            await cursor.close()

            cohort_months = [row[0] for row in rows if row[0]]

            cohort_adoption = {}
            avg_days_to_first_use = {}

            for cohort_month in cohort_months:
                # Adoption rate
                cursor = await conn.execute("""
                    WITH cohort_users AS (
                        SELECT user_id FROM users
                        WHERE strftime('%Y-%m', created_at, 'unixepoch') = ?
                    )
                    SELECT
                        (SELECT COUNT(*) FROM cohort_users) as total,
                        COUNT(DISTINCT be.user_id) as adopted
                    FROM behavior_events be
                    WHERE be.feature = ?
                      AND be.user_id IN (SELECT user_id FROM cohort_users)
                """, (cohort_month, feature_name))
                row = await cursor.fetchone()
                await cursor.close()

                if row and row[0] > 0:
                    adoption_rate = (row[1] / row[0] * 100) if row[1] else 0
                    cohort_adoption[cohort_month] = adoption_rate

                # Time to first use
                cursor = await conn.execute("""
                    SELECT AVG(time_to_first_use) as avg_days
                    FROM (
                        SELECT
                            (MIN(be.timestamp) - u.created_at) / 86400.0 as time_to_first_use
                        FROM users u
                        INNER JOIN behavior_events be ON u.user_id = be.user_id
                        WHERE strftime('%Y-%m', u.created_at, 'unixepoch') = ?
                          AND be.feature = ?
                        GROUP BY u.user_id
                    )
                """, (cohort_month, feature_name))
                row = await cursor.fetchone()
                await cursor.close()

                if row and row[0] is not None:
                    avg_days_to_first_use[cohort_month] = row[0]

            # Retention correlation
            cursor = await conn.execute("""
                WITH feature_users AS (
                    SELECT DISTINCT user_id
                    FROM behavior_events
                    WHERE feature = ?
                      AND user_id IN (
                          SELECT user_id FROM users
                          WHERE created_at > strftime('%s', 'now', '-90 day')
                      )
                ),
                retention_with AS (
                    SELECT COUNT(*) as retained
                    FROM users u
                    WHERE u.user_id IN (SELECT user_id FROM feature_users)
                      AND (strftime('%s', 'now') - u.last_active) < 2592000
                ),
                retention_without AS (
                    SELECT COUNT(*) as retained
                    FROM users u
                    WHERE u.user_id NOT IN (SELECT user_id FROM feature_users)
                      AND u.created_at > strftime('%s', 'now', '-90 day')
                      AND (strftime('%s', 'now') - u.last_active) < 2592000
                ),
                total_with AS (
                    SELECT COUNT(*) as total FROM feature_users
                ),
                total_without AS (
                    SELECT COUNT(*) as total
                    FROM users
                    WHERE user_id NOT IN (SELECT user_id FROM feature_users)
                      AND created_at > strftime('%s', 'now', '-90 day')
                )
                SELECT
                    CAST(rw.retained AS REAL) / NULLIF(tw.total, 0) * 100 as retention_with,
                    CAST(rwo.retained AS REAL) / NULLIF(two.total, 0) * 100 as retention_without
                FROM retention_with rw, retention_without rwo, total_with tw, total_without two
            """, (feature_name,))
            row = await cursor.fetchone()
            await cursor.close()

            retention_with = row[0] if row and row[0] else 0
            retention_without = row[1] if row and row[1] else 0
            retention_lift = retention_with - retention_without

            return FeatureAdoptionCohort(
                feature_name=feature_name,
                cohort_adoption=cohort_adoption,
                avg_days_to_first_use=avg_days_to_first_use,
                users_with_feature_retention=retention_with,
                users_without_feature_retention=retention_without,
                retention_lift=retention_lift
            )
