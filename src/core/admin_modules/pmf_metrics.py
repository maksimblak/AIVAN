"""
📊 Product-Market Fit (PMF) Metrics

Измеряет насколько продукт соответствует рынку через:
1. NPS (Net Promoter Score) - готовность рекомендовать
2. Sean Ellis Test - "very disappointed" метрика
3. Feature PMF Score - PMF для каждой фичи
4. Usage Intensity - как часто используют
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
import json


@dataclass
class NPSResult:
    """Результаты NPS опроса"""
    total_responses: int
    promoters: int  # Score 9-10
    passives: int  # Score 7-8
    detractors: int  # Score 0-6

    nps_score: float  # -100 to +100
    promoter_rate: float  # %
    detractor_rate: float  # %

    average_score: float  # 0-10
    response_rate: float  # % ответивших на опрос

    # Breakdown по сегментам
    nps_by_segment: dict[str, float]  # {segment_name: nps_score}

    # Динамика
    trend: str  # "improving", "stable", "declining"
    previous_nps: float | None


@dataclass
class SeanEllisResult:
    """Результаты Sean Ellis Test"""
    total_responses: int

    very_disappointed: int  # Ответ "Очень расстроюсь"
    somewhat_disappointed: int
    not_disappointed: int

    pmf_score: float  # % very_disappointed (>40% = PMF achieved)
    pmf_achieved: bool

    # По сегментам
    pmf_by_segment: dict[str, float]


@dataclass
class FeaturePMF:
    """PMF для конкретной фичи"""
    feature_name: str

    # Usage metrics
    total_users: int
    active_users: int  # Используют регулярно
    usage_frequency: float  # Среднее использований/user/week

    # Satisfaction
    satisfaction_score: float  # 0-100
    nps_for_feature: float | None  # Если есть опросы для этой фичи

    # PMF Score: satisfaction × usage_frequency × retention_lift
    pmf_score: float  # 0-100
    pmf_rating: str  # "strong", "moderate", "weak", "kill"

    # Insights
    key_insight: str  # Текстовое объяснение


@dataclass
class UsageIntensity:
    """Интенсивность использования продукта"""
    dau: int  # Daily Active Users
    wau: int  # Weekly Active Users
    mau: int  # Monthly Active Users

    # Stickiness ratios
    dau_mau_ratio: float  # DAU/MAU (>20% = good)
    dau_wau_ratio: float  # DAU/WAU (>40% = good)

    # Power User Curve
    power_user_percentage: float  # % users с >50 requests
    casual_user_percentage: float  # % users с <10 requests

    # L28 retention (% пользователей активных 14+ дней из последних 28)
    l28_retention: float


class PMFMetrics:
    """
    Product-Market Fit Metrics
    """

    def __init__(self, db):
        self.db = db

    async def get_nps(self, days: int = 30) -> NPSResult:
        """
        Получить NPS за последние N дней

        NPS = % Promoters - % Detractors
        """
        async with self.db.pool.acquire() as conn:
            # Получить все ответы на NPS опрос
            cursor = await conn.execute("""
                SELECT score, user_segment
                FROM nps_surveys
                WHERE created_at > strftime('%s', 'now', ? || ' days')
                  AND score IS NOT NULL
            """, (f'-{days}',))
            rows = await cursor.fetchall()
            await cursor.close()

            if not rows:
                return NPSResult(
                    total_responses=0,
                    promoters=0,
                    passives=0,
                    detractors=0,
                    nps_score=0,
                    promoter_rate=0,
                    detractor_rate=0,
                    average_score=0,
                    response_rate=0,
                    nps_by_segment={},
                    trend="insufficient_data",
                    previous_nps=None
                )

            # Классификация ответов
            promoters = sum(1 for row in rows if row[0] >= 9)
            passives = sum(1 for row in rows if 7 <= row[0] <= 8)
            detractors = sum(1 for row in rows if row[0] <= 6)

            total = len(rows)
            promoter_rate = (promoters / total * 100) if total > 0 else 0
            detractor_rate = (detractors / total * 100) if total > 0 else 0

            nps_score = promoter_rate - detractor_rate

            average_score = sum(row[0] for row in rows) / total if total > 0 else 0

            # Response rate (из тех кому отправили опрос)
            cursor = await conn.execute("""
                SELECT COUNT(*) as sent
                FROM nps_surveys
                WHERE created_at > strftime('%s', 'now', ? || ' days')
            """, (f'-{days}',))
            row = await cursor.fetchone()
            await cursor.close()

            sent = row[0] if row else 0
            response_rate = (total / sent * 100) if sent > 0 else 0

            # NPS по сегментам
            segments = {}
            for segment in ["power_users", "at_risk", "trial_converters", "new_users"]:
                segment_rows = [r for r in rows if r[1] == segment]
                if segment_rows:
                    seg_promoters = sum(1 for r in segment_rows if r[0] >= 9)
                    seg_detractors = sum(1 for r in segment_rows if r[0] <= 6)
                    seg_total = len(segment_rows)
                    seg_nps = ((seg_promoters / seg_total) - (seg_detractors / seg_total)) * 100
                    segments[segment] = seg_nps

            # Тренд (сравнение с предыдущим периодом)
            cursor = await conn.execute("""
                SELECT score
                FROM nps_surveys
                WHERE created_at BETWEEN strftime('%s', 'now', ? || ' days')
                                    AND strftime('%s', 'now', ? || ' days')
                  AND score IS NOT NULL
            """, (f'-{days*2}', f'-{days}'))
            prev_rows = await cursor.fetchall()
            await cursor.close()

            previous_nps = None
            trend = "stable"

            if prev_rows:
                prev_promoters = sum(1 for r in prev_rows if r[0] >= 9)
                prev_detractors = sum(1 for r in prev_rows if r[0] <= 6)
                prev_total = len(prev_rows)
                previous_nps = ((prev_promoters / prev_total) - (prev_detractors / prev_total)) * 100

                if nps_score > previous_nps + 5:
                    trend = "improving"
                elif nps_score < previous_nps - 5:
                    trend = "declining"

            return NPSResult(
                total_responses=total,
                promoters=promoters,
                passives=passives,
                detractors=detractors,
                nps_score=nps_score,
                promoter_rate=promoter_rate,
                detractor_rate=detractor_rate,
                average_score=average_score,
                response_rate=response_rate,
                nps_by_segment=segments,
                trend=trend,
                previous_nps=previous_nps
            )

    async def get_sean_ellis_score(self, days: int = 30) -> SeanEllisResult:
        """
        Sean Ellis Test: "Как вы будете себя чувствовать если продукт исчезнет?"

        >40% "Very disappointed" = PMF achieved
        """
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT
                    disappointment_level,
                    user_segment
                FROM nps_surveys
                WHERE created_at > strftime('%s', 'now', ? || ' days')
                  AND disappointment_level IS NOT NULL
            """, (f'-{days}',))
            rows = await cursor.fetchall()
            await cursor.close()

            if not rows:
                return SeanEllisResult(
                    total_responses=0,
                    very_disappointed=0,
                    somewhat_disappointed=0,
                    not_disappointed=0,
                    pmf_score=0,
                    pmf_achieved=False,
                    pmf_by_segment={}
                )

            very = sum(1 for r in rows if r[0] == "very_disappointed")
            somewhat = sum(1 for r in rows if r[0] == "somewhat_disappointed")
            not_disappointed = sum(1 for r in rows if r[0] == "not_disappointed")

            total = len(rows)
            pmf_score = (very / total * 100) if total > 0 else 0
            pmf_achieved = pmf_score >= 40

            # По сегментам
            pmf_by_segment = {}
            for segment in ["power_users", "at_risk", "trial_converters", "new_users"]:
                seg_rows = [r for r in rows if r[1] == segment]
                if seg_rows:
                    seg_very = sum(1 for r in seg_rows if r[0] == "very_disappointed")
                    pmf_by_segment[segment] = (seg_very / len(seg_rows) * 100)

            return SeanEllisResult(
                total_responses=total,
                very_disappointed=very,
                somewhat_disappointed=somewhat,
                not_disappointed=not_disappointed,
                pmf_score=pmf_score,
                pmf_achieved=pmf_achieved,
                pmf_by_segment=pmf_by_segment
            )

    async def get_feature_pmf(self, feature_name: str, days: int = 30) -> FeaturePMF:
        """
        PMF Score для конкретной фичи

        PMF = satisfaction × usage_frequency × retention_lift
        """
        async with self.db.pool.acquire() as conn:
            # Usage metrics
            cursor = await conn.execute("""
                SELECT
                    COUNT(DISTINCT user_id) as total_users,
                    COUNT(*) as total_uses,
                    AVG(uses_per_user) as avg_uses
                FROM (
                    SELECT
                        user_id,
                        COUNT(*) as uses_per_user
                    FROM behavior_events
                    WHERE feature = ?
                      AND timestamp > strftime('%s', 'now', ? || ' days')
                    GROUP BY user_id
                )
            """, (feature_name, f'-{days}'))
            row = await cursor.fetchone()
            await cursor.close()

            total_users = row[0] if row and row[0] else 0
            total_uses = row[1] if row and row[1] else 0

            # Active users (используют хотя бы 1 раз в неделю)
            cursor = await conn.execute("""
                SELECT COUNT(DISTINCT user_id) as active
                FROM behavior_events
                WHERE feature = ?
                  AND timestamp > strftime('%s', 'now', '-7 day')
            """, (feature_name,))
            row = await cursor.fetchone()
            await cursor.close()

            active_users = row[0] if row and row[0] else 0

            # Usage frequency (uses per week)
            weeks = days / 7
            usage_frequency = (total_uses / total_users / weeks) if total_users > 0 else 0

            # Satisfaction score (based on success rate + repeat usage)
            cursor = await conn.execute("""
                SELECT
                    AVG(CASE WHEN success = 1 THEN 100.0 ELSE 0.0 END) as success_rate,
                    COUNT(DISTINCT user_id) * 100.0 / (
                        SELECT COUNT(DISTINCT user_id)
                        FROM behavior_events
                        WHERE timestamp > strftime('%s', 'now', ? || ' days')
                    ) as repeat_rate
                FROM behavior_events
                WHERE feature = ?
                  AND timestamp > strftime('%s', 'now', ? || ' days')
                  AND user_id IN (
                      SELECT user_id
                      FROM behavior_events
                      WHERE feature = ?
                      GROUP BY user_id
                      HAVING COUNT(*) > 1
                  )
            """, (f'-{days}', feature_name, f'-{days}', feature_name))
            row = await cursor.fetchone()
            await cursor.close()

            success_rate = row[0] if row and row[0] else 0
            repeat_rate = row[1] if row and row[1] else 0

            satisfaction_score = (success_rate * 0.6 + repeat_rate * 0.4)

            # Retention lift (из cohort_analytics)
            cursor = await conn.execute("""
                WITH feature_users AS (
                    SELECT DISTINCT user_id
                    FROM behavior_events
                    WHERE feature = ?
                ),
                retention_with AS (
                    SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM feature_users) as rate
                    FROM users
                    WHERE user_id IN (SELECT user_id FROM feature_users)
                      AND (strftime('%s', 'now') - last_active) < 2592000
                ),
                retention_without AS (
                    SELECT COUNT(*) * 100.0 / (
                        SELECT COUNT(*) FROM users
                        WHERE user_id NOT IN (SELECT user_id FROM feature_users)
                    ) as rate
                    FROM users
                    WHERE user_id NOT IN (SELECT user_id FROM feature_users)
                      AND (strftime('%s', 'now') - last_active) < 2592000
                )
                SELECT
                    COALESCE(rw.rate, 0) as with_retention,
                    COALESCE(rwo.rate, 0) as without_retention
                FROM retention_with rw, retention_without rwo
            """, (feature_name,))
            row = await cursor.fetchone()
            await cursor.close()

            retention_with = row[0] if row else 0
            retention_without = row[1] if row else 0
            retention_lift = max(0, retention_with - retention_without)

            # PMF Score calculation
            # Нормализация компонентов к 0-1
            satisfaction_norm = satisfaction_score / 100
            frequency_norm = min(1.0, usage_frequency / 5)  # 5+ uses/week = max
            retention_norm = min(1.0, retention_lift / 30)  # +30% retention = max

            pmf_score = (satisfaction_norm * 0.4 + frequency_norm * 0.3 + retention_norm * 0.3) * 100

            # PMF Rating
            if pmf_score >= 70:
                pmf_rating = "strong"
                key_insight = "🌟 Сильный PMF - инвестировать в развитие"
            elif pmf_score >= 50:
                pmf_rating = "moderate"
                key_insight = "✅ Умеренный PMF - улучшать UX"
            elif pmf_score >= 30:
                pmf_rating = "weak"
                key_insight = "⚠️ Слабый PMF - требуется pivot"
            else:
                pmf_rating = "kill"
                key_insight = "🗑️ Нет PMF - рассмотреть удаление"

            return FeaturePMF(
                feature_name=feature_name,
                total_users=total_users,
                active_users=active_users,
                usage_frequency=usage_frequency,
                satisfaction_score=satisfaction_score,
                nps_for_feature=None,  # TODO: implement feature-specific NPS
                pmf_score=pmf_score,
                pmf_rating=pmf_rating,
                key_insight=key_insight
            )

    async def get_usage_intensity(self) -> UsageIntensity:
        """
        Метрики интенсивности использования (DAU/WAU/MAU)
        """
        async with self.db.pool.acquire() as conn:
            # DAU
            cursor = await conn.execute("""
                SELECT COUNT(DISTINCT user_id) as dau
                FROM behavior_events
                WHERE timestamp > strftime('%s', 'now', '-1 day')
            """)
            row = await cursor.fetchone()
            await cursor.close()
            dau = row[0] if row else 0

            # WAU
            cursor = await conn.execute("""
                SELECT COUNT(DISTINCT user_id) as wau
                FROM behavior_events
                WHERE timestamp > strftime('%s', 'now', '-7 day')
            """)
            row = await cursor.fetchone()
            await cursor.close()
            wau = row[0] if row else 0

            # MAU
            cursor = await conn.execute("""
                SELECT COUNT(DISTINCT user_id) as mau
                FROM behavior_events
                WHERE timestamp > strftime('%s', 'now', '-30 day')
            """)
            row = await cursor.fetchone()
            await cursor.close()
            mau = row[0] if row else 0

            # Stickiness ratios
            dau_mau_ratio = (dau / mau * 100) if mau > 0 else 0
            dau_wau_ratio = (dau / wau * 100) if wau > 0 else 0

            # Power User Curve
            cursor = await conn.execute("""
                SELECT
                    SUM(CASE WHEN request_count > 50 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as power_users,
                    SUM(CASE WHEN request_count < 10 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as casual_users
                FROM (
                    SELECT user_id, COUNT(*) as request_count
                    FROM behavior_events
                    WHERE timestamp > strftime('%s', 'now', '-30 day')
                    GROUP BY user_id
                )
            """)
            row = await cursor.fetchone()
            await cursor.close()

            power_user_pct = row[0] if row and row[0] else 0
            casual_user_pct = row[1] if row and row[1] else 0

            # L28 retention (активны 14+ дней из последних 28)
            cursor = await conn.execute("""
                SELECT
                    COUNT(*) * 100.0 / (
                        SELECT COUNT(DISTINCT user_id)
                        FROM behavior_events
                        WHERE timestamp > strftime('%s', 'now', '-28 day')
                    ) as l28
                FROM (
                    SELECT user_id, COUNT(DISTINCT DATE(timestamp, 'unixepoch')) as active_days
                    FROM behavior_events
                    WHERE timestamp > strftime('%s', 'now', '-28 day')
                    GROUP BY user_id
                    HAVING active_days >= 14
                )
            """)
            row = await cursor.fetchone()
            await cursor.close()

            l28_retention = row[0] if row and row[0] else 0

            return UsageIntensity(
                dau=dau,
                wau=wau,
                mau=mau,
                dau_mau_ratio=dau_mau_ratio,
                dau_wau_ratio=dau_wau_ratio,
                power_user_percentage=power_user_pct,
                casual_user_percentage=casual_user_pct,
                l28_retention=l28_retention
            )

    async def send_nps_survey(self, user_id: int, trigger: str = "after_payment") -> bool:
        """
        Отправить NPS опрос пользователю

        Returns: True если опрос успешно создан
        """
        # Проверить не отправляли ли недавно
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT COUNT(*) as recent_surveys
                FROM nps_surveys
                WHERE user_id = ?
                  AND created_at > strftime('%s', 'now', '-30 day')
            """, (user_id,))
            row = await cursor.fetchone()
            await cursor.close()

            if row and row[0] > 0:
                return False  # Уже отправляли недавно

            # Определить segment пользователя
            cursor = await conn.execute("""
                SELECT
                    CASE
                        WHEN total_requests > 50 AND (
                            SELECT COUNT(*) FROM payments
                            WHERE payments.user_id = users.user_id AND status = 'completed'
                        ) >= 2 THEN 'power_users'
                        WHEN total_requests < 10 THEN 'new_users'
                        ELSE 'regular_users'
                    END as segment
                FROM users
                WHERE user_id = ?
            """, (user_id,))
            row = await cursor.fetchone()
            await cursor.close()

            user_segment = row[0] if row else "unknown"

            # Создать запись опроса
            cursor = await conn.execute("""
                INSERT INTO nps_surveys (user_id, trigger_event, user_segment, created_at)
                VALUES (?, ?, ?, strftime('%s', 'now'))
            """, (user_id, trigger, user_segment))
            await conn.commit()
            await cursor.close()

            return True

    async def record_nps_response(self, user_id: int, score: int, disappointment_level: str | None = None, feedback: str | None = None) -> bool:
        """
        Записать ответ на NPS опрос

        Args:
            score: 0-10
            disappointment_level: "very_disappointed", "somewhat_disappointed", "not_disappointed"
            feedback: Текстовый комментарий
        """
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                UPDATE nps_surveys
                SET score = ?,
                    disappointment_level = ?,
                    feedback = ?,
                    responded_at = strftime('%s', 'now')
                WHERE user_id = ?
                  AND score IS NULL
                ORDER BY created_at DESC
                LIMIT 1
            """, (score, disappointment_level, feedback, user_id))
            await conn.commit()
            affected = cursor.rowcount
            await cursor.close()

            return affected > 0
