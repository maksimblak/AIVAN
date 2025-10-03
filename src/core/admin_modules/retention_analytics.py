"""
Retention Analytics - глубокий анализ повторных покупок
Понимание что удерживает пользователей vs что заставляет уходить
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetainedUserProfile:
    """Профиль пользователя который продлевает подписку"""
    user_id: int
    payment_count: int
    total_spent: int
    lifetime_days: int
    total_requests: int
    avg_requests_per_day: float

    # Поведенческие метрики
    favorite_features: list[tuple[str, int]]  # [(feature, usage_count)]
    usage_patterns: dict[str, Any]  # hour_of_day, day_of_week
    session_duration_avg_minutes: float

    # Engagement метрики
    days_active_per_week: float
    streak_max_days: int  # максимальная серия дней подряд

    # Ценность
    feature_diversity: float  # % от всех фич которые используют
    power_user_score: float  # 0-100

    # Демография
    time_to_first_payment_days: int
    retention_probability: float  # ML prediction


@dataclass
class ChurnedUserProfile:
    """Профиль пользователя который НЕ продлил"""
    user_id: int
    payment_count: int  # должно быть 1
    total_spent: int
    lifetime_days: int
    total_requests: int

    # Причины оттока
    churn_indicators: list[str]  # ['low_usage', 'had_errors', 'price_sensitive']
    last_active_days_ago: int
    drop_off_feature: str | None  # где последний раз был активен

    # Что НЕ использовал
    unused_features: list[str]
    had_technical_issues: bool
    received_poor_responses: bool  # negative ratings

    # Потенциал возврата
    winback_probability: float  # 0-100
    recommended_action: str


class RetentionAnalytics:
    """Анализ retention: кто остается vs кто уходит"""

    def __init__(self, db):
        self.db = db

    # ==================== RETAINED USERS (Продлевают) ====================

    async def get_retained_users(self, min_payments: int = 2) -> list[RetainedUserProfile]:
        """Пользователи с повторными платежами"""

        async with self.db.pool.acquire() as conn:
            # Получаем пользователей с несколькими платежами
            cursor = await conn.execute("""
                SELECT
                    u.user_id,
                    COUNT(t.id) as payment_count,
                    SUM(t.amount) as total_spent,
                    u.total_requests,
                    u.created_at,
                    MIN(t.created_at) as first_payment_at
                FROM users u
                INNER JOIN payments t ON u.user_id = t.user_id
                WHERE t.status = 'completed'
                GROUP BY u.user_id
                HAVING payment_count >= ?
                ORDER BY payment_count DESC, total_spent DESC
            """, (min_payments,))

            rows = await cursor.fetchall()
            await cursor.close()

            profiles = []
            for row in rows:
                user_id, payment_count, total_spent, total_requests, created_at, first_payment_at = row

                now = int(time.time())
                lifetime_days = (now - created_at) // 86400
                avg_requests_per_day = total_requests / max(lifetime_days, 1)

                # Анализируем любимые фичи
                favorite_features = await self._get_favorite_features(user_id)

                # Паттерны использования
                usage_patterns = await self._get_usage_patterns(user_id)

                # Session duration
                session_duration = await self._get_avg_session_duration(user_id)

                # Активность по дням
                days_active = await self._get_days_active_per_week(user_id)

                # Streak
                max_streak = await self._get_max_streak(user_id)

                # Feature diversity
                feature_diversity = await self._get_feature_diversity(user_id)

                # Power user score
                power_score = self._calculate_power_user_score(
                    payment_count=payment_count,
                    avg_requests_per_day=avg_requests_per_day,
                    days_active_per_week=days_active,
                    feature_diversity=feature_diversity,
                    lifetime_days=lifetime_days
                )

                # Time to first payment
                time_to_first_payment = (first_payment_at - created_at) // 86400

                # Retention probability (простая эвристика, можно заменить на ML)
                retention_prob = self._predict_retention_probability(
                    payment_count=payment_count,
                    power_score=power_score,
                    lifetime_days=lifetime_days
                )

                profiles.append(RetainedUserProfile(
                    user_id=user_id,
                    payment_count=payment_count,
                    total_spent=total_spent,
                    lifetime_days=lifetime_days,
                    total_requests=total_requests,
                    avg_requests_per_day=round(avg_requests_per_day, 2),
                    favorite_features=favorite_features,
                    usage_patterns=usage_patterns,
                    session_duration_avg_minutes=session_duration,
                    days_active_per_week=days_active,
                    streak_max_days=max_streak,
                    feature_diversity=feature_diversity,
                    power_user_score=power_score,
                    time_to_first_payment_days=time_to_first_payment,
                    retention_probability=retention_prob
                ))

            return profiles

    async def _get_favorite_features(self, user_id: int) -> list[tuple[str, int]]:
        """Топ-5 любимых фич пользователя"""
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT request_type, COUNT(*) as count
                FROM requests
                WHERE user_id = ? AND success = 1
                GROUP BY request_type
                ORDER BY count DESC
                LIMIT 5
            """, (user_id,))

            rows = await cursor.fetchall()
            await cursor.close()

            return [(row[0], row[1]) for row in rows]

    async def _get_usage_patterns(self, user_id: int) -> dict[str, Any]:
        """Паттерны использования по времени"""
        async with self.db.pool.acquire() as conn:
            # Peak hour
            cursor = await conn.execute("""
                SELECT
                    CAST(strftime('%H', created_at, 'unixepoch') AS INTEGER) as hour,
                    COUNT(*) as count
                FROM requests
                WHERE user_id = ?
                GROUP BY hour
                ORDER BY count DESC
                LIMIT 1
            """, (user_id,))

            hour_row = await cursor.fetchone()
            await cursor.close()

            peak_hour = hour_row[0] if hour_row else None

            # Day of week (0 = Monday, 6 = Sunday)
            cursor = await conn.execute("""
                SELECT
                    CAST(strftime('%w', created_at, 'unixepoch') AS INTEGER) as dow,
                    COUNT(*) as count
                FROM requests
                WHERE user_id = ?
                GROUP BY dow
                ORDER BY count DESC
                LIMIT 1
            """, (user_id,))

            dow_row = await cursor.fetchone()
            await cursor.close()

            peak_day = dow_row[0] if dow_row else None

            return {
                'peak_hour': peak_hour,
                'peak_day_of_week': peak_day,
                'is_weekday_user': peak_day is not None and peak_day < 6,
                'is_daytime_user': peak_hour is not None and 9 <= peak_hour <= 18
            }

    async def _get_avg_session_duration(self, user_id: int) -> float:
        """Средняя длительность сессии в минутах"""
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT AVG(response_time_ms) FROM requests
                WHERE user_id = ? AND response_time_ms > 0
            """, (user_id,))

            row = await cursor.fetchone()
            await cursor.close()

            avg_ms = row[0] if row and row[0] else 0
            return round(avg_ms / 60000, 2)  # в минуты

    async def _get_days_active_per_week(self, user_id: int) -> float:
        """Сколько дней в неделю в среднем активен"""
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT
                    COUNT(DISTINCT DATE(created_at, 'unixepoch')) as unique_days,
                    (MAX(created_at) - MIN(created_at)) / 604800.0 as weeks
                FROM requests
                WHERE user_id = ?
            """, (user_id,))

            row = await cursor.fetchone()
            await cursor.close()

            if row and row[1] > 0:
                return round(row[0] / row[1], 2)
            return 0.0

    async def _get_max_streak(self, user_id: int) -> int:
        """Максимальная серия дней подряд использования"""
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT DISTINCT DATE(created_at, 'unixepoch') as day
                FROM requests
                WHERE user_id = ?
                ORDER BY day
            """, (user_id,))

            rows = await cursor.fetchall()
            await cursor.close()

            if not rows:
                return 0

            max_streak = 1
            current_streak = 1

            for i in range(1, len(rows)):
                prev_date = datetime.strptime(rows[i-1][0], '%Y-%m-%d')
                curr_date = datetime.strptime(rows[i][0], '%Y-%m-%d')

                if (curr_date - prev_date).days == 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1

            return max_streak

    async def _get_feature_diversity(self, user_id: int) -> float:
        """% фич которые пользователь использовал"""
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT COUNT(DISTINCT request_type) as used_features
                FROM requests
                WHERE user_id = ?
            """, (user_id,))

            row = await cursor.fetchone()
            await cursor.close()

            used_features = row[0] if row else 0

            # Всего доступных фич (hardcoded, можно вынести в конфиг)
            total_features = 10  # legal_question, voice, document_*, etc.

            return round((used_features / total_features) * 100, 2)

    def _calculate_power_user_score(
        self,
        payment_count: int,
        avg_requests_per_day: float,
        days_active_per_week: float,
        feature_diversity: float,
        lifetime_days: int
    ) -> float:
        """Scoring 0-100 насколько пользователь power user"""

        score = 0.0

        # 1. Payments (до 30 баллов)
        score += min(payment_count * 5, 30)

        # 2. Activity (до 25 баллов)
        score += min(avg_requests_per_day * 2, 25)

        # 3. Consistency (до 20 баллов)
        score += min(days_active_per_week * 3, 20)

        # 4. Feature exploration (до 15 баллов)
        score += min(feature_diversity * 0.15, 15)

        # 5. Longevity bonus (до 10 баллов)
        score += min(lifetime_days / 30, 10)

        return round(min(score, 100), 2)

    def _predict_retention_probability(
        self, payment_count: int, power_score: float, lifetime_days: int
    ) -> float:
        """Простая эвристика вероятности retention (можно заменить на ML)"""

        prob = 50.0  # base

        # Каждый дополнительный payment +15%
        prob += (payment_count - 1) * 15

        # Power score влияет
        prob += (power_score - 50) * 0.3

        # Longevity влияет
        if lifetime_days > 90:
            prob += 10

        return round(min(max(prob, 0), 100), 2)

    # ==================== CHURNED USERS (Не продлевают) ====================

    async def get_churned_users(self, days_since_expiry: int = 30) -> list[ChurnedUserProfile]:
        """Пользователи которые не продлили после 1 платежа"""

        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            cutoff = now - (days_since_expiry * 86400)

            # Пользователи с ровно 1 платежом и истекшей подпиской
            cursor = await conn.execute("""
                SELECT
                    u.user_id,
                    COUNT(t.id) as payment_count,
                    SUM(t.amount) as total_spent,
                    u.created_at,
                    u.subscription_until,
                    u.total_requests,
                    u.last_request_at
                FROM users u
                INNER JOIN payments t ON u.user_id = t.user_id
                WHERE t.status = 'completed'
                  AND u.subscription_until > ?
                  AND u.subscription_until < ?
                GROUP BY u.user_id
                HAVING payment_count = 1
                ORDER BY u.subscription_until DESC
            """, (cutoff, now))

            rows = await cursor.fetchall()
            await cursor.close()

            profiles = []
            for row in rows:
                user_id, payment_count, total_spent, created_at, sub_until, total_requests, last_request = row

                lifetime_days = (sub_until - created_at) // 86400
                last_active_days_ago = (now - last_request) // 86400

                # Определяем индикаторы оттока
                churn_indicators = await self._identify_churn_indicators(user_id, total_requests, lifetime_days)

                # Последняя активность
                drop_off_feature = await self._get_last_feature_used(user_id)

                # Неиспользованные фичи
                unused_features = await self._get_unused_features(user_id)

                # Были ли технические проблемы
                had_issues = await self._had_technical_issues(user_id)

                # Были ли негативные отзывы
                poor_responses = await self._had_poor_responses(user_id)

                # Вероятность win-back
                winback_prob = self._calculate_winback_probability(
                    total_requests=total_requests,
                    lifetime_days=lifetime_days,
                    had_issues=had_issues,
                    last_active_days_ago=last_active_days_ago
                )

                # Рекомендация по возврату
                recommended_action = self._recommend_winback_action(
                    churn_indicators=churn_indicators,
                    winback_prob=winback_prob,
                    had_issues=had_issues
                )

                profiles.append(ChurnedUserProfile(
                    user_id=user_id,
                    payment_count=payment_count,
                    total_spent=total_spent,
                    lifetime_days=lifetime_days,
                    total_requests=total_requests,
                    churn_indicators=churn_indicators,
                    last_active_days_ago=last_active_days_ago,
                    drop_off_feature=drop_off_feature,
                    unused_features=unused_features,
                    had_technical_issues=had_issues,
                    received_poor_responses=poor_responses,
                    winback_probability=winback_prob,
                    recommended_action=recommended_action
                ))

            return profiles

    async def _identify_churn_indicators(
        self, user_id: int, total_requests: int, lifetime_days: int
    ) -> list[str]:
        """Определение причин оттока"""

        indicators = []

        # 1. Низкая активность
        avg_per_day = total_requests / max(lifetime_days, 1)
        if avg_per_day < 1:
            indicators.append('low_usage')

        # 2. Технические проблемы
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM requests
                WHERE user_id = ? AND success = 0
            """, (user_id,))
            failed = (await cursor.fetchone())[0]
            await cursor.close()

            if failed > total_requests * 0.2:  # >20% failures
                indicators.append('had_errors')

        # 3. Не изучил продукт
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT COUNT(DISTINCT request_type) FROM requests
                WHERE user_id = ?
            """, (user_id,))
            unique_features = (await cursor.fetchone())[0]
            await cursor.close()

            if unique_features <= 2:
                indicators.append('limited_exploration')

        # 4. Негативный опыт
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM ratings r
                INNER JOIN requests req ON r.request_id = req.id
                WHERE req.user_id = ? AND r.rating = -1
            """, (user_id,))
            dislikes = (await cursor.fetchone())[0]
            await cursor.close()

            if dislikes > 2:
                indicators.append('poor_experience')

        # 5. Быстро бросил после покупки (объединенный запрос для атомарности)
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT
                    (SELECT MIN(created_at) FROM payments WHERE user_id = ? AND status = 'completed') as first_payment,
                    (SELECT MAX(created_at) FROM requests WHERE user_id = ?) as last_request
            """, (user_id, user_id))
            row = await cursor.fetchone()
            await cursor.close()

            if row and row[0] is not None and row[1] is not None:
                first_payment, last_request = row
                days_after_payment = (last_request - first_payment) // 86400

                if days_after_payment < 3:
                    indicators.append('immediate_abandonment')

        # 6. Вероятно цена слишком высокая
        if total_requests < 10:  # мало использовал перед окончанием
            indicators.append('price_sensitive')

        return indicators

    async def _get_last_feature_used(self, user_id: int) -> str | None:
        """Последняя фича которую использовал пользователь"""
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT request_type FROM requests
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id,))

            row = await cursor.fetchone()
            await cursor.close()

            return row[0] if row else None

    async def _get_unused_features(self, user_id: int) -> list[str]:
        """Какие фичи не попробовал"""
        all_features = [
            'legal_question',
            'voice_message',
            'document_summary',
            'document_risks',
            'document_chat',
            'ocr',
            'judicial_practice'
        ]

        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT DISTINCT request_type FROM requests
                WHERE user_id = ?
            """, (user_id,))

            rows = await cursor.fetchall()
            await cursor.close()

            used = {row[0] for row in rows}

            return [f for f in all_features if f not in used]

    async def _had_technical_issues(self, user_id: int) -> bool:
        """Были ли серьезные технические проблемы"""
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM requests
                WHERE user_id = ? AND success = 0
            """, (user_id,))

            failures = (await cursor.fetchone())[0]
            await cursor.close()

            return failures > 3

    async def _had_poor_responses(self, user_id: int) -> bool:
        """Были ли плохие ответы (negative ratings)"""
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM ratings r
                INNER JOIN requests req ON r.request_id = req.id
                WHERE req.user_id = ? AND r.rating = -1
            """, (user_id,))

            dislikes = (await cursor.fetchone())[0]
            await cursor.close()

            return dislikes >= 2

    def _calculate_winback_probability(
        self,
        total_requests: int,
        lifetime_days: int,
        had_issues: bool,
        last_active_days_ago: int
    ) -> float:
        """Вероятность вернуть пользователя"""

        prob = 50.0

        # Если был активен - выше шанс вернуть
        if total_requests > 20:
            prob += 20
        elif total_requests > 10:
            prob += 10

        # Если долго был клиентом - выше лояльность
        if lifetime_days > 60:
            prob += 15

        # Если были проблемы - сложнее вернуть
        if had_issues:
            prob -= 25

        # Если давно не заходил - сложнее
        if last_active_days_ago > 60:
            prob -= 20
        elif last_active_days_ago > 30:
            prob -= 10

        return round(min(max(prob, 0), 100), 2)

    def _recommend_winback_action(
        self,
        churn_indicators: list[str],
        winback_prob: float,
        had_issues: bool
    ) -> str:
        """Рекомендация что делать для возврата"""

        if winback_prob > 60:
            if 'price_sensitive' in churn_indicators:
                return "Discount offer (30-50% off)"
            elif 'limited_exploration' in churn_indicators:
                return "Onboarding email + free trial extension"
            else:
                return "Simple reminder + case study"

        elif winback_prob > 30:
            if had_issues:
                return "Apology + fixed issues announcement + discount"
            elif 'poor_experience' in churn_indicators:
                return "Product improvements announcement + free month"
            else:
                return "Win-back campaign with strong incentive"

        else:
            return "Low priority - focus on high-probability users first"

    # ==================== COMPARATIVE ANALYSIS ====================

    async def compare_retained_vs_churned(self) -> dict[str, Any]:
        """Сравнение retained vs churned пользователей"""

        retained = await self.get_retained_users(min_payments=2)
        churned = await self.get_churned_users(days_since_expiry=90)

        if not retained or not churned:
            return {"error": "Insufficient data"}

        comparison = {
            "retained": {
                "count": len(retained),
                "avg_requests": sum(u.total_requests for u in retained) / len(retained),
                "avg_lifetime_days": sum(u.lifetime_days for u in retained) / len(retained),
                "avg_power_score": sum(u.power_user_score for u in retained) / len(retained),
                "avg_feature_diversity": sum(u.feature_diversity for u in retained) / len(retained),
                "top_features": self._get_top_features_for_segment(retained),
                "common_patterns": self._find_common_patterns(retained)
            },
            "churned": {
                "count": len(churned),
                "avg_requests": sum(u.total_requests for u in churned) / len(churned),
                "avg_lifetime_days": sum(u.lifetime_days for u in churned) / len(churned),
                "top_churn_indicators": self._get_top_churn_indicators(churned),
                "unused_features_common": self._get_common_unused_features(churned),
                "winback_distribution": self._winback_distribution(churned)
            }
        }

        return comparison

    def _get_top_features_for_segment(self, users: list[RetainedUserProfile]) -> dict[str, int]:
        """Какие фичи больше всего любят retained users"""
        feature_counts = defaultdict(int)

        for user in users:
            for feature, count in user.favorite_features[:3]:  # топ-3 для каждого
                feature_counts[feature] += count

        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:5])

    def _find_common_patterns(self, users: list[RetainedUserProfile]) -> dict[str, Any]:
        """Общие паттерны среди retained users"""
        weekday_users = sum(1 for u in users if u.usage_patterns.get('is_weekday_user', False))
        daytime_users = sum(1 for u in users if u.usage_patterns.get('is_daytime_user', False))

        return {
            "weekday_preference_pct": round((weekday_users / len(users)) * 100, 1),
            "daytime_preference_pct": round((daytime_users / len(users)) * 100, 1),
            "avg_days_active_per_week": round(
                sum(u.days_active_per_week for u in users) / len(users), 2
            )
        }

    def _get_top_churn_indicators(self, users: list[ChurnedUserProfile]) -> dict[str, int]:
        """Самые частые причины оттока"""
        indicator_counts = defaultdict(int)

        for user in users:
            for indicator in user.churn_indicators:
                indicator_counts[indicator] += 1

        sorted_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_indicators)

    def _get_common_unused_features(self, users: list[ChurnedUserProfile]) -> dict[str, int]:
        """Какие фичи чаще всего не используют churned users"""
        feature_counts = defaultdict(int)

        for user in users:
            for feature in user.unused_features:
                feature_counts[feature] += 1

        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:5])

    def _winback_distribution(self, users: list[ChurnedUserProfile]) -> dict[str, int]:
        """Распределение по вероятности win-back"""
        distribution = {"high": 0, "medium": 0, "low": 0}

        for user in users:
            if user.winback_probability > 60:
                distribution["high"] += 1
            elif user.winback_probability > 30:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution
