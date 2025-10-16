"""
Модуль аналитики для администраторов
Отслеживание активности пользователей, конверсии, оттока
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class UserSegment:
    """Сегмент пользователей для аналитики"""
    segment_id: str
    name: str
    description: str
    user_count: int
    users: list[dict[str, Any]]
    metrics: dict[str, Any]


@dataclass
class ConversionMetrics:
    """Метрики конверсии"""
    total_trial_users: int
    converted_to_paid: int
    conversion_rate: float
    avg_trial_requests_before_conversion: float
    avg_time_to_conversion_days: float


@dataclass
class ChurnMetrics:
    """Метрики оттока"""
    total_expired: int
    renewed_count: int
    churned_count: int
    retention_rate: float
    avg_requests_before_churn: float
    churn_by_usage: dict[str, int]  # low/medium/high usage


METRIC_LABELS: dict[str, str] = {
    "total_revenue_potential": "Потенциальная выручка сегмента",
    "avg_requests_per_user": "Среднее запросов на пользователя",
    "most_active_user_id": "ID самого активного пользователя",
    "high_risk_count": "Пользователей в высокой зоне риска",
    "potential_revenue_loss": "Потенциальная потеря выручки",
    "avg_lifetime_days": "Средний срок жизни (дней)",
    "total_lost_revenue": "Потерянная выручка",
    "avg_time_to_conversion": "Среднее время до оплаты (дней)",
    "recurring_customers": "Повторно оплатившие",
    "total_revenue": "Выручка сегмента",
    "potential_conversions": "Потенциальные конверсии",
    "already_paid": "Уже оплатили",
    "avg_requests": "Среднее число запросов",
    "active_subscribers": "Активных подписок",
    "monthly_revenue_estimate": "Оценка месячной выручки",
}

PLAN_SEGMENT_DEFS: dict[str, dict[str, Any]] = {
    "base_1m": {
        "name": "💼 Тариф «Базовый»",
        "button": "💼 Базовый",
        "description": "Подписчики тарифа «Базовый»",
        "price": 1499,
    },
    "standard_1m": {
        "name": "📦 Тариф «Стандарт»",
        "button": "📦 Стандарт",
        "description": "Подписчики тарифа «Стандарт»",
        "price": 2500,
    },
    "premium_1m": {
        "name": "🚀 Тариф «Премиум»",
        "button": "🚀 Премиум",
        "description": "Подписчики тарифа «Премиум»",
        "price": 4000,
    },
}

PLAN_SEGMENT_ORDER: tuple[str, ...] = tuple(PLAN_SEGMENT_DEFS.keys())


class AdminAnalytics:
    """Система аналитики для администраторов"""

    def __init__(self, db):
        self.db = db

    async def get_user_segments(self) -> dict[str, UserSegment]:
        """Получение всех сегментов пользователей"""

        segments = {}

        # 1. Power Users - активные платные пользователи
        segments['power_users'] = await self._get_power_users()

        # 2. At Risk - риск оттока
        segments['at_risk'] = await self._get_at_risk_users()

        # 3. Churned - ушедшие
        segments['churned'] = await self._get_churned_users()

        # 4. Trial Converters - успешная конверсия
        segments['trial_converters'] = await self._get_trial_converters()

        # 5. Freeloaders - использовали trial и пропали
        segments['freeloaders'] = await self._get_freeloaders()

        # 6. New Users - новые за последние 7 дней
        segments['new_users'] = await self._get_new_users()

        # 7. Подписки по тарифам
        segments.update(await self._get_subscription_plan_segments())

        return segments

    async def _get_power_users(self) -> UserSegment:
        """Активные платные пользователи (> 5 запросов/день)"""

        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            week_ago = now - (7 * 86400)

            cursor = await conn.execute("""
                SELECT
                    u.user_id,
                    u.total_requests,
                    u.last_request_at,
                    u.subscription_until,
                    u.created_at,
                    COUNT(r.id) as week_requests,
                    AVG(r.response_time_ms) as avg_response_time
                FROM users u
                LEFT JOIN requests r ON u.user_id = r.user_id
                    AND r.created_at >= ?
                WHERE u.subscription_until > ?
                GROUP BY u.user_id
                HAVING week_requests > 35  -- > 5/день за неделю
                ORDER BY week_requests DESC
            """, (week_ago, now))

            rows = await cursor.fetchall()
            await cursor.close()

            users = []
            for row in rows:
                users.append({
                    'user_id': row[0],
                    'total_requests': row[1],
                    'last_active': datetime.fromtimestamp(row[2]).strftime('%Y-%m-%d %H:%M'),
                    'subscription_until': datetime.fromtimestamp(row[3]).strftime('%Y-%m-%d'),
                    'days_since_registration': (now - row[4]) // 86400,
                    'week_requests': row[5],
                    'avg_requests_per_day': round(row[5] / 7, 1),
                    'avg_response_time_ms': int(row[6]) if row[6] else 0
                })

            metrics = {
                'total_revenue_potential': len(users) * 300,  # примерная стоимость подписки
                'avg_requests_per_user': sum(u['week_requests'] for u in users) / max(len(users), 1),
                'most_active_user_id': users[0]['user_id'] if users else None
            }

            return UserSegment(
                segment_id='power_users',
                name='⚡ Суперактивные',
                description='Активные платные пользователи (>5 запросов/день)',
                user_count=len(users),
                users=users,
                metrics=metrics
            )

    async def _get_at_risk_users(self) -> UserSegment:
        """Пользователи с риском оттока"""

        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            week_ago = now - (7 * 86400)
            expires_soon = now + (7 * 86400)  # подписка истекает через 7 дней

            cursor = await conn.execute("""
                SELECT
                    u.user_id,
                    u.total_requests,
                    u.last_request_at,
                    u.subscription_until,
                    COUNT(r.id) as week_requests
                FROM users u
                LEFT JOIN requests r ON u.user_id = r.user_id
                    AND r.created_at >= ?
                WHERE u.subscription_until > ?
                  AND u.subscription_until < ?
                GROUP BY u.user_id
                HAVING week_requests < 14  -- < 2/день
                ORDER BY u.subscription_until ASC
            """, (week_ago, now, expires_soon))

            rows = await cursor.fetchall()
            await cursor.close()

            users = []
            for row in rows:
                days_until_expiry = (row[3] - now) // 86400
                users.append({
                    'user_id': row[0],
                    'total_requests': row[1],
                    'last_active': datetime.fromtimestamp(row[2]).strftime('%Y-%m-%d %H:%M'),
                    'days_until_expiry': days_until_expiry,
                    'week_requests': row[4],
                    'risk_level': 'high' if days_until_expiry < 3 else 'medium'
                })

            metrics = {
                'high_risk_count': sum(1 for u in users if u['risk_level'] == 'high'),
                'potential_revenue_loss': len(users) * 300
            }

            return UserSegment(
                segment_id='at_risk',
                name='⚠️ Группа риска',
                description='Мало используют, подписка истекает скоро',
                user_count=len(users),
                users=users,
                metrics=metrics
            )

    async def _get_churned_users(self) -> UserSegment:
        """Ушедшие пользователи (не продлили подписку)"""

        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            month_ago = now - (30 * 86400)

            cursor = await conn.execute("""
                SELECT
                    u.user_id,
                    u.total_requests,
                    u.last_request_at,
                    u.subscription_until,
                    u.created_at,
                    COUNT(t.id) as total_payments
                FROM users u
                LEFT JOIN payments t ON u.user_id = t.user_id
                WHERE u.subscription_until > ?
                  AND u.subscription_until < ?
                  AND u.last_request_at < u.subscription_until
                GROUP BY u.user_id
                HAVING total_payments > 0
                ORDER BY u.subscription_until DESC
            """, (month_ago, now))

            rows = await cursor.fetchall()
            await cursor.close()

            users = []
            for row in rows:
                days_since_expiry = (now - row[3]) // 86400
                lifetime_days = (row[3] - row[4]) // 86400
                users.append({
                    'user_id': row[0],
                    'total_requests': row[1],
                    'last_active': datetime.fromtimestamp(row[2]).strftime('%Y-%m-%d %H:%M'),
                    'expired_at': datetime.fromtimestamp(row[3]).strftime('%Y-%m-%d'),
                    'days_since_expiry': days_since_expiry,
                    'lifetime_days': lifetime_days,
                    'total_payments': row[5],
                    'ltv': row[5] * 300  # примерная lifetime value
                })

            metrics = {
                'avg_lifetime_days': sum(u['lifetime_days'] for u in users) / max(len(users), 1),
                'total_lost_revenue': sum(u['ltv'] for u in users)
            }

            return UserSegment(
                segment_id='churned',
                name='📉 Ушедшие',
                description='Не продлили подписку после истечения',
                user_count=len(users),
                users=users,
                metrics=metrics
            )

    async def _get_trial_converters(self) -> UserSegment:
        """Успешно конвертировались из trial в paid"""

        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT
                    u.user_id,
                    u.total_requests,
                    u.created_at,
                    u.subscription_until,
                    MIN(t.created_at) as first_payment_at,
                    COUNT(t.id) as payment_count
                FROM users u
                INNER JOIN payments t ON u.user_id = t.user_id
                    AND t.status = 'completed'
                GROUP BY u.user_id
                ORDER BY first_payment_at DESC
                LIMIT 100
            """)

            rows = await cursor.fetchall()
            await cursor.close()

            users = []
            for row in rows:
                time_to_conversion_days = (row[4] - row[2]) // 86400
                users.append({
                    'user_id': row[0],
                    'total_requests': row[1],
                    'registered_at': datetime.fromtimestamp(row[2]).strftime('%Y-%m-%d'),
                    'first_payment_at': datetime.fromtimestamp(row[4]).strftime('%Y-%m-%d'),
                    'time_to_conversion_days': time_to_conversion_days,
                    'payment_count': row[5],
                    'is_recurring': row[5] > 1
                })

            metrics = {
                'avg_time_to_conversion': sum(u['time_to_conversion_days'] for u in users) / max(len(users), 1),
                'recurring_customers': sum(1 for u in users if u['is_recurring']),
                'total_revenue': len(users) * 300
            }

            return UserSegment(
                segment_id='trial_converters',
                name='💰 Из триала в оплату',
                description='Успешно конвертировались в платных клиентов',
                user_count=len(users),
                users=users,
                metrics=metrics
            )

    async def _get_freeloaders(self) -> UserSegment:
        """Использовали trial и не купили"""

        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            week_ago = now - (7 * 86400)

            cursor = await conn.execute("""
                SELECT
                    u.user_id,
                    u.total_requests,
                    u.trial_remaining,
                    u.last_request_at,
                    u.created_at
                FROM users u
                LEFT JOIN payments t ON u.user_id = t.user_id
                WHERE u.trial_remaining < 3
                  AND u.subscription_until <= ?
                  AND t.id IS NULL
                  AND u.last_request_at < ?
                ORDER BY u.total_requests DESC
            """, (now, week_ago))

            rows = await cursor.fetchall()
            await cursor.close()

            users = []
            for row in rows:
                days_inactive = (now - row[3]) // 86400
                users.append({
                    'user_id': row[0],
                    'total_requests': row[1],
                    'trial_remaining': row[2],
                    'days_inactive': days_inactive,
                    'registered_at': datetime.fromtimestamp(row[4]).strftime('%Y-%m-%d')
                })

            return UserSegment(
                segment_id='freeloaders',
                name='🚫 Бесплатники',
                description='Использовали trial, не купили, неактивны',
                user_count=len(users),
                users=users,
                metrics={'potential_conversions': len(users)}
            )

    async def _get_new_users(self) -> UserSegment:
        """Новые пользователи за последние 7 дней"""

        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            week_ago = now - (7 * 86400)

            cursor = await conn.execute("""
                SELECT
                    u.user_id,
                    u.total_requests,
                    u.trial_remaining,
                    u.created_at,
                    u.subscription_until
                FROM users u
                WHERE u.created_at >= ?
                ORDER BY u.created_at DESC
            """, (week_ago,))

            rows = await cursor.fetchall()
            await cursor.close()

            users = []
            for row in rows:
                users.append({
                    'user_id': row[0],
                    'total_requests': row[1],
                    'trial_remaining': row[2],
                    'registered_at': datetime.fromtimestamp(row[3]).strftime('%Y-%m-%d %H:%M'),
                    'has_subscription': row[4] > now
                })

            metrics = {
                'already_paid': sum(1 for u in users if u['has_subscription']),
                'avg_requests': sum(u['total_requests'] for u in users) / max(len(users), 1)
            }

            return UserSegment(
                segment_id='new_users',
                name='🆕 Новые пользователи',
                description='Зарегистрировались за последние 7 дней',
                user_count=len(users),
                users=users,
                metrics=metrics
            )

    async def _get_subscription_plan_segments(self) -> dict[str, UserSegment]:
        """Разделение пользователей по текущему тарифному плану"""

        now = int(time.time())
        segments: dict[str, UserSegment] = {}

        async with self.db.pool.acquire() as conn:
            for plan_id, config in PLAN_SEGMENT_DEFS.items():
                cursor = await conn.execute(
                    """
                    SELECT
                        u.user_id,
                        u.total_requests,
                        u.subscription_until,
                        u.subscription_requests_balance,
                        u.subscription_last_purchase_at
                    FROM users u
                    WHERE u.subscription_plan = ?
                    ORDER BY (u.subscription_last_purchase_at IS NOT NULL) DESC,
                             u.subscription_last_purchase_at DESC,
                             u.user_id
                    LIMIT 50
                    """,
                    (plan_id,),
                )

                rows = await cursor.fetchall()
                await cursor.close()

                users: list[dict[str, Any]] = []
                total_requests = 0
                active_users = 0

                for user_id, total_reqs, subscription_until, balance, last_purchase in rows:
                    total_request_value = total_reqs or 0
                    subscription_until_value = subscription_until or 0
                    last_purchase_value = last_purchase or 0

                    if subscription_until_value and subscription_until_value >= now:
                        active_users += 1

                    total_requests += total_request_value

                    users.append({
                        'user_id': user_id,
                        'subscription_until': datetime.fromtimestamp(subscription_until_value).strftime('%Y-%m-%d') if subscription_until_value else '—',
                        'last_purchase': datetime.fromtimestamp(last_purchase_value).strftime('%Y-%m-%d') if last_purchase_value else '—',
                        'total_requests': total_request_value,
                        'requests_balance': balance if balance is not None else '—',
                    })

                user_count = len(users)
                avg_requests = round(total_requests / user_count, 1) if user_count else 0.0
                monthly_revenue = active_users * config['price']
                segment_key = f'plan_{plan_id}'

                segments[segment_key] = UserSegment(
                    segment_id=segment_key,
                    name=config['name'],
                    description=config['description'],
                    user_count=user_count,
                    users=users,
                    metrics={
                        'active_subscribers': active_users,
                        'avg_requests': avg_requests,
                        'monthly_revenue_estimate': f"{monthly_revenue:,}₽".replace(",", " "),
                    },
                )

        return segments

    async def get_conversion_metrics(self) -> ConversionMetrics:
        """Метрики конверсии trial -> paid"""

        async with self.db.pool.acquire() as conn:
            # Всего trial пользователей
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM users WHERE trial_remaining < 10"
            )
            total_trial = (await cursor.fetchone())[0]
            await cursor.close()

            # Конвертировались в paid
            cursor = await conn.execute("""
                SELECT
                    COUNT(DISTINCT u.user_id) as converted,
                    AVG(u.total_requests) as avg_requests,
                    AVG(t.created_at - u.created_at) as avg_time_to_conversion
                FROM users u
                INNER JOIN payments t ON u.user_id = t.user_id
                WHERE u.trial_remaining < 10
            """)
            row = await cursor.fetchone()
            await cursor.close()

            converted = row[0] or 0
            avg_requests = row[1] or 0
            avg_time = (row[2] or 0) / 86400  # в днях

            return ConversionMetrics(
                total_trial_users=total_trial,
                converted_to_paid=converted,
                conversion_rate=round((converted / max(total_trial, 1)) * 100, 2),
                avg_trial_requests_before_conversion=round(avg_requests, 1),
                avg_time_to_conversion_days=round(avg_time, 1)
            )

    async def get_churn_metrics(self, period_days: int = 30) -> ChurnMetrics:
        """Метрики оттока"""

        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            period_start = now - (period_days * 86400)

            # Истекшие подписки за период
            cursor = await conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN renewed.user_id IS NOT NULL THEN 1 ELSE 0 END) as renewed
                FROM users u
                LEFT JOIN (
                    SELECT DISTINCT user_id
                    FROM payments
                    WHERE created_at >= ? AND status = 'completed'
                ) renewed ON u.user_id = renewed.user_id
                WHERE u.subscription_until >= ?
                  AND u.subscription_until < ?
            """, (period_start, period_start, now))

            row = await cursor.fetchone()
            await cursor.close()

            total_expired = row[0] or 0
            renewed = row[1] or 0
            churned = total_expired - renewed

            return ChurnMetrics(
                total_expired=total_expired,
                renewed_count=renewed,
                churned_count=churned,
                retention_rate=round((renewed / max(total_expired, 1)) * 100, 2),
                avg_requests_before_churn=0.0,  # TODO: рассчитать
                churn_by_usage={}  # TODO: рассчитать
            )

    async def get_daily_stats(self, days: int = 7) -> list[dict[str, Any]]:
        """Ежедневная статистика за период"""

        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            period_start = now - (days * 86400)

            cursor = await conn.execute("""
                SELECT
                    DATE(created_at, 'unixepoch') as date,
                    COUNT(*) as requests,
                    COUNT(DISTINCT user_id) as active_users,
                    SUM(tokens_used) as total_tokens,
                    AVG(response_time_ms) as avg_response_time
                FROM requests
                WHERE created_at >= ?
                GROUP BY date
                ORDER BY date DESC
            """, (period_start,))

            rows = await cursor.fetchall()
            await cursor.close()

            return [
                {
                    'date': row[0],
                    'requests': row[1],
                    'active_users': row[2],
                    'total_tokens': row[3],
                    'avg_response_time_ms': int(row[4]) if row[4] else 0
                }
                for row in rows
            ]

    async def get_feature_usage_stats(self, days: int = 30) -> dict[str, int]:
        """Статистика использования функций (кнопок) документов"""

        try:
            async with self.db.pool.acquire() as conn:
                now = int(time.time())
                period_start = now - (days * 86400)

                # Используем request_type из таблицы requests (основной источник данных)
                cursor = await conn.execute("""
                    SELECT request_type, COUNT(*) as count
                    FROM requests
                    WHERE created_at >= ?
                      AND request_type NOT IN ('legal_question', 'command', 'unknown')
                      AND request_type IS NOT NULL
                    GROUP BY request_type
                    ORDER BY count DESC
                """, (period_start,))

                rows = await cursor.fetchall()
                await cursor.close()

                if rows:
                    return {row[0]: int(row[1]) for row in rows}

                # Fallback: если нет данных в requests, попробуем behavior_events
                try:
                    cursor = await conn.execute("""
                        SELECT feature, COUNT(*) as count
                        FROM behavior_events
                        WHERE timestamp >= ?
                          AND event_type = 'feature_use'
                        GROUP BY feature
                        ORDER BY count DESC
                    """, (period_start,))

                    rows = await cursor.fetchall()
                    await cursor.close()

                    if rows:
                        return {row[0]: int(row[1]) for row in rows}
                except Exception as behavior_exc:
                    logger.debug("behavior_events query failed: %s", behavior_exc)

                return {}
        except Exception as exc:
            logger.error("Failed to get feature usage stats: %s", exc, exc_info=True)
            return {}  # Возвращаем пустой словарь при ошибке

    def format_segment_summary(self, segment: UserSegment, max_users: int = 5) -> str:
        """Форматирование сегмента для вывода"""

        summary = f"<b>{segment.name}</b>\n"
        summary += f"<i>{segment.description}</i>\n\n"
        summary += f"📊 Всего пользователей: <b>{segment.user_count}</b>\n"

        if segment.metrics:
            summary += "\n<b>Метрики:</b>\n"
            for key, value in segment.metrics.items():
                label = METRIC_LABELS.get(key, key.replace('_', ' ').title())
                summary += f"• {label}: {value}\n"

        if segment.users:
            summary += f"\n<b>Топ-{min(max_users, len(segment.users))} пользователей:</b>\n"
            for user in segment.users[:max_users]:
                summary += f"\n👤 ID: <code>{user['user_id']}</code>\n"
                for k, v in list(user.items())[1:4]:  # первые 3 поля кроме ID
                    summary += f"   • {k}: {v}\n"

        return summary


# Static analysis hint: preserve dataclass field names used dynamically
_CHURN_METRIC_FIELDS = (
    ChurnMetrics.__annotations__["avg_requests_before_churn"],
    ChurnMetrics.__annotations__["churn_by_usage"],
)

__all__ = (
    "UserSegment",
    "ConversionMetrics",
    "ChurnMetrics",
    "AdminAnalytics",
    "PLAN_SEGMENT_DEFS",
    "PLAN_SEGMENT_ORDER",
)
