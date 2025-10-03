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

        # 7. VIP - топ по платежам
        segments['vip'] = await self._get_vip_users()

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
                name='⚡ Power Users',
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
                name='⚠️ At Risk',
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
                name='📉 Churned',
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
                name='💰 Trial Converters',
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
                name='🚫 Freeloaders',
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
                name='🆕 New Users',
                description='Зарегистрировались за последние 7 дней',
                user_count=len(users),
                users=users,
                metrics=metrics
            )

    async def _get_vip_users(self) -> UserSegment:
        """VIP пользователи - топ по платежам"""

        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT
                    u.user_id,
                    u.total_requests,
                    u.subscription_until,
                    COUNT(t.id) as payment_count,
                    SUM(t.amount) as total_spent,
                    MIN(t.created_at) as first_payment,
                    MAX(t.created_at) as last_payment
                FROM users u
                INNER JOIN payments t ON u.user_id = t.user_id
                    AND t.status = 'completed'
                GROUP BY u.user_id
                HAVING payment_count >= 2
                ORDER BY total_spent DESC
                LIMIT 20
            """)

            rows = await cursor.fetchall()
            await cursor.close()

            users = []
            for row in rows:
                users.append({
                    'user_id': row[0],
                    'total_requests': row[1],
                    'subscription_until': datetime.fromtimestamp(row[2]).strftime('%Y-%m-%d'),
                    'payment_count': row[3],
                    'total_spent': row[4],
                    'first_payment': datetime.fromtimestamp(row[5]).strftime('%Y-%m-%d'),
                    'last_payment': datetime.fromtimestamp(row[6]).strftime('%Y-%m-%d')
                })

            return UserSegment(
                segment_id='vip',
                name='👑 VIP Users',
                description='Топ-20 по количеству платежей',
                user_count=len(users),
                users=users,
                metrics={'total_vip_revenue': sum(u['total_spent'] for u in users)}
            )

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

    def format_segment_summary(self, segment: UserSegment, max_users: int = 5) -> str:
        """Форматирование сегмента для вывода"""

        summary = f"<b>{segment.name}</b>\n"
        summary += f"<i>{segment.description}</i>\n\n"
        summary += f"📊 Всего пользователей: <b>{segment.user_count}</b>\n"

        if segment.metrics:
            summary += "\n<b>Метрики:</b>\n"
            for key, value in segment.metrics.items():
                summary += f"• {key.replace('_', ' ').title()}: {value}\n"

        if segment.users:
            summary += f"\n<b>Топ-{min(max_users, len(segment.users))} пользователей:</b>\n"
            for user in segment.users[:max_users]:
                summary += f"\n👤 ID: <code>{user['user_id']}</code>\n"
                for k, v in list(user.items())[1:4]:  # первые 3 поля кроме ID
                    summary += f"   • {k}: {v}\n"

        return summary

