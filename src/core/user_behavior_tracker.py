"""
Продвинутое отслеживание поведения пользователей для стартапа
Понимание что нравится, что не нравится, где отваливаются
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class UserBehaviorEvent:
    """Событие поведения пользователя"""

    user_id: int
    event_type: str  # 'feature_used', 'error', 'abandoned', 'success'
    feature: str  # 'voice', 'document', 'question', etc.
    timestamp: int
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    duration_ms: int | None = None
    success: bool = True


@dataclass
class FeatureEngagement:
    """Метрики вовлеченности по фиче"""

    feature_name: str
    total_uses: int
    unique_users: int
    avg_duration_ms: float
    success_rate: float
    abandonment_rate: float
    repeat_usage_rate: float  # % пользователей использовавших >1 раза
    satisfaction_score: float  # на основе ratings
    trend: str  # 'rising', 'stable', 'declining'


@dataclass
class UserJourney:
    """Путь пользователя через продукт"""

    user_id: int
    journey_steps: list[dict[str, Any]]
    drop_off_point: str | None
    completed: bool
    total_time_seconds: int
    friction_points: list[str]  # где пользователь застрял


@dataclass
class FrictionPoint:
    """Точка трения в продукте"""

    location: str  # 'payment_flow', 'document_upload', etc.
    friction_type: str  # 'error', 'timeout', 'abandon', 'confusion'
    affected_users: int
    avg_recovery_time_ms: float
    resolution_rate: float  # % кто преодолел трение
    impact_score: float  # насколько критично (0-100)


@dataclass
class FeatureFeedback:
    """Обратная связь по фиче"""

    feature: str
    positive_signals: int  # likes, продолжили использовать
    negative_signals: int  # dislikes, abandoned
    explicit_feedback: list[str]  # текстовые комментарии
    net_sentiment: float  # -100 to +100


class UserBehaviorTracker:
    """Система отслеживания поведения пользователей"""

    def __init__(self, db):
        self.db = db
        self.events_buffer: list[UserBehaviorEvent] = []
        self.buffer_size = 100

    # ==================== ОТСЛЕЖИВАНИЕ СОБЫТИЙ ====================

    async def track_feature_use(
        self,
        user_id: int,
        feature: str,
        duration_ms: int | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ):
        """Отслеживание использования фичи"""
        event = UserBehaviorEvent(
            user_id=user_id,
            event_type="feature_used",
            feature=feature,
            timestamp=int(time.time()),
            metadata=metadata or {},
            duration_ms=duration_ms,
            success=success,
        )

        self.events_buffer.append(event)

        # Сохраняем в БД если буфер заполнен
        if len(self.events_buffer) >= self.buffer_size:
            await self._flush_events()

        # Логируем для admin dashboard
        logger.info(
            f"Feature use: user={user_id}, feature={feature}, "
            f"duration={duration_ms}ms, success={success}"
        )

    async def track_abandonment(
        self, user_id: int, feature: str, stage: str, metadata: dict[str, Any] | None = None
    ):
        """Отслеживание когда пользователь бросил действие"""
        await self.track_feature_use(
            user_id=user_id,
            feature=feature,
            success=False,
            metadata={"stage": stage, "abandoned": True, **(metadata or {})},
        )

        logger.warning(f"Abandonment: user={user_id}, feature={feature}, stage={stage}")

    async def track_error(
        self, user_id: int, feature: str, error_type: str, error_message: str | None = None
    ):
        """Отслеживание ошибок"""
        await self.track_feature_use(
            user_id=user_id,
            feature=feature,
            success=False,
            metadata={"error": True, "error_type": error_type, "error_message": error_message},
        )

    async def track_user_journey_step(
        self,
        user_id: int,
        step_name: str,
        completed: bool = True,
        metadata: dict[str, Any] | None = None,
    ):
        """Отслеживание шагов пути пользователя"""
        async with self.db.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_journey_events
                (user_id, step_name, completed, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (user_id, step_name, 1 if completed else 0, str(metadata or {}), int(time.time())),
            )

    async def _flush_events(self):
        """Сохранение накопленных событий в БД"""
        if not self.events_buffer:
            return

        async with self.db.pool.acquire() as conn:
            payload = [
                (
                    event.user_id,
                    event.event_type,
                    event.feature,
                    event.timestamp,
                    str(event.metadata),
                    event.duration_ms,
                    1 if event.success else 0,
                )
                for event in self.events_buffer
            ]

            await conn.executemany(
                """
                INSERT INTO behavior_events
                (user_id, event_type, feature, timestamp, metadata, duration_ms, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

        logger.info(f"Flushed {len(self.events_buffer)} behavior events to DB")
        self.events_buffer.clear()

    # ==================== АНАЛИТИКА ФИЧЕЙ ====================

    async def get_feature_engagement(self, days: int = 30) -> list[FeatureEngagement]:
        """Метрики вовлеченности по всем фичам"""
        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            period_start = now - (days * 86400)

            cursor = await conn.execute(
                """
                SELECT
                    feature,
                    COUNT(*) as total_uses,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(duration_ms) as avg_duration,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    SUM(CASE WHEN metadata LIKE '%abandoned%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as abandonment_rate
                FROM behavior_events
                WHERE timestamp >= ?
                GROUP BY feature
                ORDER BY total_uses DESC
            """,
                (period_start,),
            )

            rows = await cursor.fetchall()
            await cursor.close()

            engagements = []
            for row in rows:
                # Рассчитываем repeat usage rate
                repeat_cursor = await conn.execute(
                    """
                    SELECT COUNT(DISTINCT user_id) as repeat_users
                    FROM (
                        SELECT user_id, COUNT(*) as use_count
                        FROM behavior_events
                        WHERE feature = ? AND timestamp >= ?
                        GROUP BY user_id
                        HAVING use_count > 1
                    )
                """,
                    (row[0], period_start),
                )
                repeat_users = (await repeat_cursor.fetchone())[0]
                await repeat_cursor.close()

                repeat_rate = (repeat_users / max(row[2], 1)) * 100

                # Получаем ratings для satisfaction
                rating_cursor = await conn.execute(
                    """
                    SELECT AVG(rating) as avg_rating
                    FROM ratings r
                    INNER JOIN requests req ON r.request_id = req.id
                    WHERE req.request_type = ?
                    AND r.created_at >= ?
                """,
                    (row[0], period_start),
                )
                rating_row = await rating_cursor.fetchone()
                await rating_cursor.close()

                satisfaction = ((rating_row[0] or 0) + 1) * 50  # normalize -1..1 to 0..100

                # Определяем тренд (сравнивая с предыдущим периодом)
                trend = await self._calculate_trend(row[0], period_start)

                engagements.append(
                    FeatureEngagement(
                        feature_name=row[0],
                        total_uses=row[1],
                        unique_users=row[2],
                        avg_duration_ms=row[3] or 0,
                        success_rate=row[4],
                        abandonment_rate=row[5],
                        repeat_usage_rate=repeat_rate,
                        satisfaction_score=satisfaction,
                        trend=trend,
                    )
                )

            return engagements

    async def _calculate_trend(self, feature: str, current_period_start: int) -> str:
        """Определение тренда использования фичи"""
        async with self.db.pool.acquire() as conn:
            # Текущий период
            current_cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM behavior_events
                WHERE feature = ? AND timestamp >= ?
            """,
                (feature, current_period_start),
            )
            current_count = (await current_cursor.fetchone())[0]
            await current_cursor.close()

            # Предыдущий период (той же длины)
            now = int(time.time())
            period_length = now - current_period_start
            prev_period_start = current_period_start - period_length

            prev_cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM behavior_events
                WHERE feature = ? AND timestamp >= ? AND timestamp < ?
            """,
                (feature, prev_period_start, current_period_start),
            )
            prev_count = (await prev_cursor.fetchone())[0]
            await prev_cursor.close()

            if prev_count == 0:
                return "new"

            change_pct = ((current_count - prev_count) / prev_count) * 100

            if change_pct > 20:
                return "rising"
            elif change_pct < -20:
                return "declining"
            else:
                return "stable"

    # ==================== ТОЧКИ ТРЕНИЯ ====================

    async def identify_friction_points(self, days: int = 30) -> list[FrictionPoint]:
        """Определение точек трения в продукте"""
        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            period_start = now - (days * 86400)

            # Точки с высоким % ошибок
            cursor = await conn.execute(
                """
                SELECT
                    feature,
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures,
                    SUM(CASE WHEN metadata LIKE '%abandoned%' THEN 1 ELSE 0 END) as abandonments,
                    COUNT(DISTINCT user_id) as affected_users,
                    AVG(duration_ms) as avg_duration
                FROM behavior_events
                WHERE timestamp >= ?
                GROUP BY feature
                HAVING failures > 0 OR abandonments > 0
                ORDER BY (failures + abandonments) DESC
            """,
                (period_start,),
            )

            rows = await cursor.fetchall()
            await cursor.close()

            friction_points = []
            for row in rows:
                feature, total, failures, abandonments, users, avg_duration = row

                failure_rate = (failures / total) * 100
                abandonment_rate = (abandonments / total) * 100

                # Определяем тип трения
                if failure_rate > 30:
                    friction_type = "error"
                elif abandonment_rate > 40:
                    friction_type = "abandon"
                elif avg_duration and avg_duration > 60000:  # >1 min
                    friction_type = "timeout"
                else:
                    friction_type = "confusion"

                # Resolution rate - сколько в итоге успешно завершили
                resolution_rate = ((total - failures - abandonments) / total) * 100

                # Impact score учитывает количество пользователей и серьезность
                impact_score = (users / 100) * 50 + (  # чем больше пользователей, тем выше impact
                    failure_rate + abandonment_rate
                ) / 2  # серьезность проблемы

                friction_points.append(
                    FrictionPoint(
                        location=feature,
                        friction_type=friction_type,
                        affected_users=users,
                        avg_recovery_time_ms=avg_duration or 0,
                        resolution_rate=resolution_rate,
                        impact_score=min(impact_score, 100),
                    )
                )

            return sorted(friction_points, key=lambda x: x.impact_score, reverse=True)

    # ==================== ПУТЬ ПОЛЬЗОВАТЕЛЯ ====================

    async def get_user_journey(self, user_id: int, session_id: str | None = None) -> UserJourney:
        """Получение пути пользователя через продукт"""
        async with self.db.pool.acquire() as conn:
            # Получаем все события пользователя
            cursor = await conn.execute(
                """
                SELECT event_type, feature, timestamp, metadata, success, duration_ms
                FROM behavior_events
                WHERE user_id = ?
                ORDER BY timestamp ASC
            """,
                (user_id,),
            )

            rows = await cursor.fetchall()
            await cursor.close()

            journey_steps = []
            friction_points = []
            drop_off_point = None

            for i, row in enumerate(rows):
                event_type, feature, timestamp, metadata, success, duration = row

                step = {
                    "step_number": i + 1,
                    "feature": feature,
                    "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                    "success": bool(success),
                    "duration_ms": duration,
                }

                journey_steps.append(step)

                # Определяем friction points
                if not success:
                    friction_points.append(feature)

                    # Если после неудачи нет активности в течение часа - это drop off
                    if i < len(rows) - 1:
                        next_timestamp = rows[i + 1][2]
                        if next_timestamp - timestamp > 3600:  # 1 hour
                            drop_off_point = feature
                            break
                    else:
                        drop_off_point = feature

            # Определяем completed (дошли до payment или active subscription)
            completed = any(
                step["feature"] in ["payment_success", "subscription_active"]
                for step in journey_steps
            )

            total_time = 0
            if journey_steps:
                total_time = (
                    datetime.fromisoformat(journey_steps[-1]["timestamp"]).timestamp()
                    - datetime.fromisoformat(journey_steps[0]["timestamp"]).timestamp()
                )

            return UserJourney(
                user_id=user_id,
                journey_steps=journey_steps,
                drop_off_point=drop_off_point,
                completed=completed,
                total_time_seconds=int(total_time),
                friction_points=list(set(friction_points)),
            )

    # ==================== ОБРАТНАЯ СВЯЗЬ ПО ФИЧАМ ====================

    async def get_feature_feedback(self, feature: str, days: int = 30) -> FeatureFeedback:
        """Анализ обратной связи по конкретной фиче"""
        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            period_start = now - (days * 86400)

            # Положительные сигналы: успехи, повторное использование
            pos_cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM behavior_events
                WHERE feature = ? AND timestamp >= ? AND success = 1
            """,
                (feature, period_start),
            )
            positive_signals = (await pos_cursor.fetchone())[0]
            await pos_cursor.close()

            # Негативные сигналы: ошибки, abandonment
            neg_cursor = await conn.execute(
                """
                SELECT COUNT(*) FROM behavior_events
                WHERE feature = ? AND timestamp >= ?
                AND (success = 0 OR metadata LIKE '%abandoned%')
            """,
                (feature, period_start),
            )
            negative_signals = (await neg_cursor.fetchone())[0]
            await neg_cursor.close()

            # Явные feedback (ratings)
            feedback_cursor = await conn.execute(
                """
                SELECT feedback_text FROM ratings r
                INNER JOIN requests req ON r.request_id = req.id
                WHERE req.request_type = ? AND r.created_at >= ?
                AND feedback_text IS NOT NULL
            """,
                (feature, period_start),
            )
            feedback_rows = await feedback_cursor.fetchall()
            await feedback_cursor.close()

            explicit_feedback = [row[0] for row in feedback_rows if row[0]]

            # Net sentiment
            total_signals = positive_signals + negative_signals
            if total_signals > 0:
                net_sentiment = ((positive_signals - negative_signals) / total_signals) * 100
            else:
                net_sentiment = 0.0

            return FeatureFeedback(
                feature=feature,
                positive_signals=positive_signals,
                negative_signals=negative_signals,
                explicit_feedback=explicit_feedback,
                net_sentiment=net_sentiment,
            )

    # ==================== ПОПУЛЯРНЫЕ ФИЧИ ====================

    async def get_top_features(self, days: int = 7, limit: int = 10) -> list[dict[str, Any]]:
        """Топ самых используемых фичей"""
        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            period_start = now - (days * 86400)

            cursor = await conn.execute(
                """
                SELECT
                    feature,
                    COUNT(*) as uses,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(duration_ms) as avg_duration,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM behavior_events
                WHERE timestamp >= ?
                GROUP BY feature
                ORDER BY uses DESC
                LIMIT ?
            """,
                (period_start, limit),
            )

            rows = await cursor.fetchall()
            await cursor.close()

            return [
                {
                    "feature": row[0],
                    "uses": row[1],
                    "unique_users": row[2],
                    "avg_duration_ms": int(row[3] or 0),
                    "success_rate": round(row[4], 2),
                }
                for row in rows
            ]

    # ==================== НЕИСПОЛЬЗУЕМЫЕ ФИЧИ ====================

    async def get_underutilized_features(self, days: int = 30) -> list[dict[str, Any]]:
        """Фичи которые доступны но не используются"""

        # Все доступные фичи (из конфига или списка)
        all_features = [
            "legal_question",
            "voice_message",
            "document_upload",
            "document_summary",
            "document_risks",
            "document_chat",
            "document_translate",
            "document_anonymize",
            "ocr",
            "judicial_practice",
            "document_draft",
        ]

        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            period_start = now - (days * 86400)

            # Получаем статистику по всем фичам
            cursor = await conn.execute(
                """
                SELECT feature, COUNT(*) as uses
                FROM behavior_events
                WHERE timestamp >= ?
                GROUP BY feature
            """,
                (period_start,),
            )

            rows = await cursor.fetchall()
            await cursor.close()

            usage_map = {row[0]: row[1] for row in rows}

            # Находим неиспользуемые или редко используемые
            underutilized = []
            for feature in all_features:
                uses = usage_map.get(feature, 0)
                if uses < 10:  # порог недоиспользования
                    underutilized.append(
                        {
                            "feature": feature,
                            "uses": uses,
                            "status": "unused" if uses == 0 else "underutilized",
                        }
                    )

            return underutilized

    # ==================== ВРЕМЯ ИСПОЛЬЗОВАНИЯ ====================

    async def get_usage_by_hour(self, days: int = 7) -> dict[int, dict[str, Any]]:
        """Распределение использования по часам суток"""
        async with self.db.pool.acquire() as conn:
            now = int(time.time())
            period_start = now - (days * 86400)

            cursor = await conn.execute(
                """
                SELECT
                    CAST(strftime('%H', timestamp, 'unixepoch') AS INTEGER) as hour,
                    COUNT(*) as total_events,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(duration_ms) as avg_duration
                FROM behavior_events
                WHERE timestamp >= ?
                GROUP BY hour
                ORDER BY hour
            """,
                (period_start,),
            )

            rows = await cursor.fetchall()
            await cursor.close()

            return {
                row[0]: {
                    "total_events": row[1],
                    "unique_users": row[2],
                    "avg_duration_ms": int(row[3] or 0),
                }
                for row in rows
            }

    # ==================== ФОРМАТИРОВАНИЕ ДЛЯ АДМИНА ====================

    def format_engagement_report(self, engagements: list[FeatureEngagement]) -> str:
        """Форматирование отчета о вовлеченности для админа"""
        if not engagements:
            return "<b>📊 Нет данных о вовлеченности</b>"

        report = "<b>📊 ВОВЛЕЧЕННОСТЬ ПО ФИЧАМ</b>\n\n"

        for eng in engagements[:10]:  # топ-10
            trend_emoji = {"rising": "📈", "stable": "➡️", "declining": "📉", "new": "🆕"}.get(
                eng.trend, "➡️"
            )

            satisfaction_emoji = (
                "😍"
                if eng.satisfaction_score > 75
                else "😊" if eng.satisfaction_score > 50 else "😐"
            )

            report += f"<b>{eng.feature_name}</b> {trend_emoji}\n"
            report += f"  • Использований: {eng.total_uses} ({eng.unique_users} польз.)\n"
            report += f"  • Успешность: {eng.success_rate:.1f}%\n"
            report += f"  • Повторное использование: {eng.repeat_usage_rate:.1f}%\n"
            report += (
                f"  • Удовлетворенность: {satisfaction_emoji} {eng.satisfaction_score:.0f}/100\n"
            )

            if eng.abandonment_rate > 20:
                report += f"  ⚠️ Высокий abandonment: {eng.abandonment_rate:.1f}%\n"

            report += "\n"

        return report

    def format_friction_report(self, frictions: list[FrictionPoint]) -> str:
        """Форматирование отчета о точках трения"""
        if not frictions:
            return "<b>✅ Точек трения не обнаружено!</b>"

        report = "<b>🔥 ТОЧКИ ТРЕНИЯ</b>\n\n"

        for friction in frictions[:5]:  # топ-5 самых критичных
            impact_emoji = (
                "🔴" if friction.impact_score > 70 else "🟡" if friction.impact_score > 40 else "🟢"
            )

            report += f"{impact_emoji} <b>{friction.location}</b>\n"
            report += f"  • Тип: {friction.friction_type}\n"
            report += f"  • Затронуто пользователей: {friction.affected_users}\n"
            report += f"  • Решают проблему: {friction.resolution_rate:.1f}%\n"
            report += f"  • Impact score: {friction.impact_score:.0f}/100\n\n"

        return report
