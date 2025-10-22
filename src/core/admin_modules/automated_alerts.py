"""
🔔 Automated Alerts System

Автоматические алерты для критических метрик:
- MRR drops
- Churn spikes
- NPS declines
- Feature errors
- Retention issues
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
import logging
from typing import Any, Awaitable, Iterable
import time

logger = logging.getLogger(__name__)


from src.core.admin_modules.cohort_analytics import CohortAnalytics
from src.core.admin_modules.pmf_metrics import PMFMetrics
from src.core.admin_modules.revenue_analytics import RevenueAnalytics
from src.core.admin_modules.retention_analytics import RetentionAnalytics
from src.core.user_behavior_tracker import UserBehaviorTracker


@dataclass
class Alert:
    """Структура alert"""
    severity: str  # "critical", "warning", "info"
    category: str  # "revenue", "retention", "pmf", "technical"
    title: str
    message: str
    metric_value: Any
    threshold: Any
    action_required: str  # Что нужно сделать
    timestamp: int


def group_alerts_by_severity(alerts: Iterable["Alert"]) -> dict[str, list["Alert"]]:
    """Group alerts by severity for UI presentation."""
    groups: dict[str, list[Alert]] = defaultdict(list)
    for alert in alerts:
        groups[alert.severity].append(alert)
    for severity in ("critical", "warning", "info"):
        groups.setdefault(severity, [])
    return groups





@dataclass
class AlertConfig:
    """Конфигурация alert правил"""
    # Revenue alerts
    mrr_drop_threshold: float = 10.0  # % падение MRR
    churn_spike_threshold: float = 20.0  # % рост churn
    quick_ratio_min: float = 1.0

    # Retention alerts
    day_30_retention_min: float = 30.0  # %
    power_user_churn_threshold: int = 2  # Количество power users ушедших

    # PMF alerts
    nps_min: float = 0.0  # Минимальный NPS
    nps_drop_threshold: float = 10.0  # Падение NPS
    dau_mau_min: float = 10.0  # Минимальный DAU/MAU ratio

    # Technical alerts
    error_rate_threshold: float = 10.0  # % ошибок
    feature_success_rate_min: float = 80.0  # %
    # Garant API limits
    garant_min_remaining: int = 10  # Остаток вызовов, ниже которого шлём предупреждение

    # Cache
    alerts_cache_ttl_seconds: int = 60


class AutomatedAlerts:
    """
    Система автоматических алертов
    """

    def __init__(self, db, bot, admin_chat_ids: list[int], config: AlertConfig | None = None):
        self.db = db
        self.bot = bot
        self.admin_chat_ids = admin_chat_ids
        self.config = config or AlertConfig()

        # Analytics instances
        self.cohort_analytics = CohortAnalytics(db)
        self.pmf_metrics = PMFMetrics(db)
        self.revenue_analytics = RevenueAnalytics(db)
        self.retention_analytics = RetentionAnalytics(db)
        self.behavior_tracker = UserBehaviorTracker(db)

        self._alerts_cache: list[Alert] = []
        self._alerts_cache_timestamp: float = 0.0
        self._alerts_cache_lock = asyncio.Lock()

    async def check_all_alerts(self, force_refresh: bool = False) -> list[Alert]:
        """
        Проверить все метрики и сгенерировать alerts

        Args:
            force_refresh: игнорировать кэш и пересчитать метрики заново

        Returns: Список alerts
        """
        now = time.time()
        ttl = self.config.alerts_cache_ttl_seconds

        if not force_refresh and self._alerts_cache and (now - self._alerts_cache_timestamp) < ttl:
            return list(self._alerts_cache)

        async with self._alerts_cache_lock:
            now = time.time()
            if not force_refresh and self._alerts_cache and (now - self._alerts_cache_timestamp) < ttl:
                return list(self._alerts_cache)

            tasks = [
                self._run_check("revenue", self._check_revenue_alerts()),
                self._run_check("retention", self._check_retention_alerts()),
                self._run_check("pmf", self._check_pmf_alerts()),
                self._run_check("technical", self._check_technical_alerts()),
            ]

            results = await asyncio.gather(*tasks)

            alerts: list[Alert] = []
            for result in results:
                alerts.extend(result)

            self._alerts_cache = list(alerts)
            self._alerts_cache_timestamp = now
            return list(alerts)


    async def _run_check(self, name: str, coroutine: Awaitable[list[Alert]]) -> list[Alert]:
        """Запустить проверку метрик с логированием ошибок"""
        try:
            return await coroutine
        except Exception:
            logger.exception("Error checking %s alerts", name)
            return []

    async def _check_revenue_alerts(self) -> list[Alert]:
        """Проверка revenue метрик"""
        alerts = []

        try:
            # MRR drop
            current_mrr = await self.revenue_analytics.get_mrr_breakdown()

            if current_mrr.mrr_growth_rate < -self.config.mrr_drop_threshold:
                alerts.append(Alert(
                    severity="critical",
                    category="revenue",
                    title="🔴 Обнаружено падение MRR",
                    message=f"MRR упал на {abs(current_mrr.mrr_growth_rate):.1f}% в {current_mrr.month}",
                    metric_value=current_mrr.mrr_growth_rate,
                    threshold=-self.config.mrr_drop_threshold,
                    action_required="Проанализировать churn reasons, провести retention campaign",
                    timestamp=int(datetime.now().timestamp())
                ))

            # Churn spike
            if current_mrr.customer_churn_rate > self.config.churn_spike_threshold:
                alerts.append(Alert(
                    severity="critical",
                    category="revenue",
                    title="🚨 Высокий уровень оттока",
                    message=f"Уровень оттока клиентов: {current_mrr.customer_churn_rate:.1f}% ({current_mrr.churned_customers} пользователей)",
                    metric_value=current_mrr.customer_churn_rate,
                    threshold=self.config.churn_spike_threshold,
                    action_required="Запустить winback campaign, провести exit interviews",
                    timestamp=int(datetime.now().timestamp())
                ))

            # Quick Ratio too low
            arr_metrics = await self.revenue_analytics.get_arr_metrics()

            if arr_metrics.quick_ratio < self.config.quick_ratio_min:
                alerts.append(Alert(
                    severity="warning",
                    category="revenue",
                    title="⚠️ Низкий Quick Ratio",
                    message=f"Quick Ratio: {arr_metrics.quick_ratio:.2f} (цель: >{self.config.quick_ratio_min})",
                    metric_value=arr_metrics.quick_ratio,
                    threshold=self.config.quick_ratio_min,
                    action_required="Фокус на уменьшении churn и увеличении expansion revenue",
                    timestamp=int(datetime.now().timestamp())
                ))

        except Exception:
            logger.exception("Error checking revenue alerts")

        return alerts

    async def _check_retention_alerts(self) -> list[Alert]:
        """Проверка retention метрик"""
        alerts = []

        try:
            # Cohort retention
            comparison = await self.cohort_analytics.compare_cohorts(months_back=3)

            if comparison.cohorts_data:
                latest_cohort = comparison.cohorts_data[0]

                if latest_cohort.day_30_retention < self.config.day_30_retention_min:
                    alerts.append(Alert(
                        severity="warning",
                        category="retention",
                        title="📉 Низкое удержание на 30-й день",
                        message=f"Когорта {latest_cohort.cohort_month}: {latest_cohort.day_30_retention:.1f}% удержание",
                        metric_value=latest_cohort.day_30_retention,
                        threshold=self.config.day_30_retention_min,
                        action_required="Улучшить onboarding, добавить engagement hooks",
                        timestamp=int(datetime.now().timestamp())
                    ))

            # Power user churn
            churned_power_users = await self._count_churned_power_users(days=7)

            if churned_power_users >= self.config.power_user_churn_threshold:
                alerts.append(Alert(
                    severity="critical",
                    category="retention",
                    title="🔴 Уходят активные пользователи",
                    message=f"{churned_power_users} активных пользователей ушли за последние 7 дней",
                    metric_value=churned_power_users,
                    threshold=self.config.power_user_churn_threshold,
                    action_required="СРОЧНО связаться с ушедшими активными пользователями, выяснить причины",
                    timestamp=int(datetime.now().timestamp())
                ))

        except Exception:
            logger.exception("Error checking retention alerts")

        return alerts

    async def _count_churned_power_users(self, days: int = 7) -> int:
        """Подсчитать сколько power users ушли"""
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT COUNT(*) as churned
                FROM users u
                WHERE u.total_requests > 50
                  AND (
                      SELECT COUNT(*) FROM payments
                      WHERE payments.user_id = u.user_id AND status = 'completed'
                  ) >= 2
                  AND u.subscription_until < strftime('%s', 'now')
                  AND u.last_active IS NOT NULL
                  AND (strftime('%s', 'now') - u.last_active) BETWEEN 0 AND ?
            """, (86400 * days,))
            row = await cursor.fetchone()
            await cursor.close()

            return row[0] if row else 0

    async def _check_pmf_alerts(self) -> list[Alert]:
        """Проверка PMF метрик"""
        alerts = []

        try:
            # NPS drop
            nps = await self.pmf_metrics.get_nps(days=30)

            if nps.nps_score < self.config.nps_min:
                alerts.append(Alert(
                    severity="critical",
                    category="pmf",
                    title="🔴 Отрицательный NPS",
                    message=f"NPS Score: {nps.nps_score:+.0f} (Критики: {nps.detractor_rate:.1f}%)",
                    metric_value=nps.nps_score,
                    threshold=self.config.nps_min,
                    action_required="Опросить критиков, выявить основные проблемы",
                    timestamp=int(datetime.now().timestamp())
                ))

            if (nps.previous_nps is not None) and (nps.nps_score < nps.previous_nps - self.config.nps_drop_threshold):
                alerts.append(Alert(
                    severity="warning",
                    category="pmf",
                    title="📉 Падение NPS",
                    message=f"NPS упал с {nps.previous_nps:+.0f} до {nps.nps_score:+.0f}",
                    metric_value=nps.nps_score - nps.previous_nps,
                    threshold=-self.config.nps_drop_threshold,
                    action_required="Проанализировать недавние изменения, отзывы пользователей",
                    timestamp=int(datetime.now().timestamp())
                ))

            # DAU/MAU stickiness
            usage = await self.pmf_metrics.get_usage_intensity()

            if usage.dau_mau_ratio < self.config.dau_mau_min:
                alerts.append(Alert(
                    severity="warning",
                    category="pmf",
                    title="⚠️ Низкая вовлеченность",
                    message=f"DAU/MAU: {usage.dau_mau_ratio:.1f}% (цель: >{self.config.dau_mau_min}%)",
                    metric_value=usage.dau_mau_ratio,
                    threshold=self.config.dau_mau_min,
                    action_required="Добавить ежедневные механики вовлечения, push-уведомления",
                    timestamp=int(datetime.now().timestamp())
                ))

        except Exception:
            logger.exception("Error checking PMF alerts")

        return alerts

    async def _check_technical_alerts(self) -> list[Alert]:
        """Проверка technical метрик"""
        alerts = []

        try:
            # Feature error rates
            engagements = await self.behavior_tracker.get_feature_engagement(days=7)

            for engagement in engagements:
                if engagement.success_rate < self.config.feature_success_rate_min:
                    alerts.append(Alert(
                        severity="critical",
                        category="technical",
                        title=f"🔴 Высокий уровень ошибок: {engagement.feature_name}",
                        message=f"Успешность: {engagement.success_rate:.1f}% (использований: {engagement.total_uses})",
                        metric_value=engagement.success_rate,
                        threshold=self.config.feature_success_rate_min,
                        action_required=f"Проверить логи для {engagement.feature_name}, исправить ошибки",
                        timestamp=int(datetime.now().timestamp())
                    ))

            # Friction points with high impact
            frictions = await self.behavior_tracker.identify_friction_points(days=7)

            for friction in frictions:
                if friction.impact_score > 80:
                    alerts.append(Alert(
                        severity="critical",
                        category="technical",
                        title=f"🚨 Критическая проблема: {friction.location}",
                        message=f"Влияние: {friction.impact_score}/100, затронуто {friction.affected_users} пользователей",
                        metric_value=friction.impact_score,
                        threshold=80,
                        action_required=f"Исправить {friction.friction_type} в {friction.location}",
                        timestamp=int(datetime.now().timestamp())
                    ))

            # Garant API limits (diagnostics): предупреждать при низком остатке
            try:
                from src.core.bot_app import context as simple_context  # noqa: WPS433

                garant_client = getattr(simple_context, "garant_client", None)
                if getattr(garant_client, "enabled", False):
                    limits = await garant_client.get_limits()  # type: ignore[attr-defined]
                    warn_threshold = max(0, int(self.config.garant_min_remaining))
                    for item in limits or []:
                        # Если явный ноль — критично; если ниже порога — warning
                        if item.value <= 0:
                            alerts.append(Alert(
                                severity="critical",
                                category="technical",
                                title="🔴 ГАРАНТ: исчерпан лимит",
                                message=f"{item.title}: 0 оставшихся вызовов",
                                metric_value=item.value,
                                threshold=0,
                                action_required="Приостановить сценарии, увеличить квоту или подождать новый месяц",
                                timestamp=int(datetime.now().timestamp()),
                            ))
                        elif item.value <= warn_threshold:
                            alerts.append(Alert(
                                severity="warning",
                                category="technical",
                                title="⚠️ ГАРАНТ: низкий остаток",
                                message=f"{item.title}: {item.value} вызовов осталось",
                                metric_value=item.value,
                                threshold=warn_threshold,
                                action_required="Планировать экономию запросов или пополнить квоту",
                                timestamp=int(datetime.now().timestamp()),
                            ))
            except Exception:
                logger.debug("Garant limits check skipped", exc_info=True)

        except Exception:
            logger.exception("Error checking technical alerts")

        return alerts

    async def send_alerts(self, alerts: list[Alert]):
        """
        Отправить alerts админам в Telegram

        Args:
            alerts: Список alerts для отправки
        """
        if not alerts:
            return

        grouped = group_alerts_by_severity(alerts)
        critical = grouped.get("critical", [])
        warnings = grouped.get("warning", [])
        info = grouped.get("info", [])

        # Send to all admins
        for admin_id in self.admin_chat_ids:
            try:
                # Critical alerts first
                if critical:
                    message = "🚨 <b>CRITICAL ALERTS</b>\n\n"
                    for alert in critical:
                        message += self._format_alert(alert) + "\n"

                    await self.bot.send_message(admin_id, message, parse_mode="HTML")

                # Warnings
                if warnings:
                    message = "⚠️ <b>Предупреждения</b>\n\n"
                    for alert in warnings:
                        message += self._format_alert(alert) + "\n"

                    await self.bot.send_message(admin_id, message, parse_mode="HTML")

                # Info (only if no critical/warnings)
                if info and not critical and not warnings:
                    message = "ℹ️ <b>Информация</b>\n\n"
                    for alert in info:
                        message += self._format_alert(alert) + "\n"

                    await self.bot.send_message(admin_id, message, parse_mode="HTML")

            except Exception:
                logger.exception("Error sending alert to admin %s", admin_id)

    def _format_alert(self, alert: Alert) -> str:
        """Форматировать alert для Telegram"""
        title = html_escape(alert.title or "")
        message = html_escape(alert.message or "")
        action = html_escape(alert.action_required or "")
        text = f"<b>{title}</b>\n"
        text += f"{message}\n"
        text += f"<i>Действие: {action}</i>\n"
        return text

    async def monitoring_loop(self, check_interval_seconds: int = 3600):
        """
        Background task для периодической проверки alerts

        Args:
            check_interval_seconds: Интервал проверки (default: 1 час)
        """
        while True:
            try:
                alerts = await self.check_all_alerts()

                if alerts:
                    await self.send_alerts(alerts)

                    # Log alerts
                    logger.info("Sent %d alerts at %s", len(alerts), datetime.now())

            except Exception:
                logger.exception("Error in monitoring loop")

            await asyncio.sleep(check_interval_seconds)

    async def send_daily_digest(self):
        """
        Отправить ежедневный digest всех ключевых метрик

        Вызывается по расписанию (напр., каждое утро в 9:00)
        """
        try:
            # Собрать метрики
            mrr = await self.revenue_analytics.get_mrr_breakdown()
            nps = await self.pmf_metrics.get_nps(days=30)
            usage = await self.pmf_metrics.get_usage_intensity()
            comparison = await self.cohort_analytics.compare_cohorts(months_back=3)

            # Формирование digest
            text = "☀️ <b>Ежедневный отчет по метрикам</b>\n"
            text += f"📅 {datetime.now().strftime('%Y-%m-%d')}\n\n"

            # Revenue
            text += "<b>💰 Выручка</b>\n"
            text += f"  MRR: {mrr.total_mrr:,}₽ ({mrr.mrr_growth_rate:+.1f}%)\n"
            text += f"  Клиентов: {mrr.total_paying_customers}\n"
            text += f"  Отток: {mrr.customer_churn_rate:.1f}%\n\n"

            # PMF
            text += "<b>📊 Product-Market Fit</b>\n"
            text += f"  NPS: {nps.nps_score:+.0f}\n"
            text += f"  DAU: {usage.dau} | WAU: {usage.wau} | MAU: {usage.mau}\n"
            text += f"  Вовлеченность (DAU/MAU): {usage.dau_mau_ratio:.1f}%\n\n"

            # Retention
            if comparison.cohorts_data:
                latest = comparison.cohorts_data[0]
                text += "<b>🎯 Удержание</b>\n"
                text += f"  Последняя когорта ({latest.cohort_month})\n"
                text += f"  День 30: {latest.day_30_retention:.1f}%\n"
                text += f"  Конверсия: {latest.conversion_rate:.1f}%\n\n"

            # Alerts
            alerts = await self.check_all_alerts()
            critical_count = len(group_alerts_by_severity(alerts).get("critical", []))

            if critical_count > 0:
                text += f"🚨 <b>{critical_count} критических алертов</b>\n"
                text += "Используйте /alerts для просмотра деталей\n"

            # Send to admins
            for admin_id in self.admin_chat_ids:
                await self.bot.send_message(admin_id, text, parse_mode="HTML")

        except Exception:
            logger.exception("Error sending daily digest")


# Helper function для запуска monitoring в background
async def start_monitoring(db, bot, admin_chat_ids: list[int], config: AlertConfig | None = None):
    """
    Запустить background monitoring task

    Usage in main():
    ```python
    # Create task
    alert_system = AutomatedAlerts(db, bot, admin_chat_ids=[ADMIN_ID])
    asyncio.create_task(alert_system.monitoring_loop(check_interval_seconds=3600))
    ```
    """
    alert_system = AutomatedAlerts(db, bot, admin_chat_ids, config)
    await alert_system.monitoring_loop()

__all__ = (
    "AutomatedAlerts",
    "start_monitoring",
)
