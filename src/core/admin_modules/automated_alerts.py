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
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable
import json

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

    async def check_all_alerts(self) -> list[Alert]:
        """
        Проверить все метрики и сгенерировать alerts

        Returns: Список alerts
        """
        alerts = []

        # Revenue alerts
        revenue_alerts = await self._check_revenue_alerts()
        alerts.extend(revenue_alerts)

        # Retention alerts
        retention_alerts = await self._check_retention_alerts()
        alerts.extend(retention_alerts)

        # PMF alerts
        pmf_alerts = await self._check_pmf_alerts()
        alerts.extend(pmf_alerts)

        # Technical alerts
        technical_alerts = await self._check_technical_alerts()
        alerts.extend(technical_alerts)

        return alerts

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
                    title="🔴 MRR Drop Detected",
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
                    title="🚨 High Churn Rate",
                    message=f"Customer churn rate: {current_mrr.customer_churn_rate:.1f}% ({current_mrr.churned_customers} users)",
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
                    title="⚠️ Low Quick Ratio",
                    message=f"Quick Ratio: {arr_metrics.quick_ratio:.2f} (target: >{self.config.quick_ratio_min})",
                    metric_value=arr_metrics.quick_ratio,
                    threshold=self.config.quick_ratio_min,
                    action_required="Фокус на уменьшении churn и увеличении expansion revenue",
                    timestamp=int(datetime.now().timestamp())
                ))

        except Exception as e:
            print(f"Error checking revenue alerts: {e}")

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
                        title="📉 Low Day-30 Retention",
                        message=f"Cohort {latest_cohort.cohort_month}: {latest_cohort.day_30_retention:.1f}% retention",
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
                    title="🔴 Power Users Churning",
                    message=f"{churned_power_users} power users ушли за последние 7 дней",
                    metric_value=churned_power_users,
                    threshold=self.config.power_user_churn_threshold,
                    action_required="СРОЧНО связаться с ушедшими power users, выяснить причины",
                    timestamp=int(datetime.now().timestamp())
                ))

        except Exception as e:
            print(f"Error checking retention alerts: {e}")

        return alerts

    async def _count_churned_power_users(self, days: int = 7) -> int:
        """Подсчитать сколько power users ушли"""
        async with self.db.pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT COUNT(*) as churned
                FROM users
                WHERE total_requests > 50
                  AND (
                      SELECT COUNT(*) FROM payments
                      WHERE payments.user_id = users.user_id AND status = 'completed'
                  ) >= 2
                  AND (strftime('%s', 'now') - last_active) BETWEEN ? AND ?
            """, (86400 * days, 86400 * (days + 30)))
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
                    title="🔴 Negative NPS",
                    message=f"NPS Score: {nps.nps_score:+.0f} (Detractors: {nps.detractor_rate:.1f}%)",
                    metric_value=nps.nps_score,
                    threshold=self.config.nps_min,
                    action_required="Survey detractors, identify main pain points",
                    timestamp=int(datetime.now().timestamp())
                ))

            if nps.previous_nps and (nps.nps_score < nps.previous_nps - self.config.nps_drop_threshold):
                alerts.append(Alert(
                    severity="warning",
                    category="pmf",
                    title="📉 NPS Declining",
                    message=f"NPS упал с {nps.previous_nps:+.0f} до {nps.nps_score:+.0f}",
                    metric_value=nps.nps_score - nps.previous_nps,
                    threshold=-self.config.nps_drop_threshold,
                    action_required="Проанализировать recent changes, user feedback",
                    timestamp=int(datetime.now().timestamp())
                ))

            # DAU/MAU stickiness
            usage = await self.pmf_metrics.get_usage_intensity()

            if usage.dau_mau_ratio < self.config.dau_mau_min:
                alerts.append(Alert(
                    severity="warning",
                    category="pmf",
                    title="⚠️ Low Stickiness",
                    message=f"DAU/MAU: {usage.dau_mau_ratio:.1f}% (target: >{self.config.dau_mau_min}%)",
                    metric_value=usage.dau_mau_ratio,
                    threshold=self.config.dau_mau_min,
                    action_required="Добавить daily engagement hooks, push notifications",
                    timestamp=int(datetime.now().timestamp())
                ))

        except Exception as e:
            print(f"Error checking PMF alerts: {e}")

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
                        title=f"🔴 High Error Rate: {engagement.feature_name}",
                        message=f"Success rate: {engagement.success_rate:.1f}% (uses: {engagement.total_uses})",
                        metric_value=engagement.success_rate,
                        threshold=self.config.feature_success_rate_min,
                        action_required=f"Check logs for {engagement.feature_name}, fix errors",
                        timestamp=int(datetime.now().timestamp())
                    ))

            # Friction points with high impact
            frictions = await self.behavior_tracker.identify_friction_points(days=7)

            for friction in frictions:
                if friction.impact_score > 80:
                    alerts.append(Alert(
                        severity="critical",
                        category="technical",
                        title=f"🚨 Critical Friction: {friction.location}",
                        message=f"Impact: {friction.impact_score}/100, {friction.affected_users} users affected",
                        metric_value=friction.impact_score,
                        threshold=80,
                        action_required=f"Fix {friction.friction_type} at {friction.location}",
                        timestamp=int(datetime.now().timestamp())
                    ))

        except Exception as e:
            print(f"Error checking technical alerts: {e}")

        return alerts

    async def send_alerts(self, alerts: list[Alert]):
        """
        Отправить alerts админам в Telegram

        Args:
            alerts: Список alerts для отправки
        """
        if not alerts:
            return

        # Group alerts by severity
        critical = [a for a in alerts if a.severity == "critical"]
        warnings = [a for a in alerts if a.severity == "warning"]
        info = [a for a in alerts if a.severity == "info"]

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
                    message = "⚠️ <b>Warnings</b>\n\n"
                    for alert in warnings:
                        message += self._format_alert(alert) + "\n"

                    await self.bot.send_message(admin_id, message, parse_mode="HTML")

                # Info (only if no critical/warnings)
                if info and not critical and not warnings:
                    message = "ℹ️ <b>Info</b>\n\n"
                    for alert in info:
                        message += self._format_alert(alert) + "\n"

                    await self.bot.send_message(admin_id, message, parse_mode="HTML")

            except Exception as e:
                print(f"Error sending alert to {admin_id}: {e}")

    def _format_alert(self, alert: Alert) -> str:
        """Форматировать alert для Telegram"""
        text = f"<b>{alert.title}</b>\n"
        text += f"{alert.message}\n"
        text += f"<i>Action: {alert.action_required}</i>\n"
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
                    print(f"[AutoAlerts] Sent {len(alerts)} alerts at {datetime.now()}")

            except Exception as e:
                print(f"Error in monitoring loop: {e}")

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
            text = "☀️ <b>Daily Metrics Digest</b>\n"
            text += f"📅 {datetime.now().strftime('%Y-%m-%d')}\n\n"

            # Revenue
            text += "<b>💰 Revenue</b>\n"
            text += f"  MRR: {mrr.total_mrr:,}₽ ({mrr.mrr_growth_rate:+.1f}%)\n"
            text += f"  Customers: {mrr.total_paying_customers}\n"
            text += f"  Churn: {mrr.customer_churn_rate:.1f}%\n\n"

            # PMF
            text += "<b>📊 Product-Market Fit</b>\n"
            text += f"  NPS: {nps.nps_score:+.0f}\n"
            text += f"  DAU: {usage.dau} | WAU: {usage.wau} | MAU: {usage.mau}\n"
            text += f"  Stickiness (DAU/MAU): {usage.dau_mau_ratio:.1f}%\n\n"

            # Retention
            if comparison.cohorts_data:
                latest = comparison.cohorts_data[0]
                text += "<b>🎯 Retention</b>\n"
                text += f"  Latest Cohort ({latest.cohort_month})\n"
                text += f"  Day 30: {latest.day_30_retention:.1f}%\n"
                text += f"  Conversion: {latest.conversion_rate:.1f}%\n\n"

            # Alerts
            alerts = await self.check_all_alerts()
            critical_count = len([a for a in alerts if a.severity == "critical"])

            if critical_count > 0:
                text += f"🚨 <b>{critical_count} Critical Alerts</b>\n"
                text += "Use /alerts to view details\n"

            # Send to admins
            for admin_id in self.admin_chat_ids:
                await self.bot.send_message(admin_id, text, parse_mode="HTML")

        except Exception as e:
            print(f"Error sending daily digest: {e}")


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
