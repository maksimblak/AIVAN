"""
Retention Notifier - система автоматических напоминаний для неактивных пользователей
Помогает вернуть пользователей которые зарегистрировались но не используют бота
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiogram import Bot
    from src.core.db_advanced import DatabaseAdvanced

logger = logging.getLogger(__name__)


@dataclass
class NotificationTemplate:
    """Шаблон уведомления для разных сценариев"""
    name: str
    delay_hours: int
    message: str
    show_buttons: bool = True


# Сценарии уведомлений
NOTIFICATION_SCENARIOS = [
    NotificationTemplate(
        name="registered_no_request",
        delay_hours=24,  # через 24 часа после регистрации
        message=(
            "👋 <b>Привет!</b>\n\n"
            "Ты зарегистрировался в боте, но ещё не задал ни одного вопроса.\n\n"
            "💡 <b>Попробуй спросить:</b>\n"
            "• Что делать, если нарушили права?\n"
            "• Как правильно составить договор?\n"
            "• Какие есть права у арендатора?\n\n"
            "У тебя есть <b>10 бесплатных вопросов</b> — воспользуйся ими! 🎁"
        ),
        show_buttons=True
    ),
    NotificationTemplate(
        name="inactive_3days",
        delay_hours=72,  # через 3 дня неактивности
        message=(
            "🔔 <b>Давно не виделись!</b>\n\n"
            "Прошло уже 3 дня с момента твоего последнего вопроса.\n\n"
            "📚 <b>У меня для тебя есть:</b>\n"
            "• Консультации по любым правовым вопросам\n"
            "• Анализ документов и договоров\n"
            "• Поиск судебной практики\n"
            "• Голосовые ответы\n\n"
            "Возвращайся, если нужна помощь! 💼"
        ),
        show_buttons=True
    ),
    NotificationTemplate(
        name="inactive_7days",
        delay_hours=168,  # через 7 дней неактивности
        message=(
            "⚡ <b>Специальное предложение!</b>\n\n"
            "Не видел тебя уже неделю. Вот что нового:\n\n"
            "🆕 <b>Новые возможности:</b>\n"
            "• Распознание текста из фото\n"
            "• Составление документов\n"
            "• Работа с голосовыми сообщениями\n\n"
            "У тебя всё ещё есть бесплатные запросы — используй их! 🚀"
        ),
        show_buttons=True
    ),
]


class RetentionNotifier:
    """
    Сервис для отправки retention уведомлений неактивным пользователям

    Отслеживает:
    - Пользователей без запросов после регистрации (24ч)
    - Неактивных пользователей (3 дня, 7 дней)
    - Отписавшихся пользователей (для win-back кампаний)
    """

    def __init__(self, bot: Bot, db: DatabaseAdvanced):
        self.bot = bot
        self.db = db
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Запустить фоновую задачу отправки уведомлений"""
        if self._running:
            logger.warning("RetentionNotifier already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._notification_loop())
        logger.info("RetentionNotifier started")

    async def stop(self) -> None:
        """Остановить фоновую задачу"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("RetentionNotifier stopped")

    async def _notification_loop(self) -> None:
        """Основной цикл проверки и отправки уведомлений"""
        while self._running:
            try:
                # Проверяем каждый час
                await asyncio.sleep(3600)  # 1 час

                if not self._running:
                    break

                # Обрабатываем каждый сценарий
                for scenario in NOTIFICATION_SCENARIOS:
                    await self._process_scenario(scenario)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in retention notification loop: {e}", exc_info=True)
                await asyncio.sleep(300)  # 5 минут перед повтором при ошибке

    async def _process_scenario(self, scenario: NotificationTemplate) -> None:
        """Обработать один сценарий уведомлений"""
        try:
            # Получаем пользователей для этого сценария
            users = await self._get_users_for_scenario(scenario)

            if not users:
                return

            logger.info(f"Processing {len(users)} users for scenario '{scenario.name}'")

            # Отправляем уведомления
            sent_count = 0
            for user_id in users:
                try:
                    await self._send_notification(user_id, scenario)
                    await self._mark_notification_sent(user_id, scenario.name)
                    sent_count += 1

                    # Небольшая задержка между отправками
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"Failed to send notification to user {user_id}: {e}")

            logger.info(f"Sent {sent_count} notifications for scenario '{scenario.name}'")

        except Exception as e:
            logger.error(f"Error processing scenario '{scenario.name}': {e}", exc_info=True)

    async def _get_users_for_scenario(self, scenario: NotificationTemplate) -> list[int]:
        """Получить список пользователей для конкретного сценария"""
        now = int(time.time())
        delay_seconds = scenario.delay_hours * 3600

        async with self.db.pool.acquire() as conn:
            if scenario.name == "registered_no_request":
                # Зарегистрировались, но не задали ни одного вопроса
                cursor = await conn.execute("""
                    SELECT u.user_id
                    FROM users u
                    WHERE u.total_requests = 0
                      AND u.created_at < ?
                      AND u.created_at > ?
                      AND NOT EXISTS (
                          SELECT 1 FROM retention_notifications rn
                          WHERE rn.user_id = u.user_id
                            AND rn.scenario = ?
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM blocked_users bu
                          WHERE bu.user_id = u.user_id
                      )
                    LIMIT 100
                """, (now - delay_seconds, now - (delay_seconds + 3600), scenario.name))

            elif scenario.name.startswith("inactive_"):
                # Были активны, но сейчас неактивны N дней
                cursor = await conn.execute("""
                    SELECT u.user_id
                    FROM users u
                    WHERE u.total_requests > 0
                      AND u.last_request_at < ?
                      AND u.last_request_at > ?
                      AND NOT EXISTS (
                          SELECT 1 FROM retention_notifications rn
                          WHERE rn.user_id = u.user_id
                            AND rn.scenario = ?
                            AND rn.sent_at > ?
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM blocked_users bu
                          WHERE bu.user_id = u.user_id
                      )
                    LIMIT 100
                """, (
                    now - delay_seconds,
                    now - (delay_seconds + 3600),
                    scenario.name,
                    now - delay_seconds  # Не отправляем повторно в этом периоде
                ))

            else:
                logger.warning(f"Unknown scenario: {scenario.name}")
                return []

            rows = await cursor.fetchall()
            await cursor.close()

            return [row[0] for row in rows]

    async def _send_notification(self, user_id: int, scenario: NotificationTemplate) -> None:
        """Отправить уведомление пользователю"""
        try:
            # Проверяем, не заблокировал ли бот пользователь
            # (если заблокировал, получим исключение)

            if scenario.show_buttons:
                # Можно добавить inline кнопки для быстрого действия
                from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(
                        text="💬 Задать вопрос",
                        callback_data="quick_question"
                    )],
                    [InlineKeyboardButton(
                        text="📚 Все возможности",
                        callback_data="show_features"
                    )]
                ])

                await self.bot.send_message(
                    chat_id=user_id,
                    text=scenario.message,
                    parse_mode="HTML",
                    reply_markup=keyboard
                )
            else:
                await self.bot.send_message(
                    chat_id=user_id,
                    text=scenario.message,
                    parse_mode="HTML"
                )

            logger.info(f"Sent '{scenario.name}' notification to user {user_id}")

        except Exception as e:
            # Если пользователь заблокировал бота, логируем и пропускаем
            if "bot was blocked" in str(e).lower() or "user is deactivated" in str(e).lower():
                logger.info(f"User {user_id} blocked the bot, skipping")
                await self._mark_user_blocked(user_id)
            else:
                raise

    async def _mark_notification_sent(self, user_id: int, scenario_name: str) -> None:
        """Отметить что уведомление отправлено"""
        now = int(time.time())

        async with self.db.pool.acquire() as conn:
            # Создаём таблицу если её нет
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS retention_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    scenario TEXT NOT NULL,
                    sent_at INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # Добавляем запись
            await conn.execute("""
                INSERT INTO retention_notifications (user_id, scenario, sent_at)
                VALUES (?, ?, ?)
            """, (user_id, scenario_name, now))

            await conn.commit()

    async def _mark_user_blocked(self, user_id: int) -> None:
        """Отметить что пользователь заблокировал бота"""
        async with self.db.pool.acquire() as conn:
            # Создаём таблицу если её нет
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS blocked_users (
                    user_id INTEGER PRIMARY KEY,
                    blocked_at INTEGER NOT NULL
                )
            """)

            await conn.execute("""
                INSERT OR REPLACE INTO blocked_users (user_id, blocked_at)
                VALUES (?, ?)
            """, (user_id, int(time.time())))

            await conn.commit()

    # ==================== Ручные методы ====================

    async def send_manual_notification(
        self,
        user_ids: list[int],
        message: str,
        with_buttons: bool = False
    ) -> dict[str, int]:
        """
        Отправить ручное уведомление списку пользователей

        Args:
            user_ids: Список ID пользователей
            message: Текст сообщения (HTML)
            with_buttons: Показывать кнопки действий

        Returns:
            {"sent": N, "failed": M, "blocked": K}
        """
        stats = {"sent": 0, "failed": 0, "blocked": 0}

        for user_id in user_ids:
            try:
                if with_buttons:
                    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

                    keyboard = InlineKeyboardMarkup(inline_keyboard=[
                        [InlineKeyboardButton(
                            text="💬 Задать вопрос",
                            callback_data="quick_question"
                        )]
                    ])

                    await self.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode="HTML",
                        reply_markup=keyboard
                    )
                else:
                    await self.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode="HTML"
                    )

                stats["sent"] += 1
                await asyncio.sleep(0.5)  # Rate limiting

            except Exception as e:
                if "bot was blocked" in str(e).lower() or "user is deactivated" in str(e).lower():
                    stats["blocked"] += 1
                    await self._mark_user_blocked(user_id)
                else:
                    stats["failed"] += 1
                    logger.error(f"Failed to send manual notification to {user_id}: {e}")

        return stats

    async def get_notification_stats(self) -> dict:
        """Получить статистику по отправленным уведомлениям"""
        async with self.db.pool.acquire() as conn:
            # Общее количество отправленных
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM retention_notifications
            """)
            total_sent = (await cursor.fetchone())[0]
            await cursor.close()

            # По сценариям
            cursor = await conn.execute("""
                SELECT scenario, COUNT(*) as count
                FROM retention_notifications
                GROUP BY scenario
                ORDER BY count DESC
            """)
            by_scenario = {row[0]: row[1] for row in await cursor.fetchall()}
            await cursor.close()

            # Заблокированные пользователи
            cursor = await conn.execute("""
                SELECT COUNT(*) FROM blocked_users
            """)
            blocked_row = await cursor.fetchone()
            blocked_count = blocked_row[0] if blocked_row else 0
            await cursor.close()

            return {
                "total_sent": total_sent,
                "by_scenario": by_scenario,
                "blocked_users": blocked_count
            }


__all__ = ["RetentionNotifier", "NotificationTemplate", "NOTIFICATION_SCENARIOS"]
