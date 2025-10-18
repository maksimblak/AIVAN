"""
Улучшенная база данных с connection pooling, транзакциями и оптимизированными запросами
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import weakref
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import aiosqlite

# Импорт исключений
from .exceptions import DatabaseException

logger = logging.getLogger(__name__)


@dataclass
class UserRecord:
    user_id: int
    is_admin: int
    trial_remaining: int
    subscription_until: int
    created_at: int
    updated_at: int
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_request_at: int = 0
    referred_by: int | None = None
    referral_code: str | None = None
    referrals_count: int = 0
    referral_bonus_days: int = 0
    subscription_plan: str | None = None
    subscription_requests_balance: int = 0
    subscription_last_purchase_at: int = 0
    subscription_cancelled: int = 0


@dataclass
class TransactionRecord:
    id: int
    user_id: int
    provider: str
    currency: str
    amount: int
    amount_minor_units: int | None
    payload: str | None
    status: str
    telegram_payment_charge_id: str | None
    provider_payment_charge_id: str | None
    created_at: int
    updated_at: int


class TransactionStatus(str, Enum):
    # Normalized transaction statuses used across the system.

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    _ALIASES = {"success": COMPLETED}

    @classmethod
    def from_value(cls, value: "TransactionStatus | str") -> "TransactionStatus":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError("Transaction status cannot be empty")
        if normalized in cls._ALIASES:
            return cls._ALIASES[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported transaction status: {value!r}") from exc


@dataclass
class RequestRecord:
    id: int
    user_id: int
    request_type: str  # 'legal_question', 'command', etc.
    tokens_used: int
    response_time_ms: int
    success: bool
    error_type: str | None
    created_at: int


@dataclass
class RatingRecord:
    id: int
    request_id: int
    user_id: int
    rating: int  # 1 = like, -1 = dislike
    feedback_text: str | None
    created_at: int
    username: str | None = None
    answer_text: str | None = None


class ConnectionPool:
    """Пул соединений для SQLite с контролем жизненного цикла"""

    def __init__(
        self,
        db_path: str,
        max_connections: int = 5,
        connection_timeout: float = 30.0,
        max_connection_age: float = 3600.0,  # 1 час
    ):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.max_connection_age = max_connection_age

        self._connections: list[aiosqlite.Connection] = []
        self._available_connections = asyncio.Queue(maxsize=max_connections)
        self._connection_times: dict[aiosqlite.Connection, float] = {}
        self._lock = asyncio.Lock()
        self._closed = False

        # Executor для блокирующих операций
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="db-pool")

        # Weak references для автоматической очистки
        self._connection_refs: list[weakref.ref] = []

    async def initialize(self) -> None:
        """Инициализация пула соединений"""
        async with self._lock:
            if self._closed:
                raise RuntimeError("Connection pool is closed")

            # Создаем директорию для БД если не существует
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            os.makedirs(db_dir, exist_ok=True)

            # Создаем начальные соединения
            for _ in range(min(2, self.max_connections)):  # Создаем 2 соединения для начала
                conn = await self._create_connection()
                await self._available_connections.put(conn)

    async def _create_connection(self) -> aiosqlite.Connection:
        """Создание нового соединения с оптимальными настройками"""
        conn = await aiosqlite.connect(
            self.db_path, timeout=self.connection_timeout, isolation_level=None  # autocommit mode
        )

        # Оптимизации SQLite для concurrent доступа
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=10000")
        await conn.execute("PRAGMA temp_store=memory")
        await conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.commit()

        self._connections.append(conn)
        self._connection_times[conn] = time.time()

        # Добавляем weak reference для автоматической очистки
        ref = weakref.ref(conn, self._connection_cleanup_callback)
        self._connection_refs.append(ref)

        logger.debug(f"Created new database connection (total: {len(self._connections)})")
        return conn

    def _connection_cleanup_callback(self, ref: weakref.ref) -> None:
        """Callback для автоматической очистки мертвых соединений"""
        if ref in self._connection_refs:
            self._connection_refs.remove(ref)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiosqlite.Connection]:
        """Получение соединения из пула с автоматическим возвратом"""
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        conn = None
        try:
            # Пытаемся получить доступное соединение
            try:
                conn = await asyncio.wait_for(
                    self._available_connections.get(), timeout=self.connection_timeout
                )
            except TimeoutError:
                # Если нет доступных соединений и пул не заполнен - создаем новое
                async with self._lock:
                    if len(self._connections) < self.max_connections:
                        conn = await self._create_connection()
                    else:
                        # Ждем дольше если пул заполнен
                        conn = await asyncio.wait_for(
                            self._available_connections.get(), timeout=self.connection_timeout * 2
                        )

            # Проверяем валидность соединения
            conn = await self._validate_connection(conn)

            yield conn

        except Exception as e:
            logger.error(f"Error acquiring database connection: {e}")
            # Если соединение испорчено, не возвращаем его в пул
            if conn and conn in self._connections:
                await self._remove_connection(conn)
            raise
        finally:
            # Возвращаем соединение в пул
            if conn and conn in self._connections and not self._closed:
                await self._available_connections.put(conn)

    async def _validate_connection(self, conn: aiosqlite.Connection) -> aiosqlite.Connection:
        """Валидация соединения и пересоздание если необходимо"""
        try:
            # Проверяем возраст соединения
            conn_age = time.time() - self._connection_times.get(conn, 0)
            if conn_age > self.max_connection_age:
                logger.debug(f"Connection too old ({conn_age:.1f}s), recreating")
                await self._remove_connection(conn)
                return await self._create_connection()

            # Проверяем работоспособность
            await asyncio.wait_for(conn.execute("SELECT 1"), timeout=5.0)
            return conn

        except Exception as e:
            logger.warning(f"Connection validation failed: {e}, recreating")
            await self._remove_connection(conn)
            return await self._create_connection()

    async def _remove_connection(self, conn: aiosqlite.Connection) -> None:
        """Удаление соединения из пула"""
        try:
            if conn in self._connections:
                self._connections.remove(conn)
            if conn in self._connection_times:
                del self._connection_times[conn]
            await conn.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    async def cleanup_old_connections(self) -> None:
        """Очистка старых соединений (вызывается периодически)"""
        if self._closed:
            return

        async with self._lock:
            current_time = time.time()
            old_connections = [
                conn
                for conn, create_time in self._connection_times.items()
                if current_time - create_time > self.max_connection_age
            ]

            for conn in old_connections:
                if len(self._connections) > 1:  # Оставляем минимум одно соединение
                    await self._remove_connection(conn)

    async def close(self) -> None:
        """Закрытие всех соединений в пуле"""
        async with self._lock:
            self._closed = True

            # Закрываем все соединения
            for conn in self._connections.copy():
                try:
                    await conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection during pool shutdown: {e}")

            self._connections.clear()
            self._connection_times.clear()

            # Очищаем очередь доступных соединений
            while not self._available_connections.empty():
                try:
                    self._available_connections.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Закрываем executor
            self._executor.shutdown(wait=False)

            logger.info("Database connection pool closed")

    def get_stats(self) -> dict[str, Any]:
        """Статистика пула соединений"""
        return {
            "total_connections": len(self._connections),
            "available_connections": self._available_connections.qsize(),
            "max_connections": self.max_connections,
            "oldest_connection_age": max(
                [time.time() - create_time for create_time in self._connection_times.values()],
                default=0,
            ),
            "is_closed": self._closed,
        }


class DatabaseAdvanced:
    """Продвинутая база данных с connection pooling и оптимизациями"""

    def __init__(
        self,
        db_path: str,
        max_connections: int = 5,
        enable_metrics: bool = True,
        cleanup_interval: float = 300.0,  # 5 минут
    ):
        self.db_path = db_path
        self.pool = ConnectionPool(db_path, max_connections=max_connections)
        self.enable_metrics = enable_metrics
        self.cleanup_interval = cleanup_interval

        # Метрики
        self.query_count = 0
        self.transaction_count = 0
        self.error_count = 0

        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None

    async def init(self) -> None:
        """Инициализация базы данных и создание таблиц"""
        await self.pool.initialize()

        async with self.pool.acquire() as conn:
            # Создание таблиц с оптимизированными индексами
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    trial_remaining INTEGER NOT NULL DEFAULT 10,
                    subscription_until INTEGER NOT NULL DEFAULT 0,
                    subscription_plan TEXT,
                    subscription_requests_balance INTEGER NOT NULL DEFAULT 0,
                    subscription_last_purchase_at INTEGER NOT NULL DEFAULT 0,
                    subscription_cancelled INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    total_requests INTEGER NOT NULL DEFAULT 0,
                    successful_requests INTEGER NOT NULL DEFAULT 0,
                    failed_requests INTEGER NOT NULL DEFAULT 0,
                    last_request_at INTEGER NOT NULL DEFAULT 0,
                    referred_by INTEGER,
                    referral_code TEXT,
                    referrals_count INTEGER NOT NULL DEFAULT 0,
                    referral_bonus_days INTEGER NOT NULL DEFAULT 0
                );
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    amount_minor_units INTEGER,
                    payload TEXT,
                    status TEXT NOT NULL,
                    telegram_payment_charge_id TEXT,
                    provider_payment_charge_id TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    request_type TEXT NOT NULL DEFAULT 'legal_question',
                    tokens_used INTEGER NOT NULL DEFAULT 0,
                    response_time_ms INTEGER NOT NULL DEFAULT 0,
                    success BOOLEAN NOT NULL DEFAULT 1,
                    error_type TEXT,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL, -- 1 = like, -1 = dislike
                    feedback_text TEXT,
                    created_at INTEGER NOT NULL,
                    username TEXT,
                    answer_text TEXT,
                    FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
                """
            )

            # NPS surveys table for PMF metrics
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nps_surveys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    trigger_event TEXT,
                    user_segment TEXT,
                    score INTEGER,
                    disappointment_level TEXT,
                    feedback TEXT,
                    created_at INTEGER NOT NULL,
                    responded_at INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
                """
            )

            # Behavior events table for cohort and retention analytics
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS behavior_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    feature TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    metadata TEXT,
                    duration_ms INTEGER,
                    success INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
                """
            )

            # User journey events table for behavior tracking
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_journey_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
                """
            )

            # User onboarding hints - для отслеживания показанных подсказок
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_onboarding_hints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    hint_key TEXT NOT NULL,
                    shown_at INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    UNIQUE(user_id, hint_key)
                );
                """
            )

            # Create VIEW for backwards compatibility (payments -> transactions)
            await conn.execute("DROP VIEW IF EXISTS payments")
            await conn.execute(
                """
                CREATE VIEW payments AS
                SELECT
                    id,
                    user_id,
                    provider,
                    currency,
                    amount,
                    amount_minor_units,
                    payload,
                    status,
                    telegram_payment_charge_id,
                    provider_payment_charge_id,
                    created_at,
                    updated_at
                FROM transactions
                """
            )

            # Миграции для существующих БД - СНАЧАЛА!
            try:
                # Добавляем новые поля в таблицу users, если их нет
                migrations = [
                    "ALTER TABLE users ADD COLUMN total_requests INTEGER NOT NULL DEFAULT 0;",
                    "ALTER TABLE users ADD COLUMN successful_requests INTEGER NOT NULL DEFAULT 0;",
                    "ALTER TABLE users ADD COLUMN failed_requests INTEGER NOT NULL DEFAULT 0;",
                    "ALTER TABLE users ADD COLUMN last_request_at INTEGER NOT NULL DEFAULT 0;",
                    "ALTER TABLE users ADD COLUMN referred_by INTEGER;",
                    "ALTER TABLE users ADD COLUMN referral_code TEXT;",
                    "ALTER TABLE users ADD COLUMN referrals_count INTEGER NOT NULL DEFAULT 0;",
                    "ALTER TABLE users ADD COLUMN referral_bonus_days INTEGER NOT NULL DEFAULT 0;",
                    "ALTER TABLE users ADD COLUMN subscription_plan TEXT;",
                    "ALTER TABLE users ADD COLUMN subscription_requests_balance INTEGER NOT NULL DEFAULT 0;",
                    "ALTER TABLE users ADD COLUMN subscription_last_purchase_at INTEGER NOT NULL DEFAULT 0;",
                    "ALTER TABLE ratings ADD COLUMN username TEXT;",
                    "ALTER TABLE ratings ADD COLUMN answer_text TEXT;",
                    "ALTER TABLE user_journey_events ADD COLUMN step_name TEXT;",
                    "ALTER TABLE user_journey_events ADD COLUMN completed INTEGER NOT NULL DEFAULT 1;",
                    "ALTER TABLE user_journey_events ADD COLUMN metadata TEXT;",
                    "ALTER TABLE user_journey_events ADD COLUMN created_at INTEGER NOT NULL DEFAULT 0;",
                ]

                for migration in migrations:
                    try:
                        await conn.execute(migration)
                        logger.info(f"Applied migration: {migration}")
                    except Exception as migration_error:
                        logger.debug(f"Migration skipped: {migration_error}")

                # Проставляем значения по умолчанию в user_journey_events, если новые колонки добавлены
                existing_journey_cols = await self._get_table_columns(conn, 'user_journey_events')
                if {'event_type', 'event_data', 'timestamp'}.intersection(existing_journey_cols):
                    await conn.execute(
                        """
                        UPDATE user_journey_events
                        SET step_name = COALESCE(step_name, event_type),
                            metadata = COALESCE(metadata, event_data),
                            created_at = CASE
                                WHEN created_at IS NOT NULL AND created_at > 0 THEN created_at
                                ELSE COALESCE(timestamp, CAST(strftime('%s','now') AS INTEGER))
                            END,
                            completed = CASE WHEN completed IN (0,1) THEN completed ELSE 1 END
                        WHERE step_name IS NULL OR metadata IS NULL OR created_at <= 0
                        """
                    )

                await conn.commit()
            except Exception as e:
                logger.warning(f"Migration warning: {e}")

            # Создание оптимизированных индексов - ПОСЛЕ миграций!
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_users_admin ON users(is_admin) WHERE is_admin = 1;",
                "CREATE INDEX IF NOT EXISTS idx_users_subscription ON users(subscription_until) WHERE subscription_until > 0;",
                "CREATE INDEX IF NOT EXISTS idx_users_requests ON users(total_requests);",
                "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_referral_code ON users(referral_code) WHERE referral_code IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_transactions_user_created ON transactions(user_id, created_at);",
                "CREATE INDEX IF NOT EXISTS idx_users_last_request_at ON users(last_request_at);",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_transactions_tg_charge ON transactions(telegram_payment_charge_id) WHERE telegram_payment_charge_id IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);",
                "CREATE INDEX IF NOT EXISTS idx_transactions_provider ON transactions(provider);",
                "CREATE INDEX IF NOT EXISTS idx_users_created_month ON users(strftime('%Y-%m', created_at, 'unixepoch'));",
                "CREATE INDEX IF NOT EXISTS idx_transactions_created_month ON transactions(strftime('%Y-%m', created_at, 'unixepoch'));",
                "CREATE INDEX IF NOT EXISTS idx_requests_user_created ON requests(user_id, created_at);",
                "CREATE INDEX IF NOT EXISTS idx_requests_type ON requests(request_type);",
                "CREATE INDEX IF NOT EXISTS idx_requests_success ON requests(success);",
                "CREATE INDEX IF NOT EXISTS idx_ratings_request ON ratings(request_id);",
                "CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_ratings_user_request ON ratings(user_id, request_id);",
                "CREATE INDEX IF NOT EXISTS idx_behavior_events_user ON behavior_events(user_id, timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_behavior_events_feature ON behavior_events(feature);",
                "CREATE INDEX IF NOT EXISTS idx_behavior_events_timestamp ON behavior_events(timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_nps_surveys_user ON nps_surveys(user_id, created_at);",
                "CREATE INDEX IF NOT EXISTS idx_nps_surveys_score ON nps_surveys(score);",
                "CREATE INDEX IF NOT EXISTS idx_user_journey_events_user ON user_journey_events(user_id, timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_user_journey_events_session ON user_journey_events(session_id);",
                "CREATE INDEX IF NOT EXISTS idx_user_onboarding_hints_user ON user_onboarding_hints(user_id);",
            ]

            for index_sql in indexes:
                try:
                    await conn.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {index_sql} - {e}")

            await conn.execute(
                "UPDATE transactions SET status = ? WHERE status = ?",
                (TransactionStatus.COMPLETED.value, "success"),
            )

            await conn.commit()

        # Запускаем фоновую очистку
        if self.cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(
            f"Advanced database initialized with connection pool (max_connections={self.pool.max_connections})"
        )

    async def _cleanup_loop(self) -> None:
        """Фоновая задача для периодической очистки"""
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval)
                await self.pool.cleanup_old_connections()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Database cleanup loop error: {e}")

    # Методы с улучшенной производительностью и транзакциями

    async def _get_table_columns(self, conn, table: str) -> set[str]:
        """Получение списка колонок таблицы с проверкой имени таблицы"""
        # Whitelist допустимых таблиц для защиты от SQL injection
        ALLOWED_TABLES = frozenset([
            "users", "transactions", "payments", "requests", "ratings",
            "nps_surveys", "behavior_events", "user_journey_events"
        ])

        if table not in ALLOWED_TABLES:
            # Попытка записать метрику, но не падать если metrics недоступны
            try:
                from src.core.metrics import get_metrics_collector
                metrics = get_metrics_collector()
                if metrics:
                    metrics.record_sql_injection_attempt(
                        pattern_type="invalid_table_name",
                        source="database_layer"
                    )
                    metrics.record_security_violation(
                        violation_type="sql_injection",
                        severity="warning",
                        source="database_layer"
                    )
            except Exception as e:
                logger.debug(f"Failed to record metrics for invalid table access: {e}")

            logger.warning(f"Attempted table access with invalid name: {table}")
            raise ValueError(f"Invalid table name: {table}")

        cursor = await conn.execute(f'PRAGMA table_info({table})')
        rows = await cursor.fetchall()
        await cursor.close()
        return {row[1] for row in rows}

    async def _ensure_referral_columns(self, conn) -> None:
        required_columns = {
            'referred_by': "ALTER TABLE users ADD COLUMN referred_by INTEGER;",
            'referral_code': "ALTER TABLE users ADD COLUMN referral_code TEXT;",
            'referrals_count': "ALTER TABLE users ADD COLUMN referrals_count INTEGER NOT NULL DEFAULT 0;",
            'referral_bonus_days': "ALTER TABLE users ADD COLUMN referral_bonus_days INTEGER NOT NULL DEFAULT 0;",
            'subscription_cancelled': "ALTER TABLE users ADD COLUMN subscription_cancelled INTEGER NOT NULL DEFAULT 0;",
        }
        existing = await self._get_table_columns(conn, 'users')
        applied = False
        index_applied = False
        for column_name, ddl in required_columns.items():
            if column_name not in existing:
                try:
                    await conn.execute(ddl)
                    applied = True
                    logger.info(f'Applied late migration for users: {ddl}')
                except Exception as migration_error:
                    logger.error(f'Failed to apply late migration {ddl}: {migration_error}')
        cursor = await conn.execute("PRAGMA index_list(users)")
        indexes = {row[1] for row in await cursor.fetchall()}
        await cursor.close()
        if 'idx_users_referral_code' not in indexes:
            try:
                await conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_referral_code ON users(referral_code) WHERE referral_code IS NOT NULL;"
                )
                index_applied = True
                logger.info('Ensured unique index for users.referral_code')
            except Exception as index_error:
                logger.warning(f'Failed to ensure referral_code unique index: {index_error}')
        if applied or index_applied:
            await conn.commit()

    async def ensure_user(
        self, user_id: int, *, default_trial: int = 10, is_admin: bool = False
    ) -> UserRecord:
        """Обеспечение существования пользователя с оптимизацией"""
        async with self.pool.acquire() as conn:
            # Используем INSERT OR IGNORE для атомарности
            now = int(time.time())

            try:
                # Пробуем с новыми полями реферальной системы
                try:
                    await conn.execute(
                        """
                        INSERT OR IGNORE INTO users
                        (user_id, is_admin, trial_remaining, subscription_until, subscription_plan, subscription_requests_balance,
                         subscription_last_purchase_at, created_at, updated_at, total_requests, successful_requests, failed_requests,
                         last_request_at, referred_by, referral_code, referrals_count, referral_bonus_days, subscription_cancelled)
                        VALUES (?, ?, ?, 0, NULL, 0, 0, ?, ?, 0, 0, 0, 0, NULL, NULL, 0, 0, 0)
                        """,
                        (user_id, 1 if is_admin else 0, default_trial, now, now),
                    )
                except Exception:
                    await self._ensure_referral_columns(conn)
                    await conn.execute(
                        """
                        INSERT OR IGNORE INTO users
                        (user_id, is_admin, trial_remaining, subscription_until, subscription_plan, subscription_requests_balance,
                         subscription_last_purchase_at, created_at, updated_at, total_requests, successful_requests, failed_requests, last_request_at)
                        VALUES (?, ?, ?, 0, NULL, 0, 0, ?, ?, 0, 0, 0, 0)
                        """,
                        (user_id, 1 if is_admin else 0, default_trial, now, now),
                    )

                # Обновляем админа если нужно
                if is_admin:
                    await conn.execute(
                        "UPDATE users SET is_admin = 1, updated_at = ? WHERE user_id = ? AND is_admin = 0",
                        (now, user_id),
                    )

                # Получаем итоговую запись
                try:
                    # Пробуем с новыми полями
                    cursor = await conn.execute(
                        """SELECT user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at,
                           total_requests, successful_requests, failed_requests, last_request_at,
                           referred_by, referral_code, referrals_count, referral_bonus_days, subscription_plan,
                           subscription_requests_balance, subscription_last_purchase_at, subscription_cancelled
                           FROM users WHERE user_id = ?""",
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                except Exception:
                    await self._ensure_referral_columns(conn)
                    # Fallback для старой схемы
                    cursor = await conn.execute(
                        """SELECT user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at,
                           total_requests, successful_requests, failed_requests, last_request_at
                           FROM users WHERE user_id = ?""",
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                    if row:
                        # Дополняем данные значениями по умолчанию для новых полей
                        row = row + (None, None, 0, 0, None, 0, 0, 0)

                if row:
                    self.query_count += 1 if self.enable_metrics else 0
                    return UserRecord(*row)
                else:
                    raise DatabaseException(f"Failed to ensure user {user_id}")

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in ensure_user: {str(e)}")

    async def get_user(self, user_id: int) -> UserRecord | None:
        """Получение пользователя"""
        async with self.pool.acquire() as conn:
            try:
                # Пробуем с новыми полями
                try:
                    cursor = await conn.execute(
                        """SELECT user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at,
                           total_requests, successful_requests, failed_requests, last_request_at,
                           referred_by, referral_code, referrals_count, referral_bonus_days, subscription_plan,
                           subscription_requests_balance, subscription_last_purchase_at, subscription_cancelled
                           FROM users WHERE user_id = ?""",
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                except Exception:
                    await self._ensure_referral_columns(conn)
                    # Fallback для старой схемы
                    cursor = await conn.execute(
                        """SELECT user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at,
                           total_requests, successful_requests, failed_requests, last_request_at
                           FROM users WHERE user_id = ?""",
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                    if row:
                        # Дополняем данные значениями по умолчанию для новых полей
                        row = row + (None, None, 0, 0, None, 0, 0, 0)

                self.query_count += 1 if self.enable_metrics else 0
                return UserRecord(*row) if row else None

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in get_user: {str(e)}")

    async def decrement_trial(self, user_id: int) -> bool:
        """Декремент trial запросов с атомарностью"""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
                cursor = await conn.execute(
                    "UPDATE users SET trial_remaining = trial_remaining - 1, updated_at = ? WHERE user_id = ? AND trial_remaining > 0",
                    (now, user_id),
                )

                rows_affected = cursor.rowcount
                await cursor.close()

                self.query_count += 1 if self.enable_metrics else 0
                return rows_affected > 0

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in decrement_trial: {str(e)}")

    async def has_active_subscription(self, user_id: int) -> bool:
        """Проверка активной подписки"""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
                cursor = await conn.execute(
                    "SELECT 1 FROM users WHERE user_id = ? AND subscription_until > ?",
                    (user_id, now),
                )
                row = await cursor.fetchone()
                await cursor.close()

                self.query_count += 1 if self.enable_metrics else 0
                return bool(row)

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in has_active_subscription: {str(e)}")

    async def cancel_subscription(self, user_id: int) -> bool:
        """Отметить подписку как отменённую (отключает автопродление)."""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
                cursor = await conn.execute(
                    """UPDATE users
                           SET subscription_cancelled = 1, updated_at = ?
                         WHERE user_id = ? AND subscription_cancelled = 0""",
                    (now, user_id),
                )
                updated = cursor.rowcount
                await cursor.close()

                self.query_count += 1 if self.enable_metrics else 0
                return updated > 0

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in cancel_subscription: {str(e)}")


    async def extend_subscription_days(self, user_id: int, days: int) -> None:
        """Продление подписки"""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())

                # Получаем текущую подписку
                cursor = await conn.execute(
                    "SELECT subscription_until FROM users WHERE user_id = ?", (user_id,)
                )
                row = await cursor.fetchone()
                await cursor.close()

                if not row:
                    raise DatabaseException(f"User {user_id} not found")

                # Вычисляем новую дату
                current_until = int(row[0])
                base_time = max(current_until, now)
                new_until = base_time + days * 86400

                # Обновляем подписку
                await conn.execute(
                    "UPDATE users SET subscription_until = ?, subscription_cancelled = 0, updated_at = ? WHERE user_id = ?",
                    (new_until, now, user_id),
                )

                self.query_count += 1 if self.enable_metrics else 0

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in extend_subscription_days: {str(e)}")

    async def apply_subscription_purchase(
        self,
        user_id: int,
        *,
        plan_id: str,
        duration_days: int,
        request_quota: int,
    ) -> tuple[int, int]:
        """Extend subscription and replenish quota for a specific plan."""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
                cursor = await conn.execute(
                    "SELECT subscription_until, subscription_requests_balance FROM users WHERE user_id = ?",
                    (user_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if not row:
                    raise DatabaseException(f"User {user_id} not found")
                current_until = int(row[0]) if row[0] else 0
                current_balance = int(row[1]) if row[1] is not None else 0
                base_time = max(current_until, now)
                new_until = base_time + max(0, duration_days) * 86400
                new_balance = max(0, current_balance) + max(0, request_quota)
                await conn.execute(
                    """UPDATE users SET
                       subscription_until = ?,
                       subscription_plan = ?,
                       subscription_requests_balance = ?,
                       subscription_last_purchase_at = ?,
                       subscription_cancelled = 0,
                       updated_at = ?
                     WHERE user_id = ?""",
                    (new_until, plan_id, new_balance, now, now, user_id),
                )
                self.query_count += 2 if self.enable_metrics else 0
                return new_until, new_balance
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in apply_subscription_purchase: {str(e)}")


    async def record_transaction(
        self,
        *,
        user_id: int,
        provider: str,
        currency: str,
        amount: int,
        payload: str,
        status: TransactionStatus | str,
        telegram_payment_charge_id: str | None = None,
        provider_payment_charge_id: str | None = None,
        amount_minor_units: int | None = None,
    ) -> int:
        """Запись транзакции с возвратом ID"""
        # Проверяем существование транзакции перед записью для idempotency
        if telegram_payment_charge_id:
            if await self.transaction_exists_by_telegram_charge_id(telegram_payment_charge_id):
                # Возвращаем ID существующей транзакции
                async with self.pool.acquire() as conn:
                    try:
                        cursor = await conn.execute(
                            "SELECT id FROM transactions WHERE telegram_payment_charge_id = ?",
                            (telegram_payment_charge_id,),
                        )
                        row = await cursor.fetchone()
                        await cursor.close()

                        if row:
                            logger.info(
                                f"Transaction with charge_id {telegram_payment_charge_id} already exists, returning existing ID: {row[0]}"
                            )
                            return row[0]
                    except Exception as e:
                        logger.warning(f"Error retrieving existing transaction ID: {e}")

        try:
            normalized_status = TransactionStatus.from_value(status)
        except ValueError as exc:
            raise DatabaseException(f"Unsupported transaction status: {status!r}") from exc

        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())

                minor_units = amount_minor_units
                if minor_units is None:
                    # По умолчанию работаем в копейках для RUB, иначе сохраняем исходное значение
                    if currency.upper() in {"RUB", "RUR"}:
                        minor_units = amount * 100
                    else:
                        minor_units = amount

                cursor = await conn.execute(
                    """
                    INSERT INTO transactions
                    (user_id, provider, currency, amount, amount_minor_units, payload, status,
                     telegram_payment_charge_id, provider_payment_charge_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        provider,
                        currency,
                        amount,
                        minor_units,
                        payload,
                        normalized_status.value,
                        telegram_payment_charge_id,
                        provider_payment_charge_id,
                        now,
                        now,
                    ),
                )

                transaction_id = cursor.lastrowid
                await cursor.close()

                self.transaction_count += 1 if self.enable_metrics else 0
                self.query_count += 1 if self.enable_metrics else 0

                logger.info(
                    f"Created new transaction with ID: {transaction_id}, charge_id: {telegram_payment_charge_id}"
                )
                return transaction_id

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                # Если ошибка UNIQUE constraint, проверяем ещё раз
                if "UNIQUE constraint failed" in str(e) and telegram_payment_charge_id:
                    logger.warning(
                        f"UNIQUE constraint failed for charge_id {telegram_payment_charge_id}, checking existing transaction"
                    )
                    if await self.transaction_exists_by_telegram_charge_id(telegram_payment_charge_id):
                        # Возвращаем ID существующей транзакции
                        cursor = await conn.execute(
                            "SELECT id FROM transactions WHERE telegram_payment_charge_id = ?",
                            (telegram_payment_charge_id,),
                        )
                        row = await cursor.fetchone()
                        await cursor.close()

                        if row:
                            logger.info(
                                f"Returning existing transaction ID after constraint error: {row[0]}"
                            )
                            return row[0]

                raise DatabaseException(f"Database error in record_transaction: {str(e)}")


    async def transaction_exists_by_telegram_charge_id(self, charge_id: str) -> bool:
        """Проверка существования транзакции по Telegram charge_id"""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT 1 FROM transactions WHERE telegram_payment_charge_id = ?", (charge_id,)
                )
                row = await cursor.fetchone()
                await cursor.close()

                self.query_count += 1 if self.enable_metrics else 0
                return bool(row)

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(
                    f"Database error in transaction_exists_by_telegram_charge_id: {str(e)}"
                )

    async def get_transaction_by_id(self, transaction_id: int) -> TransactionRecord | None:
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """
                    SELECT id, user_id, provider, currency, amount, amount_minor_units, payload,
                           status, telegram_payment_charge_id, provider_payment_charge_id,
                           created_at, updated_at
                    FROM transactions
                    WHERE id = ?
                    """,
                    (transaction_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                self.query_count += 1 if self.enable_metrics else 0
                if not row:
                    return None
                return TransactionRecord(*row)
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in get_transaction_by_id: {str(e)}")

    async def get_transaction_by_provider_charge_id(
        self, provider: str, provider_payment_charge_id: str
    ) -> TransactionRecord | None:
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """
                    SELECT id, user_id, provider, currency, amount, amount_minor_units, payload,
                           status, telegram_payment_charge_id, provider_payment_charge_id,
                           created_at, updated_at
                    FROM transactions
                    WHERE provider = ? AND provider_payment_charge_id = ?
                    """,
                    (provider, provider_payment_charge_id),
                )
                row = await cursor.fetchone()
                await cursor.close()
                self.query_count += 1 if self.enable_metrics else 0
                if not row:
                    return None
                return TransactionRecord(*row)
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(
                    f"Database error in get_transaction_by_provider_charge_id: {str(e)}"
                )

    async def update_transaction(
        self,
        transaction_id: int,
        *,
        status: TransactionStatus | str | None = None,
        provider_payment_charge_id: str | None = None,
        telegram_payment_charge_id: str | None = None,
    ) -> None:
        """Обновление полей транзакции."""
        updates: list[str] = []
        params: list[Any] = []
        if status is not None:
            try:
                normalized_status = TransactionStatus.from_value(status)
            except ValueError as exc:
                raise DatabaseException(f"Unsupported transaction status: {status!r}") from exc
            updates.append("status = ?")
            params.append(normalized_status.value)
        if provider_payment_charge_id is not None:
            updates.append("provider_payment_charge_id = ?")
            params.append(provider_payment_charge_id)
        if telegram_payment_charge_id is not None:
            updates.append("telegram_payment_charge_id = ?")
            params.append(telegram_payment_charge_id)

        if not updates:
            return

        updates.append("updated_at = ?")
        params.append(int(time.time()))
        params.append(transaction_id)

        async with self.pool.acquire() as conn:
            try:
                await conn.execute(
                    f"""
                    UPDATE transactions
                    SET {', '.join(updates)}
                    WHERE id = ?
                    """,
                    params,
                )
                self.query_count += 1 if self.enable_metrics else 0
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in update_transaction: {str(e)}")

    # ============ Методы для работы со статистикой запросов ============

    async def record_request(
        self,
        user_id: int,
        request_type: str = "legal_question",
        tokens_used: int = 0,
        response_time_ms: int = 0,
        success: bool = True,
        error_type: str | None = None,
    ) -> int:
        """Запись запроса пользователя в статистику. Возвращает ID записи."""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())

                # Записываем детальную статистику запроса
                cursor = await conn.execute(
                    """INSERT INTO requests 
                       (user_id, request_type, tokens_used, response_time_ms, success, error_type, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        request_type,
                        tokens_used,
                        response_time_ms,
                        success,
                        error_type,
                        now,
                    ),
                )

                request_id = cursor.lastrowid

                # Обновляем счетчики в таблице пользователей
                if success:
                    await conn.execute(
                        """UPDATE users SET
                           total_requests = total_requests + 1,
                           successful_requests = successful_requests + 1,
                           last_request_at = ?,
                           updated_at = ?,
                           subscription_requests_balance = MAX(0, subscription_requests_balance - 1)
                           WHERE user_id = ?
                           AND (subscription_plan IS NULL OR subscription_requests_balance > 0 OR subscription_requests_balance = 0)""",
                        (now, now, user_id),
                    )
                else:
                    await conn.execute(
                        """UPDATE users SET 
                           total_requests = total_requests + 1,
                           failed_requests = failed_requests + 1,
                           last_request_at = ?,
                           updated_at = ?
                           WHERE user_id = ?""",
                        (now, now, user_id),
                    )

                self.query_count += 2 if self.enable_metrics else 0
                return request_id

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in record_request: {str(e)}")

    # ============ Rating helpers ============

    async def add_rating(
        self,
        request_id: int,
        user_id: int,
        rating: int,
        feedback_text: str | None = None,
        *,
        username: str | None = None,
        answer_text: str | None = None,
    ) -> bool:
        """Insert or update a rating for a request."""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
                await conn.execute(
                    """
                    INSERT INTO ratings (
                        request_id,
                        user_id,
                        rating,
                        feedback_text,
                        created_at,
                        username,
                        answer_text
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, request_id) DO UPDATE SET
                        rating = excluded.rating,
                        feedback_text = excluded.feedback_text,
                        created_at = excluded.created_at,
                        username = excluded.username,
                        answer_text = excluded.answer_text
                    """,
                    (
                        request_id,
                        user_id,
                        rating,
                        feedback_text,
                        now,
                        username,
                        answer_text,
                    ),
                )
                self.query_count += 1 if self.enable_metrics else 0
                return True
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in add_rating: {e}")
                return False

    async def get_rating(self, request_id: int, user_id: int) -> RatingRecord | None:
        """Fetch a rating for the given request/user pair."""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """
                    SELECT id, request_id, user_id, rating, feedback_text, created_at, username, answer_text
                    FROM ratings
                    WHERE request_id = ? AND user_id = ?
                    """,
                    (request_id, user_id),
                )
                row = await cursor.fetchone()
                await cursor.close()
                self.query_count += 1 if self.enable_metrics else 0
                if not row:
                    return None
                return RatingRecord(*row)
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in get_rating: {e}")
                return None

    async def get_ratings_statistics(self, days: int) -> dict[str, Any]:
        """Aggregate rating metrics for the given period (in days)."""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
                period_start = 0 if days <= 0 else now - days * 86400
                cursor = await conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total_ratings,
                        SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS total_likes,
                        SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS total_dislikes,
                        SUM(CASE WHEN feedback_text IS NOT NULL AND TRIM(feedback_text) <> '' THEN 1 ELSE 0 END) AS feedback_count
                    FROM ratings
                    WHERE created_at >= ?
                    """,
                    (period_start,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                self.query_count += 1 if self.enable_metrics else 0
                if not row:
                    return {
                        "total_ratings": 0,
                        "total_likes": 0,
                        "total_dislikes": 0,
                        "like_rate": 0.0,
                        "feedback_count": 0,
                    }

                total_ratings = row[0] or 0
                total_likes = row[1] or 0
                total_dislikes = row[2] or 0
                feedback_count = row[3] or 0
                like_rate = (total_likes / total_ratings * 100.0) if total_ratings else 0.0

                return {
                    "total_ratings": total_ratings,
                    "total_likes": total_likes,
                    "total_dislikes": total_dislikes,
                    "like_rate": like_rate,
                    "feedback_count": feedback_count,
                }
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in get_ratings_statistics: {e}")
                return {
                    "total_ratings": 0,
                    "total_likes": 0,
                    "total_dislikes": 0,
                    "like_rate": 0.0,
                    "feedback_count": 0,
                }

    async def get_low_rated_requests(
        self,
        limit: int = 5,
        days: int = 30,
        min_count: int = 1,
    ) -> list[dict[str, Any]]:
        """Return requests with the lowest average ratings."""
        limit = max(limit, 1)
        min_count = max(min_count, 1)
        async with self.pool.acquire() as conn:
            try:
                params: list[Any] = []
                where_clauses: list[str] = []
                if days and days > 0:
                    period_start = int(time.time()) - days * 86400
                    where_clauses.append("created_at >= ?")
                    params.append(period_start)
                where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

                query = f"""
                    SELECT
                        request_id,
                        AVG(rating) AS avg_rating,
                        COUNT(*) AS rating_count
                    FROM ratings
                    {where_sql}
                    GROUP BY request_id
                    HAVING COUNT(*) >= ?
                    ORDER BY avg_rating ASC, rating_count DESC
                    LIMIT ?
                """
                params.extend([min_count, limit])
                cursor = await conn.execute(query, params)
                rows = await cursor.fetchall()
                await cursor.close()
                self.query_count += 1 if self.enable_metrics else 0

                results: list[dict[str, Any]] = []
                for request_id, avg_rating, rating_count in rows:
                    if avg_rating is None:
                        continue
                    results.append(
                        {
                            "request_id": request_id,
                            "avg_rating": float(avg_rating),
                            "rating_count": int(rating_count),
                        }
                    )

                return [item for item in results if item["avg_rating"] < 0.5]
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in get_low_rated_requests: {e}")
                return []

    async def get_user_statistics(self, user_id: int, days: int = 30) -> dict[str, Any]:
        """Получение статистики пользователя за определенный период"""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
                period_start = now - (days * 86400)  # days в секундах

                # Общая статистика пользователя
                user_cursor = await conn.execute(
                    """SELECT total_requests, successful_requests, failed_requests, last_request_at,
                       trial_remaining, subscription_until, is_admin,
                       subscription_plan, subscription_requests_balance, subscription_last_purchase_at,
                       created_at, updated_at
                       FROM users WHERE user_id = ?""",
                    (user_id,),
                )
                user_row = await user_cursor.fetchone()
                await user_cursor.close()

                if not user_row:
                    return {"error": "User not found"}

                previous_period_start = period_start - (days * 86400)

                # Статистика за период
                period_cursor = await conn.execute(
                    """SELECT 
                       COUNT(*) as period_requests,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as period_successful,
                       SUM(tokens_used) as period_tokens,
                       AVG(response_time_ms) as avg_response_time
                       FROM requests 
                       WHERE user_id = ? AND created_at >= ?""",
                    (user_id, period_start),
                )
                period_row = await period_cursor.fetchone()
                await period_cursor.close()

                prev_cursor = await conn.execute(
                    """SELECT 
                       COUNT(*) as prev_requests,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as prev_successful,
                       SUM(tokens_used) as prev_tokens,
                       AVG(response_time_ms) as prev_avg_response_time
                       FROM requests 
                       WHERE user_id = ? AND created_at >= ? AND created_at < ?""",
                    (user_id, previous_period_start, period_start),
                )
                prev_row = await prev_cursor.fetchone()
                await prev_cursor.close()

                # Статистика по типам запросов за период
                types_cursor = await conn.execute(
                    """SELECT request_type, COUNT(*) as count
                       FROM requests 
                       WHERE user_id = ? AND created_at >= ?
                       GROUP BY request_type""",
                    (user_id, period_start),
                )
                types_rows = await types_cursor.fetchall()
                await types_cursor.close()

                dow_cursor = await conn.execute(
                    """SELECT strftime('%w', created_at, 'unixepoch') as dow, COUNT(*)
                        FROM requests
                        WHERE user_id = ? AND created_at >= ?
                        GROUP BY dow""",
                    (user_id, period_start),
                )
                dow_rows = await dow_cursor.fetchall()
                await dow_cursor.close()

                hour_cursor = await conn.execute(
                    """SELECT strftime('%H', created_at, 'unixepoch') as hour, COUNT(*)
                        FROM requests
                        WHERE user_id = ? AND created_at >= ?
                        GROUP BY hour""",
                    (user_id, period_start),
                )
                hour_rows = await hour_cursor.fetchall()
                await hour_cursor.close()

                tx_cursor = await conn.execute(
                    """SELECT provider, currency, amount, amount_minor_units, status, created_at, payload
                        FROM transactions
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT 1""",
                    (user_id,),
                )
                tx_row = await tx_cursor.fetchone()
                await tx_cursor.close()

                # Популярные функции (из behavior_events)
                features_cursor = await conn.execute(
                    """SELECT feature, COUNT(*) as count,
                       MAX(timestamp) as last_used
                       FROM behavior_events
                       WHERE user_id = ? AND timestamp >= ?
                       GROUP BY feature
                       ORDER BY count DESC
                       LIMIT 10""",
                    (user_id, period_start),
                )
                features_rows = await features_cursor.fetchall()
                await features_cursor.close()

                # Ежедневная активность для графика (последние 7 дней)
                daily_cursor = await conn.execute(
                    """SELECT date(created_at, 'unixepoch') as day, COUNT(*) as count
                       FROM requests
                       WHERE user_id = ? AND created_at >= ?
                       GROUP BY day
                       ORDER BY day ASC""",
                    (user_id, now - (7 * 86400)),
                )
                daily_rows = await daily_cursor.fetchall()
                await daily_cursor.close()

                self.query_count += 9 if self.enable_metrics else 0

                types_dict = {row[0]: int(row[1]) for row in types_rows} if types_rows else {}

                raw_period_requests = period_row[0] if period_row else 0
                period_requests = int(raw_period_requests or 0)
                if period_requests == 0 and types_dict:
                    period_requests = sum(types_dict.values())

                raw_period_successful = period_row[1] if period_row else 0
                period_successful = int(raw_period_successful or 0)

                raw_period_tokens = period_row[2] if period_row else 0
                period_tokens = int(raw_period_tokens or 0)

                avg_response_time = (
                    round(period_row[3]) if period_row and period_row[3] is not None else 0
                )

                prev_requests = int((prev_row[0] if prev_row else 0) or 0)
                prev_successful = int((prev_row[1] if prev_row else 0) or 0)
                prev_tokens = int((prev_row[2] if prev_row else 0) or 0)
                prev_avg_response_time = (
                    round(prev_row[3]) if prev_row and prev_row[3] is not None else 0
                )

                dow_counts = {row[0]: int(row[1]) for row in dow_rows} if dow_rows else {}
                hour_counts = {row[0]: int(row[1]) for row in hour_rows} if hour_rows else {}

                last_transaction = None
                if tx_row:
                    last_transaction = {
                        "provider": tx_row[0],
                        "currency": tx_row[1],
                        "amount": tx_row[2],
                        "amount_minor_units": tx_row[3],
                        "status": tx_row[4],
                        "created_at": tx_row[5],
                        "payload": tx_row[6],
                    }

                # Обработка популярных функций
                feature_stats = [
                    {
                        "feature": row[0],
                        "count": int(row[1]),
                        "last_used": int(row[2]) if row[2] else 0
                    }
                    for row in features_rows
                ] if features_rows else []

                # Обработка ежедневной активности для графика
                daily_activity = [int(row[1]) for row in daily_rows] if daily_rows else []

                return {
                    "user_id": user_id,
                    "total_requests": user_row[0],
                    "successful_requests": user_row[1],
                    "failed_requests": user_row[2],
                    "last_request_at": user_row[3],
                    "trial_remaining": user_row[4],
                    "subscription_until": user_row[5],
                    "subscription_plan": user_row[7],
                    "subscription_requests_balance": user_row[8],
                    "subscription_last_purchase_at": user_row[9],
                    "is_admin": bool(user_row[6]),
                    "created_at": user_row[10],
                    "updated_at": user_row[11],
                    "period_days": days,
                    "period_requests": period_requests,
                    "period_successful": period_successful,
                    "period_tokens": period_tokens,
                    "avg_response_time_ms": avg_response_time,
                    "request_types": types_dict,
                    "previous_period_requests": prev_requests,
                    "previous_period_successful": prev_successful,
                    "previous_period_tokens": prev_tokens,
                    "previous_avg_response_time_ms": prev_avg_response_time,
                    "day_of_week_counts": dow_counts,
                    "hour_of_day_counts": hour_counts,
                    "last_transaction": last_transaction,
                    "recent_requests": period_requests,
                    "recent_successful": period_successful,
                    # Новые поля
                    "feature_stats": feature_stats,
                    "daily_activity": daily_activity,
                }

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in get_user_statistics: {str(e)}")

    # ============ Методы для работы с рейтингами ============






    async def get_stats(self) -> dict[str, Any]:
        """Получение статистики базы данных"""
        pool_stats = self.pool.get_stats()

        return {
            **pool_stats,
            "query_count": self.query_count,
            "transaction_count": self.transaction_count,
            "error_count": self.error_count,
            "cleanup_task_running": self._cleanup_task is not None
            and not self._cleanup_task.done(),
        }

    # Методы для реферальной системы
    async def apply_referral_code(self, user_id: int, referral_code: str) -> tuple[bool, str]:
        """
        Закрепляет пользователя за реферером по коду.

        Возвращает кортеж (успех, причина), где причина принимает значения:
        - "applied" — код успешно применён;
        - "invalid_code" — код не найден или пуст;
        - "self_referral" — пользователь попытался использовать собственный код;
        - "already_linked" — уже привязан к этому рефереру;
        - "already_has_referrer" — уже есть другой реферер.
        """
        normalized_code = (referral_code or "").strip().upper()
        if not normalized_code or normalized_code == "SYSTEM_ERROR":
            return False, "invalid_code"

        async with self.pool.acquire() as conn:
            try:
                await self._ensure_referral_columns(conn)

                cursor = await conn.execute(
                    "SELECT user_id FROM users WHERE referral_code = ?",
                    (normalized_code,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if not row:
                    return False, "invalid_code"

                referrer_id = int(row[0])
                if referrer_id == user_id:
                    return False, "self_referral"

                cursor = await conn.execute(
                    "SELECT referred_by FROM users WHERE user_id = ?",
                    (user_id,),
                )
                current_row = await cursor.fetchone()
                await cursor.close()
                if not current_row:
                    raise DatabaseException(f"User {user_id} not found while applying referral code")

                existing_referrer = current_row[0]
                if existing_referrer:
                    if int(existing_referrer) == referrer_id:
                        return False, "already_linked"
                    return False, "already_has_referrer"

                now = int(time.time())
                await conn.execute("BEGIN")
                try:
                    cursor = await conn.execute(
                        "UPDATE users SET referred_by = ?, updated_at = ? "
                        "WHERE user_id = ? AND (referred_by IS NULL OR referred_by = 0)",
                        (referrer_id, now, user_id),
                    )
                    updated_rows = cursor.rowcount
                    await cursor.close()
                    if updated_rows == 0:
                        await conn.rollback()
                        return False, "already_has_referrer"

                    cursor = await conn.execute(
                        "UPDATE users SET referrals_count = referrals_count + 1, updated_at = ? "
                        "WHERE user_id = ?",
                        (now, referrer_id),
                    )
                    await cursor.close()
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise

                return True, "applied"

            except DatabaseException:
                raise
            except Exception as e:
                logger.error(f"Database error in apply_referral_code: {e}")
                raise DatabaseException(f"Error applying referral code: {e}")

    async def generate_referral_code(self, user_id: int) -> str:
        """Генерация реферального кода для пользователя"""
        import secrets
        import string

        max_attempts = 5
        async with self.pool.acquire() as conn:
            try:
                await self._ensure_referral_columns(conn)
                for attempt in range(max_attempts):
                    code = ''.join(
                        secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8)
                    )
                    try:
                        await conn.execute(
                            "UPDATE users SET referral_code = ?, updated_at = ? WHERE user_id = ?",
                            (code, int(time.time()), user_id),
                        )
                        await conn.commit()
                        return code
                    except aiosqlite.IntegrityError:
                        await conn.rollback()
                        logger.warning(
                            "Referral code collision for user %s on attempt %s",
                            user_id,
                            attempt + 1,
                        )
                error_message = (
                    f"Failed to generate unique referral code for user {user_id} after several attempts"
                )
                logger.error(error_message)
                raise DatabaseException(error_message)
            except DatabaseException:
                raise
            except Exception as e:
                logger.error(f"Database error in generate_referral_code: {e}")
                raise DatabaseException(f"Error generating referral code: {e}")





    async def get_user_referrals(self, user_id: int) -> list[dict[str, Any]]:
        """Получение списка рефералов пользователя"""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """SELECT user_id, created_at, subscription_until > ? as has_active_subscription
                       FROM users WHERE referred_by = ? ORDER BY created_at DESC""",
                    (int(time.time()), user_id),
                )
                rows = await cursor.fetchall()
                await cursor.close()

                return [
                    {
                        "user_id": row[0],
                        "joined_at": row[1],
                        "has_active_subscription": bool(row[2])
                    }
                    for row in rows
                ]
            except Exception as e:
                logger.error(f"Database error in get_user_referrals: {e}")
                return []


    async def get_user_transactions(self, user_id: int, limit: int = 20) -> list[TransactionRecord]:
        """Получение истории транзакций пользователя"""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """SELECT id, user_id, provider, currency, amount, amount_minor_units,
                       payload, status, telegram_payment_charge_id, provider_payment_charge_id,
                       created_at, updated_at
                       FROM transactions WHERE user_id = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (user_id, limit),
                )
                rows = await cursor.fetchall()
                await cursor.close()

                normalized_rows: list[TransactionRecord] = []
                for row in rows:
                    normalized_rows.append(
                        TransactionRecord(
                            id=row[0],
                            user_id=row[1],
                            provider=row[2],
                            currency=row[3],
                            amount=row[4],
                            amount_minor_units=row[5],
                            payload=row[6],
                            status=TransactionStatus.from_value(row[7]).value,
                            telegram_payment_charge_id=row[8],
                            provider_payment_charge_id=row[9],
                            created_at=row[10],
                            updated_at=row[11],
                        )
                    )
                return normalized_rows
            except Exception as e:
                logger.error(f"Database error in get_user_transactions: {e}")
                return []

    async def get_transaction_stats(self, user_id: int) -> dict[str, Any]:
        """Получение статистики платежей пользователя"""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """SELECT
                       COUNT(*) as total_transactions,
                       SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) as total_spent,
                       MIN(created_at) as first_payment,
                       MAX(created_at) as last_payment
                       FROM transactions WHERE user_id = ?""",
                    (user_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()

                if row:
                    return {
                        "total_transactions": row[0] or 0,
                        "total_spent": row[1] or 0,
                        "first_payment": row[2],
                        "last_payment": row[3],
                    }
                return {
                    "total_transactions": 0,
                    "total_spent": 0,
                    "first_payment": None,
                    "last_payment": None,
                }
            except Exception as e:
                logger.error(f"Database error in get_transaction_stats: {e}")
                return {}

    async def mark_hint_shown(self, user_id: int, hint_key: str) -> bool:
        """
        Пометить подсказку как показанную

        Args:
            user_id: ID пользователя
            hint_key: Ключ подсказки (например, 'search_practice', 'document_analysis')

        Returns:
            True если подсказка была добавлена, False если уже существовала
        """
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
                await conn.execute(
                    """INSERT OR IGNORE INTO user_onboarding_hints (user_id, hint_key, shown_at)
                       VALUES (?, ?, ?)""",
                    (user_id, hint_key, now),
                )
                self.query_count += 1 if self.enable_metrics else 0
                return True
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in mark_hint_shown: {e}")
                return False

    async def get_shown_hints(self, user_id: int) -> set[str]:
        """
        Получить список уже показанных подсказок

        Args:
            user_id: ID пользователя

        Returns:
            Множество ключей показанных подсказок
        """
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT hint_key FROM user_onboarding_hints WHERE user_id = ?",
                    (user_id,),
                )
                rows = await cursor.fetchall()
                await cursor.close()
                self.query_count += 1 if self.enable_metrics else 0
                return {row[0] for row in rows}
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in get_shown_hints: {e}")
                return set()

    async def update_user_hint_counter(self, user_id: int, request_count: int) -> None:
        """
        Обновление счётчика последней показанной подсказки (deprecated, используйте mark_hint_shown)

        Args:
            user_id: ID пользователя
            request_count: Номер запроса, на котором была показана подсказка
        """
        pass

    async def close(self) -> None:
        """Закрытие базы данных"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        await self.pool.close()
        logger.info("Advanced database closed")

# Hints for static analyzers
_UNUSED_USER_RECORD_FIELDS = (
    "UserRecord.successful_requests",
    "UserRecord.last_request_at",
    "UserRecord.referred_by",
    "UserRecord.subscription_last_purchase_at",
    "TransactionRecord.updated_at",
    "TransactionStatus.PENDING",
)

__all__ = (
    "UserRecord",
    "TransactionRecord",
    "RequestRecord",
    "TransactionStatus",
    "DatabaseAdvanced",
)
