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
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

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
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    total_requests INTEGER NOT NULL DEFAULT 0,
                    successful_requests INTEGER NOT NULL DEFAULT 0,
                    failed_requests INTEGER NOT NULL DEFAULT 0,
                    last_request_at INTEGER NOT NULL DEFAULT 0
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
                    FOREIGN KEY (request_id) REFERENCES requests(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
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
                    "ALTER TABLE users ADD COLUMN referral_code TEXT UNIQUE;",
                    "ALTER TABLE users ADD COLUMN referrals_count INTEGER NOT NULL DEFAULT 0;",
                    "ALTER TABLE users ADD COLUMN referral_bonus_days INTEGER NOT NULL DEFAULT 0;",
                ]

                for migration in migrations:
                    try:
                        await conn.execute(migration)
                        logger.info(f"Applied migration: {migration}")
                    except Exception:
                        # Колонка уже существует, игнорируем
                        pass

                await conn.commit()
            except Exception as e:
                logger.warning(f"Migration warning: {e}")

            # Создание оптимизированных индексов - ПОСЛЕ миграций!
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_users_admin ON users(is_admin) WHERE is_admin = 1;",
                "CREATE INDEX IF NOT EXISTS idx_users_subscription ON users(subscription_until) WHERE subscription_until > 0;",
                "CREATE INDEX IF NOT EXISTS idx_users_requests ON users(total_requests);",
                "CREATE INDEX IF NOT EXISTS idx_transactions_user_created ON transactions(user_id, created_at);",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_transactions_tg_charge ON transactions(telegram_payment_charge_id) WHERE telegram_payment_charge_id IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);",
                "CREATE INDEX IF NOT EXISTS idx_transactions_provider ON transactions(provider);",
                "CREATE INDEX IF NOT EXISTS idx_requests_user_created ON requests(user_id, created_at);",
                "CREATE INDEX IF NOT EXISTS idx_requests_type ON requests(request_type);",
                "CREATE INDEX IF NOT EXISTS idx_requests_success ON requests(success);",
                "CREATE INDEX IF NOT EXISTS idx_ratings_request ON ratings(request_id);",
                "CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_ratings_user_request ON ratings(user_id, request_id);",
            ]

            for index_sql in indexes:
                try:
                    await conn.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {index_sql} - {e}")

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
                        (user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at,
                         total_requests, successful_requests, failed_requests, last_request_at,
                         referred_by, referral_code, referrals_count, referral_bonus_days)
                        VALUES (?, ?, ?, 0, ?, ?, 0, 0, 0, 0, NULL, NULL, 0, 0)
                        """,
                        (user_id, 1 if is_admin else 0, default_trial, now, now),
                    )
                except Exception:
                    # Fallback для старой схемы без реферальных полей
                    await conn.execute(
                        """
                        INSERT OR IGNORE INTO users
                        (user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at,
                         total_requests, successful_requests, failed_requests, last_request_at)
                        VALUES (?, ?, ?, 0, ?, ?, 0, 0, 0, 0)
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
                           referred_by, referral_code, referrals_count, referral_bonus_days
                           FROM users WHERE user_id = ?""",
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                except Exception:
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
                        row = row + (None, None, 0, 0)

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
                           referred_by, referral_code, referrals_count, referral_bonus_days
                           FROM users WHERE user_id = ?""",
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    await cursor.close()
                except Exception:
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
                        row = row + (None, None, 0, 0)

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
                    "UPDATE users SET subscription_until = ?, updated_at = ? WHERE user_id = ?",
                    (new_until, now, user_id),
                )

                self.query_count += 1 if self.enable_metrics else 0

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in extend_subscription_days: {str(e)}")

    async def record_transaction(
        self,
        *,
        user_id: int,
        provider: str,
        currency: str,
        amount: int,
        payload: str,
        status: str,
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

        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
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
                        amount_minor_units if amount_minor_units is not None else amount,
                        payload,
                        status,
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
                    if await self.transaction_exists_by_telegram_charge_id(
                        telegram_payment_charge_id
                    ):
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
                           updated_at = ?
                           WHERE user_id = ?""",
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

    async def get_user_statistics(self, user_id: int, days: int = 30) -> dict[str, Any]:
        """Получение статистики пользователя за определенный период"""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
                period_start = now - (days * 86400)  # days в секундах

                # Общая статистика пользователя
                user_cursor = await conn.execute(
                    """SELECT total_requests, successful_requests, failed_requests, last_request_at,
                       trial_remaining, subscription_until, is_admin 
                       FROM users WHERE user_id = ?""",
                    (user_id,),
                )
                user_row = await user_cursor.fetchone()
                await user_cursor.close()

                if not user_row:
                    return {"error": "User not found"}

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

                self.query_count += 3 if self.enable_metrics else 0

                return {
                    "user_id": user_id,
                    "total_requests": user_row[0],
                    "successful_requests": user_row[1],
                    "failed_requests": user_row[2],
                    "last_request_at": user_row[3],
                    "trial_remaining": user_row[4],
                    "subscription_until": user_row[5],
                    "is_admin": bool(user_row[6]),
                    "period_days": days,
                    "period_requests": period_row[0] if period_row else 0,
                    "period_successful": period_row[1] if period_row else 0,
                    "period_tokens": period_row[2] if period_row else 0,
                    "avg_response_time_ms": (
                        round(period_row[3]) if period_row and period_row[3] else 0
                    ),
                    "request_types": {row[0]: row[1] for row in types_rows} if types_rows else {},
                }

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in get_user_statistics: {str(e)}")

    # ============ Методы для работы с рейтингами ============

    async def add_rating(
        self,
        request_id: int,
        user_id: int,
        rating: int,  # 1 = like, -1 = dislike
        feedback_text: str | None = None,
    ) -> bool:
        """Добавление/обновление рейтинга для запроса"""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())

                # Используем INSERT OR REPLACE для обновления существующего рейтинга
                await conn.execute(
                    """INSERT OR REPLACE INTO ratings 
                       (request_id, user_id, rating, feedback_text, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (request_id, user_id, rating, feedback_text, now),
                )

                self.query_count += 1 if self.enable_metrics else 0
                return True

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in add_rating: {e}")
                return False

    async def get_rating(self, request_id: int, user_id: int) -> RatingRecord | None:
        """Получение рейтинга пользователя для конкретного запроса"""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """SELECT id, request_id, user_id, rating, feedback_text, created_at 
                       FROM ratings WHERE request_id = ? AND user_id = ?""",
                    (request_id, user_id),
                )
                row = await cursor.fetchone()
                await cursor.close()

                self.query_count += 1 if self.enable_metrics else 0
                return RatingRecord(*row) if row else None

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in get_rating: {e}")
                return None

    async def get_request_ratings_summary(self, request_id: int) -> dict[str, Any]:
        """Получение суммарной статистики рейтингов для запроса"""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """SELECT 
                       COUNT(*) as total_ratings,
                       SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as likes,
                       SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as dislikes,
                       AVG(rating) as avg_rating
                       FROM ratings WHERE request_id = ?""",
                    (request_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()

                self.query_count += 1 if self.enable_metrics else 0

                if row:
                    return {
                        "total_ratings": row[0],
                        "likes": row[1] or 0,
                        "dislikes": row[2] or 0,
                        "avg_rating": row[3] or 0.0,
                    }
                return {"total_ratings": 0, "likes": 0, "dislikes": 0, "avg_rating": 0.0}

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in get_request_ratings_summary: {e}")
                return {"total_ratings": 0, "likes": 0, "dislikes": 0, "avg_rating": 0.0}

    async def get_ratings_statistics(self, days: int = 30) -> dict[str, Any]:
        """Получение общей статистики рейтингов за период"""
        async with self.pool.acquire() as conn:
            try:
                now = int(time.time())
                period_start = now - (days * 86400)

                cursor = await conn.execute(
                    """SELECT 
                       COUNT(*) as total_ratings,
                       SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as total_likes,
                       SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as total_dislikes,
                       AVG(rating) as avg_rating,
                       COUNT(CASE WHEN feedback_text IS NOT NULL THEN 1 END) as feedback_count
                       FROM ratings WHERE created_at >= ?""",
                    (period_start,),
                )
                row = await cursor.fetchone()
                await cursor.close()

                self.query_count += 1 if self.enable_metrics else 0

                if row:
                    return {
                        "period_days": days,
                        "total_ratings": row[0] or 0,
                        "total_likes": row[1] or 0,
                        "total_dislikes": row[2] or 0,
                        "avg_rating": row[3] or 0.0,
                        "feedback_count": row[4] or 0,
                        "like_rate": (row[1] or 0) / max(row[0] or 1, 1) * 100,
                    }
                return {
                    "period_days": days,
                    "total_ratings": 0,
                    "total_likes": 0,
                    "total_dislikes": 0,
                    "avg_rating": 0.0,
                    "feedback_count": 0,
                    "like_rate": 0.0,
                }

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in get_ratings_statistics: {e}")
                return {}

    async def get_low_rated_requests(self, limit: int = 10) -> list[dict[str, Any]]:
        """Получение запросов с низкими рейтингами для анализа"""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """SELECT 
                       r.id, r.user_id, r.request_type, r.created_at,
                       AVG(rt.rating) as avg_rating,
                       COUNT(rt.rating) as rating_count
                       FROM requests r 
                       JOIN ratings rt ON r.id = rt.request_id
                       WHERE r.success = 1
                       GROUP BY r.id
                       HAVING avg_rating < 0
                       ORDER BY avg_rating ASC, rating_count DESC
                       LIMIT ?""",
                    (limit,),
                )
                rows = await cursor.fetchall()
                await cursor.close()

                self.query_count += 1 if self.enable_metrics else 0

                return [
                    {
                        "request_id": row[0],
                        "user_id": row[1],
                        "request_type": row[2],
                        "created_at": row[3],
                        "avg_rating": row[4],
                        "rating_count": row[5],
                    }
                    for row in rows
                ]

            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                logger.error(f"Database error in get_low_rated_requests: {e}")
                return []

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
    async def generate_referral_code(self, user_id: int) -> str:
        """Генерация реферального кода для пользователя"""
        import string
        import random

        # Генерируем уникальный код
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

        async with self.pool.acquire() as conn:
            try:
                await conn.execute(
                    "UPDATE users SET referral_code = ?, updated_at = ? WHERE user_id = ?",
                    (code, int(time.time()), user_id),
                )
                await conn.commit()
                return code
            except Exception as e:
                logger.error(f"Database error in generate_referral_code: {e}")
                raise DatabaseException(f"Error generating referral code: {e}")

    async def get_user_by_referral_code(self, referral_code: str) -> UserRecord | None:
        """Получение пользователя по реферальному коду"""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    """SELECT user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at,
                       total_requests, successful_requests, failed_requests, last_request_at,
                       referred_by, referral_code, referrals_count, referral_bonus_days
                       FROM users WHERE referral_code = ?""",
                    (referral_code,),
                )
                row = await cursor.fetchone()
                await cursor.close()

                return UserRecord(*row) if row else None
            except Exception as e:
                logger.error(f"Database error in get_user_by_referral_code: {e}")
                return None

    async def set_user_referrer(self, user_id: int, referrer_id: int) -> bool:
        """Установка реферера для пользователя"""
        async with self.pool.acquire() as conn:
            try:
                # Проверяем, что пользователь не ссылается сам на себя
                if user_id == referrer_id:
                    return False

                await conn.execute(
                    "UPDATE users SET referred_by = ?, updated_at = ? WHERE user_id = ? AND referred_by IS NULL",
                    (referrer_id, int(time.time()), user_id),
                )

                # Увеличиваем счетчик рефералов у реферера
                await conn.execute(
                    "UPDATE users SET referrals_count = referrals_count + 1, updated_at = ? WHERE user_id = ?",
                    (int(time.time()), referrer_id),
                )

                await conn.commit()
                return True
            except Exception as e:
                logger.error(f"Database error in set_user_referrer: {e}")
                return False

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

    async def add_referral_bonus(self, user_id: int, bonus_days: int) -> bool:
        """Добавление бонусных дней пользователю"""
        async with self.pool.acquire() as conn:
            try:
                await conn.execute(
                    "UPDATE users SET referral_bonus_days = referral_bonus_days + ?, updated_at = ? WHERE user_id = ?",
                    (bonus_days, int(time.time()), user_id),
                )
                await conn.commit()
                return True
            except Exception as e:
                logger.error(f"Database error in add_referral_bonus: {e}")
                return False

    # Методы для истории платежей
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

                return [TransactionRecord(*row) for row in rows]
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
