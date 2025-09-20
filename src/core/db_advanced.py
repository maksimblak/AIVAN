"""
Улучшенная база данных с connection pooling, транзакциями и оптимизированными запросами
"""

from __future__ import annotations
import asyncio
import aiosqlite
import time
import os
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional, AsyncIterator, Dict, List, Union
from concurrent.futures import ThreadPoolExecutor
import weakref

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

@dataclass
class TransactionRecord:
    id: int
    user_id: int
    provider: str
    currency: str
    amount: int
    amount_minor_units: Optional[int]
    payload: Optional[str]
    status: str
    telegram_payment_charge_id: Optional[str]
    provider_payment_charge_id: Optional[str]
    created_at: int
    updated_at: int

class ConnectionPool:
    """Пул соединений для SQLite с контролем жизненного цикла"""
    
    def __init__(
        self, 
        db_path: str, 
        max_connections: int = 5,
        connection_timeout: float = 30.0,
        max_connection_age: float = 3600.0  # 1 час
    ):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.max_connection_age = max_connection_age
        
        self._connections: List[aiosqlite.Connection] = []
        self._available_connections = asyncio.Queue(maxsize=max_connections)
        self._connection_times: Dict[aiosqlite.Connection, float] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        
        # Executor для блокирующих операций
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="db-pool")
        
        # Weak references для автоматической очистки
        self._connection_refs: List[weakref.ref] = []
    
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
            self.db_path,
            timeout=self.connection_timeout,
            isolation_level=None  # autocommit mode
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
                    self._available_connections.get(), 
                    timeout=self.connection_timeout
                )
            except asyncio.TimeoutError:
                # Если нет доступных соединений и пул не заполнен - создаем новое
                async with self._lock:
                    if len(self._connections) < self.max_connections:
                        conn = await self._create_connection()
                    else:
                        # Ждем дольше если пул заполнен
                        conn = await asyncio.wait_for(
                            self._available_connections.get(),
                            timeout=self.connection_timeout * 2
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
                conn for conn, create_time in self._connection_times.items()
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика пула соединений"""
        return {
            "total_connections": len(self._connections),
            "available_connections": self._available_connections.qsize(),
            "max_connections": self.max_connections,
            "oldest_connection_age": max(
                [time.time() - create_time for create_time in self._connection_times.values()],
                default=0
            ),
            "is_closed": self._closed
        }

class DatabaseAdvanced:
    """Продвинутая база данных с connection pooling и оптимизациями"""
    
    def __init__(
        self, 
        db_path: str, 
        max_connections: int = 5,
        enable_metrics: bool = True,
        cleanup_interval: float = 300.0  # 5 минут
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
        self._cleanup_task: Optional[asyncio.Task] = None
    
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
                    updated_at INTEGER NOT NULL
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
            
            # Создание оптимизированных индексов
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_users_admin ON users(is_admin) WHERE is_admin = 1;",
                "CREATE INDEX IF NOT EXISTS idx_users_subscription ON users(subscription_until) WHERE subscription_until > 0;",
                "CREATE INDEX IF NOT EXISTS idx_transactions_user_created ON transactions(user_id, created_at);",
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_transactions_tg_charge ON transactions(telegram_payment_charge_id) WHERE telegram_payment_charge_id IS NOT NULL;",
                "CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);",
                "CREATE INDEX IF NOT EXISTS idx_transactions_provider ON transactions(provider);",
            ]
            
            for index_sql in indexes:
                await conn.execute(index_sql)
            
            await conn.commit()
        
        # Запускаем фоновую очистку
        if self.cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"Advanced database initialized with connection pool (max_connections={self.pool.max_connections})")
    
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
        self, 
        user_id: int, 
        *, 
        default_trial: int = 10, 
        is_admin: bool = False
    ) -> UserRecord:
        """Обеспечение существования пользователя с оптимизацией"""
        async with self.pool.acquire() as conn:
            # Используем INSERT OR IGNORE для атомарности
            now = int(time.time())
            
            try:
                await conn.execute(
                    """
                    INSERT OR IGNORE INTO users 
                    (user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at)
                    VALUES (?, ?, ?, 0, ?, ?)
                    """,
                    (user_id, 1 if is_admin else 0, default_trial, now, now)
                )
                
                # Обновляем админа если нужно
                if is_admin:
                    await conn.execute(
                        "UPDATE users SET is_admin = 1, updated_at = ? WHERE user_id = ? AND is_admin = 0",
                        (now, user_id)
                    )
                
                # Получаем итоговую запись
                cursor = await conn.execute(
                    "SELECT user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = await cursor.fetchone()
                await cursor.close()
                
                if row:
                    self.query_count += 1 if self.enable_metrics else 0
                    return UserRecord(*row)
                else:
                    raise DatabaseException(f"Failed to ensure user {user_id}")
                    
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in ensure_user: {str(e)}")
    
    async def get_user(self, user_id: int) -> Optional[UserRecord]:
        """Получение пользователя"""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = await cursor.fetchone()
                await cursor.close()
                
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
                    (now, user_id)
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
                    (user_id, now)
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
                    "SELECT subscription_until FROM users WHERE user_id = ?",
                    (user_id,)
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
                    (new_until, now, user_id)
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
        telegram_payment_charge_id: Optional[str] = None, 
        provider_payment_charge_id: Optional[str] = None, 
        amount_minor_units: Optional[int] = None
    ) -> int:
        """Запись транзакции с возвратом ID"""
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
                        user_id, provider, currency, amount, 
                        amount_minor_units if amount_minor_units is not None else amount,
                        payload, status, telegram_payment_charge_id, provider_payment_charge_id,
                        now, now
                    )
                )
                
                transaction_id = cursor.lastrowid
                await cursor.close()
                
                self.transaction_count += 1 if self.enable_metrics else 0
                self.query_count += 1 if self.enable_metrics else 0
                
                return transaction_id
                
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in record_transaction: {str(e)}")
    
    async def transaction_exists_by_telegram_charge_id(self, charge_id: str) -> bool:
        """Проверка существования транзакции по Telegram charge_id"""
        async with self.pool.acquire() as conn:
            try:
                cursor = await conn.execute(
                    "SELECT 1 FROM transactions WHERE telegram_payment_charge_id = ?",
                    (charge_id,)
                )
                row = await cursor.fetchone()
                await cursor.close()
                
                self.query_count += 1 if self.enable_metrics else 0
                return bool(row)
                
            except Exception as e:
                self.error_count += 1 if self.enable_metrics else 0
                raise DatabaseException(f"Database error in transaction_exists_by_telegram_charge_id: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики базы данных"""
        pool_stats = self.pool.get_stats()
        
        return {
            **pool_stats,
            "query_count": self.query_count,
            "transaction_count": self.transaction_count,
            "error_count": self.error_count,
            "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done()
        }
    
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
