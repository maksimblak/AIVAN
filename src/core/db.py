from __future__ import annotations
import asyncio
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Optional


def _now_ts() -> int:
    return int(time.time())


@dataclass
class UserRecord:
    user_id: int
    is_admin: int
    trial_remaining: int
    subscription_until: int
    created_at: int
    updated_at: int


class Database:
    """Lightweight SQLite wrapper for users and transactions.

    Uses sqlite3 with a single connection (check_same_thread=False) and an asyncio.Lock
    to avoid concurrent writes from the event loop. All queries are executed in
    a background thread via asyncio.to_thread to prevent blocking.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,  # autocommit
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._exec(
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

        await self._exec(
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

        # Helpful indexes for queries and reporting
        await self._exec("CREATE INDEX IF NOT EXISTS idx_transactions_user_created ON transactions(user_id, created_at);")
        await self._exec("CREATE UNIQUE INDEX IF NOT EXISTS idx_transactions_tg_charge ON transactions(telegram_payment_charge_id);")

        # Backfill schema for existing DBs: ensure amount_minor_units column exists
        await self._exec("PRAGMA table_info(transactions);")
        # SQLite can't conditionally add columns without checking, so try-catch
        try:
            await self._exec("ALTER TABLE transactions ADD COLUMN amount_minor_units INTEGER;")
        except Exception:
            pass

    # ---------------- Internal helpers ----------------

    async def _exec(self, query: str, params: tuple[Any, ...] = ()) -> None:
        async with self._lock:
            await asyncio.to_thread(self._conn.execute, query, params)  # type: ignore[arg-type]

    async def _fetchone(self, query: str, params: tuple[Any, ...] = ()) -> Optional[tuple]:
        async with self._lock:
            cur = await asyncio.to_thread(self._conn.execute, query, params)  # type: ignore[arg-type]
            row = await asyncio.to_thread(cur.fetchone)
        return row

    async def _fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple]:
        async with self._lock:
            cur = await asyncio.to_thread(self._conn.execute, query, params)  # type: ignore[arg-type]
            rows = await asyncio.to_thread(cur.fetchall)
        return rows

    # ---------------- Users ----------------

    async def ensure_user(self, user_id: int, *, default_trial: int = 10, is_admin: bool = False) -> UserRecord:
        row = await self._fetchone("SELECT user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at FROM users WHERE user_id = ?", (user_id,))
        now = _now_ts()
        if row:
            # Optionally upgrade admin flag
            if is_admin and row[1] == 0:
                await self._exec("UPDATE users SET is_admin = 1, updated_at = ? WHERE user_id = ?", (now, user_id))
            return UserRecord(*row)

        await self._exec(
            "INSERT INTO users (user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at) VALUES (?, ?, ?, 0, ?, ?)",
            (user_id, 1 if is_admin else 0, default_trial, now, now),
        )
        return UserRecord(user_id=user_id, is_admin=1 if is_admin else 0, trial_remaining=default_trial, subscription_until=0, created_at=now, updated_at=now)

    async def get_user(self, user_id: int) -> Optional[UserRecord]:
        row = await self._fetchone("SELECT user_id, is_admin, trial_remaining, subscription_until, created_at, updated_at FROM users WHERE user_id = ?", (user_id,))
        return UserRecord(*row) if row else None

    async def set_admin(self, user_id: int, is_admin: bool) -> None:
        now = _now_ts()
        await self._exec("UPDATE users SET is_admin = ?, updated_at = ? WHERE user_id = ?", (1 if is_admin else 0, now, user_id))

    async def decrement_trial(self, user_id: int) -> bool:
        now = _now_ts()
        row = await self._fetchone("SELECT trial_remaining FROM users WHERE user_id = ?", (user_id,))
        if not row:
            return False
        remaining = int(row[0])
        if remaining <= 0:
            return False
        await self._exec("UPDATE users SET trial_remaining = trial_remaining - 1, updated_at = ? WHERE user_id = ?", (now, user_id))
        return True

    async def has_active_subscription(self, user_id: int) -> bool:
        row = await self._fetchone("SELECT subscription_until FROM users WHERE user_id = ?", (user_id,))
        if not row:
            return False
        return int(row[0]) > _now_ts()

    async def extend_subscription_days(self, user_id: int, days: int) -> None:
        now = _now_ts()
        row = await self._fetchone("SELECT subscription_until FROM users WHERE user_id = ?", (user_id,))
        base = int(row[0]) if row and int(row[0]) > now else now
        new_until = base + days * 86400
        await self._exec("UPDATE users SET subscription_until = ?, updated_at = ? WHERE user_id = ?", (new_until, now, user_id))

    # ---------------- Transactions ----------------

    async def record_transaction(self, *, user_id: int, provider: str, currency: str, amount: int, payload: str, status: str, telegram_payment_charge_id: Optional[str] = None, provider_payment_charge_id: Optional[str] = None, amount_minor_units: Optional[int] = None) -> None:
        now = _now_ts()
        await self._exec(
            """
            INSERT INTO transactions (user_id, provider, currency, amount, amount_minor_units, payload, status, telegram_payment_charge_id, provider_payment_charge_id, created_at, updated_at)
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

    async def mark_transaction_success(self, *, telegram_payment_charge_id: str, provider_payment_charge_id: Optional[str]) -> None:
        now = _now_ts()
        await self._exec(
            "UPDATE transactions SET status = 'success', provider_payment_charge_id = COALESCE(?, provider_payment_charge_id), updated_at = ? WHERE telegram_payment_charge_id = ?",
            (provider_payment_charge_id, now, telegram_payment_charge_id),
        )

    async def transaction_exists_by_telegram_charge_id(self, charge_id: str) -> bool:
        row = await self._fetchone("SELECT 1 FROM transactions WHERE telegram_payment_charge_id = ?", (charge_id,))
        return bool(row)

    # ---------------- Lifecycle ----------------

    async def close(self) -> None:
        """Close the underlying connection."""
        if self._conn is not None:
            conn = self._conn
            self._conn = None
            try:
                await asyncio.to_thread(conn.close)
            except Exception:
                pass


