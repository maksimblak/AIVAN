"""
Тесты для продвинутой базы данных
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.core.db_advanced import DatabaseAdvanced, UserRecord, TransactionRecord


class TestDatabaseAdvanced:

    @pytest.fixture
    async def db(self):
        # Создаем временную базу данных для тестов
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")

        database = DatabaseAdvanced(db_path, max_connections=2)
        await database.init()

        yield database

        await database.close()
        # Очистка
        if os.path.exists(db_path):
            os.unlink(db_path)
        os.rmdir(temp_dir)

    @pytest.mark.asyncio
    async def test_init_creates_tables(self, db):
        # Act & Assert - проверяем что таблицы созданы
        async with db.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = await cursor.fetchall()
            table_names = [row[0] for row in tables]

            assert "users" in table_names
            assert "transactions" in table_names
            assert "requests" in table_names
            assert "ratings" in table_names

    @pytest.mark.asyncio
    async def test_ensure_user_creates_new_user(self, db):
        # Arrange
        user_id = 123

        # Act
        user = await db.ensure_user(user_id, default_trial=10, is_admin=False)

        # Assert
        assert user.user_id == user_id
        assert user.trial_remaining == 10
        assert user.is_admin == 0

    @pytest.mark.asyncio
    async def test_ensure_user_returns_existing_user(self, db):
        # Arrange
        user_id = 123
        await db.ensure_user(user_id, default_trial=10, is_admin=False)

        # Act
        user = await db.ensure_user(user_id, default_trial=5, is_admin=True)

        # Assert - должен вернуть существующего пользователя, а не создать нового
        assert user.user_id == user_id
        assert user.trial_remaining == 10  # Не изменилось
        assert user.is_admin == 0  # Не изменилось

    @pytest.mark.asyncio
    async def test_decrement_trial_success(self, db):
        # Arrange
        user_id = 123
        await db.ensure_user(user_id, default_trial=5, is_admin=False)

        # Act
        result = await db.decrement_trial(user_id)

        # Assert
        assert result is True
        user = await db.get_user(user_id)
        assert user.trial_remaining == 4

    @pytest.mark.asyncio
    async def test_decrement_trial_no_remaining(self, db):
        # Arrange
        user_id = 123
        await db.ensure_user(user_id, default_trial=0, is_admin=False)

        # Act
        result = await db.decrement_trial(user_id)

        # Assert
        assert result is False
        user = await db.get_user(user_id)
        assert user.trial_remaining == 0

    @pytest.mark.asyncio
    async def test_create_transaction(self, db):
        # Arrange
        user_id = 123
        await db.ensure_user(user_id, default_trial=10, is_admin=False)

        # Act
        transaction_id = await db.create_transaction(
            user_id=user_id,
            provider="telegram",
            currency="XTR",
            amount=100,
            payload="test_payload",
            status="pending"
        )

        # Assert
        assert transaction_id is not None
        transaction = await db.get_transaction(transaction_id)
        assert transaction.user_id == user_id
        assert transaction.provider == "telegram"
        assert transaction.amount == 100

    @pytest.mark.asyncio
    async def test_update_subscription(self, db):
        # Arrange
        user_id = 123
        await db.ensure_user(user_id, default_trial=10, is_admin=False)
        subscription_until = 1234567890

        # Act
        await db.update_subscription(user_id, subscription_until)

        # Assert
        user = await db.get_user(user_id)
        assert user.subscription_until == subscription_until

    @pytest.mark.asyncio
    async def test_has_active_subscription(self, db):
        # Arrange
        user_id = 123
        await db.ensure_user(user_id, default_trial=10, is_admin=False)

        # Тест без подписки
        result = await db.has_active_subscription(user_id)
        assert result is False

        # Добавляем активную подписку (далеко в будущем)
        future_timestamp = 9999999999
        await db.update_subscription(user_id, future_timestamp)

        result = await db.has_active_subscription(user_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_connection_pool_multiple_connections(self, db):
        # Arrange & Act - получаем несколько соединений одновременно
        connections = []
        try:
            async with db.get_connection() as conn1:
                connections.append(conn1)
                async with db.get_connection() as conn2:
                    connections.append(conn2)

                    # Assert - должны быть разные соединения
                    assert conn1 is not conn2

                    # Проверяем что оба работают
                    cursor1 = await conn1.execute("SELECT 1")
                    result1 = await cursor1.fetchone()

                    cursor2 = await conn2.execute("SELECT 2")
                    result2 = await cursor2.fetchone()

                    assert result1[0] == 1
                    assert result2[0] == 2
        except Exception as e:
            pytest.fail(f"Connection pool failed: {e}")

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db):
        # Arrange
        user_id = 123
        await db.ensure_user(user_id, default_trial=10, is_admin=False)

        # Act - пытаемся выполнить транзакцию с ошибкой
        try:
            async with db.get_connection() as conn:
                async with conn.execute("BEGIN"):
                    await conn.execute(
                        "UPDATE users SET trial_remaining = ? WHERE user_id = ?",
                        (5, user_id)
                    )
                    # Имитируем ошибку
                    raise Exception("Test error")
        except Exception:
            pass  # Ожидаем ошибку

        # Assert - изменения должны быть отменены
        user = await db.get_user(user_id)
        assert user.trial_remaining == 10  # Не изменилось