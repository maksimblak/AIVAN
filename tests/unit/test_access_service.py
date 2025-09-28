"""
Тесты для сервиса контроля доступа
"""

import pytest
from unittest.mock import AsyncMock, Mock
from src.core.access import AccessService, AccessDecision
from src.core.db_advanced import UserRecord


class TestAccessService:

    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        return db

    @pytest.fixture
    def access_service(self, mock_db):
        admin_ids = {123, 456}
        trial_limit = 10
        return AccessService(db=mock_db, trial_limit=trial_limit, admin_ids=admin_ids)

    @pytest.mark.asyncio
    async def test_admin_user_gets_access(self, access_service, mock_db):
        # Arrange
        user_id = 123  # Admin ID
        user_record = UserRecord(
            user_id=user_id,
            is_admin=1,
            trial_remaining=5,
            subscription_until=0,
            created_at=1234567890,
            updated_at=1234567890
        )
        mock_db.ensure_user.return_value = user_record

        # Act
        decision = await access_service.check_and_consume(user_id)

        # Assert
        assert decision.allowed is True
        assert decision.is_admin is True
        mock_db.ensure_user.assert_called_once_with(user_id, default_trial=10, is_admin=True)

    @pytest.mark.asyncio
    async def test_subscriber_gets_access(self, access_service, mock_db):
        # Arrange
        user_id = 789  # Not admin
        user_record = UserRecord(
            user_id=user_id,
            is_admin=0,
            trial_remaining=5,
            subscription_until=9999999999,  # Future date
            created_at=1234567890,
            updated_at=1234567890
        )
        mock_db.ensure_user.return_value = user_record
        mock_db.has_active_subscription.return_value = True

        # Act
        decision = await access_service.check_and_consume(user_id)

        # Assert
        assert decision.allowed is True
        assert decision.has_subscription is True
        assert decision.subscription_until == 9999999999

    @pytest.mark.asyncio
    async def test_trial_user_with_remaining_trials(self, access_service, mock_db):
        # Arrange
        user_id = 789
        user_record = UserRecord(
            user_id=user_id,
            is_admin=0,
            trial_remaining=5,
            subscription_until=0,
            created_at=1234567890,
            updated_at=1234567890
        )
        mock_db.ensure_user.return_value = user_record
        mock_db.has_active_subscription.return_value = False
        mock_db.decrement_trial.return_value = True

        # Act
        decision = await access_service.check_and_consume(user_id)

        # Assert
        assert decision.allowed is True
        assert decision.trial_used == 5
        assert decision.trial_remaining == 4
        mock_db.decrement_trial.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_trial_user_no_remaining_trials(self, access_service, mock_db):
        # Arrange
        user_id = 789
        user_record = UserRecord(
            user_id=user_id,
            is_admin=0,
            trial_remaining=0,
            subscription_until=0,
            created_at=1234567890,
            updated_at=1234567890
        )
        mock_db.ensure_user.return_value = user_record
        mock_db.has_active_subscription.return_value = False
        mock_db.decrement_trial.return_value = False

        # Act
        decision = await access_service.check_and_consume(user_id)

        # Assert
        assert decision.allowed is False
        assert decision.trial_used == 0
        assert decision.trial_remaining == 0

    @pytest.mark.asyncio
    async def test_admin_by_id_not_by_database_flag(self, access_service, mock_db):
        # Arrange - пользователь админ по ID, но не по флагу в БД
        user_id = 456  # Admin ID
        user_record = UserRecord(
            user_id=user_id,
            is_admin=0,  # Не админ в БД
            trial_remaining=5,
            subscription_until=0,
            created_at=1234567890,
            updated_at=1234567890
        )
        mock_db.ensure_user.return_value = user_record

        # Act
        decision = await access_service.check_and_consume(user_id)

        # Assert
        assert decision.allowed is True
        assert decision.is_admin is True

    @pytest.mark.asyncio
    async def test_admin_by_database_flag(self, access_service, mock_db):
        # Arrange - пользователь админ по флагу в БД, но не по ID
        user_id = 999  # Not in admin_ids
        user_record = UserRecord(
            user_id=user_id,
            is_admin=1,  # Админ в БД
            trial_remaining=5,
            subscription_until=0,
            created_at=1234567890,
            updated_at=1234567890
        )
        mock_db.ensure_user.return_value = user_record

        # Act
        decision = await access_service.check_and_consume(user_id)

        # Assert
        assert decision.allowed is True
        assert decision.is_admin is True