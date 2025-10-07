"""Тесты для DI контейнера"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.di_container import DIContainer, create_container, get_container, reset_container
from src.core.settings import AppSettings


class MockService:
    def __init__(self, dependency: str = "default"):
        self.dependency = dependency
        self.initialized = False

    async def init(self):
        self.initialized = True


class MockDependency:
    def __init__(self):
        self.value = "mock_dependency"


def _make_settings(tmp_path: Path | None = None) -> AppSettings:
    db_path = tmp_path / "test.db" if tmp_path else Path("test.db")
    env = {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "OPENAI_API_KEY": "test-key",
        "DB_PATH": str(db_path),
    }
    return AppSettings.load(env)


class TestDIContainer:

    def setup_method(self):
        self.container = DIContainer()

    def test_register_and_get_singleton(self):
        # Arrange
        mock_instance = MockService()

        # Act
        self.container.register_singleton(MockService, mock_instance)
        result = self.container.get(MockService)

        # Assert
        assert result is mock_instance
        assert self.container.get(MockService) is mock_instance  # Same instance

    def test_register_and_get_transient(self):
        # Arrange & Act
        self.container.register_transient(MockService, MockService)
        result1 = self.container.get(MockService)
        result2 = self.container.get(MockService)

        # Assert
        assert isinstance(result1, MockService)
        assert isinstance(result2, MockService)
        assert result1 is not result2  # Different instances

    def test_register_factory(self):
        # Arrange
        def factory():
            return MockService("factory_created")

        # Act
        self.container.register_factory(MockService, factory)
        result = self.container.get(MockService)

        # Assert
        assert isinstance(result, MockService)
        assert result.dependency == "factory_created"

    def test_dependency_injection(self):
        # Arrange
        self.container.register_singleton(MockDependency, MockDependency())
        self.container.register_transient(MockService, MockService)

        # Note: This test would require MockService to accept MockDependency
        # For now, just test basic functionality
        result = self.container.get(MockService)
        assert isinstance(result, MockService)

    def test_config_registration(self):
        # Arrange & Act
        self.container.register_config("test_key", "test_value")
        result = self.container.get_config("test_key")

        # Assert
        assert result == "test_value"

    def test_config_default_value(self):
        # Act
        result = self.container.get_config("non_existent", "default")

        # Assert
        assert result == "default"

    @pytest.mark.asyncio
    async def test_init_async_services(self):
        # Arrange
        mock_service = MockService()
        self.container.register_singleton(MockService, mock_service)

        # Act
        await self.container.init_async_services()

        # Assert
        assert mock_service.initialized

    @pytest.mark.asyncio
    async def test_cleanup(self):
        # Arrange
        mock_service = Mock()
        mock_service.close = AsyncMock()
        self.container.register_singleton(Mock, mock_service)

        # Act
        await self.container.cleanup()

        # Assert
        mock_service.close.assert_called_once()


class TestContainerFactory:

    def teardown_method(self):
        reset_container()

    def test_create_container(self, tmp_path: Path):
        settings = _make_settings(tmp_path)

        container = create_container(settings)

        assert isinstance(container, DIContainer)
        assert container.get(AppSettings) is settings
        assert container.get_config("subscription_price_rub") == settings.subscription_price_rub

        asyncio.run(container.cleanup())

    def test_get_container_singleton(self, tmp_path: Path):
        reset_container()
        settings = _make_settings(tmp_path)

        container1 = get_container(settings)
        container2 = get_container()

        assert container1 is container2

        asyncio.run(container1.cleanup())

    def test_reset_container(self, tmp_path: Path):
        settings = _make_settings(tmp_path)
        container1 = get_container(settings)

        reset_container()
        container2 = get_container(settings)

        assert container1 is not container2

        asyncio.run(container1.cleanup())
        asyncio.run(container2.cleanup())
