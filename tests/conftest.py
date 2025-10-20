"""
Pytest конфигурация и общие fixtures для тестов AIVAN
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock


# Настройка event loop для asyncio тестов
@pytest.fixture(scope="session")
def event_loop():
    """Создание event loop для всех тестов"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Создание временной директории для тестов"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Очистка после теста
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_db_path(temp_dir):
    """Путь к временной базе данных"""
    return os.path.join(temp_dir, "test.db")


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI клиента"""
    client = AsyncMock()

    # Mock для chat completions
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.return_value = Mock(
        choices=[
            Mock(
                message=Mock(content="Test response"),
                finish_reason="stop"
            )
        ],
        usage=Mock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30
        )
    )

    # Mock для audio
    client.audio.speech.create = AsyncMock()
    client.audio.transcriptions.create = AsyncMock()
    client.audio.transcriptions.create.return_value = Mock(text="Test transcription")

    return client


@pytest.fixture
def mock_telegram_bot():
    """Mock Telegram бота"""
    bot = AsyncMock()
    bot.send_message = AsyncMock()
    bot.send_voice = AsyncMock()
    bot.send_document = AsyncMock()
    bot.edit_message_text = AsyncMock()
    return bot


@pytest.fixture
def sample_user_data():
    """Образцы данных пользователей для тестов"""
    return {
        "admin_user": {
            "user_id": 123,
            "is_admin": True,
            "trial_remaining": 10,
            "subscription_until": 0
        },
        "subscriber_user": {
            "user_id": 456,
            "is_admin": False,
            "trial_remaining": 5,
            "subscription_until": 9999999999
        },
        "trial_user": {
            "user_id": 789,
            "is_admin": False,
            "trial_remaining": 3,
            "subscription_until": 0
        },
        "expired_user": {
            "user_id": 999,
            "is_admin": False,
            "trial_remaining": 0,
            "subscription_until": 1234567890  # Past date
        }
    }


@pytest.fixture
def mock_config():
    """Mock конфигурации приложения"""
    return {
        "telegram_bot_token": "test_bot_token",
        "openai_api_key": "test_openai_key",
        "db_path": "test.db",
        "trial_requests": 10,
        "admin_ids": {123, 456},
        "subscription_price": 300,
        "enable_voice": True,
        "log_level": "DEBUG"
    }


@pytest.fixture
def sample_documents():
    """Образцы документов для тестирования"""
    return {
        "simple_text": "Это простой текстовый документ для тестирования.",
        "legal_document": """
        ИСКОВОЕ ЗАЯВЛЕНИЕ

        В Арбитражный суд города Москвы

        Истец: ООО "Тест"
        Ответчик: ООО "Ответчик"

        Прошу взыскать задолженность в размере 100 000 рублей.
        """,
        "contract": """
        ДОГОВОР ПОСТАВКИ № 123

        Поставщик: ООО "Поставщик"
        Покупатель: ООО "Покупатель"

        Предмет договора: поставка товаров
        Цена: 50 000 рублей
        """
    }


# Настройка для pytest-asyncio
pytest_plugins = ("pytest_asyncio",)

collect_ignore = ["scripts/test_rag.py", "scripts\\test_rag.py"]


# Маркеры для категоризации тестов
def pytest_configure(config):
    """Регистрация custom маркеров"""
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: slow tests")
    config.addinivalue_line("markers", "network: tests requiring network access")
    config.addinivalue_line("markers", "database: tests requiring database")


# Фикстура для пропуска тестов требующих внешних сервисов
@pytest.fixture
def skip_if_no_openai_key():
    """Пропуск тестов если нет OpenAI API ключа"""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture
def skip_if_no_telegram_token():
    """Пропуск тестов если нет Telegram токена"""
    import os
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        pytest.skip("TELEGRAM_BOT_TOKEN not set")





collect_ignore_glob = ["scripts/test_*.py"]
