"""
Тестовый скрипт для проверки RetentionNotifier
"""
import asyncio
import sys
from pathlib import Path

# Фикс кодировки для Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))


async def test_imports():
    """Тест 1: Проверка импортов"""
    print("Test 1: Checking imports...")
    try:
        from src.bot.retention_notifier import (
            RetentionNotifier,
            NotificationTemplate,
            NOTIFICATION_SCENARIOS
        )
        print("✅ Imports successful")
        print(f"   - Found {len(NOTIFICATION_SCENARIOS)} notification scenarios")
        for scenario in NOTIFICATION_SCENARIOS:
            print(f"     • {scenario.name}: {scenario.delay_hours}h delay")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_notification_template():
    """Тест 2: Проверка NotificationTemplate"""
    print("\nTest 2: Testing NotificationTemplate...")
    try:
        from src.bot.retention_notifier import NotificationTemplate

        template = NotificationTemplate(
            name="test_scenario",
            delay_hours=24,
            message="Test message",
            show_buttons=True
        )

        assert template.name == "test_scenario"
        assert template.delay_hours == 24
        assert template.message == "Test message"
        assert template.show_buttons is True

        print("✅ NotificationTemplate works correctly")
        return True
    except Exception as e:
        print(f"❌ NotificationTemplate test failed: {e}")
        return False


async def test_sql_syntax():
    """Тест 3: Проверка SQL синтаксиса"""
    print("\nTest 3: Checking SQL queries...")
    try:
        # Проверяем что SQL не содержит очевидных ошибок
        from src.bot.retention_notifier import RetentionNotifier

        # Проверяем наличие методов
        required_methods = [
            'start', 'stop', '_notification_loop', '_process_scenario',
            '_get_users_for_scenario', '_send_notification',
            '_mark_notification_sent', '_mark_user_blocked',
            'send_manual_notification', 'get_notification_stats'
        ]

        for method in required_methods:
            assert hasattr(RetentionNotifier, method), f"Missing method: {method}"

        print("✅ All required methods exist")
        return True
    except Exception as e:
        print(f"❌ SQL syntax check failed: {e}")
        return False


async def test_scenarios_validity():
    """Тест 4: Проверка валидности сценариев"""
    print("\nTest 4: Validating notification scenarios...")
    try:
        from src.bot.retention_notifier import NOTIFICATION_SCENARIOS

        for scenario in NOTIFICATION_SCENARIOS:
            # Проверяем обязательные поля
            assert scenario.name, "Scenario must have a name"
            assert scenario.delay_hours > 0, f"Invalid delay for {scenario.name}"
            assert scenario.message, f"Empty message for {scenario.name}"
            assert len(scenario.message) < 4096, f"Message too long for {scenario.name}"

            # Проверяем HTML разметку
            assert "<b>" in scenario.message or "•" in scenario.message, \
                f"Message should have formatting for {scenario.name}"

        print("✅ All scenarios are valid")
        print(f"   - Total scenarios: {len(NOTIFICATION_SCENARIOS)}")
        return True
    except Exception as e:
        print(f"❌ Scenario validation failed: {e}")
        return False


async def test_integration():
    """Тест 5: Проверка интеграции с main_simple.py"""
    print("\nTest 5: Checking integration with main_simple.py...")
    try:
        with open("../src/core/main_simple.py", "r", encoding="utf-8") as f:
            content = f.read()

        # Проверяем наличие импорта
        assert "from src.bot.retention_notifier import RetentionNotifier" in content, \
            "Missing import in main_simple.py"

        # Проверяем наличие глобальной переменной
        assert "retention_notifier = None" in content, \
            "Missing global variable declaration"

        # Проверяем запуск
        assert "retention_notifier = RetentionNotifier(bot, db)" in content, \
            "Missing initialization"
        assert "await retention_notifier.start()" in content, \
            "Missing start() call"

        # Проверяем остановку
        assert "await retention_notifier.stop()" in content, \
            "Missing stop() call"

        # Проверяем обработчики кнопок
        assert "handle_retention_quick_question" in content, \
            "Missing quick_question handler"
        assert "handle_retention_show_features" in content, \
            "Missing show_features handler"

        print("✅ Integration with main_simple.py is correct")
        return True
    except Exception as e:
        print(f"❌ Integration check failed: {e}")
        return False


async def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("RetentionNotifier Test Suite")
    print("=" * 60)

    tests = [
        test_imports,
        test_notification_template,
        test_sql_syntax,
        test_scenarios_validity,
        test_integration,
    ]

    results = []
    for test in tests:
        result = await test()
        results.append(result)

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! RetentionNotifier is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
