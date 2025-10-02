#!/usr/bin/env python3
"""
Скрипт валидации проекта AIVAN
Проверяет все основные компоненты и зависимости
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from src.core.app_context import get_settings, set_settings

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def validate_dependencies():
    """Проверка зависимостей"""
    print("🔍 Проверка зависимостей...")

    try:
        import aiogram
        import openai
        import aiosqlite
        import httpx
        import pytest
        print("✅ Основные зависимости найдены")
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        return False

    # Проверка опциональных зависимостей
    optional_deps = {
        "redis": "Redis (опционально)",
        "prometheus_client": "Prometheus метрики (опционально)",
        "psutil": "Системный мониторинг (опционально)"
    }

    for dep, desc in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {desc}")
        except ImportError:
            print(f"⚠️  {desc} - не установлено")

    return True


async def validate_database():
    """Проверка базы данных"""
    print("\n🗄️  Проверка базы данных...")

    try:
        from src.core.db_advanced import DatabaseAdvanced

        # Создаем временную БД для тестов
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name

        try:
            db = DatabaseAdvanced(db_path, max_connections=2)
            await db.init()

            # Тест создания пользователя
            user = await db.ensure_user(123, default_trial=10, is_admin=False)
            assert user.user_id == 123

            # Тест декремента trial
            result = await db.decrement_trial(123)
            assert result is True

            await db.close()
            print("✅ База данных работает корректно")
            return True

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    except Exception as e:
        print(f"❌ Ошибка базы данных: {e}")
        return False


async def validate_di_container():
    """Проверка DI контейнера"""
    print("\n📦 Проверка DI контейнера...")

    container = None
    try:
        from src.core.di_container import create_container
        from src.core.settings import AppSettings
        from src.core.db_advanced import DatabaseAdvanced

        settings = get_settings()
        set_settings(settings)
        container = create_container(settings)
        assert container is not None

        resolved_settings = container.get(AppSettings)
        assert resolved_settings is settings

        db = container.get(DatabaseAdvanced)
        await db.init()
        await db.close()

        print("✅ DI контейнер работает корректно")
        return True

    except Exception as e:
        print(f"❌ Ошибка DI контейнера: {e}")
        return False

    finally:
        if container is not None:
            try:
                await container.cleanup()
            except Exception:
                pass



async def validate_performance():
    """Проверка компонентов производительности"""
    print("\n⚡ Проверка оптимизаций производительности...")

    try:
        from src.core.performance import (
            PerformanceMetrics, LRUCache, timing,
            get_performance_summary
        )

        # Тест метрик
        metrics = PerformanceMetrics()
        metrics.record_timing("test", 0.1)
        avg = metrics.get_average_timing("test")
        assert avg == 0.1

        # Тест кеша
        cache = LRUCache(max_size=10, ttl_seconds=3600)
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value"

        # Тест декоратора timing
        @timing("test_function")
        async def test_func():
            await asyncio.sleep(0.01)
            return "result"

        result = await test_func()
        assert result == "result"

        print("✅ Компоненты производительности работают корректно")
        return True

    except Exception as e:
        print(f"❌ Ошибка компонентов производительности: {e}")
        return False


async def validate_configuration():
    """Проверка конфигурации проекта"""
    print("\n⚙️  Проверка конфигурации...")

    # Проверка pyproject.toml
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        print("❌ Файл pyproject.toml не найден")
        return False

    with open(pyproject_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if "[project]" in content and "dependencies" in content:
            print("✅ pyproject.toml корректно настроен")
        else:
            print("❌ pyproject.toml имеет некорректную структуру")
            return False

    # Проверка Docker файлов
    dockerfile_path = project_root / "Dockerfile"
    if dockerfile_path.exists():
        print("✅ Dockerfile найден")
    else:
        print("❌ Dockerfile отсутствует")
        return False

    docker_compose_path = project_root / "docker-compose.yml"
    if docker_compose_path.exists():
        print("✅ docker-compose.yml найден")
    else:
        print("❌ docker-compose.yml отсутствует")
        return False

    # Проверка README
    readme_path = project_root / "README.md"
    if readme_path.exists():
        print("✅ README.md найден")
    else:
        print("❌ README.md отсутствует")
        return False

    return True


async def validate_tests():
    """Проверка тестов"""
    print("\n🧪 Проверка тестов...")

    tests_dir = project_root / "tests"
    if not tests_dir.exists():
        print("❌ Директория tests не найдена")
        return False

    # Проверка структуры тестов
    test_files = [
        "conftest.py",
        "unit/test_di_container.py",
        "unit/test_db_advanced.py",
        "unit/test_access_service.py"
    ]

    missing_files = []
    for test_file in test_files:
        if not (tests_dir / test_file).exists():
            missing_files.append(test_file)

    if missing_files:
        print(f"❌ Отсутствуют тестовые файлы: {missing_files}")
        return False

    print("✅ Структура тестов корректна")
    return True


async def main():
    """Основная функция валидации"""
    print("🚀 AIVAN Project Validation")
    print("=" * 50)

    validators = [
        validate_dependencies,
        validate_configuration,
        validate_database,
        validate_di_container,
        validate_performance,
        validate_tests,
    ]

    results = []
    for validator in validators:
        try:
            result = await validator()
            results.append(result)
        except Exception as e:
            print(f"❌ Критическая ошибка в {validator.__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 Все проверки пройдены! ({passed}/{total})")
        print("✅ Проект готов к использованию")
        return 0
    else:
        print(f"⚠️  Пройдено проверок: {passed}/{total}")
        print("❌ Требуются исправления")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Валидация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка валидации: {e}")
        sys.exit(1)