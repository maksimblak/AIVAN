#!/usr/bin/env python3
"""
Скрипт запуска тестов AIVAN с отчетностью
"""

import os
import subprocess
import sys
from pathlib import Path


# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
os.environ["PYTHONPATH"] = str(project_root)


def run_command(command: str, description: str) -> bool:
    """Выполнение команды с выводом результата"""
    print(f"\n🔍 {description}")
    print("-" * 50)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=project_root
        )

        if result.returncode == 0:
            print("✅ Успешно")
            if result.stdout.strip():
                print(result.stdout)
        else:
            print("❌ Ошибка")
            if result.stderr.strip():
                print(result.stderr)
            if result.stdout.strip():
                print(result.stdout)

        return result.returncode == 0

    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False


def main():
    """Основная функция запуска тестов"""
    print("🧪 AIVAN Test Suite")
    print("=" * 50)

    success_count = 0
    total_count = 0

    has_tests = (project_root / "tests").exists()
    commands = []

    if (project_root / "pyproject.toml").exists():
        commands.append(("poetry check", "Проверка корректности pyproject.toml"))

    if has_tests:
        commands.append(("poetry run pytest tests -v", "Запуск pytest"))
        commands.append(("poetry run pytest --cov=src --cov-report=term-missing", "Отчет о покрытии"))

    if (project_root / "src").exists():
        commands.append(("poetry run ruff check src tests", "Проверка стиля кода (Ruff)"))
        commands.append(("poetry run black --check src tests", "Проверка форматирования (Black)"))
        commands.append(("poetry run mypy src", "Проверка типов (MyPy)"))

    for command, description in commands:
        total_count += 1
        if run_command(command, description):
            success_count += 1

    # Дополнительная проверка - запуск валидации проекта
    validation_cmd = "poetry run python scripts/validate_project.py"
    if (project_root / "scripts" / "validate_project.py").exists():
        total_count += 1
        if run_command(validation_cmd, "Валидация проекта"):
            success_count += 1

    # Итоги
    print("\n" + "=" * 50)
    print(f"📊 Результаты тестирования: {success_count}/{total_count}")

    if success_count == total_count:
        print("🎉 Все тесты пройдены успешно!")
        return 0
    else:
        print(f"⚠️  Провалено тестов: {total_count - success_count}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Тестирование прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)
