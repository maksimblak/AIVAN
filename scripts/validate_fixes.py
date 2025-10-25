#!/usr/bin/env python3
"""
Валидация критических исправлений в проекте AIVAN
"""

import re
import sys
from pathlib import Path


def check_user_message_fixes():
    """Проверка исправления user_message → user_text"""
    print("🔍 Проверка исправления параметров LLM вызовов...")

    files_to_check = [
        "src/documents/translator.py",
        "src/documents/summarizer.py",
        "src/documents/risk_analyzer.py",
        "src/documents/document_chat.py",
    ]

    issues = []
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            # Ищем проблемные вызовы user_message= в контексте ask_legal
            pattern = r"ask_legal\([^)]*user_message="
            matches = re.findall(pattern, content)
            if matches:
                issues.append(f"❌ {file_path}: найдены вызовы с user_message")
            else:
                print(f"✅ {file_path}: user_message исправлен")
        else:
            issues.append(f"⚠️  {file_path}: файл не найден")

    return len(issues) == 0, issues


def check_path_traversal_fix():
    """Проверка исправления path traversal"""
    print("\n🔍 Проверка исправления path traversal...")

    base_file = Path("src/documents/base.py")
    if not base_file.exists():
        return False, ["❌ src/documents/base.py не найден"]

    content = base_file.read_text(encoding="utf-8")

    # Проверяем наличие метода _sanitize_filename
    if "_sanitize_filename" not in content:
        return False, ["❌ Метод _sanitize_filename не найден"]

    # Проверяем использование метода в save_document
    if "self._sanitize_filename(original_name)" not in content:
        return False, ["❌ _sanitize_filename не используется в save_document"]

    # Проверяем что старый небезопасный код заменен
    if "original_name.replace(' ', '_')" in content:
        return False, ["❌ Остался небезопасный код замены пробелов"]

    print("✅ Path traversal исправлен")
    return True, []


def check_html_markdown_fixes():
    """Проверка исправления HTML/Markdown"""
    print("\n🔍 Проверка исправления HTML/Markdown форматирования...")

    manager_file = Path("src/documents/document_manager.py")
    if not manager_file.exists():
        return False, ["❌ src/documents/document_manager.py не найден"]

    content = manager_file.read_text(encoding="utf-8")

    issues = []

    # Проверяем что markdown ** заменен на HTML <b>
    markdown_patterns = [
        r"\*\*[^*]+\*\*",  # **текст**
        r"\*[^*]+\*(?!\*)",  # *текст* (но не **)
    ]

    for pattern in markdown_patterns:
        matches = re.findall(pattern, content)
        if matches:
            # Исключаем комментарии и строки
            actual_issues = []
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if re.search(pattern, line) and not line.strip().startswith("#") and 'f"' in line:
                    actual_issues.append(f"Строка {i+1}: {line.strip()}")

            if actual_issues:
                issues.extend(actual_issues)

    # Проверяем использование html_escape
    if "html_escape" not in content:
        issues.append("❌ html_escape не импортирован или не используется")

    # Проверяем что ошибки в format_result_for_telegram исправлены
    if "**Ошибка обработки**" in content:
        issues.append("❌ Остался markdown в error сообщениях")

    if issues:
        return False, issues
    else:
        print("✅ HTML/Markdown форматирование исправлено")
        return True, []


def main():
    """Основная функция валидации"""
    print("🚨 Валидация критических исправлений AIVAN")
    print("=" * 50)

    all_checks = [
        ("Параметры LLM вызовов", check_user_message_fixes),
        ("Path traversal уязвимость", check_path_traversal_fix),
        ("HTML/Markdown форматирование", check_html_markdown_fixes),
    ]

    total_passed = 0
    total_checks = len(all_checks)

    for check_name, check_func in all_checks:
        try:
            passed, issues = check_func()
            if passed:
                total_passed += 1
            else:
                print(f"\n❌ {check_name} - ПРОБЛЕМЫ:")
                for issue in issues:
                    print(f"   {issue}")
        except Exception as e:
            print(f"\n💥 Ошибка в проверке '{check_name}': {e}")

    print("\n" + "=" * 50)
    if total_passed == total_checks:
        print("🎉 ВСЕ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ!")
        print("✅ Проект готов к безопасному использованию")
        return 0
    else:
        print(f"⚠️  Пройдено проверок: {total_passed}/{total_checks}")
        print("❌ Требуются дополнительные исправления")
        return 1


if __name__ == "__main__":
    sys.exit(main())
