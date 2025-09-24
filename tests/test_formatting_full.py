#!/usr/bin/env python3
"""
Полный тест форматирования без вызова OpenAI API
Имитирует весь процесс обработки сообщения
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Импортируем функции из основного файла
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))
from src.core.main_simple import render_legal_html, _split_html_safely

# Имитируем ответ от OpenAI API с типичным юридическим текстом
MOCK_OPENAI_RESPONSE = """Коротко: да — расторгнуть договор поставки за просрочку можно, но только при выполнении определённых условий и с соблюдением процессуальной порядочности.

1) Правовая основа — что говорит закон

— Односторонний отказ от исполнения договора поставки допускается в случае существенного нарушения договора одной из сторон (ст. 523 ГК РФ). Существенным нарушение для поставщика в том числе признаётся неоднократное нарушение сроков поставки.

— Право на односторонний отказ закреплено в ст. 450.1 ГК РФ — договор считается расторгнутым с момента получения уведомления, если иное не установлено законом или договором.

2) Практический вывод: когда расторжение — реально, а когда — маловероятно

— Реально (высокая вероятность успеха): неоднократные просрочки поставщика; системные срывы графика поставок по партиям.

— Менее реально: единичная, несущественная просрочка без доказательств вреда/убытков.

3) Рекомендуемая практическая последовательность

1. Соберите доказательства: договора, графики поставок, акты недопоставки.
2. Направьте претензию поставщику с указанием срока устранения нарушений.
3. При отсутствии реакции — уведомление об одностороннем отказе от договора."""

async def test_full_formatting_process():
    """
    Тест полного процесса форматирования как в реальном боте
    """
    print("=" * 60)
    print("ТЕСТ ПОЛНОГО ПРОЦЕССА ФОРМАТИРОВАНИЯ")
    print("=" * 60)

    print("\n1. Исходный текст от 'OpenAI':")
    print("-" * 40)
    print(MOCK_OPENAI_RESPONSE)

    print("\n2. Применяем функцию render_legal_html:")
    print("-" * 40)

    # Шаг 1: Форматируем текст
    formatted_html = render_legal_html(MOCK_OPENAI_RESPONSE)
    print(formatted_html)

    print("\n3. Разбиваем на чанки для Telegram:")
    print("-" * 40)

    # Шаг 2: Разбиваем на части (как в реальном боте)
    chunks = _split_html_safely(formatted_html, hard_limit=3900)

    print(f"Количество чанков: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Чанк {i} (длина: {len(chunk)}) ---")
        print(chunk)

    print("\n4. Проверяем качество форматирования:")
    print("-" * 40)

    # Проверки
    checks = {
        "Жирные номера пунктов": any('<b>1) </b>' in chunk for chunk in chunks),
        "Жирные статьи ГК": any('<b>ст. 523</b>' in chunk for chunk in chunks),
        "Жирные заголовки": any('<b>Коротко:' in chunk for chunk in chunks),
        "Переносы строк": any('<br>' in chunk for chunk in chunks),
        "Нет двойного экранирования": not any('&lt;b&gt;' in chunk for chunk in chunks),
        "Правильные параграфы": any('<br><br>' in chunk for chunk in chunks)
    }

    all_passed = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status} - {check_name}")
        if not passed:
            all_passed = False

    print("\n5. Имитируем отправку в Telegram:")
    print("-" * 40)

    # Имитируем отправку каждого чанка
    for i, chunk in enumerate(chunks, 1):
        print(f"\nОтправляем чанк {i} в Telegram с ParseMode.HTML")
        print(f"Размер: {len(chunk)} символов")

        # Проверяем, что чанк не превышает лимит Telegram
        if len(chunk) > 4096:
            print("ВНИМАНИЕ: Чанк превышает лимит Telegram (4096 символов)!")
        else:
            print("Размер чанка в пределах лимита")

    print("\n6. Результат теста:")
    print("=" * 60)

    if all_passed:
        print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("Текст будет отображаться в Telegram с правильным форматированием:")
        print("   • Жирные номера пунктов")
        print("   • Жирные ссылки на статьи")
        print("   • Структурированные параграфы")
        print("   • Переносы строк")
    else:
        print("НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ!")
        print("Требуется дополнительная настройка форматирования")

    return all_passed

async def test_specific_cases():
    """
    Тест специфических случаев форматирования
    """
    print("\n" + "=" * 60)
    print("ТЕСТ СПЕЦИФИЧЕСКИХ СЛУЧАЕВ")
    print("=" * 60)

    test_cases = [
        {
            "name": "Номерованный список",
            "input": "1) Первый пункт\n2) Второй пункт\n3) Третий пункт",
            "expected": ["<b>1) </b>", "<b>2) </b>", "<b>3) </b>"]
        },
        {
            "name": "Ссылки на статьи",
            "input": "Согласно ст. 304 ГК РФ и Статья 222 ГК РФ",
            "expected": ["<b>ст. 304</b>", "<b>Статья 222</b>"]
        },
        {
            "name": "Заголовки",
            "input": "Коротко: основные моменты\nПрактический вывод:",
            "expected": ["<b>Коротко:", "<b>Практический вывод:</b>"]
        }
    ]

    for case in test_cases:
        print(f"\nТест: {case['name']}")
        print(f"Вход: {case['input']}")

        result = render_legal_html(case['input'])
        print(f"Результат: {result}")

        all_expected_found = all(expected in result for expected in case['expected'])
        status = "PASS" if all_expected_found else "FAIL"
        print(f"Статус: {status}")

        if not all_expected_found:
            missing = [exp for exp in case['expected'] if exp not in result]
            print(f"Не найдено: {missing}")

def test_telegram_html_limits():
    """
    Тест лимитов Telegram для HTML сообщений
    """
    print("\n" + "=" * 60)
    print("ТЕСТ ЛИМИТОВ TELEGRAM")
    print("=" * 60)

    # Создаем очень длинный текст
    long_text = "Очень длинный юридический текст. " * 200
    formatted = render_legal_html(long_text)
    chunks = _split_html_safely(formatted, hard_limit=3900)

    print(f"Исходный текст: {len(long_text)} символов")
    print(f"После форматирования: {len(formatted)} символов")
    print(f"Количество чанков: {len(chunks)}")

    for i, chunk in enumerate(chunks, 1):
        size = len(chunk)
        status = "OK" if size <= 4096 else "FAIL"
        print(f"Чанк {i}: {size} символов {status}")

async def main():
    """Запуск всех тестов"""
    print("ЗАПУСК ТЕСТОВ ФОРМАТИРОВАНИЯ БЕЗ API ВЫЗОВОВ")

    # Основной тест
    success = await test_full_formatting_process()

    # Дополнительные тесты
    await test_specific_cases()
    test_telegram_html_limits()

    print("\n" + "=" * 60)
    if success:
        print("ЗАКЛЮЧЕНИЕ: Форматирование работает корректно!")
        print("Можно тестировать в реальном Telegram боте")
    else:
        print("ЗАКЛЮЧЕНИЕ: Требуется доработка форматирования")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())