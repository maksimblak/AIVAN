#!/usr/bin/env python3
"""
Тест streaming функциональности без реального OpenAI API
Симулирует streaming ответы для проверки работы механизма
"""

import asyncio
import os
import sys
import time

# Импортируем необходимые модули
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "core"))
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "bot"))

from src.bot.stream_manager import StreamingCallback, StreamManager
from src.bot.ui_components import render_legal_html


# Мок бот для тестирования
class MockBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, parse_mode=None, disable_web_page_preview=True):
        message = {
            "message_id": len(self.messages) + 1,
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        self.messages.append(message)
        print(f"[SEND] Message {message['message_id']}: {text[:100]}...")
        return MockMessage(message)

    async def edit_message_text(self, chat_id, message_id, text, parse_mode=None):
        # Находим сообщение и обновляем
        for msg in self.messages:
            if msg["message_id"] == message_id and msg["chat_id"] == chat_id:
                msg["text"] = text
                msg["parse_mode"] = parse_mode
                print(f"[EDIT] Message {message_id}: {text[:100]}...")
                return
        raise Exception("Message not found")


class MockMessage:
    def __init__(self, data):
        self.message_id = data["message_id"]
        self.chat = type("Chat", (), {"id": data["chat_id"]})()


# Симуляция streaming ответа OpenAI
STREAMING_CHUNKS = [
    "Коротко: да",
    "Коротко: да — расторгнуть договор поставки",
    "Коротко: да — расторгнуть договор поставки за просрочку можно",
    "Коротко: да — расторгнуть договор поставки за просрочку можно, но только",
    "Коротко: да — расторгнуть договор поставки за просрочку можно, но только при выполнении определённых условий.",
    "Коротко: да — расторгнуть договор поставки за просрочку можно, но только при выполнении определённых условий.\n\n1) Правовая основа",
    "Коротко: да — расторгнуть договор поставки за просрочку можно, но только при выполнении определённых условий.\n\n1) Правовая основа — что говорит закон\n\n— Односторонний отказ",
    "Коротко: да — расторгнуть договор поставки за просрочку можно, но только при выполнении определённых условий.\n\n1) Правовая основа — что говорит закон\n\n— Односторонний отказ от исполнения договора поставки допускается в случае существенного нарушения договора одной из сторон (ст. 523 ГК РФ).",
    """Коротко: да — расторгнуть договор поставки за просрочку можно, но только при выполнении определённых условий.

1) Правовая основа — что говорит закон

— Односторонний отказ от исполнения договора поставки допускается в случае существенного нарушения договора одной из сторон (ст. 523 ГК РФ). Существенным нарушение для поставщика в том числе признаётся неоднократное нарушение сроков поставки.

— Право на односторонний отказ закреплено в ст. 450.1 ГК РФ — договор считается расторгнутым с момента получения уведомления.

2) Практический вывод: когда расторжение реально

— Реально (высокая вероятность успеха): неоднократные просрочки поставщика; системные срывы графика поставок.

— Менее реально: единичная, несущественная просрочка без доказательств вреда.""",
]


async def simulate_streaming_callback(callback_func):
    """Симулирует streaming от OpenAI с постепенным добавлением текста"""
    print("\n=== СИМУЛЯЦИЯ STREAMING ОТВЕТА ===")

    for i, chunk in enumerate(STREAMING_CHUNKS):
        is_final = i == len(STREAMING_CHUNKS) - 1

        print(f"\nChunk {i+1}/{len(STREAMING_CHUNKS)} ({'FINAL' if is_final else 'PARTIAL'})")
        print(f"Length: {len(chunk)} chars")

        # Вызываем callback
        await callback_func(chunk, is_final)

        # Пауза между чанками (как в реальном streaming)
        if not is_final:
            await asyncio.sleep(0.5)


async def test_streaming_manager():
    """Тестирует StreamManager с симуляцией streaming"""
    print("=" * 60)
    print("ТЕСТ STREAMING MANAGER")
    print("=" * 60)

    # Создаем мок объекты
    bot = MockBot()
    chat_id = 12345

    # Создаем StreamManager
    stream_manager = StreamManager(
        bot=bot, chat_id=chat_id, update_interval=1.0, buffer_size=50  # Быстрее для теста
    )

    print("\n1. Начинаем streaming...")
    initial_message = await stream_manager.start_streaming("Обдумываю ваш вопрос...")
    print(f"Начальное сообщение ID: {initial_message.message_id}")

    print("\n2. Создаем callback...")
    callback = StreamingCallback(stream_manager)

    print("\n3. Симулируем streaming ответ...")
    await simulate_streaming_callback(callback)

    print("\n4. Проверяем финальное сообщение...")
    final_message = bot.messages[-1]
    print(f"Финальная длина: {len(final_message['text'])} символов")
    print(f"Содержит HTML теги: {'<b>' in final_message['text']}")
    print(f"Содержит переносы: {'<br>' in final_message['text']}")

    print("\n5. Проверяем форматирование...")
    formatted_text = render_legal_html(STREAMING_CHUNKS[-1])
    print(f"Форматированный текст: {len(formatted_text)} символов")

    # Проверяем наличие ключевых элементов форматирования
    checks = {
        "Жирные номера": "<b>1) </b>" in formatted_text,
        "Жирные статьи": "<b>ст. 523</b>" in formatted_text,
        "Жирные заголовки": "<b>Коротко:" in formatted_text,
        "Переносы строк": "<br>" in formatted_text,
    }

    print("\n6. Результаты проверки форматирования:")
    all_passed = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status} - {check_name}")
        if not passed:
            all_passed = False

    print("\n7. История сообщений:")
    for i, msg in enumerate(bot.messages):
        print(f"  Сообщение {i+1}: {len(msg['text'])} символов, parse_mode: {msg['parse_mode']}")

    return all_passed


async def test_streaming_performance():
    """Тестирует производительность streaming"""
    print("\n" + "=" * 60)
    print("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ STREAMING")
    print("=" * 60)

    # Создаем мок объекты
    bot = MockBot()
    stream_manager = StreamManager(
        bot=bot, chat_id=12345, update_interval=0.1, buffer_size=20  # Быстрое обновление для теста
    )

    start_time = time.time()

    await stream_manager.start_streaming("Тест производительности...")
    callback = StreamingCallback(stream_manager)

    # Симулируем быстрый streaming
    for i in range(10):
        text = f"Тест сообщение {i+1}: " + "Текст " * (i * 10)
        await callback(text, i == 9)
        await asyncio.sleep(0.05)

    end_time = time.time()
    duration = end_time - start_time

    print(f"Время выполнения: {duration:.2f} секунд")
    print(f"Количество сообщений: {len(bot.messages)}")
    print(f"Среднее время на сообщение: {duration/len(bot.messages):.3f} сек")

    return duration < 5.0  # Должно выполниться быстро


async def test_streaming_error_handling():
    """Тестирует обработку ошибок в streaming"""
    print("\n" + "=" * 60)
    print("ТЕСТ ОБРАБОТКИ ОШИБОК")
    print("=" * 60)

    # Создаем бот который выбрасывает ошибки
    class ErrorBot(MockBot):
        def __init__(self):
            super().__init__()
            self.error_count = 0

        async def edit_message_text(self, chat_id, message_id, text, parse_mode=None):
            self.error_count += 1
            if self.error_count <= 2:  # Первые 2 вызова - ошибка
                print(f"[ERROR] Симуляция ошибки #{self.error_count}")
                raise Exception("Simulated Telegram error")
            else:
                return await super().edit_message_text(chat_id, message_id, text, parse_mode)

    error_bot = ErrorBot()
    stream_manager = StreamManager(
        bot=error_bot,
        chat_id=12345,
        update_interval=0.5,
        max_retries=3,  # Должно хватить для восстановления
    )

    try:
        await stream_manager.start_streaming("Тест ошибок...")
        callback = StreamingCallback(stream_manager)

        # Отправляем несколько обновлений
        await callback("Тест 1", False)
        await asyncio.sleep(0.6)  # Ждем обновления
        await callback("Тест 2 - это должно пройти", False)
        await asyncio.sleep(0.6)
        await callback("Финальное сообщение", True)

        print("Тест обработки ошибок завершен успешно")
        return True

    except Exception as e:
        print(f"Тест обработки ошибок провален: {e}")
        return False


async def main():
    """Запуск всех тестов streaming"""
    print("ЗАПУСК ТЕСТОВ STREAMING ФУНКЦИОНАЛЬНОСТИ")
    print("=" * 60)

    tests = [
        ("Основной функционал streaming", test_streaming_manager),
        ("Производительность streaming", test_streaming_performance),
        ("Обработка ошибок", test_streaming_error_handling),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\nЗапуск теста: {test_name}")
        try:
            success = await test_func()
            results.append((test_name, success))
            status = "PASS" if success else "FAIL"
            print(f"Результат: {status}")
        except Exception as e:
            print(f"Ошибка в тесте: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)

    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1

    print(f"\nПройдено тестов: {passed}/{len(results)}")

    if passed == len(results):
        print("ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("Streaming готов к использованию в продакшене")
    else:
        print("НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ!")
        print("Требуется дополнительная отладка")


if __name__ == "__main__":
    asyncio.run(main())
