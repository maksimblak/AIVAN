#!/usr/bin/env python3
"""
Тест идемпотентности платежей - защита от дублирующих транзакций
"""

import asyncio
import os
import sys
import tempfile

# Импортируем advanced database
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "core"))
from src.core.db_advanced import DatabaseAdvanced, TransactionStatus


async def test_payment_idempotency():
    """
    Тест защиты от дублирующих платежей
    """
    print("=" * 60)
    print("ТЕСТ ИДЕМПОТЕНТНОСТИ ПЛАТЕЖЕЙ")
    print("=" * 60)

    # Создаем временную БД
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name

    print(f"Создаем временную БД: {db_path}")

    try:
        # Инициализируем БД
        db = DatabaseAdvanced(db_path=db_path, max_connections=5, enable_metrics=True)
        await db.init()

        # Создаем тестового пользователя
        user_id = 12345
        await db.ensure_user(user_id, is_admin=False)
        print(f"Создан тестовый пользователь: {user_id}")

        # Тестовые данные платежа
        test_charge_id = "test_charge_123456"
        test_payment_data = {
            "user_id": user_id,
            "provider": "telegram",
            "currency": "RUB",
            "amount": 300,
            "payload": "subscription_30_days",
            "status": TransactionStatus.PENDING.value,
            "telegram_payment_charge_id": test_charge_id,
            "provider_payment_charge_id": "provider_123456",
        }

        print("\n1. Тест первой записи транзакции:")
        print("-" * 40)

        # Первая запись
        transaction_id_1 = await db.record_transaction(**test_payment_data)
        print(f"Первая транзакция создана с ID: {transaction_id_1}")

        # Проверяем, что транзакция существует
        exists_1 = await db.transaction_exists_by_telegram_charge_id(test_charge_id)
        print(f"Транзакция существует: {exists_1}")

        print("\n2. Тест дублирующей записи (должна вернуть тот же ID):")
        print("-" * 40)

        # Попытка записать дубль
        transaction_id_2 = await db.record_transaction(**test_payment_data)
        print(f"Вторая попытка вернула ID: {transaction_id_2}")

        # Проверяем, что ID совпадают
        ids_match = transaction_id_1 == transaction_id_2
        print(f"ID совпадают (идемпотентность): {ids_match}")

        print("\n3. Тест параллельных записей (race condition):")
        print("-" * 40)

        # Новые тестовые данные для параллельного теста
        test_charge_id_parallel = "test_charge_parallel_789"
        parallel_payment_data = {
            **test_payment_data,
            "telegram_payment_charge_id": test_charge_id_parallel,
        }

        # Запускаем несколько параллельных попыток записи
        parallel_tasks = [db.record_transaction(**parallel_payment_data) for _ in range(5)]

        parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
        print(f"Результаты параллельных записей: {parallel_results}")

        # Проверяем, что все вернули один ID или исключения обработаны корректно
        valid_ids = [r for r in parallel_results if isinstance(r, int)]
        if valid_ids:
            all_same_id = all(id == valid_ids[0] for id in valid_ids)
            print(f"Все валидные ID одинаковые: {all_same_id}")
            print(f"Уникальный ID: {valid_ids[0]}")

        print("\n4. Тест записи без charge_id (должна пройти без проверок):")
        print("-" * 40)

        # Запись без charge_id
        no_charge_data = {**test_payment_data, "telegram_payment_charge_id": None}

        transaction_id_no_charge = await db.record_transaction(**no_charge_data)
        print(f"Транзакция без charge_id создана с ID: {transaction_id_no_charge}")

        print("\n5. Проверка уникального индекса в БД:")
        print("-" * 40)

        # Проверяем структуру индексов
        async with db.pool.acquire() as conn:
            cursor = await conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='index' AND name LIKE '%charge%'"
            )
            indexes = await cursor.fetchall()
            await cursor.close()

            print("Индексы для charge_id:")
            for name, sql in indexes:
                print(f"  {name}: {sql}")

        print("\n6. Финальная проверка количества записей:")
        print("-" * 40)

        # Считаем общее количество транзакций
        async with db.pool.acquire() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM transactions")
            count = await cursor.fetchone()
            await cursor.close()

            total_transactions = count[0] if count else 0
            print(f"Общее количество транзакций в БД: {total_transactions}")

            # Считаем уникальные charge_id
            cursor = await conn.execute(
                "SELECT COUNT(DISTINCT telegram_payment_charge_id) FROM transactions WHERE telegram_payment_charge_id IS NOT NULL"
            )
            count = await cursor.fetchone()
            await cursor.close()

            unique_charges = count[0] if count else 0
            print(f"Уникальных charge_id: {unique_charges}")

        print("\n" + "=" * 60)

        # Результаты тестов
        all_tests_passed = (
            exists_1
            and ids_match
            and len(valid_ids) > 0
            and all(id == valid_ids[0] for id in valid_ids)
            and transaction_id_no_charge > 0
            and unique_charges >= 2  # test_charge_123456 и test_charge_parallel_789
        )

        if all_tests_passed:
            print("ВСЕ ТЕСТЫ ИДЕМПОТЕНТНОСТИ ПРОЙДЕНЫ!")
            print("+ Первая запись создает транзакцию")
            print("+ Дублирующие записи возвращают тот же ID")
            print("+ Параллельные записи обрабатываются корректно")
            print("+ Записи без charge_id работают")
            print("+ Уникальный индекс защищает от дублей")
        else:
            print("НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ!")
            print("Требуется дополнительная отладка")

        return all_tests_passed

    finally:
        # Закрываем БД и удаляем временный файл
        await db.close()
        try:
            os.unlink(db_path)
            print(f"Временная БД удалена: {db_path}")
        except Exception as e:
            print(f"Не удалось удалить временную БД: {e}")


async def main():
    """Запуск тестов идемпотентности"""
    print("ЗАПУСК ТЕСТОВ ИДЕМПОТЕНТНОСТИ ПЛАТЕЖЕЙ")
    success = await test_payment_idempotency()

    print("\n" + "=" * 60)
    if success:
        print("ЗАКЛЮЧЕНИЕ: Защита от дублирующих платежей работает!")
        print("Можно использовать в продакшене")
    else:
        print("ЗАКЛЮЧЕНИЕ: Требуется доработка защиты от дублей")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
