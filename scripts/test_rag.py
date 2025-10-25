"""
Тестовый скрипт для проверки работы RAG поиска судебной практики.

Использование:
    python scripts/test_rag.py "Найди практику по взысканию неустойки"
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover - pytest not available in runtime
    pytest = None

if pytest and os.getenv("RUN_FULL_TESTS") != "1":
    pytestmark = pytest.mark.skip(reason="Manual RAG script; set RUN_FULL_TESTS=1 to enable.")

# Добавляем путь к проекту до импорта внутренних модулей
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pydantic import ValidationError

from src.core.rag.judicial_rag import JudicialPracticeRAG
from src.core.settings import AppSettings


async def test_rag(query: str) -> None:
    """Тестировать RAG поиск."""

    print(f"\n{'='*60}")
    print(f"Тестирование RAG поиска")
    print(f"{'='*60}\n")

    # Инициализация настроек
    try:
        settings = AppSettings.load()
    except ValidationError as exc:
        print("❌ Не удалось загрузить настройки приложения")
        print(f"Причина: {exc}")
        print("\nУбедитесь, что заданы переменные окружения TELEGRAM_BOT_TOKEN и OPENAI_API_KEY")
        return

    rag = JudicialPracticeRAG(settings)

    if not rag.enabled:
        print("❌ RAG не включен!")
        print("\nПроверьте настройки в .env:")
        print("  - RAG_ENABLED=true")
        print("  - RAG_QDRANT_URL или RAG_QDRANT_HOST")
        return

    print(f"✅ RAG инициализирован")
    print(f"📝 Запрос: {query}\n")

    try:
        # Поиск
        print("🔍 Ищу релевантную практику...\n")
        context, fragments = await rag.build_context(query)

        if not fragments:
            print("❌ Ничего не найдено")
            print("\nВозможные причины:")
            print("  1. База данных пуста - загрузите дела через load_judicial_practice.py")
            print("  2. Слишком высокий RAG_SCORE_THRESHOLD - попробуйте снизить или убрать")
            print("  3. Запрос не соответствует содержимому базы")
            return

        print(f"✅ Найдено дел: {len(fragments)}\n")
        print(f"{'='*60}")
        print("Найденные дела:")
        print(f"{'='*60}\n")

        for idx, fragment in enumerate(fragments, 1):
            print(f"\n[{idx}] {fragment.header or 'Без заголовка'}")
            print(f"    Score: {fragment.match.score:.4f}")
            print(f"    ID: {fragment.match.id}")
            print(f"    Excerpt: {fragment.excerpt[:200]}...")
            if fragment.match.metadata.get("url"):
                print(f"    URL: {fragment.match.metadata['url']}")

        print(f"\n{'='*60}")
        print("Контекст для GPT:")
        print(f"{'='*60}\n")
        print(context[:500] + "..." if len(context) > 500 else context)
        print(f"\nОбщая длина контекста: {len(context)} символов")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await rag.close()


async def main() -> None:
    if len(sys.argv) < 2:
        print('Использование: python scripts/test_rag.py "ваш запрос"')
        print("\nПримеры:")
        print('  python scripts/test_rag.py "неустойка с застройщика"')
        print('  python scripts/test_rag.py "отказ администрации в согласовании"')
        print('  python scripts/test_rag.py "трудовой договор срочный"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    await test_rag(query)


if __name__ == "__main__":
    asyncio.run(main())
