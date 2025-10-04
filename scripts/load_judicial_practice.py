"""
Скрипт для загрузки судебной практики в Qdrant векторную базу данных.

Использование:
    python scripts/load_judicial_practice.py --input data/judicial_cases.jsonl

Формат входных данных (JSONL):
    {"text": "Решение суда...", "case_number": "А40-12345/2023", "court": "Арбитражный суд", "date": "2023-05-15", "url": "https://..."}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Добавляем путь к проекту для импорта модулей
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import AppSettings
from src.core.rag.embedding_service import EmbeddingService
from src.core.rag.vector_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Загрузить данные из JSONL файла."""
    cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                if not isinstance(case, dict):
                    logger.warning(f"Line {line_num}: expected dict, got {type(case)}")
                    continue
                if 'text' not in case:
                    logger.warning(f"Line {line_num}: missing 'text' field")
                    continue
                cases.append(case)
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error: {e}")
                continue
    return cases


async def load_to_qdrant(
    cases: list[dict[str, Any]],
    *,
    settings: AppSettings,
    batch_size: int = 64,
) -> None:
    """Загрузить дела в Qdrant."""

    # Инициализация embedding service
    embedding_model = settings.get_str("RAG_EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_service = EmbeddingService(embedding_model or "text-embedding-3-large")

    # Инициализация vector store
    collection = settings.get_str("RAG_COLLECTION", "judicial_practice") or "judicial_practice"
    url = settings.get_str("RAG_QDRANT_URL")
    host = settings.get_str("RAG_QDRANT_HOST")
    api_key = settings.get_str("RAG_QDRANT_API_KEY")

    port_value = settings.get_str("RAG_QDRANT_PORT")
    port = int(port_value) if port_value else None

    prefer_grpc = settings.get_bool("RAG_QDRANT_GRPC", False)
    timeout_value = settings.get_str("RAG_QDRANT_TIMEOUT")
    timeout = float(timeout_value) if timeout_value else None

    vector_store = QdrantVectorStore(
        collection=collection,
        url=url,
        host=host,
        port=port,
        api_key=api_key,
        prefer_grpc=prefer_grpc,
        timeout=timeout,
    )

    logger.info(f"Подключение к Qdrant: collection={collection}")

    # Получаем тексты для эмбеддинга
    texts = [case['text'] for case in cases]

    # Создаём эмбеддинги батчами
    logger.info(f"Создание эмбеддингов для {len(texts)} документов...")
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Обработка батча {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
        vectors = await embedding_service.embed(batch)
        all_vectors.extend(vectors)

    # Определяем размер вектора
    if all_vectors:
        vector_size = len(all_vectors[0])
        logger.info(f"Размер вектора: {vector_size}")

        # Создаём коллекцию если не существует
        await vector_store.ensure_collection(vector_size)

        # Подготавливаем payloads (сохраняем все метаданные)
        payloads = []
        ids = []
        for idx, case in enumerate(cases):
            payload = {
                "text": case['text'],
                "case_number": case.get('case_number', ''),
                "court": case.get('court', ''),
                "date": case.get('date', ''),
                "url": case.get('url', ''),
                "title": case.get('title', ''),
                "region": case.get('region', ''),
            }
            # Удаляем пустые поля
            payload = {k: v for k, v in payload.items() if v}
            payloads.append(payload)

            # Генерируем ID из номера дела или индекса
            case_id = case.get('case_number', f"case_{idx}")
            ids.append(case_id)

        # Загружаем в Qdrant
        logger.info(f"Загрузка {len(all_vectors)} векторов в Qdrant...")
        await vector_store.upsert(
            vectors=all_vectors,
            payloads=payloads,
            ids=ids,
            batch_size=batch_size,
        )

        logger.info("✅ Загрузка завершена успешно!")
    else:
        logger.error("Не удалось создать эмбеддинги")

    await vector_store.close()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description='Загрузка судебной практики в Qdrant'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Путь к JSONL файлу с делами'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Размер батча для обработки (default: 64)'
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Файл не найден: {args.input}")
        sys.exit(1)

    # Загружаем настройки из .env
    settings = AppSettings()

    # Проверяем необходимые настройки
    if not settings.get_str("RAG_QDRANT_URL") and not settings.get_str("RAG_QDRANT_HOST"):
        logger.error("Ошибка: необходимо указать RAG_QDRANT_URL или RAG_QDRANT_HOST в .env")
        sys.exit(1)

    # Загружаем дела из файла
    logger.info(f"Загрузка данных из {args.input}")
    cases = load_jsonl(args.input)
    logger.info(f"Загружено дел: {len(cases)}")

    if not cases:
        logger.error("Нет данных для загрузки")
        sys.exit(1)

    # Загружаем в Qdrant
    await load_to_qdrant(cases, settings=settings, batch_size=args.batch_size)


if __name__ == '__main__':
    asyncio.run(main())
