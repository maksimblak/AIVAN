# RAG Setup Guide

Документ описывает развёртывание Retrieval-Augmented Generation для юридической практики на основе Qdrant и OpenAI embeddings.

## Архитектура
1. Пользовательский запрос → базовый ответ OpenAI.
2. Если RAG включён, выполняется семантический поиск по Qdrant (коллекция `judicial_practice`).
3. Подобранные документы внедряются в prompt и формируют аргументированные ответы.

## Компоненты
- **Qdrant** — векторное хранилище (локальный Docker или Qdrant Cloud).
- **EmbeddingService** (`src/core/rag/embedding_service.py`) — обёртка над OpenAI `text-embedding-3-large`.
- **VectorStore** (`src/core/rag/vector_store.py`) — абстракция поверх Qdrant API.
- **JudicialRAG** (`src/core/rag/judicial_rag.py`) — логика поиска и преобразования контекста.

## Развёртывание Qdrant
```bash
docker run -d \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```
Для Qdrant Cloud используйте выданный URL и API ключ.

## Настройки окружения
```env
RAG_ENABLED=true
RAG_QDRANT_URL=http://localhost:6333
RAG_QDRANT_API_KEY=           # только для облака
RAG_COLLECTION=judicial_practice
RAG_TOP_K=6
RAG_SCORE_THRESHOLD=0.65
RAG_CONTEXT_CHAR_LIMIT=8000
```

## Загрузка данных
1. Подготовьте JSONL (см. `data/judicial_cases_example.jsonl`).
2. Выполните:
   ```bash
   poetry run python scripts/load_judicial_practice.py --input data/judicial_cases_example.jsonl
   ```
3. Проверьте логи и убедитесь, что коллекция создана.

## Эксплуатация
- При старте бота (`poetry run telegram-legal-bot`) убедитесь, что подключение к Qdrant проходит без ошибок.
- Метрики RAG: `rag_queries_total`, `rag_fallback_total`, `rag_fetch_latency_seconds`.
- В случае недоступности Qdrant бот автоматически переключается на fallback (ответ без ссылок).

## Тюнинг
- Повышайте `RAG_TOP_K` для более полного контекста (при достаточном лимите токенов).
- Уменьшайте `RAG_SCORE_THRESHOLD`, если результаты пустые.
- Настройте префиксы/суффиксы подсказок в `src/bot/promt.py` для улучшения стилистики ответов.

