# RAG Quickstart

Сборка Retrieval-Augmented Generation над юридическими кейсами позволяет боту ссылаться на судебную практику при ответах.

## 1. Запуск Qdrant локально
```bash
docker run -d \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

## 2. Настройка окружения
Добавьте в `.env` или секрет-менеджер:
```env
RAG_ENABLED=true
RAG_QDRANT_URL=http://localhost:6333
RAG_COLLECTION=judicial_practice
RAG_TOP_K=6
RAG_SCORE_THRESHOLD=0.65
```
При использовании Qdrant Cloud добавьте `RAG_QDRANT_API_KEY`.

## 3. Загрузка корпуса
Подготовьте JSONL с полями `text`, `case_number`, `court`, `date`, `url` (см. пример в `data/judicial_cases_example.jsonl`) и выполните:
```bash
poetry run python scripts/load_judicial_practice.py --input data/judicial_cases_example.jsonl
```
Скрипт создаст коллекцию, рассчитает эмбеддинги (через OpenAI) и загрузит документы в Qdrant.

## 4. Локальная проверка
```bash
poetry run telegram-legal-bot
```
Задайте вопрос в духе «Какая практика по спорам с контрагентами о поставках?» и убедитесь, что ответ содержит выдержки и ссылки.

## 5. Диагностика
- Проверить коллекции: `curl http://localhost:6333/collections`.
- Проверить количество документов: `curl http://localhost:6333/collections/judicial_practice`.
- Логи загрузки — в `logs/load_judicial_practice.log` (создаётся автоматически).

## 6. Тюнинг
- `RAG_TOP_K` — сколько документов подтягивать на ответ.
- `RAG_SCORE_THRESHOLD` — минимальный скор; понизьте, если ответы не содержат фактов.
- `RAG_CONTEXT_CHAR_LIMIT` — ограничение длины вставляемых выдержек.

Если RAG недоступен (Qdrant не запущен или сеть закрыта), бот автоматически продолжит работу без расширения контекстом.

