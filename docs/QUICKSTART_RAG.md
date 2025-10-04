# Быстрый старт RAG для поиска судебной практики

## За 5 минут до первого поиска

### 1. Запустите Qdrant (Docker)

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 2. Настройте .env

Добавьте минимальные параметры:

```env
RAG_ENABLED=true
RAG_QDRANT_URL=http://localhost:6333
RAG_COLLECTION=judicial_practice
```

### 3. Загрузите тестовые данные

```bash
# Используйте пример из репозитория
python scripts/load_judicial_practice.py --input data/judicial_cases_example.jsonl
```

### 4. Запустите бота

```bash
python -m src.core.main_simple
```

### 5. Протестируйте

1. Откройте бота в Telegram
2. Нажмите **🔍 Поиск судебной практики**
3. Спросите: "Найди практику по взысканию неустойки с застройщика"

## Что дальше?

### Добавить свои дела

Создайте файл `data/my_cases.jsonl`:

```jsonl
{"text": "Текст решения...", "case_number": "А40-12345/2023", "court": "АС Москвы", "date": "2023-10-15", "url": "https://sudact.ru/..."}
{"text": "Текст решения...", "case_number": "А56-78901/2023", "court": "АС СПб", "date": "2023-09-20"}
```

Загрузите:

```bash
python scripts/load_judicial_practice.py --input data/my_cases.jsonl
```

### Настроить качество поиска

```env
RAG_TOP_K=10                    # Больше результатов
RAG_SCORE_THRESHOLD=0.75        # Выше порог релевантности
RAG_CONTEXT_CHAR_LIMIT=8000     # Больше контекста
```

### Использовать Qdrant Cloud

```env
RAG_QDRANT_URL=https://your-cluster.cloud.qdrant.io
RAG_QDRANT_API_KEY=your-api-key
```

## Полная документация

См. `docs/RAG_SETUP.md` для детальной информации.

## Проверка работы

### Qdrant запущен?

```bash
curl http://localhost:6333/collections
```

### Данные загружены?

```bash
curl http://localhost:6333/collections/judicial_practice
```

Должно показать количество точек (points_count).

## Troubleshooting

**RAG не находит дела** → Снизьте `RAG_SCORE_THRESHOLD` до 0.5 или уберите совсем

**Ошибка подключения к Qdrant** → Проверьте `docker ps` и `RAG_QDRANT_URL`

**Долгий поиск** → Уменьшите `RAG_TOP_K` до 3-5
