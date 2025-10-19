# AIVAN – Telegram Legal Assistant

Modern Telegram бот для юридических консультаций с поддержкой OpenAI, документов, RAG-поиска и голосового взаимодействия. Репозиторий подготовлен для безопасного релиза и локальной разработки с Poetry и Docker.

## Возможности
- Диалог с GPT‑моделью (стриминг ответов, статус анимации, rich UI-компоненты).
- Обработка документов: OCR (PaddleOCR/Tesseract), анонимизация, анализ исков, генерация сводок и черновиков.
- RAG по судебной практике (Qdrant) и внутренним материалам.
- Платёжные функции: подписки, CryptoPay, Telegram Stars, Robokassa, YooKassa.
- Наблюдаемость: Prometheus-метрики, healthchecks, фоновый мониторинг и алёрты.

## Структура проекта
- `src/telegram_legal_bot` — CLI-энтрипоинт и healthcheck.
- `src/core` — конфигурация (`settings.py`), DI, расширенная БД (`db_advanced.py`), кэш, метрики, фоновые задачи, RAG.
- `src/bot` — шлюз OpenAI, промпты, управление статусами/стримингом, удержание пользователей.
- `src/documents` — инструменты OCR, анонимизация, риск-оценка и менеджмент файлов.
- `scripts/` — тестовые и диагностические утилиты (`run_tests.py`, `validate_project.py`, `load_judicial_practice.py`).
- `tests/` — pytest-сценарии (юнит и интеграционные), ориентир для новых тестов.

## Быстрый старт
```bash
poetry install --with dev
cp .env.example .env  # заполните секреты через секрет-менеджер
poetry run telegram-legal-bot
```

### Docker
```bash
docker compose up --build aivan
```
Параметры окружения прокидываются из секретного файла (`--env-file`), а данные сохраняются в `./data` и `./logs`.

## Конфигурация
Все значения собираются через `AppSettings` (`src/core/settings.py`). Минимальные переменные:
- `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY` — обязательно.
- `DB_PATH` — путь к SQLite (по умолчанию `data/bot.sqlite3`).
- Платёжные токены (`CRYPTO_PAY_TOKEN`, `TELEGRAM_PROVIDER_TOKEN_*`) включают премиум-функции.
- Для RAG выставьте `RAG_ENABLED=true` и координаты Qdrant (`docs/RAG_SETUP.md`).

Перечень и описание переменных смотрите в `.env.example`. **Не храните реальные ключи в репозитории** — используйте секрет-менеджер и GitHub Actions secrets.

## Тестирование и качество
- Запуск полного пайплайна: `poetry run python scripts/run_tests.py`.
- Раздельные проверки:
  - `poetry run pytest`
  - `poetry run ruff check src tests`
  - `poetry run black --check src tests`
  - `poetry run mypy src tests`
- Для ключевых потоков (платежи, retention, документы) есть тесты в `tests/` и `tests/unit/`.

## Наблюдаемость и эксплуатация
- Healthcheck: `python -m telegram_legal_bot.healthcheck`.
- Prometheus-метрики доступны при `ENABLE_PROMETHEUS=1` и `PROMETHEUS_PORT` (см. `docker-compose.yml`).
- Скрипт `scripts/validate_project.py` выполняет bootstrap-проверки (DI, БД, зависимости).

## Release checklist
1. Смените все секреты, присутствовавшие в предыдущих коммитах (`TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY` и т.д.).
2. Убедитесь, что локальные `.env` и SQLite базы не попадают в коммиты (`.gitignore` уже настроен).
3. `poetry run python scripts/run_tests.py` — должен завершиться без ошибок.
4. Соберите образ: `docker build -t aivan-bot .` и выполните healthcheck.
5. Заполните релизное описание (фичи, миграции, известные ограничения) и загрузите артефакты.

## Контрибуции
Перед PR:
- Прогоните форматирование (`black`, `isort`) и линтеры (`ruff`, `mypy`).
- Добавьте тесты к новой функциональности.
- Опишите изменения, приложите логи/скриншоты для UI и ссылку на задачу.

Обратная связь и вопросы — через issues или support@aivan.ai.

