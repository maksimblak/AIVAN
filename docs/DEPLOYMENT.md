# AIVAN Production Deployment Guide

This runbook описывает безопасный запуск Telegram-бота AIVAN в staging и production средах.

## 1. Предварительные требования
- Python 3.12+ или Docker/Compose.
- Telegram Bot API токен и OpenAI API ключ (создайте свежие перед релизом).
- (Опционально) Redis для rate-limit'ов и горизонтального масштабирования.
- (Опционально) Prometheus + Grafana для метрик.
- VPN/прокси, если инфраструктура требует выхода в Telegram/OpenAI через прокси.

## 2. Подготовка конфигурации
1. Скопируйте `.env.example` в защищённое место, но **не коммитьте** реальные секреты.
2. Заполните обязательные переменные:
   - `TELEGRAM_BOT_TOKEN`
   - `OPENAI_API_KEY`
   - `DB_PATH` (по умолчанию `data/bot.sqlite3`)
3. Настройте дополнительные блоки:
   - Голосовой режим (`ENABLE_VOICE_MODE`, `VOICE_*`).
   - Платежи (`CRYPTO_PAY_TOKEN`, `TELEGRAM_PROVIDER_TOKEN_*`, `YOOKASSA_*` и т.д.).
   - RAG (`RAG_ENABLED`, `RAG_QDRANT_URL`, `RAG_COLLECTION`).
4. Убедитесь, что секреты хранятся в секрет-менеджере (Vault, AWS Secrets Manager, GitHub Actions secrets).

## 3. Запуск для разработки / QA
```bash
poetry install --with dev
poetry run python -m telegram_legal_bot.main
```
Отладочные логи включаются через `LOG_LEVEL=DEBUG`, healthcheck доступен через `python -m telegram_legal_bot.healthcheck`.

## 4. Docker-развёртывание
```bash
docker build -t aivan-bot .
docker run --env-file path/to/prod.env --volume $(pwd)/data:/app/data aivan-bot
```

### Docker Compose
```bash
docker compose up --build aivan
```
Включает Redis и (опционально) стек мониторинга. Проверьте монтирование `./data`, `./logs` и профили `monitoring` для Prometheus/Grafana.

## 5. Наблюдаемость
- Метрики Prometheus активируются `ENABLE_PROMETHEUS=1` и `PROMETHEUS_PORT` (по умолчанию 9000).
- Healthcheck в Dockerfile автоматически выполняет `python -m telegram_legal_bot.healthcheck`.
- `scripts/validate_project.py` проверяет зависимости, DI-контейнер и SQLite.

## 6. Release checklist
1. Поменяйте все ключи, которые могли находиться в репозитории или в старых окружениях.
2. `poetry run python scripts/run_tests.py` — без ошибок.
3. Соберите Docker-образ и прогоните healthcheck локально.
4. Обновите changelog/релиз-ноты, опишите миграции и особенности деплоя.
5. Добавьте артефакты (Docker image tag, helm chart и т.п.) в релиз.
6. Убедитесь, что CI содержит шаги lint→test→build→scan и использует секреты из менеджера.

## 7. Дополнительные рекомендации
- При ротации секретов используйте rolling reload контейнеров.
- Настройте alerting в Prometheus/Grafana на ключевые метрики (`openai_errors_total`, `payment_failures_total`, `security_warnings_total`).
- Пересматривайте лимиты OpenAI и Telegram, чтобы избежать блокировок.
- Планируйте резервное копирование `data/` либо переход на внешнюю БД (PostgreSQL) для продакшена.

