# Telegram Legal Bot

Телеграм-бот для предоставления юридических консультаций на основе модели OpenAI GPT.

## Возможности

- Принимает текстовые запросы от пользователей в Telegram.
- Обеспечивает интеграцию со специально обученной моделью OpenAI GPT.
- Форматирует ответы с учётом UX/UI требований (эмодзи, выделение, списки).
- Ограничивает количество запросов от пользователя и показывает индикатор набора текста.
- Логирует действия и корректно завершается по сигналам остановки.

## Стек

- Python 3.11+
- [python-telegram-bot](https://docs.python-telegram-bot.org/) v21 (асинхронный API)
- [OpenAI Python SDK](https://github.com/openai/openai-python) с использованием Responses API и модели `gpt-5`
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- Poetry для управления зависимостями и запуском

## Структура проекта

```
telegram_legal_bot/
├── main.py
├── config.py
├── handlers/
│   ├── __init__.py
│   ├── start.py
│   └── legal_query.py
├── services/
│   ├── __init__.py
│   └── openai_service.py
├── utils/
│   ├── __init__.py
│   ├── message_formatter.py
│   └── rate_limiter.py
├── .env.example
├── requirements.txt
└── pyproject.toml
```

## Подготовка окружения

1. Установите Poetry согласно [официальной инструкции](https://python-poetry.org/docs/).
2. Склонируйте репозиторий и перейдите в корень проекта.
3. Скопируйте `.env.example` в `.env` и заполните значения переменных:

```env
TELEGRAM_BOT_TOKEN=ваш_токен
OPENAI_API_KEY=ваш_api_ключ
OPENAI_MODEL=gpt-5
MAX_REQUESTS_PER_HOUR=10
```

4. Установите зависимости и активируйте виртуальное окружение:

```bash
poetry install
poetry shell
```

5. Запустите бота:

```bash
poetry run python -m telegram_legal_bot.main
```

## Переменные окружения

| Переменная              | Описание                                           |
|-------------------------|----------------------------------------------------|
| `TELEGRAM_BOT_TOKEN`    | Токен Telegram бота                                 |
| `OPENAI_API_KEY`        | API-ключ OpenAI                                     |
| `OPENAI_MODEL`          | Идентификатор модели OpenAI (по умолчанию `gpt-5`)  |
| `MAX_REQUESTS_PER_HOUR` | Максимум запросов в час на пользователя (дефолт 10) |

## Ограничения и безопасность

- Бот не хранит персональные данные пользователей.
- Встроенный rate-limiter предотвращает злоупотребления.
- Ответы носят информационный характер и не являются юридической услугой.

## Генерация `requirements.txt`

Проект использует только Poetry. При необходимости создайте файл зависимостей для деплоя:

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Graceful Shutdown

Бот корректно завершается по сигналам `SIGINT`/`SIGTERM`, освобождая ресурсы и останавливая polling.

## Логирование

Логи пишутся в stdout с уровнем `INFO`. При необходимости измените конфигурацию в `telegram_legal_bot/main.py`.

