# telegram_legal_bot

Асинхронный Telegram-бот для юридических консультаций на базе OpenAI GPT.

## 🚀 Возможности

- Приём текстовых вопросов из Telegram
- Интеграция с OpenAI (чат-комплишены)
- Красивое форматирование (MarkdownV2): заголовки, списки, эмодзи
- Показ индикатора "печатает"
- Антиспам + per-user rate limit (в час)
- Автоматическое разбиение длинных ответов на части
- История последних кратких ответов для минимального контекста
- Логирование (обычное или JSON)
- Graceful shutdown

## 🧱 Стек

- `aiogram` v3 (asyncio)
- `openai` (AsyncOpenAI, >=1.40)
- `httpx` (прокси/клиент)
- `python-dotenv`
- (опционально) `redis` — если вынести лимиты/историю

## 📦 Установка (Poetry)

```bash
git clone <repo> && cd AIVAN
poetry install
cp .env.example .env  # или создайте .env вручную
# Установите TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, при необходимости OPENAI_MODEL
```

## ▶️ Запуск

```bash
poetry run telegram-legal-bot
```

Альтернативно:

```bash
poetry run python -m telegram_legal_bot.main
