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

## 📦 Установка

```bash
git clone <repo> && cd telegram_legal_bot
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Заполни TELEGRAM_BOT_TOKEN и OPENAI_API_KEY
