# Интеграция Web App с AIVAN Telegram ботом

Это руководство описывает процесс интеграции React Web App с существующим Python Telegram ботом AIVAN.

## Шаг 1: Настройка Web App

### 1.1 Установка зависимостей

```bash
cd webapp
npm install
```

### 1.2 Локальный запуск для тестирования

```bash
npm run dev
```

Приложение будет доступно на `http://localhost:3000`

## Шаг 2: Создание публичного URL

### Вариант A: ngrok (быстрый старт)

```bash
# Установка ngrok
npm install -g ngrok

# Запуск туннеля
ngrok http 3000
```

Скопируйте HTTPS URL (например: `https://abc123.ngrok-free.app`)

### Вариант B: Cloudflare Tunnel (рекомендуется)

```bash
# Установка cloudflared
# Windows: scoop install cloudflared
# macOS: brew install cloudflared
# Linux: wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64

# Запуск туннеля
cloudflared tunnel --url http://localhost:3000
```

Скопируйте полученный URL (например: `https://your-subdomain.trycloudflare.com`)

## Шаг 3: Модификация Python бота

### 3.1 Добавление Web App кнопки в главное меню

Откройте файл `src/core/bot_app/menus.py` и найдите функцию `_main_menu_keyboard()` (около строки 94).

**Добавьте импорт:**

```python
from aiogram.types import WebAppInfo
```

**Модифицируйте функцию `_main_menu_keyboard()`:**

```python
def _main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⚖️ Юридический вопрос", callback_data="legal_question")],
            [
                InlineKeyboardButton(
                    text="🔍 Поиск и анализ судебной практики", callback_data="search_practice"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🗂️ Работа с документами", callback_data="document_processing"
                )
            ],
            # ⬇️ НОВАЯ КНОПКА WEB APP
            [
                InlineKeyboardButton(
                    text="🚀 Открыть Web интерфейс",
                    web_app=WebAppInfo(url="https://your-url-here.ngrok-free.app")
                )
            ],
            # ⬆️ ЗАМЕНИТЕ URL НА ВАШ
            [
                InlineKeyboardButton(text="👤 Мой профиль", callback_data="my_profile"),
                InlineKeyboardButton(text="💬 Поддержка", callback_data="help_info"),
            ],
        ]
    )
```

### 3.2 Обработка данных от Web App

Создайте новый файл `src/core/bot_app/webapp_handlers.py`:

```python
"""Обработчики для Telegram Web App"""

from __future__ import annotations

import json
import logging
from typing import Any

from aiogram import Dispatcher, F, Router
from aiogram.types import Message

logger = logging.getLogger("ai-ivan.webapp")

router = Router()


@router.message(F.web_app_data)
async def handle_web_app_data(message: Message) -> None:
    """
    Обработка данных от Web App.

    Web App может отправлять данные используя:
    window.Telegram.WebApp.sendData(JSON.stringify({...}))
    """
    if not message.web_app_data:
        return

    try:
        # Парсим JSON данные от Web App
        data: dict[str, Any] = json.loads(message.web_app_data.data)

        logger.info(
            "Received Web App data from user %s: %s",
            message.from_user.id if message.from_user else "unknown",
            data
        )

        # Обработка разных типов действий
        action = data.get("action")

        if action == "legal_question":
            # Пользователь отправил юридический вопрос из Web App
            category = data.get("category", "unknown")
            question = data.get("question", "")

            await message.answer(
                f"✅ Получен вопрос по категории <b>{category}</b>:\n\n"
                f"{question}\n\n"
                f"Обрабатываю запрос...",
                parse_mode="HTML"
            )

            # Здесь можно вызвать вашу логику обработки вопроса
            # Например, из openai_gateway.py

        elif action == "document_upload":
            # Пользователь запросил загрузку документа
            doc_type = data.get("type", "unknown")

            await message.answer(
                f"📄 Отправьте документ ({doc_type}) для обработки",
                parse_mode="HTML"
            )

        elif action == "search_practice":
            # Поиск судебной практики
            query = data.get("query", "")

            await message.answer(
                f"🔍 Ищу судебную практику по запросу:\n<b>{query}</b>",
                parse_mode="HTML"
            )

            # Вызовите вашу логику поиска

        else:
            # Неизвестное действие
            await message.answer(
                "Данные получены, но действие не распознано.",
                parse_mode="HTML"
            )

    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Web App data: %s", exc)
        await message.answer(
            "❌ Ошибка обработки данных от Web App",
            parse_mode="HTML"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error handling Web App data: %s", exc)
        await message.answer(
            "❌ Произошла ошибка при обработке запроса",
            parse_mode="HTML"
        )


def register_webapp_handlers(dp: Dispatcher) -> None:
    """Регистрация обработчиков Web App"""
    dp.include_router(router)
    logger.info("Web App handlers registered")
```

### 3.3 Регистрация обработчиков

Откройте файл `src/core/bot_app/app.py` и добавьте регистрацию обработчиков Web App:

```python
# В импортах добавьте:
from src.core.bot_app import webapp_handlers

# В функции create_dispatcher() добавьте после других регистраций:
def create_dispatcher(...) -> Dispatcher:
    # ... существующий код ...

    # Регистрация обработчиков
    menus.register_menu_handlers(dp)
    questions.register_question_handlers(dp)
    # ... другие регистрации ...

    # ⬇️ ДОБАВЬТЕ ЭТУ СТРОКУ
    webapp_handlers.register_webapp_handlers(dp)

    return dp
```

## Шаг 4: Отправка данных из Web App в бота

В вашем React компоненте `LegalAIPro.jsx` уже подготовлена интеграция с Telegram Web App API.

### Пример отправки данных:

```javascript
// В компоненте LegalAIPro.jsx
const sendQuestionToBot = () => {
  const tg = window.Telegram?.WebApp;

  if (tg) {
    // Отправляем данные в бота
    tg.sendData(JSON.stringify({
      action: 'legal_question',
      category: selectedCategory?.id || 'general',
      question: questionText,
      timestamp: new Date().toISOString()
    }));

    // Закрываем Web App после отправки
    tg.close();
  }
};
```

### Модифицируйте кнопку "Получить консультацию" в QuestionInputScreen:

```javascript
<button
  onClick={() => {
    // Получаем текст из textarea
    const questionText = document.querySelector('textarea').value;

    if (questionText.trim()) {
      const tg = window.Telegram?.WebApp;
      if (tg) {
        tg.sendData(JSON.stringify({
          action: 'legal_question',
          category: selectedCategory?.id || 'general',
          question: questionText,
          timestamp: new Date().toISOString()
        }));
        // Показываем подтверждение
        tg.showAlert('Вопрос отправлен!');
      }
    }
  }}
  className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 rounded-xl font-bold flex items-center justify-center space-x-2 hover:shadow-2xl hover:scale-105 transition-all duration-300"
>
  <Sparkles size={20} />
  <span>Получить консультацию</span>
</button>
```

## Шаг 5: Тестирование

1. **Запустите бота:**
```bash
poetry run telegram-legal-bot
```

2. **Запустите Web App:**
```bash
cd webapp
npm run dev
```

3. **Создайте туннель (ngrok/cloudflare):**
```bash
ngrok http 3000
# или
cloudflared tunnel --url http://localhost:3000
```

4. **Обновите URL в боте** (в `menus.py`)

5. **Отправьте `/start` боту в Telegram**

6. **Нажмите кнопку "🚀 Открыть Web интерфейс"**

7. **Проверьте функционал:**
   - Навигация между экранами
   - Отправка данных в бота
   - Темная тема
   - Адаптивность

## Шаг 6: Production деплой

### Вариант A: Vercel (рекомендуется)

```bash
npm install -g vercel
cd webapp
vercel --prod
```

### Вариант B: Netlify

```bash
npm install -g netlify-cli
cd webapp
netlify deploy --prod
```

### Вариант C: Собственный сервер

```bash
# Сборка
npm run build

# Скопируйте содержимое dist/ на ваш сервер
# Настройте Nginx:
```

```nginx
server {
    listen 443 ssl http2;
    server_name webapp.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    root /var/www/webapp/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Кэширование статики
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## Шаг 7: Обновление URL в production

После деплоя обновите URL в `src/core/bot_app/menus.py`:

```python
WebAppInfo(url="https://your-production-domain.com")
```

И перезапустите бота.

## Дополнительные возможности

### Передача данных пользователя в Web App

```python
# В menus.py при создании кнопки:
user_id = message.from_user.id if message.from_user else 0
webapp_url = f"https://your-domain.com?user_id={user_id}&lang=ru"

InlineKeyboardButton(
    text="🚀 Открыть Web интерфейс",
    web_app=WebAppInfo(url=webapp_url)
)
```

В React:

```javascript
// Получение параметров из URL
const params = new URLSearchParams(window.location.search);
const userId = params.get('user_id');
const lang = params.get('lang');
```

### Использование InitData для аутентификации

```javascript
// В React компоненте
const tg = window.Telegram?.WebApp;
const initData = tg?.initData;

// Отправьте initData на ваш backend для валидации
fetch('https://your-bot-backend.com/api/validate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ initData })
});
```

В Python (backend):

```python
import hmac
import hashlib
from urllib.parse import parse_qs

def validate_init_data(init_data: str, bot_token: str) -> bool:
    """Валидация Telegram initData"""
    try:
        parsed = parse_qs(init_data)
        hash_value = parsed.get('hash', [''])[0]

        # Удаляем hash для проверки
        data_check = '\n'.join(
            f"{k}={v[0]}"
            for k, v in sorted(parsed.items())
            if k != 'hash'
        )

        secret_key = hmac.new(
            b"WebAppData",
            bot_token.encode(),
            hashlib.sha256
        ).digest()

        calculated_hash = hmac.new(
            secret_key,
            data_check.encode(),
            hashlib.sha256
        ).hexdigest()

        return calculated_hash == hash_value
    except Exception:
        return False
```

## Troubleshooting

### Web App не открывается

- Проверьте, что URL использует HTTPS
- Убедитесь, что ngrok/cloudflare туннель работает
- Проверьте консоль браузера (F12)

### Данные не отправляются в бота

- Убедитесь, что обработчик `webapp_handlers.py` зарегистрирован
- Проверьте логи бота
- Убедитесь, что используется `tg.sendData()`

### Темная тема не работает

- Проверьте `tg.colorScheme`
- Убедитесь, что Tailwind `darkMode: 'class'` настроен
- Добавьте `dark` класс к `<html>` элементу

## Поддержка

Вопросы и предложения: support@aivan.ai

---

**AIVAN Legal Bot - Web App Integration Guide** 🚀
