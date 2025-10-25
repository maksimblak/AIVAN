# LegalAI Pro - Telegram Web App

Премиум интерфейс для юридического бота-ассистента AIVAN. Production-ready React приложение с профессиональным дизайном уровня топовых приложений.

## Особенности

### 🎨 Дизайн
- **Современный UI/UX** - Glassmorphism, градиенты, плавные анимации
- **Темная тема** - Автоматическое переключение по системным настройкам
- **Адаптивность** - Оптимизировано для мобильных устройств (390x844px)
- **Премиум элементы** - Градиентные кнопки, тени, hover-эффекты

### ⚡ Функционал
- **Юридические консультации** - 10 категорий права с умной навигацией
- **Поиск судебной практики** - Анализ дел с релевантностью
- **Работа с документами** - OCR, анализ рисков, генерация документов
- **Профиль пользователя** - Статистика, подписка, настройки
- **Telegram интеграция** - Полная поддержка Telegram Web App API

### 🚀 Технологии
- React 18.3+ с хуками
- Tailwind CSS 3.4+
- Vite 5.3+
- Lucide React (иконки)
- Telegram Web App SDK

## Структура проекта

```
webapp/
├── src/
│   ├── main.jsx           # Точка входа
│   └── index.css          # Глобальные стили
├── LegalAIPro.jsx         # Главный компонент
├── index.html             # HTML шаблон
├── package.json           # Зависимости
├── vite.config.js         # Vite конфигурация
├── tailwind.config.js     # Tailwind конфигурация
└── postcss.config.js      # PostCSS конфигурация
```

## Быстрый старт

### 1. Установка зависимостей

```bash
cd webapp
npm install
```

### 2. Локальная разработка

```bash
npm run dev
```

Приложение откроется на `http://localhost:3000`

### 3. Сборка для production

```bash
npm run build
```

Результат в папке `dist/`

## Интеграция с Telegram ботом

### Вариант 1: Через ngrok (для разработки)

1. Запустите dev-сервер:
```bash
npm run dev
```

2. Установите ngrok:
```bash
npm install -g ngrok
```

3. Создайте туннель:
```bash
ngrok http 3000
```

4. Получите HTTPS URL (например: `https://abc123.ngrok.io`)

5. Добавьте Web App в бота (Python код):

```python
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo

# В src/core/bot_app/menus.py добавьте кнопку:
def _main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            # ... существующие кнопки ...
            [
                InlineKeyboardButton(
                    text="🚀 Открыть Web App",
                    web_app=WebAppInfo(url="https://abc123.ngrok.io")
                )
            ],
        ]
    )
```

### Вариант 2: Через Cloudflare Tunnel (рекомендуется)

1. Установите cloudflared:
```bash
# Windows (через Scoop)
scoop install cloudflared

# macOS
brew install cloudflared

# Linux
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
```

2. Запустите туннель:
```bash
npm run dev
cloudflared tunnel --url http://localhost:3000
```

3. Используйте полученный URL в боте

### Вариант 3: Production deployment

1. Соберите проект:
```bash
npm run build
```

2. Разместите содержимое `dist/` на хостинге:
   - **Vercel**: `vercel --prod`
   - **Netlify**: `netlify deploy --prod`
   - **GitHub Pages**: настройте GitHub Actions
   - **Ваш сервер**: Nginx/Apache

3. Получите HTTPS URL и добавьте в бота

## Интеграция с Python ботом

### Добавление Web App кнопки в меню

В файле `src/core/bot_app/menus.py`:

```python
from aiogram.types import WebAppInfo

# В функции _main_menu_keyboard():
def _main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⚖️ Юридический вопрос", callback_data="legal_question")],
            [InlineKeyboardButton(
                text="🔍 Поиск и анализ судебной практики",
                callback_data="search_practice"
            )],
            [InlineKeyboardButton(
                text="🗂️ Работа с документами",
                callback_data="document_processing"
            )],
            # ⬇️ НОВАЯ КНОПКА WEB APP
            [InlineKeyboardButton(
                text="🚀 Web интерфейс",
                web_app=WebAppInfo(url="https://your-domain.com")
            )],
            [
                InlineKeyboardButton(text="👤 Мой профиль", callback_data="my_profile"),
                InlineKeyboardButton(text="💬 Поддержка", callback_data="help_info"),
            ],
        ]
    )
```

### Получение данных от Web App

```python
from aiogram import Router, F
from aiogram.types import Message

router = Router()

@router.message(F.web_app_data)
async def handle_web_app_data(message: Message):
    """Обработка данных от Web App"""
    data = message.web_app_data.data
    # data содержит JSON строку от Web App
    import json
    payload = json.loads(data)

    # Обработайте данные от Web App
    await message.answer(f"Получены данные: {payload}")
```

### Отправка данных из Web App в бота

В `LegalAIPro.jsx` добавьте:

```javascript
// Отправка данных в бота
const sendDataToBot = (data) => {
  if (window.Telegram?.WebApp) {
    window.Telegram.WebApp.sendData(JSON.stringify(data));
  }
};

// Пример использования:
<button onClick={() => sendDataToBot({
  action: 'legal_question',
  category: 'civil',
  question: 'Мой вопрос...'
})}>
  Отправить вопрос
</button>
```

## Кастомизация

### Изменение цветовой схемы

В `webapp/tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: '#2563EB',      // Синий
      secondary: '#8B5CF6',    // Фиолетовый
      accent: '#EC4899',       // Розовый
    }
  }
}
```

### Добавление новых экранов

1. Создайте компонент экрана в `LegalAIPro.jsx`:

```javascript
const MyNewScreen = () => (
  <div className="flex flex-col h-full">
    <ScreenHeader title="Новый экран" onBack={() => setCurrentScreen('main')} />
    {/* Ваш контент */}
  </div>
);
```

2. Добавьте в роутинг:

```javascript
const renderScreen = () => {
  switch (currentScreen) {
    case 'my-new-screen':
      return <MyNewScreen />;
    // ...
  }
};
```

3. Добавьте кнопку навигации:

```javascript
<ActionCard
  title="Моя функция"
  subtitle="Описание"
  icon={Star}
  gradient="from-green-600 to-green-700"
  onClick={() => setCurrentScreen('my-new-screen')}
/>
```

## Telegram Web App API

### Доступные методы:

```javascript
const tg = window.Telegram.WebApp;

// Инициализация
tg.ready();

// Развернуть на весь экран
tg.expand();

// Получить данные пользователя
const user = tg.initDataUnsafe.user;
console.log(user.id, user.first_name);

// Цветовая схема
console.log(tg.colorScheme); // 'light' | 'dark'

// Показать главную кнопку
tg.MainButton.setText('Продолжить');
tg.MainButton.show();
tg.MainButton.onClick(() => {
  console.log('Главная кнопка нажата');
});

// Показать кнопку "Назад"
tg.BackButton.show();
tg.BackButton.onClick(() => {
  setCurrentScreen('main');
});

// Вибрация
tg.HapticFeedback.impactOccurred('medium');

// Закрыть Web App
tg.close();

// Отправить данные в бота
tg.sendData(JSON.stringify({ key: 'value' }));
```

## Production чеклист

- [ ] Обновить URL в боте на production домен
- [ ] Включить HTTPS (обязательно для Telegram)
- [ ] Настроить CSP заголовки
- [ ] Оптимизировать изображения
- [ ] Включить компрессию gzip/brotli
- [ ] Настроить CDN для статики
- [ ] Добавить сервис для аналитики
- [ ] Протестировать на разных устройствах
- [ ] Проверить производительность (Lighthouse)
- [ ] Настроить error tracking (Sentry)

## Развертывание на Vercel

1. Установите Vercel CLI:
```bash
npm i -g vercel
```

2. Деплой:
```bash
cd webapp
vercel --prod
```

3. Получите URL и обновите бота

## Развертывание на Netlify

1. Создайте `netlify.toml`:
```toml
[build]
  command = "npm run build"
  publish = "dist"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

2. Деплой:
```bash
netlify deploy --prod
```

## Поддержка

При возникновении проблем:

1. Проверьте консоль браузера (F12)
2. Проверьте логи Telegram бота
3. Убедитесь, что используется HTTPS
4. Проверьте совместимость Telegram Web App API

## Лицензия

MIT - используйте свободно в коммерческих и некоммерческих проектах.

---

**Создано для AIVAN Legal Bot** 🤖⚖️

Техподдержка: support@aivan.ai
