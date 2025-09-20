# 🤖 ИИ-Иван - Advanced Legal AI Assistant

Продвинутый асинхронный Telegram-бот для юридических консультаций с enterprise-level возможностями.

## 🆕 ОСНОВНЫЕ УЛУЧШЕНИЯ

### ✅ **Критические исправления**
- ✅ **Входная валидация**: Защита от XSS, SQL injection, спам-атак
- ✅ **Улучшенная обработка ошибок**: Централизованная система с автовосстановлением
- ✅ **Connection Pooling**: Оптимизированная работа с базой данных через aiosqlite
- ✅ **Кеширование ответов**: Redis/in-memory кеш для OpenAI ответов

### 🚀 **Производительность и масштабирование**
- ✅ **Background cleanup**: Автоматическая очистка старых данных
- ✅ **Prometheus метрики**: Детальный мониторинг всех компонентов
- ✅ **Health checks**: Автоматическая проверка состояния системы
- ✅ **Horizontal scaling**: Service registry, load balancing, session affinity

### 🛡️ **Надёжность**
- ✅ **Graceful degradation**: Fallback на локальное хранение при недоступности Redis
- ✅ **Recovery mechanisms**: Автоматическое восстановление после сбоев
- ✅ **Circuit breakers**: Защита от каскадных отказов
- ✅ **Distributed locks**: Координация в кластере

## 🚀 Ключевые возможности

### 💼 **Для пользователей**
- 🔍 Анализ российского права и судебной практики
- 📋 Подготовка процессуальных документов
- 💡 Правовые рекомендации и оценка рисков
- ⚡ Быстрые ответы благодаря кешированию
- 🎯 Красивый интерфейс с анимированными статусами

### 🔧 **Для администраторов**
- 📊 **Prometheus метрики** на порту 8000
- 🏥 **Health checks** всех компонентов
- 📈 **Detailed logging** с JSON структурированием
- 🔄 **Background tasks** для обслуживания системы
- 🎛️ **Flexible configuration** через environment variables

### 🏗️ **Для DevOps**
- 🔄 **Horizontal scaling** с service discovery
- ⚖️ **Load balancing** с несколькими стратегиями
- 🔐 **Session affinity** для sticky sessions
- 🐳 **Docker-ready** конфигурация
- ☸️ **Kubernetes-ready** с health endpoints

## 🧱 Архитектура

### **Core Stack**
- `aiogram` v3 - Telegram Bot API
- `openai` v1.40+ - AI responses с новым Responses API
- `aiosqlite` - Async database с connection pooling
- `redis` - Кеширование и координация кластера
- `prometheus-client` - Метрики и мониторинг
- `psutil` - Системный мониторинг

### **Модули системы**
```
src/
├── core/                    # Бизнес-логика
│   ├── validation.py        # Валидация входных данных
│   ├── exceptions.py        # Централизованная обработка ошибок
│   ├── db_advanced.py       # База данных с pooling
│   ├── cache.py             # Система кеширования
│   ├── background_tasks.py  # Фоновые задачи
│   ├── metrics.py           # Prometheus метрики
│   ├── health.py            # Health checks
│   └── scaling.py           # Horizontal scaling
├── bot/                     # Telegram интерфейс
└── telegram_legal_bot/      # Конфигурация
```

## 📦 Установка

### **Базовая установка**
```bash
git clone <repo> && cd AIVAN
poetry install
cp config.example.env .env
# Настройте TELEGRAM_BOT_TOKEN и OPENAI_API_KEY
```

### **Production установка**
```bash
# Полная установка со всеми возможностями
poetry install --extras full

# Только мониторинг
poetry install --extras monitoring

# Только scaling
poetry install --extras scaling
```

### **Docker установка**
```bash
# Базовый запуск
docker build -t ai-ivan .
docker run -d --env-file .env ai-ivan

# С Redis и Prometheus
docker-compose up -d
```

## ⚙️ Конфигурация

### **Основные настройки**
```env
# Обязательные
TELEGRAM_BOT_TOKEN=your_token
OPENAI_API_KEY=your_key

# Расширенная БД (рекомендуется)
USE_ADVANCED_DB=1
DB_MAX_CONNECTIONS=5

# Кеширование (рекомендуется)
ENABLE_OPENAI_CACHE=1
REDIS_URL=redis://localhost:6379/0
```

### **Мониторинг**
```env
# Prometheus метрики
ENABLE_PROMETHEUS=1
PROMETHEUS_PORT=8000

# Health checks
HEALTH_CHECK_INTERVAL=30.0
ENABLE_SYSTEM_MONITORING=1
```

### **Масштабирование**
```env
# Horizontal scaling
ENABLE_SCALING=1
HEARTBEAT_INTERVAL=15.0
SESSION_AFFINITY_TTL=3600
```

## ▶️ Запуск

### **Разработка**
```bash
# Простой запуск
poetry run python run_simple_bot.py

# С полным логированием
LOG_LEVEL=DEBUG poetry run python run_simple_bot.py
```

### **Production**
```bash
# Базовая production конфигурация
USE_ADVANCED_DB=1 ENABLE_PROMETHEUS=1 poetry run python run_simple_bot.py

# Полная production конфигурация
docker-compose -f docker-compose.prod.yml up -d
```

### **Кластер**
```bash
# Первая нода
ENABLE_SCALING=1 NODE_ID=node-1 PORT=8001 poetry run python run_simple_bot.py

# Вторая нода  
ENABLE_SCALING=1 NODE_ID=node-2 PORT=8002 poetry run python run_simple_bot.py
```

## 📊 Мониторинг

### **Prometheus метрики**
- `http://localhost:8000/metrics` - Все метрики
- Счетчики: telegram_messages_total, openai_requests_total
- Гистограммы: response_time, database_query_duration
- Gauges: active_sessions, database_connections

### **Health checks**
- `GET /health` - Общий статус системы
- `GET /health/database` - Статус БД
- `GET /health/openai` - Статус OpenAI
- `GET /health/redis` - Статус Redis

### **Логирование**
```bash
# Структурированные JSON логи
LOG_JSON=1 poetry run python run_simple_bot.py

# Фильтрация по компонентам
LOG_LEVEL=INFO poetry run python run_simple_bot.py | jq 'select(.logger | contains("openai"))'
```

## 🔧 Обслуживание

### **Фоновые задачи**
- **Database cleanup**: Удаление старых транзакций (каждый час)
- **Cache cleanup**: Очистка истекшего кеша (каждые 5 минут)
- **Session cleanup**: Удаление неактивных сессий (каждые 10 минут)
- **Health monitoring**: Проверка всех компонентов (каждые 30 секунд)

### **Диагностика проблем**
```bash
# Проверка health checks
curl http://localhost:8000/health

# Просмотр метрик
curl http://localhost:8000/metrics | grep error

# Анализ логов
tail -f logs/ai-ivan.log | jq 'select(.lvl == "ERROR")'
```

## 🚀 Производительность

### **Бенчмарки**
- **Без кеша**: ~2-5 секунд на запрос OpenAI
- **С кешем**: ~50-100ms на кешированный запрос
- **Connection pooling**: до 3x ускорение работы с БД
- **Memory usage**: ~50-100MB в базовой конфигурации

### **Оптимизации**
- **Advanced DB**: Connection pooling, WAL mode, оптимизированные индексы
- **Smart caching**: Сжатие ответов, TTL, LRU eviction
- **Background cleanup**: Автоматическая очистка без блокировки
- **Graceful degradation**: Fallback при недоступности external сервисов

## 🤝 Участие в разработке

### **Добавление новых возможностей**
1. Создайте новый модуль в `src/core/`
2. Добавьте health check в `src/core/health.py`
3. Добавьте метрики в `src/core/metrics.py`
4. Интегрируйте в `src/core/main_simple.py`

### **Тестирование**
```bash
# Установка dev зависимостей
poetry install --with dev

# Запуск тестов
poetry run pytest

# Линтинг
poetry run ruff check
poetry run black --check .
```

## 📄 Лицензия

MIT License - см. файл LICENSE для деталей.
