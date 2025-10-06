# 🤖 AIVAN - ИИ Юридический Ассистент

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**AIVAN** (AI Virtual Assistant for Negotiations) — это интеллектуальный Telegram-бот для юридических консультаций, построенный на базе OpenAI GPT и предоставляющий комплексные решения для юристов и правовых компаний.

## 📋 Содержание

- [Основные возможности](#-основные-возможности)
- [Архитектура](#-архитектура)
- [Установка](#-установка)
- [Конфигурация](#-конфигурация)
- [Запуск](#-запуск)
- [Использование](#-использование)
- [API Documentation](#-api-documentation)
- [Разработка](#-разработка)
- [Тестирование](#-тестирование)
- [Развертывание](#-развертывание)
- [Мониторинг](#-мониторинг)
- [Вклад в проект](#-вклад-в-проект)
- [Лицензия](#-лицензия)

## 🚀 Основные возможности

### 💼 Юридический ассистент
- **Анализ судебной практики** - поиск и анализ релевантных судебных решений
- **Консультации по НПА** - разъяснение нормативно-правовых актов
- **Подготовка документов** - генерация исковых заявлений, ходатайств, жалоб
- **Стратегическое планирование** - построение правовых стратегий для дел
- **Структурированные ответы** - форматированные ответы с ссылками на законы

### 📄 Обработка документов
- **OCR сканирование** - извлечение текста из изображений (PaddleOCR, Tesseract)
- **Анализ документов** - PDF, DOCX, изображения
- **Суммаризация** - краткое изложение объемных документов
- **Анализ рисков** - выявление юридических рисков в документах
- **Обезличивание** - удаление персональных данных
- **Перевод** - поддержка множественных языков
- **Чат с документами** - вопросы-ответы по содержимому

### 🎙️ Мультимедиа возможности
- **Голосовые сообщения** - STT/TTS через OpenAI
- **Streaming ответы** - real-time обновления ответов
- **Интерактивные элементы** - кнопки, клавиатуры, прогресс-бары

### 💳 Система платежей
- **Telegram Stars** - встроенные платежи Telegram
- **Crypto Pay** - криптовалютные платежи
- **Подписки** - гибкая система подписок
- **Пробные периоды** - бесплатное тестирование

## 🏗 Архитектура

```
AIVAN/
├── src/
│   ├── bot/                    # Telegram интерфейс
│   │   ├── promt.py           # Системные промпты
│   │   ├── openai_gateway.py  # OpenAI API интеграция
│   │   ├── status_manager.py  # Управление статусами
│   │   └── ui_components.py   # UI компоненты
│   ├── core/                  # Бизнес-логика
│   │   ├── main_simple.py     # Основная точка входа
│   │   ├── db_advanced.py     # База данных
│   │   ├── audio_service.py   # Аудио сервисы
│   │   ├── crypto_pay.py      # Платежи
│   │   └── scaling.py         # Масштабирование
│   ├── documents/             # Обработка документов
│   │   ├── document_manager.py # Центральный менеджер
│   │   ├── ocr_converter.py   # OCR конвертация
│   │   ├── summarizer.py      # Суммаризация
│   │   └── risk_analyzer.py   # Анализ рисков
│   └── telegram_legal_bot/    # Конфигурация
├── data/                      # Данные пользователей
├── tests/                     # Тесты
└── docs/                      # Документация
```

### Технологический стек

- **Python 3.12+** - основной язык программирования
- **aiogram 3.4+** - асинхронная библиотека для Telegram Bot API
- **OpenAI API** - языковые модели для ИИ-ассистента
- **aiosqlite** - асинхронная работа с SQLite
- **httpx** - HTTP клиент с поддержкой HTTP/2
- **Poetry** - управление зависимостями

#### Дополнительные компоненты
- **Redis** - кеширование и сессии (опционально)
- **Prometheus** - метрики и мониторинг (опционально)
- **PaddleOCR/Tesseract** - распознавание текста
- **PyPDF2/pymupdf** - работа с PDF
- **python-docx** - работа с DOCX

## 🛠 Установка

### Требования

- Python 3.12 или выше
- Poetry для управления зависимостями
- Git для клонирования репозитория

### Быстрая установка

```bash
# Клонировать репозиторий
git clone https://github.com/your-username/AIVAN.git
cd AIVAN

# Установить зависимости
poetry install

# Установить дополнительные компоненты (опционально)
poetry install --extras "prod"  # Redis, Prometheus, psutil
poetry install --extras "full"  # Все дополнительные компоненты
```

### Системные зависимости

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-rus poppler-utils
```

#### macOS
```bash
brew install tesseract poppler
```

#### Windows
1. Установите [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
2. Установите [Poppler](https://blog.alivate.com.au/poppler-windows/)
3. Добавьте пути в переменную PATH

## ⚙️ Конфигурация

### Переменные окружения

Создайте файл `.env` в корневой директории:

```env
# Основные настройки
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
TELEGRAM_CHAT_ID=your_chat_id

# Конфигурация OpenAI
OPENAI_VERBOSITY=medium           # low | medium | high
OPENAI_REASONING_EFFORT=medium    # minimal | medium | high
TEMPERATURE=0.2
TOP_P=1
MAX_OUTPUT_TOKENS=4096
USE_STREAMING=1

# База данных
DB_PATH=data/bot.sqlite3

# Подписки и платежи
TRIAL_REQUESTS=10
SUB_DURATION_DAYS=30
SUBSCRIPTION_PRICE_RUB=300
RUB_PER_XTR=3.5
ADMIN_IDS=your_user_id
TELEGRAM_PROVIDER_TOKEN_STARS=STARS
CRYPTO_PAY_TOKEN=your_crypto_pay_token

# Прокси (опционально)
TELEGRAM_PROXY_URL=http://proxy:port
TELEGRAM_PROXY_USER=username
TELEGRAM_PROXY_PASS=password
OPENAI_PROXY_URL=http://proxy:port
OPENAI_PROXY_USER=username
OPENAI_PROXY_PASS=password

# Голосовые функции
ENABLE_VOICE_MODE=1
VOICE_STT_MODEL=gpt-4o-mini-transcribe
VOICE_TTS_MODEL=gpt-4o-mini-tts
VOICE_TTS_VOICE=alloy
VOICE_TTS_VOICE_MALE=verse
VOICE_TTS_SPEED=0.95
VOICE_TTS_STYLE=formal
VOICE_TTS_FORMAT=ogg
VOICE_TTS_BACKEND=auto
VOICE_MAX_DURATION_SECONDS=300

# Мониторинг
ENABLE_PROMETHEUS=0
ENABLE_SYSTEM_MONITORING=0
LOG_LEVEL=INFO
```

### Конфигурация Telegram Bot

1. Создайте бота через [@BotFather](https://t.me/botfather)
2. Получите токен и добавьте в `TELEGRAM_BOT_TOKEN`
3. Настройте команды бота:

```
start - Начать работу с ботом
help - Помощь и инструкции
profile - Профиль пользователя
subscription - Управление подпиской
settings - Настройки
```

## 🚀 Запуск

### Разработка

```bash
# Активировать виртуальное окружение
poetry shell

# Запустить бота
python src/core/main_simple.py

# Или через Poetry
poetry run python src/core/main_simple.py
```

### Production

```bash
# Запуск с дополнительными компонентами
poetry install --extras "prod"
poetry run python src/core/main_simple.py
```

### Docker (рекомендуется)

```bash
# Сборка образа
docker build -t aivan .

# Запуск контейнера
docker run -d --name aivan-bot \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  aivan
```

## 📖 Использование

### Основные команды

- `/start` - Начать работу с ботом
- `/help` - Получить справку
- `/profile` - Просмотр профиля и статистики
- `/subscription` - Управление подпиской

### Работа с документами

1. **Загрузка документов** - отправьте PDF, DOCX или изображение
2. **OCR обработка** - автоматическое извлечение текста
3. **Анализ** - получите суммарию и анализ рисков
4. **Вопросы** - задавайте вопросы по содержимому

### Юридические консультации

1. **Опишите ситуацию** - подробно изложите правовой вопрос
2. **Получите анализ** - структурированный ответ с ссылками на НПА
3. **Запросите документы** - попросите составить исковое заявление
4. **Уточните детали** - задавайте дополнительные вопросы

### Голосовые сообщения

1. Отправьте голосовое сообщение
2. Бот автоматически распознает речь
3. Получите текстовый или голосовой ответ

## 📚 API Documentation

### Основные модули

#### DocumentManager
```python
from src.documents.document_manager import DocumentManager

# Инициализация
doc_manager = DocumentManager()

# Обработка документа
result = await doc_manager.process_document(file_path, user_id)
```

#### OpenAI Gateway
```python
from src.bot.openai_gateway import OpenAIGateway

# Создание клиента
gateway = OpenAIGateway(api_key="your_key")

# Генерация ответа
response = await gateway.generate_response(prompt, user_id)
```

### REST API (планируется)

Документация API будет доступна по адресу `/docs` после запуска сервера.

## 🔧 Разработка

### Настройка среды разработки

```bash
# Клонировать репозиторий
git clone https://github.com/your-username/AIVAN.git
cd AIVAN

# Установить зависимости разработки
poetry install --with dev

# Настроить pre-commit hooks
poetry run pre-commit install
```

### Стиль кода

Проект использует современные инструменты для поддержания качества кода:

```bash
# Форматирование кода
poetry run black .
poetry run isort .

# Линтинг
poetry run ruff check .

# Проверка типов
poetry run mypy src/
```

### Структура проекта

- **src/bot/** - Telegram интерфейс и UI компоненты
- **src/core/** - Основная бизнес-логика
- **src/documents/** - Обработка документов
- **tests/** - Юнит и интеграционные тесты
- **docs/** - Документация проекта

## 🧪 Тестирование

### Запуск тестов

```bash
# Все тесты
poetry run pytest

# С покрытием
poetry run pytest --cov=src

# Только юнит-тесты
poetry run pytest tests/unit/

# Только интеграционные тесты
poetry run pytest tests/integration/
```

### Типы тестов

- **Unit tests** - тестирование отдельных модулей
- **Integration tests** - тестирование взаимодействия компонентов
- **E2E tests** - сквозное тестирование через Telegram API

### Добавление тестов

```python
# tests/unit/test_document_manager.py
import pytest
from src.documents.document_manager import DocumentManager

class TestDocumentManager:
    @pytest.fixture
    def manager(self):
        return DocumentManager()

    async def test_process_pdf(self, manager):
        result = await manager.process_document("test.pdf", 123)
        assert result.success
```

## 🚀 Развертывание

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  aivan:
    build: .
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
```

### Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aivan
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aivan
  template:
    metadata:
      labels:
        app: aivan
    spec:
      containers:
      - name: aivan
        image: aivan:latest
        env:
        - name: TELEGRAM_BOT_TOKEN
          valueFrom:
            secretKeyRef:
              name: aivan-secrets
              key: telegram-token
```

### CI/CD

Проект поддерживает автоматическое развертывание через GitHub Actions:

```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Deploy to production
      run: |
        docker build -t aivan .
        docker push registry/aivan:latest
```

## 📊 Мониторинг

### Метрики

Бот собирает следующие метрики:

- **Requests per minute** - количество запросов в минуту
- **Response time** - время ответа
- **Error rate** - процент ошибок
- **Active users** - количество активных пользователей
- **Document processing** - статистика обработки документов

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "AIVAN Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(telegram_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### Логирование

```python
# Структурированное логирование
import structlog

logger = structlog.get_logger()

logger.info(
    "User request processed",
    user_id=123,
    request_type="document_analysis",
    processing_time=1.5
)
```

## 🤝 Вклад в проект

Мы приветствуем вклад в развитие проекта!

### Как внести вклад

1. **Fork** проекта
2. Создайте **feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit** изменения (`git commit -m 'Add amazing feature'`)
4. **Push** в branch (`git push origin feature/amazing-feature`)
5. Откройте **Pull Request**

### Стандарты разработки

- Следуйте стилю кода (Black + Ruff)
- Добавляйте тесты для новой функциональности
- Обновляйте документацию
- Используйте semantic commit messages

### Отчеты об ошибках

Используйте [GitHub Issues](https://github.com/your-username/AIVAN/issues) для отчетов об ошибках.

**Template для отчета:**
```markdown
## Описание ошибки
Краткое описание проблемы

## Шаги воспроизведения
1. Перейти к '...'
2. Нажать на '...'
3. Увидеть ошибку

## Ожидаемое поведение
Что должно было произойти

## Скриншоты
При необходимости

## Окружение
- OS: [e.g. Ubuntu 22.04]
- Python: [e.g. 3.12]
- Version: [e.g. 0.1.0]
```

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

## 🙏 Благодарности

- [OpenAI](https://openai.com/) за API
- [aiogram](https://github.com/aiogram/aiogram) за Telegram Bot framework
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) за OCR возможности
- Сообществу разработчиков за вклад и обратную связь

## 📞 Поддержка

- **Документация**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-username/AIVAN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/AIVAN/discussions)
- **Email**: support@aivan.ai

---

<p align="center">
  Сделано с ❤️ для юридического сообщества
</p>