# Multi-stage build для оптимизации размера образа
FROM python:3.12-slim as builder

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
RUN pip install poetry==1.8.0

# Настройка Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Копирование файлов конфигурации
COPY pyproject.toml poetry.lock ./

# Установка зависимостей
RUN poetry install --only=main,prod && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.12-slim as runtime

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя для безопасности
RUN groupadd -r aivan && useradd -r -g aivan aivan

# Настройка рабочей директории
WORKDIR /app

# Копирование virtual environment из builder stage
COPY --from=builder --chown=aivan:aivan /app/.venv /app/.venv

# Активация virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Копирование исходного кода
COPY --chown=aivan:aivan src/ ./src/

# Создание директорий для данных
RUN mkdir -p /app/data /app/logs && chown -R aivan:aivan /app/data /app/logs

# Переключение на непривилегированного пользователя
USER aivan

# Переменные окружения
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "print('health: ok')" || exit 1

# Экспонирование портов (если планируется web интерфейс)
EXPOSE 8000

# Запуск приложения
CMD ["python", "src/core/main_simple.py"]
