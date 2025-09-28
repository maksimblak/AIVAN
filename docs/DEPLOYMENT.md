# 🚀 Руководство по развертыванию AIVAN

Это руководство поможет развернуть AIVAN в production окружении.

## 📋 Предварительные требования

### Системные требования
- **Linux/Ubuntu 20.04+** (рекомендуется)
- **Python 3.12+**
- **Docker 20.10+** и **Docker Compose v2**
- **Git**
- **4GB RAM** минимум (8GB рекомендуется)
- **20GB** свободного места на диске

### Внешние сервисы
- **OpenAI API ключ** - обязательно
- **Telegram Bot Token** - обязательно
- **Redis** - опционально (можно использовать встроенный)
- **Crypto Pay токен** - для платежей (опционально)

## 🛠 Быстрое развертывание (Docker)

### 1. Клонирование и подготовка

```bash
# Клонирование репозитория
git clone https://github.com/your-username/AIVAN.git
cd AIVAN

# Создание директорий
mkdir -p data logs monitoring/grafana/{dashboards,datasources}

# Настройка прав доступа
sudo chown -R 1000:1000 data logs
```

### 2. Конфигурация

Создайте файл `.env`:

```bash
cp .env.example .env
nano .env
```

**Минимальная конфигурация:**
```env
# Обязательные параметры
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
ADMIN_IDS=your_user_id

# Базовые настройки
TRIAL_REQUESTS=10
SUB_DURATION_DAYS=30
SUBSCRIPTION_PRICE_RUB=300
LOG_LEVEL=INFO

# Прокси (если требуется)
TELEGRAM_PROXY_URL=
OPENAI_PROXY_URL=
```

### 3. Запуск

```bash
# Сборка и запуск
docker-compose up -d

# Проверка статуса
docker-compose ps

# Просмотр логов
docker-compose logs -f aivan
```

### 4. Мониторинг (опционально)

```bash
# Запуск с мониторингом
docker-compose --profile monitoring up -d

# Доступ к интерфейсам:
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## 🔧 Ручное развертывание

### 1. Установка зависимостей

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip \
                 tesseract-ocr tesseract-ocr-rus poppler-utils \
                 redis-server nginx

# Установка Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Настройка проекта

```bash
# Клонирование и установка
git clone https://github.com/your-username/AIVAN.git
cd AIVAN

# Установка зависимостей
poetry install --extras "prod"

# Создание systemd сервиса
sudo nano /etc/systemd/system/aivan.service
```

**Содержимое aivan.service:**
```ini
[Unit]
Description=AIVAN Telegram Bot
After=network.target redis.service

[Service]
Type=simple
User=aivan
Group=aivan
WorkingDirectory=/opt/aivan
Environment=PATH=/opt/aivan/.venv/bin
ExecStart=/opt/aivan/.venv/bin/python src/core/main_simple.py
Restart=always
RestartSec=10

# Переменные окружения
EnvironmentFile=/opt/aivan/.env

[Install]
WantedBy=multi-user.target
```

### 3. Настройка пользователя и прав

```bash
# Создание пользователя
sudo useradd -r -d /opt/aivan -s /bin/false aivan
sudo mkdir -p /opt/aivan
sudo cp -r . /opt/aivan/
sudo chown -R aivan:aivan /opt/aivan

# Настройка логов
sudo mkdir -p /var/log/aivan
sudo chown aivan:aivan /var/log/aivan
```

### 4. Запуск и автозагрузка

```bash
# Запуск сервиса
sudo systemctl daemon-reload
sudo systemctl enable aivan
sudo systemctl start aivan

# Проверка статуса
sudo systemctl status aivan

# Просмотр логов
sudo journalctl -u aivan -f
```

## 🔒 Настройка безопасности

### 1. Firewall

```bash
# UFW (Ubuntu)
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Для мониторинга (опционально)
sudo ufw allow 3000/tcp  # Grafana
sudo ufw allow 9090/tcp  # Prometheus
```

### 2. SSL/TLS с Nginx

```bash
# Установка Certbot
sudo apt install certbot python3-certbot-nginx

# Получение сертификата
sudo certbot --nginx -d your-domain.com

# Автообновление
sudo crontab -e
# Добавить: 0 12 * * * /usr/bin/certbot renew --quiet
```

**Nginx конфигурация** (`/etc/nginx/sites-available/aivan`):
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Grafana
    location /grafana/ {
        proxy_pass http://localhost:3000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Prometheus
    location /prometheus/ {
        proxy_pass http://localhost:9090/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Секреты

```bash
# Использование systemd credentials (Ubuntu 20.04+)
sudo systemd-creds encrypt --name=telegram_token - /etc/aivan/telegram.cred
sudo systemd-creds encrypt --name=openai_key - /etc/aivan/openai.cred

# Обновление service файла для использования credentials
sudo nano /etc/systemd/system/aivan.service
```

## 📊 Мониторинг и логирование

### 1. Конфигурация Prometheus

Создайте `monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'aivan'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### 2. Настройка Grafana

1. Добавьте Prometheus как источник данных: `http://prometheus:9090`
2. Импортируйте дашборды из `monitoring/grafana/dashboards/`
3. Настройте алерты для критических метрик

### 3. Централизованное логирование

```bash
# Установка rsyslog с JSON поддержкой
sudo apt install rsyslog rsyslog-elasticsearch

# Конфигурация для отправки в ELK stack
sudo nano /etc/rsyslog.d/50-aivan.conf
```

## 🔄 Обновление и откат

### 1. Обновление Docker версии

```bash
# Создание бэкапа
docker-compose exec aivan cp -r /app/data /app/data.backup.$(date +%Y%m%d)

# Обновление кода
git pull origin main

# Пересборка и перезапуск
docker-compose build --no-cache
docker-compose up -d
```

### 2. Откат к предыдущей версии

```bash
# Остановка сервисов
docker-compose down

# Откат к предыдущему коммиту
git checkout HEAD~1

# Восстановление данных (если нужно)
docker-compose exec aivan cp -r /app/data.backup.YYYYMMDD /app/data

# Запуск
docker-compose up -d
```

## 🚨 Устранение неполадок

### Общие проблемы

**1. Ошибка подключения к Telegram API**
```bash
# Проверка прокси
curl -x $TELEGRAM_PROXY_URL https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe

# Проверка токена
docker-compose exec aivan python -c "
import os
from aiogram import Bot
bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
print('Token OK')
"
```

**2. Проблемы с базой данных**
```bash
# Проверка целостности БД
docker-compose exec aivan sqlite3 /app/data/bot.sqlite3 "PRAGMA integrity_check;"

# Бэкап и восстановление
docker-compose exec aivan sqlite3 /app/data/bot.sqlite3 ".backup /app/data/backup.db"
```

**3. Высокое использование памяти**
```bash
# Мониторинг памяти
docker stats aivan-bot

# Просмотр метрик производительности
docker-compose exec aivan python -c "
from src.core.performance import get_performance_summary
import json
print(json.dumps(get_performance_summary(), indent=2))
"
```

### Логи и диагностика

```bash
# Детальные логи
docker-compose logs --tail=100 -f aivan

# Проверка health check
docker-compose exec aivan python -c "
import asyncio
from src.core.health import check_health
asyncio.run(check_health())
"

# Системные ресурсы
docker-compose exec aivan python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"
```

## 📞 Поддержка

При возникновении проблем:

1. **Проверьте логи**: `docker-compose logs aivan`
2. **Запустите диагностику**: `python scripts/validate_project.py`
3. **Проверьте статус**: `docker-compose ps`
4. **Создайте issue**: [GitHub Issues](https://github.com/your-username/AIVAN/issues)

## 📚 Дополнительные ресурсы

- [Документация Docker Compose](https://docs.docker.com/compose/)
- [Руководство по systemd](https://systemd.io/)
- [Безопасность Nginx](https://nginx.org/en/docs/http/securing_web_traffic.html)
- [Мониторинг с Prometheus](https://prometheus.io/docs/introduction/overview/)