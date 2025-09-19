# telegram_legal_bot/constants.py
"""
Константы проекта для избежания магических чисел в коде.
"""

# Telegram API ограничения
TELEGRAM_MESSAGE_MAX_LENGTH = 4096
TELEGRAM_CAPTION_MAX_LENGTH = 1024

# Rate Limiting
DEFAULT_MAX_REQUESTS_PER_HOUR = 10
DEFAULT_RATE_LIMIT_PERIOD = 3600  # 1 час в секундах
DEFAULT_MAX_TRACKED_KEYS = 50000

# История сообщений
DEFAULT_HISTORY_SIZE = 5
MAX_HISTORY_USERS = 10000
HISTORY_CLEANUP_INTERVAL = 21600  # 6 часов

# OpenAI API
DEFAULT_OPENAI_TEMPERATURE = 0.3
DEFAULT_OPENAI_MAX_TOKENS = 1500
DEFAULT_MAX_OUTPUT_TOKENS = 1800
DEFAULT_OPENAI_MODEL = "gpt-5"

# Валидация входных данных
MIN_QUESTION_LENGTH = 20
MAX_DOMAIN_LENGTH = 253
MAX_RETRIES_API_CALLS = 10

# Таймауты и интервалы (в секундах)
DEFAULT_HTTP_TIMEOUT = 45.0
RATE_LIMITER_CLEANUP_MIN_INTERVAL = 300  # 5 минут

# Форматирование сообщений
MD_ESCAPE_CHARS = r"_*[]()~`>#+-=|{}.!\\"
MAX_CHUNK_BACKOFF_CHARS = 200  # Для предотвращения разрыва ссылок

# Логирование
DEFAULT_LOG_LEVEL = "INFO"
MAX_ERROR_MESSAGE_LENGTH = 1000
