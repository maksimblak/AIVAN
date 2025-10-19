# Security Monitoring & Alerting

Система безопасности отслеживает подозрительные запросы пользователей (SQL injection, XSS, попытки утечки секретов) и записывает события в метрики и логи.

## Архитектура
```
Telegram Update → validation.InputValidator → security monitor → Prometheus / Logs
```
- `src/core/validation.py` — базовая фильтрация и нормализация входящих сообщений.
- `MetricsCollector` (`src/core/metrics.py`) увеличивает счётчики `security_warnings_total`, `sql_injection_attempts_total`, `xss_attempts_total`.
- `src/core/background_tasks.py` может включать реакцию (уведомление админа, блокировка пользователя).

## Настройка
- Активируйте метрики через `ENABLE_PROMETHEUS=1` и настройте `PROMETHEUS_PORT`.
- `SECURITY_BLOCK_THRESHOLD` (в `AppSettings`) — количество предупреждений до авто-блокировки.
- `ADMIN_IDS` — список Telegram ID, куда отправляются уведомления о критических событиях.

## Метрики
| Metric | Описание |
|--------|----------|
| `security_warnings_total` | Общее число предупреждений. |
| `sql_injection_attempts_total` | Попытки SQL-injection, распознаваемые по паттернам. |
| `xss_attempts_total` | HTML/JS вставки в пользовательских сообщениях. |
| `blocked_users_total` | Сколько пользователей заблокировано автоматически. |

Все метрики имеют лейблы `reason`, `chat_id`, `feature` — используйте их в Grafana.

## Алёрты и действия
1. Включите алёрты в Prometheus: рост `security_warnings_total` > 10 за 5 минут.
2. Настройте уведомления в Telegram/Slack.
3. Проверяйте журналы (`logs/security.log`) и при необходимости вручную разблокируйте пользователей.

## Тестирование
- `tests/test_security_monitoring.py` содержит проверку XSS/SQL и метрик.
- При интеграционных тестах используйте sandbox-бота и тестовые сообщения.
- Скрипт `scripts/validate_project.py` выполняет sanity-check зависимостей, включая Prometheus клиент.

