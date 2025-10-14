# Security Monitoring и Alerting

## Обзор

AIVAN Bot включает комплексную систему мониторинга попыток SQL injection, XSS атак и других security violations через Prometheus метрики.

## Архитектура мониторинга

### 1. Уровни детекции

```
┌─────────────────┐
│  User Input     │
│  (Telegram)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  validation.py              │
│  • XSS паттерны             │◄─── Метрики: xss_attempts_total
│  • SQL injection паттерны   │◄─── Метрики: sql_injection_attempts_total
│  • Spam detection           │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  db_advanced.py             │
│  • Table whitelist          │◄─── Метрики: sql_injection_attempts_total
│  • PRAGMA validation        │     (source=database_layer)
└─────────────────────────────┘
```

### 2. Prometheus метрики

#### `security_violations_total`
**Тип**: Counter
**Labels**:
- `violation_type`: тип нарушения (`sql_injection`, `xss`, `spam`)
- `severity`: уровень критичности (`warning`, `error`, `critical`)
- `source`: источник детекции (`user_input`, `database_layer`)

**Пример**:
```promql
# Все критичные нарушения за последние 5 минут
security_violations_total{severity="critical"}[5m]
```

#### `sql_injection_attempts_total`
**Тип**: Counter
**Labels**:
- `pattern_type`: тип SQL injection паттерна
  - `sql_keywords` - использование SQL ключевых слов (SELECT, UNION, etc.)
  - `sql_comment` - SQL комментарии (`'; --`)
  - `sql_or_equals` - классический `OR '1'='1'`
  - `sql_numeric_equals` - числовой `OR 1=1`
  - `sql_dangerous_commands` - DROP, DELETE, TRUNCATE
  - `invalid_table_name` - попытка доступа к несуществующей таблице
- `source`: `user_input` или `database_layer`

**Пример**:
```promql
# Топ-5 паттернов SQL injection
topk(5, sum by (pattern_type) (sql_injection_attempts_total))
```

#### `xss_attempts_total`
**Тип**: Counter
**Labels**:
- `pattern_type`: тип XSS паттерна
  - `script_tag` - `<script>` теги
  - `javascript_protocol` - `javascript:` протокол
  - `event_handler` - `onclick=`, `onerror=`
  - `iframe_tag`, `object_tag`, `embed_tag`
- `source`: источник детекции

## Алертинг

### Критичные алерты

#### 1. **HighRateSQLInjectionAttempts**
```yaml
expr: rate(sql_injection_attempts_total[5m]) > 5
for: 2m
```
**Триггер**: Более 5 попыток SQL injection в секунду в течение 2 минут
**Действия**:
1. Проверить логи на user_id атакующих
2. Заблокировать IP через rate limiter
3. Проверить валидацию входных данных

#### 2. **DatabaseLayerSQLInjection**
```yaml
expr: sql_injection_attempts_total{source="database_layer"} > 0
for: 1m
```
**Триггер**: Любая попытка SQL injection в database layer
**Критичность**: ОЧЕНЬ ВЫСОКАЯ - означает bypass валидации
**Действия**:
1. НЕМЕДЛЕННО проверить код на предмет пропущенной валидации
2. Проверить логи на возможную эксплуатацию
3. Провести security аудит кодовой базы

#### 3. **XSSAttemptsDetected**
```yaml
expr: rate(xss_attempts_total[5m]) > 2
for: 2m
```
**Триггер**: Более 2 XSS попыток в секунду
**Действия**:
1. Проверить escape HTML в ответах бота
2. Убедиться, что Telegram HTML parser корректно работает
3. Проверить источник атаки

### Warning алерты

#### 4. **SingleUserSecurityViolations**
```yaml
expr: topk(5, sum by (user_id) (increase(security_violations_total[10m]))) > 10
for: 5m
```
**Триггер**: Один пользователь совершил >10 нарушений за 10 минут
**Действия**:
1. Добавить user_id в временный blacklist
2. Проанализировать паттерн поведения
3. Возможно, это автоматизированный scanner

## Интеграция с кодом

### validation.py

```python
from src.core.metrics import get_metrics_collector

metrics = get_metrics_collector()

# При детекции SQL injection
metrics.record_sql_injection_attempt(
    pattern_type="sql_keywords",
    source="user_input"
)
metrics.record_security_violation(
    violation_type="sql_injection",
    severity="warning",
    source="user_input"
)

# При детекции XSS
metrics.record_xss_attempt(
    pattern_type="script_tag",
    source="user_input"
)
metrics.record_security_violation(
    violation_type="xss",
    severity="critical",
    source="user_input"
)
```

### db_advanced.py

```python
# Whitelist проверка таблиц
ALLOWED_TABLES = frozenset([
    "users", "transactions", "payments", "requests", "ratings",
    "nps_surveys", "behavior_events", "user_journey_events"
])

if table not in ALLOWED_TABLES:
    metrics.record_sql_injection_attempt(
        pattern_type="invalid_table_name",
        source="database_layer"
    )
    raise ValueError(f"Invalid table name: {table}")
```

## Запросы для анализа

### 1. Топ атакующих пользователей (последние 24 часа)
```promql
topk(10,
  sum by (user_id) (
    increase(security_violations_total[24h])
  )
)
```

### 2. Динамика SQL injection попыток
```promql
rate(sql_injection_attempts_total[5m])
```

### 3. Распределение по типам XSS паттернов
```promql
sum by (pattern_type) (xss_attempts_total)
```

### 4. Security violations по severity
```promql
sum by (severity) (security_violations_total)
```

### 5. Попытки bypass валидации (database_layer)
```promql
sql_injection_attempts_total{source="database_layer"}
```

## Grafana Dashboard

### Рекомендуемые панели:

1. **Security Overview**
   - Total violations (last 24h)
   - SQL injection rate (5m window)
   - XSS attempt rate (5m window)
   - Top attacking users

2. **SQL Injection Analysis**
   - Timeline по pattern_type
   - Heatmap по часам дня
   - Top patterns
   - Source breakdown (user_input vs database_layer)

3. **XSS Detection**
   - Timeline по pattern_type
   - Event handler attempts vs script tag attempts

4. **Alerts Timeline**
   - Active alerts
   - Alert history

## Логирование

Все security violations также пишутся в logs:

```python
logger.warning(
    f"SQL injection attempt detected: pattern={pattern_name}, "
    f"text_preview={text[:100]}"
)
```

**Формат лога**:
```
2025-01-15 14:32:11 WARNING SQL injection attempt detected: pattern=sql_keywords, text_preview=SELECT * FROM users WHERE '1'='1'
```

## Response Playbook

### При срабатывании алерта:

1. **Immediate** (0-5 минут):
   - Проверить Grafana dashboard
   - Определить масштаб атаки
   - Проверить логи на user_id

2. **Short-term** (5-30 минут):
   - Если burst атака → включить aggressive rate limiting
   - Если один пользователь → временный бан
   - Проверить, есть ли успешные эксплуатации

3. **Investigation** (30+ минут):
   - Полный анализ логов
   - Проверка на data exfiltration
   - Патчинг если найдена уязвимость

4. **Post-mortem**:
   - Документирование инцидента
   - Обновление detection rules
   - Улучшение валидации

## Настройка

### Environment variables

```bash
# В .env
PROMETHEUS_PORT=8000
ENABLE_PROMETHEUS=true
```

### Запуск Prometheus

```bash
# prometheus.yml уже включает scrape конфигурацию
prometheus --config.file=prometheus_alerts.yml
```

### Проверка метрик

```bash
# Локально
curl http://localhost:8000/metrics | grep security_violations

# Должно показать:
# security_violations_total{violation_type="sql_injection",severity="warning",source="user_input"} 5
# sql_injection_attempts_total{pattern_type="sql_keywords",source="user_input"} 3
# xss_attempts_total{pattern_type="script_tag",source="user_input"} 2
```

## Best Practices

1. **Regular Review**: Анализировать метрики раз в неделю
2. **Threshold Tuning**: Корректировать пороги алертов на основе false positives
3. **Pattern Updates**: Обновлять detection паттерны при появлении новых векторов атак
4. **Log Retention**: Хранить security логи минимум 90 дней
5. **Alert Routing**: Критичные алерты → PagerDuty, Warning → Slack

## Дополнительные метрики

Для полного мониторинга также используйте:
- `telegram_messages_total{status="error"}` - ошибки обработки сообщений
- `database_operations_total{status="error"}` - ошибки БД
- `openai_requests_total{status="error"}` - ошибки OpenAI API
