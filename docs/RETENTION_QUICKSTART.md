# ⚡ Retention Notifications - Быстрый старт

## 🎯 Что это?

Система автоматических напоминаний для пользователей, которые:
- Зарегистрировались, но не сделали ни одного запроса
- Были активны, но перестали заходить

## 🚀 Запуск (уже готово!)

Система **уже интегрирована** в бота и работает автоматически!

```bash
python src/core/main_simple.py
```

В логах увидишь:
```
INFO: ✉️ Retention notifier started
```

Всё! Система работает в фоне.

## 📧 Какие уведомления отправляются?

### 1. Через 24 часа после регистрации
**Кому:** Нажали /start, но не задали ни одного вопроса
**Сообщение:** Напоминание о 10 бесплатных вопросах

### 2. Через 3 дня неактивности
**Кому:** Были активны, но не заходили 3 дня
**Сообщение:** Напоминание о возможностях бота

### 3. Через 7 дней неактивности
**Кому:** Были активны, но не заходили неделю
**Сообщение:** Последнее напоминание + новые фичи

## 📊 Как посмотреть статистику?

```python
from src.bot.retention_notifier import retention_notifier

stats = await retention_notifier.get_notification_stats()
print(stats)

# {
#     "total_sent": 1543,
#     "by_scenario": {
#         "registered_no_request": 842,
#         "inactive_3days": 421,
#         "inactive_7days": 280
#     },
#     "blocked_users": 45
# }
```

## 🛠 Как отправить уведомление вручную?

```python
from src.bot.retention_notifier import retention_notifier

stats = await retention_notifier.send_manual_notification(
    user_ids=[123456, 789012],
    message="<b>Важное обновление!</b>\n\nТеперь доступна новая фича...",
    with_buttons=True
)

print(stats)
# {"sent": 1, "failed": 0, "blocked": 1}
```

## ⚙️ Настройка

### Изменить время задержки

Открой `src/bot/retention_notifier.py`:

```python
NOTIFICATION_SCENARIOS = [
    NotificationTemplate(
        name="registered_no_request",
        delay_hours=48,  # Измени с 24 на 48 часов
        message="...",
        show_buttons=True
    ),
]
```

### Изменить текст сообщений

Просто отредактируй `message` в `NOTIFICATION_SCENARIOS`.

### Добавить новый сценарий

```python
NOTIFICATION_SCENARIOS.append(
    NotificationTemplate(
        name="inactive_30days",
        delay_hours=720,  # 30 дней
        message="Скучаем по тебе! 😢",
        show_buttons=True
    )
)
```

## 📈 Мониторинг

### Логи

```bash
# Смотри логи бота
grep "Retention" logs/bot.log

# Примеры:
# INFO: RetentionNotifier started
# INFO: Processing 15 users for scenario 'registered_no_request'
# INFO: Sent 14 notifications for scenario 'registered_no_request'
```

### База данных

```sql
-- Все отправленные уведомления
SELECT * FROM retention_notifications;

-- Заблокированные пользователи
SELECT * FROM blocked_users;

-- Статистика по сценариям
SELECT scenario, COUNT(*) as count
FROM retention_notifications
GROUP BY scenario;
```

## ❓ FAQ

### Как отключить?

Закомментируй в `main_simple.py`:
```python
# retention_notifier = RetentionNotifier(bot, db)
# await retention_notifier.start()
```

### Почему уведомления не отправляются?

1. Проверь логи: `grep "Retention" logs/bot.log`
2. Проверь что прошло >= delay_hours с момента регистрации/последнего запроса
3. Проверь что пользователь не заблокировал бота

### Как часто проверяется?

Каждый час (3600 секунд). Можно изменить в `_notification_loop()`:
```python
await asyncio.sleep(3600)  # Измени на нужное значение
```

### Сколько пользователей обрабатывается за раз?

Максимум 100 на каждый сценарий (LIMIT 100 в SQL).

## 🐛 Проблемы?

### Тесты
```bash
python test_retention_notifier.py
```

### Проверка импортов
```bash
python -c "from src.bot.retention_notifier import RetentionNotifier; print('OK')"
```

## 📚 Документация

- **[Полная документация](RETENTION_NOTIFICATIONS.md)** - все детали
- **[Список исправленных багов](RETENTION_BUGFIXES.md)** - что было исправлено

## ✅ Checklist перед production

- [x] Код написан
- [x] Баги исправлены
- [x] Тесты пройдены (5/5)
- [x] Документация создана
- [x] Интеграция в main_simple.py
- [x] Обработчики кнопок добавлены

**Всё готово! 🎉**
