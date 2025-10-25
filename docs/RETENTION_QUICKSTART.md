# Retention Notifications Quickstart

`RetentionNotifier` keeps users engaged by sending automated follow-ups from inside the bot runtime.
It starts automatically in `core/bot_app/startup.py` once the aiogram dispatcher is initialised.
This document explains how to customise, monitor, and test the notifier.

## What it does
- Scans the `users` table once per hour.
- Applies `NOTIFICATION_SCENARIOS` (defined in `src/core/bot_app/retention_notifier.py`) which
  describe:
  - `registered_no_request` - users who never sent a question within 24h.
  - `inactive_3days` / `inactive_7days` - churned users with no requests for N days.
  - `winback_*` scenarios for longer absences (extend the list as needed).
- Sends HTML-formatted messages (with optional inline keyboards) using aiogram’s `Bot.send_message`.
- Records deliveries in `retention_notifications` and keeps a `blocked_users` table to avoid
  retrying deactivated chats.

## Editing templates
1. Open `src/core/bot_app/retention_notifier.py`.
2. Modify or add `NotificationTemplate` entries (message body, delay in hours, buttons).
3. Keep the copy concise and use HTML tags supported by Telegram. Emojis should be added via plain
   Unicode to stay ASCII-safe in git diffs.
4. Restart the bot; templates are evaluated on startup.

## Running & monitoring
- No extra CLI flags are required. When the bot starts you should see `Retention notifier started`.
- Logs include the number of processed users per scenario and delivery errors (e.g. blocked chats).
- Metrics:
  - `retention_notifications_sent_total` (counter, exported via Prometheus).
  - `retention_errors_total`, `blocked_users_total` (see `src/core/metrics.py`).
- Admin stats: `src/core/admin_modules/admin_retention_commands.py` prints cohort summaries and
  scenario-level counts via `/retention_*` commands (available to `ADMIN_IDS`).

## Manual notifications
`RetentionNotifier.send_manual_notification([...], message, with_buttons=True)` can be called from
admin tooling or REPL sessions. The helper handles rate limits and updates the same tables so manual
messages do not collide with automated ones.

## Testing
- Unit tests: `poetry run pytest tests/test_retention_notifier.py` validates class attributes,
  database wiring, and basic integration with `core/main_simple.py` (the historical entrypoint).
- Dry runs: set `_send_concurrency` and `_send_delay` inside the class if you need throttle changes
  during stress tests.
- Schema migrations: when the notifier starts it lazily creates `retention_notifications` and
  `blocked_users` if they do not exist. If your DB is not SQLite, add explicit migrations to keep
  schemas in sync.

## Tuning tips
- The hourly cadence is controlled by `await asyncio.sleep(3600)` inside `_notification_loop`. If you
  need a different frequency, adjust the sleep and consider making it configurable.
- `_get_users_for_scenario` enforces a 1-hour window per scenario to prevent duplicate pings.
- All SQL queries run via `DatabaseAdvanced`; switch to Postgres by updating `DB_DSN` in `.env`.
- Inline keyboards use callback data (`quick_question`, `show_features`). Add handlers in
  `src/core/bot_app/menus.py` if you introduce new callbacks.

## Troubleshooting
- If no notifications are sent, confirm that the `users` table has `created_at`, `total_requests`,
  and `last_request_at` populated (see `src/core/db_advanced.py` for the schema).
- For spam/abuse reports, disable the notifier by commenting out the instantiation in
  `core/bot_app/startup.py` or by adding a feature flag around it.
- Always keep templates respectful and provide an easy CTA (button or `/start` instruction) to bring
  users back to the main flow.
