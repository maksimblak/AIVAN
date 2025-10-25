# Background Tasks

`src/core/background_tasks.py` defines the cooperative scheduler that keeps caches clean, enforces
document quotas, and exports metrics. This guide explains the default tasks and how to register your
own jobs.

## Architecture
- `BackgroundTask` is an abstract base class with an `execute()` coroutine, retry logic, and metrics
  about the last run (`TaskResult`, `TaskStatus`).
- `BackgroundTaskManager` stores tasks in a dictionary, starts them (each task gets its own asyncio
  loop via `_task`), restarts them on demand, and exposes aggregated stats through
  `get_all_stats()`.
- `src/core/bot_app/startup.py:253` instantiates the manager, registers the built-in tasks (see
  below), and calls `await task_manager.start_all()`. On shutdown the bot awaits `task_manager.stop_all()`.

## Built-in tasks
| Task | Class | Purpose | Key settings |
|------|-------|---------|--------------|
| Database cleanup | `DatabaseCleanupTask` | Removes stale transactions and trims audit tables via `DatabaseAdvanced`. | `DB_CLEANUP_INTERVAL`, `DB_CLEANUP_DAYS`. |
| Cache cleanup | `CacheCleanupTask` | Calls `clear_expired()` on OpenAI caches and other cache-like components. | `CACHE_CLEANUP_INTERVAL`. |
| Session cleanup | `SessionCleanupTask` | Prunes expired entries from `SessionStore`. | `SESSION_CLEANUP_INTERVAL`. |
| Document storage cleanup | `DocumentStorageCleanupTask` | Deletes temp OCR outputs older than the configured TTL and enforces quota. | `DOCUMENTS_CLEANUP_INTERVAL_SECONDS`, `DOCUMENTS_CLEANUP_HOURS`, `DOCUMENTS_STORAGE_QUOTA_MB`. |
| Health check | `HealthCheckTask` | Iterates through the critical services map (`database`, `rate_limiter`, etc.) and records their status. | `HEALTH_CHECK_TASK_INTERVAL`. |
| Metrics collection | `MetricsCollectionTask` | Aggregates `get_stats()` output from main components so Prometheus can scrape historical values. | `METRICS_COLLECTION_INTERVAL`. |

All intervals map back to `AppSettings` fields (see `src/core/settings.py:120` onwards) and can be
overridden per deployment.

## Adding a custom task
1. Subclass `BackgroundTask` and implement `async def execute(self)`. Raise an exception to trigger a
   retry; the manager automatically applies exponential backoff via `max_retries` / `retry_delay`.
2. Register the task inside `startup.py` after the built-in tasks:
   ```python
   task_manager.register_task(MyCoolTask(...))
   ```
3. Provide dependencies (DB handles, clients, etc.) via the constructor. Tasks are long-lived, so
   cache references on `self` instead of re-importing modules inside `execute()`.
4. Add observability: use `logger` or expose stats through a method that `MetricsCollectionTask` can
   call.

## Operational tips
- Call `task_manager.get_all_stats()` from an admin command to see which tasks are running and when
  they last completed.
- Catching up after downtime: each task keeps `_last_run` in memory. When the bot restarts, the first
  iteration fires immediately, so long maintenance windows are handled automatically.
- If a task crashes repeatedly you will see warnings in logs and its `TaskStatus` flips to `FAILED`.
  Update retry parameters or add internal guards.
- For I/O heavy tasks, respect `self._running` and cancellation: if `stop()` is called the task
  should exit its loop quickly to avoid blocking shutdown.

With this scheduler in place you can keep the primary aiogram event loop slim: anything that does not
require immediate interaction with the user should be modeled as a background task instead.
