# Project Validation Script

`scripts/validate_project.py` runs the same set of guardrails we expect in CI. Use it before every
release, infrastructure change, or when you onboard a teammate to ensure their workstation is ready.

## Running the script
```bash
poetry run python scripts/validate_project.py
```

The command prints the status of each check and exits with a non-zero status when something fails,
making it safe to wire into Git hooks or deployment pipelines.

## What it checks
1. **Dependencies** — imports `aiogram`, `aiosqlite`, `httpx`, `openai`, `pytest` and logs which
   optional extras (`redis`, `prometheus_client`, `psutil`) are missing.
2. **Configuration files** — verifies that `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, and
   `README.md` exist.
3. **SQLite bootstrap** — spins up a temp database via `DatabaseAdvanced`, creates a user, decrements
   their trial counter, and closes the pool cleanly.
4. **Dependency injection container** — instantiates `AppSettings` from a fake env, builds the DI
   container, and resolves `DatabaseAdvanced` through it.
5. **Cache and metrics** — exercises the in-memory cache backend plus the Prometheus-friendly
   `MetricsCollector` timers.
6. **Tests folder** — makes sure `tests/` exists and contains at least one `test_*.py` file.

When all six checks succeed you get a `Validation summary: 6/6` line at the end.

## Customizing
- Add new validators to the `validators` list in the script when you introduce critical services,
  such as external APIs or migrations.
- If your deployment uses PostgreSQL instead of SQLite, you can temporarily point `DB_PATH` at a
  scratch file to keep the validation lightweight, then mirror the same logic in your PG-specific
  smoke tests.
- The helper already handles Ctrl+C gracefully, so you can run it from IDE tasks without littering
  temp files (it deletes temp DB files in `finally` blocks).

## Recommended workflow
1. Run `poetry install --with dev`.
2. Copy `.env.example` to `.env` and fill in the secrets you have.
3. Execute `poetry run python scripts/validate_project.py`.
4. Fix any FAILED lines before proceeding to `poetry run python scripts/run_tests.py`.

Following this checklist keeps the local environment close to CI, reducing “works on my machine”
failures.
