# Repository Guidelines

## Project Structure & Module Organization
- Core runtime lives under `src/telegram_legal_bot`, with orchestration helpers in `src/core` and reusable document tooling in `src/documents`.
- Project assets sit in `data/`; contributor references live in `docs/`.
- Integration and regression coverage resides in `tests/`; mirror new modules with matching `test_<module>.py`.
- Utility automation lives in `scripts/`; run them with Poetry to reuse the venv.

## Build, Test, and Development Commands
- `poetry install` — create/update the Poetry environment before development.
- `poetry run telegram-legal-bot` — start the entrypoint; ensure `.env` mirrors `.env.example`.
- `poetry run pytest` — execute the suite; combine with `-k <pattern>` for focused iterations.
- `poetry run python scripts/run_tests.py` — runs CI-equivalent checks (type check, Ruff, pytest) together.
- `docker compose up --build bot` — container build for deployment parity; supply production env vars.

## Coding Style & Naming Conventions
- Format with Black (line length 100) and sort imports with isort’s Black profile; both run via `poetry run black .` and `poetry run isort .`.
- Ruff enforces linting (`poetry run ruff check .`); address warnings before submitting.
- Prefer snake_case for functions and variables, PascalCase for classes, and keep module names lowercase.
- Add or update type hints; mypy (`poetry run mypy src tests`) runs in CI and blocks merges on failures.

## Testing Guidelines
- Tests use pytest with `pytest-asyncio`; mark async cases with `@pytest.mark.asyncio`.
- New features require unit tests plus a regression scenario in `tests/` mirroring the package path (e.g., `src/core/auth.py` → `tests/test_core_auth.py`).
- Use parametrized tests for matrixed inputs and keep fixtures in `tests/conftest.py` or module-scoped `conftest.py`.
- Maintain coverage for critical flows (payment pipeline, document captioning, retention); use `poetry run pytest --cov=telegram_legal_bot`.

## Commit & Pull Request Guidelines
- Recent history favors concise, lowercase, imperative summaries (e.g., `fix bug lawsuit analiser`); follow that style while staying descriptive.
- Group related changes per commit and ensure each passes formatting and pytest locally.
- Pull requests should include: purpose paragraph, explicit test evidence (`poetry run pytest` output), linked issue ID, and screenshots/log excerpts for user-facing changes.
- Flag secrets and configuration updates in the PR body and document any `.env` additions in `docs/` or as inline comments.
