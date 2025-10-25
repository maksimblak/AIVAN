# AIVAN - Telegram Legal Assistant

AIVAN is a production-ready Telegram bot that helps users prepare legal documents, analyse cases,
and answer legal questions with OpenAI models, OCR tooling, and Retrieval-Augmented Generation
(RAG). The codebase bundles subscriptions, payments, retention, observability, and background
tasks so the bot can run unattended in production.

## Highlights
- **Conversational legal copilot** - rich `/start` menu, status animations, streaming answers, voice
  mode, and message validation driven by `src/core/bot_app` and `src/core/openai_service.py`.
- **Document automation** - OCR, anonymisation, lawsuit analysis, risk scoring, and document
  drafting powered by `src/documents/*` and the aiogram handlers in
  `src/core/bot_app/documents.py`.
- **Payments & access control** - CryptoPay, Telegram Stars, Robokassa, and YooKassa plus trials
  and subscriptions (see `src/core/payments.py`, `subscription_*`, and `core/bot_app/payments.py`).
- **RAG pipeline** - optional Qdrant vector store with OpenAI embeddings (see `src/core/rag/*`
  and `docs/QUICKSTART_RAG.md`).
- **Retention & analytics** - automated win-back notifications, admin queries, and Excel exports
  built into `src/core/bot_app/retention_notifier.py` and `src/core/admin_modules/*`.
- **Observability & safety** - Prometheus metrics, structured health checks, anti-abuse input
  validation, background tasks, and scaling helpers (`src/core/metrics.py`, `health.py`,
  `validation.py`, `background_tasks.py`, `scaling.py`).

## Architecture Overview

```
Telegram Updates
    ↓
aiogram dispatcher (src/core/bot_app/* handlers)
    ↓
Access / rate limits / session store ──┬── OpenAIService + AudioService
                                        ├── DocumentManager (OCR, drafter, analyzers)
                                        ├── Payments (CryptoPay / Robokassa / YooKassa / Stars)
                                        ├── RAG & Garant API clients
                                        └── Background + retention services
```

Configuration is centralised in `src/core/settings.AppSettings`, injected via the DI container
(`src/core/di_container.py`) and exposed at runtime through `src/core/bootstrap.py` and
`src/core/bot_app/context.py`. Shared data flows through `DatabaseAdvanced` (SQLite by default),
Redis-backed caches (optional), and the document storage backends in `src/documents/storage_backends.py`.

## Repository Layout
| Path | Purpose |
|------|---------|
| `src/telegram_legal_bot` | Entrypoints (`main.py`) and health check CLI. |
| `src/core` | Runtime service layer: DI, OpenAI, payments, metrics, scaling, health checks, excel export, etc. |
| `src/core/bot_app` | aiogram bot application (menus, questions, documents, payments, retention, admin). |
| `src/documents` | OCR, translation, anonymisation, drafter, lawsuit/risk analysis pipelines. |
| `data/` | Local assets (SQLite, fixtures). |
| `docs/` | Operational guides (deployment, RAG, retention, monitoring, voice mode). |
| `scripts/` | Automation utilities (`run_tests.py`, `validate_project.py`, `load_judicial_practice.py`, `test_rag.py`). |
| `tests/` | Pytest suite (async-friendly, mirrors src layout). |

## Getting Started
1. **Requirements**: Python 3.12+, Poetry 1.8+, Docker (for parity), OpenAI API key, Telegram bot token.
2. **Install dependencies**:
   ```bash
   poetry install --with dev
   cp .env.example .env  # update secrets locally
   ```
3. **Run the bot**:
   ```bash
   poetry run telegram-legal-bot
   ```
4. **Smoke test & CI-equivalent checks**:
   ```bash
   poetry run python scripts/run_tests.py          # lint + type-check + pytest
   poetry run pytest -k documents                  # example focused test run
   poetry run python scripts/validate_project.py   # dependency + settings sanity
   ```
5. **Docker**:
   ```bash
   docker compose up --build bot
   # or
   docker build -t aivan-bot .
   docker run --env-file .env --volume $(pwd)/data:/app/data aivan-bot
   ```
   For Retrieval-Augmented Generation (RAG) support the stack now includes a `qdrant` service:
   `docker compose up -d qdrant` will start the vector DB with its data persisted under
   `./qdrant_data` (via the named volume).

See `docs/DEPLOYMENT.md` for the full production checklist.

## Configuration Reference
Key `.env` variables loaded by `AppSettings`:

| Area | Variables |
|------|-----------|
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_SUPPORT_USERNAME`, optional proxy credentials (`TELEGRAM_PROXY_*`). |
| OpenAI | `OPENAI_API_KEY`, `USE_STREAMING`, `USE_STATUS_ANIMATION`, default model knobs in `src/core/openai_service.py`. |
| Database / cache | `DB_PATH`, `DB_MAX_CONNECTIONS`, `CACHE_BACKEND`, `CACHE_TTL`, optional `REDIS_URL`. |
| Access & trials | `TRIAL_REQUESTS`, `SUB_DURATION_DAYS`, `ADMIN_IDS` (comma-separated). |
| Payments | `TELEGRAM_PROVIDER_TOKEN_RUB`, `TELEGRAM_PROVIDER_TOKEN_STARS`, `CRYPTO_PAY_TOKEN`, `ROBOKASSA_*`, `YOOKASSA_*`. |
| Voice mode | `ENABLE_VOICE_MODE`, `VOICE_STT_MODEL`, `VOICE_TTS_MODEL`, `VOICE_TTS_VOICE(_MALE)`, `VOICE_TTS_SPEED`, `VOICE_MAX_DURATION_SECONDS`. |
| RAG | `RAG_ENABLED`, `RAG_QDRANT_URL` or (`RAG_QDRANT_HOST`/`PORT`), `RAG_QDRANT_API_KEY`, `RAG_COLLECTION`, `RAG_TOP_K`, `RAG_SCORE_THRESHOLD`, `RAG_CONTEXT_CHAR_LIMIT`. |
| Observability | `ENABLE_PROMETHEUS`, `PROMETHEUS_PORT`, `ENABLE_SYSTEM_MONITORING`, health/cleanup intervals. |
| Integrations | `GARANT_API_BASE_URL`, `GARANT_API_TOKEN`, `GARANT_API_ENV`, etc. |

Each variable accepts empty strings; `AppSettings` normalises blanks to `None` where sensible.

## Document Automation Stack
- **Uploads & OCR**: `src/core/bot_app/documents.py` routes photos, PDFs, and archives through
  `src/documents/ocr_converter.py`, anonymisation, translation, and summarisation pipelines.
- **Document drafter**: `generate_document` and `plan_document` (`src/documents/document_drafter.py`)
  convert user briefs into Markdown plans and DOCX outputs (via `python-docx`). Telegram handlers
  handle turn-by-turn clarifications, voice inputs, and attachments.
- **Lawsuit & risk analysis**: `src/documents/lawsuit_analyzer.py`, `risk_analyzer.py`, and
  `document_chat.py` expose analysis reports, structured plans, and follow-up chat sessions.
- **RAG context**: when enabled, `JudicialPracticeRAG` injects relevant case excerpts into prompts.

## Observability & Operations
- **Metrics**: `src/core/metrics.py` exports counters/gauges/histograms to Prometheus (`/metrics`
  HTTP server) or an in-memory fallback. Key metrics include `telegram_messages_total`,
  `openai_requests_total`, `payment_transactions_total`, `security_violations_total`, and
  `system_status`.
- **Health checks**: `python -m telegram_legal_bot.healthcheck` runs the same checks triggered in
  production (DB, OpenAI, session store, rate limiter, system resources).
- **Background tasks**: `src/core/background_tasks.py` defines cleanup jobs for the database, cache,
  sessions, document storage, and metrics. `RetentionNotifier` runs hourly alongside those tasks.
- **Security monitoring**: `src/core/validation.py` sanitises user prompts (XSS/SQL/spam) and
  increments the security metrics. See `docs/security_monitoring.md` for triage tips.
- **Scaling hooks**: `src/core/scaling.py` provides optional service registry + load balancer if you
  point the bot at Redis for shared session affinity.

## Development Workflow
- Formatters: `poetry run black .` and `poetry run isort .` (Black profile, 100 columns).
- Linting: `poetry run ruff check src tests` (E/F rules by default).
- Typing: `poetry run mypy src tests` (core/documents/bot packages are temporarily ignored).
- Tests: `poetry run pytest`, or targeted modules via `-k`. Async tests use `pytest-asyncio`.
- Tooling: `scripts/run_tests.py` bundles lint + type + tests; `scripts/validate_project.py` validates
  settings, dependencies, and migrations.
- Git: keep commits small, run tests before pushing, and follow lower-case imperative messages
  (e.g., `add retention notifier metrics`).

## RAG & Data Loading
- Quickstart: `docs/QUICKSTART_RAG.md` walks through launching Qdrant, populating data with
  `scripts/load_judicial_practice.py`, and verifying matches via `scripts/test_rag.py`.
  When using Docker Compose just run `docker compose up -d qdrant` instead of the standalone
  `docker run` command from the guide.
- Deep dive: `docs/RAG_SETUP.md` covers collection design, payload schema, and production tuning.

## Retention & Engagement
`RetentionNotifier` (started inside `core/bot_app/startup.py`) scans the `users` table and sends
message templates defined in `NOTIFICATION_SCENARIOS`. Metrics, admin dashboards, and a helper test
suite live in `tests/test_retention_notifier.py`. See `docs/RETENTION_QUICKSTART.md` for operating
procedures.

## Additional Guides
- `docs/SCALING.md` - enabling the optional Redis-backed service registry, load balancer, and session
  affinity.
- `docs/BACKGROUND_TASKS.md` - understanding the built-in cleanup jobs and how to register custom
  tasks.
- `docs/EXCEL_EXPORT.md` - generating XLSX attachments for RAG answers and admin exports.
- `docs/ADMIN_ANALYTICS.md` - retention/cohort/PMF tooling available to admins inside Telegram.
- `docs/PROJECT_VALIDATION.md` - running `scripts/validate_project.py` to mirror CI checks locally.

## Support & Questions
- Consult `docs/DEPLOYMENT.md`, `docs/security_monitoring.md`, and `docs/voice_mode.md` for common
  operational scenarios.
- For new integrations, mirror the existing structure: put shared services under `src/core`,
  telegram handlers under `src/core/bot_app`, and data processors under `src/documents`.

Feel free to open issues or send questions to `support@aivan.ai`.
