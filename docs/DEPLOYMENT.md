# AIVAN Deployment Guide

This runbook covers the tasks required to promote the Telegram bot from development to staging or
production. Use it alongside `docs/security_monitoring.md` and `docs/voice_mode.md` when specific
features are enabled.

## Prerequisites
- Python 3.12+, Poetry 1.8+, Docker 24+ (Compose v2).
- Telegram bot token, OpenAI API key, payment provider credentials (CryptoPay, Telegram Stars,
  Robokassa, YooKassa) as needed.
- Access to permanent storage for `./data` (SQLite DB, OCR caches) and `./logs`.
- Optional: Redis for cache/session affinity, Qdrant for RAG, VPN/proxy for Telegram/OpenAI.

## 1. Configure environment
1. Bootstrap the virtualenv and copy the sample environment file:
   ```bash
   poetry install --with dev
   cp .env.example .env
   ```
2. Fill `.env` with production secrets. At minimum set:
   - `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY`
   - `DB_PATH` (absolute path inside the container/VM, e.g. `/app/data/bot.sqlite3`)
   - Payment tokens you plan to expose to users
   - `ADMIN_IDS` for on-call maintainers
3. Optional toggles:
   - `ENABLE_PROMETHEUS=1` and `PROMETHEUS_PORT=9000` for metrics
   - `ENABLE_VOICE_MODE=1` plus voice model presets (see `docs/voice_mode.md`)
   - `RAG_*` variables if Qdrant is available (see `docs/RAG_SETUP.md`)
   - `GARANT_API_*` if you licensed the Garant integration

## 2. Build & verify
Run the CI-equivalent suite locally before packaging:
```bash
poetry run python scripts/run_tests.py        # lint, type-check, pytest
poetry run python scripts/validate_project.py # settings sanity (env, DB, Redis, etc.)
poetry run telegram-legal-bot                 # manual smoke test
python -m telegram_legal_bot.healthcheck      # health probes used by k8s/compose
```
If you touched migrations or background tasks, run targeted pytest selections (e.g. `-k retention`).

## 3. Package & run with Docker
```bash
docker build -t aivan-bot .
docker run \
  --env-file prod.env \
  --name aivan-bot \
  --restart unless-stopped \
  -v /srv/aivan/data:/app/data \
  -v /srv/aivan/logs:/app/logs \
  -p 9000:9000 \  # Prometheus (optional)
  aivan-bot
```
For multi-container deployments use the supplied Compose file:
```bash
docker compose up --build bot
```
Mount `./data` and `./logs`, or wire the container to managed Postgres/Redis/Qdrant if desired.

## 4. Observability & health
- **Metrics**: enable Prometheus via `ENABLE_PROMETHEUS` and scrape the `/metrics` endpoint
  (exposed by `src/core/metrics.py`). Alert on `system_status != running` or spikes in
  `security_violations_total`.
- **Health checks**: the health CLI mirrors the HTTP probe `python -m telegram_legal_bot.healthcheck`.
- **Background tasks**: `BackgroundTaskManager` (sessions, cache, DB cleanup) and
  `RetentionNotifier` start automatically; their status is logged during startup.
- **Logs**: aiogram events and service logs emit to stdout; redirect to your collector or mount
  `/app/logs`.

## 5. Storage & secrets
- SQLite lives under `data/bot.sqlite3` by default. Mount a persistent volume or swap in another DB
  via `DB_PATH` + `DB_DSN` (see `src/core/db_advanced.py`).
- OCR/temp files sit under `data/documents`. Document drafter headers are loaded from `images/`.
- Never commit `.env`; track secret changes in your deployment inventory (Vault, AWS Secrets
  Manager, GitHub Actions secrets, etc.).

## Release checklist
1. Confirm `.env` is updated and encrypted in the secrets manager.
2. Run `scripts/run_tests.py` and `scripts/validate_project.py`.
3. Build the Docker image and run `python -m telegram_legal_bot.healthcheck` inside the container.
4. Capture screenshots of user-facing changes (menus, payments, document flows) for the release PR.
5. Verify Prometheus metrics arrive and `system_status=running` after deploy.
6. Announce the release with the current git commit hash and image tag.

## Rollback & troubleshooting
- Stop the bot with `docker stop aivan-bot` or scale down the Compose service.
- Restore the previous `.env`/image tag and relaunch.
- Inspect `logs/` plus `docker logs -f aivan-bot` for aiogram errors, payment failures, or
  background task exceptions.
- Use `python -m telegram_legal_bot.healthcheck` to isolate failed dependencies (DB, OpenAI, Redis).
- Check Prometheus counters (`openai_requests_total`, `payment_transactions_total`,
  `security_violations_total`) for anomalies.
- If retention notifications misbehave, run `pytest -k retention_notifier` and consult
  `docs/RETENTION_QUICKSTART.md`.

For advanced configuration (RAG, scaling, monitoring) refer to the dedicated guides under `docs/`.
