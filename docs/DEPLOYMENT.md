# AIVAN Production Deployment Guide

This document summarises how to run the bot in a production-ready way after the recent
hardening pass.

## 1. Prerequisites
- Python 3.12 (or run via the provided Docker image).
- Access to the required external services: OpenAI, Telegram Bot API, payment providers.
- Redis (optional, enables persistent rate limiting and scaling scenarios).
- Prometheus + Grafana stack (optional, for observability).

## 2. Configuration
1. Copy `.env.example` to the secure location you use for secrets management.
2. Fill in the required variables:
   - `TELEGRAM_BOT_TOKEN`
   - `OPENAI_API_KEY`
   - Payment tokens (`CRYPTO_PAY_TOKEN`, `TELEGRAM_PROVIDER_TOKEN_*`, etc.).
3. Set optional knobs when you enable extra services:
   - `REDIS_URL` when deploying Redis.
   - `PROMETHEUS_PORT` when exposing metrics.
4. Do **not** commit real secret values. The repository now only ships placeholders.

## 3. Running With Poetry (development / staging)
```bash
poetry install --with dev
poetry run python -m telegram_legal_bot.main
```

> The launcher fails fast if required secrets are still placeholders.

## 4. Running With Docker
```bash
docker build -t aivan-bot .
docker run --env-file path/to/prod.env aivan-bot
```

- Health probes use `python -m telegram_legal_bot.healthcheck`.
- The container exposes port `8000` for optional HTTP services (metrics, REST, etc.).

### Docker Compose
Use `docker-compose.yml` when running the bot together with Redis / Prometheus / Grafana.
Remember to set:
```yaml
    environment:
      REDIS_URL: redis://redis:6379/0
      PROMETHEUS_PORT: 9000    # match healthcheck expectations
```

## 5. Observability & Health
- Prometheus exporter is enabled via `ENABLE_PROMETHEUS=1` and `PROMETHEUS_PORT`.
- Docker health check validates environment, storage permissions, and optional services.
- Metrics include system status, request counters, and health-check summaries.

## 6. Release Checklist
1. `poetry check` – validates the packaging metadata.
2. `poetry run pytest` – run automated tests (add a CI workflow to enforce this).
3. Build Docker image and run `python -m telegram_legal_bot.healthcheck` manually.
4. Ensure secrets are sourced from your secrets manager (Vault, AWS SM, etc.).
5. Configure logging destination (JSON logs work well with ELK / Loki).
6. Tag the image and push to your registry.

## 7. Next Steps / Open Items
- Add CI/CD pipeline (GitHub Actions / GitLab CI) to automate the checklist.
- Rotate production tokens since the previous `.env` leaked real values.
- Consider reducing image size (multi-stage build already prepared) by using wheels caches.
- Audit large modules such as `core/main_simple.py` and split into maintainable components.
- Finalise localisation: current README still contains mojibake from incorrect encoding.
