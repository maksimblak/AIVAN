# Security Monitoring & Alerting

The bot sanitises every inbound message, tracks suspicious activity, and exposes counters you can
alert on. This document summarises the pipeline and provides an operational checklist.

## Request flow
1. `InputValidator` (`src/core/validation.py`) cleans and validates user text before it reaches the
   OpenAI gateway. It checks for:
   - Oversized or empty messages
   - XSS patterns (`<script>`, event handlers, `javascript:` links, etc.)
   - SQL injection snippets (`UNION SELECT`, `OR 1=1`, `; DROP TABLE ...`)
   - Spam patterns (repeated punctuation, long repeats)
   - Forbidden secrets (password/token/credit card keywords)
2. Violations raise `ValidationError` or attach warnings. The validator also strips zero-width
   characters and normalises whitespace to make prompts easier to diff.
3. `src/core/metrics.MetricsCollector` increments counters such as `security_violations_total`,
   `sql_injection_attempts_total`, and `xss_attempts_total`. These metrics are emitted to Prometheus
   when enabled or stored in the in-memory fallback otherwise.
4. `BackgroundTaskManager` periodically polls the metrics and health of the validator, DB, cache, and
   rate limiter. Failures are surfaced through logs and `system_status` gauges.
5. Optional: `scripts/validate_project.py` checks environment variables and DB schemas to ensure
   attack surfaces (e.g., missing indexes) are caught before deployment.

## Metrics to watch
| Metric | Meaning |
|--------|---------|
| `security_violations_total{severity=…}` | Number of rejected prompts grouped by severity. |
| `sql_injection_attempts_total{pattern_type}` | Input patterns that matched the SQL blacklist. |
| `xss_attempts_total{pattern_type}` | XSS payloads detected by the validator. |
| `telegram_messages_total{status=blocked}` | Requests dropped before hitting OpenAI. |
| `blocked_users_total` | Users who blocked the bot after repeated alerts (from retention module). |

Set alerts on sudden spikes (e.g., more than 10 SQL attempts in 5 minutes) or when
`security_violations_total` increases while `telegram_messages_total{status=processed}` drops.

## Configuration
- `ADMIN_IDS` determine who receives admin menus with `/errors`, `/ratings`, etc.
- `.env` has no explicit toggles for validation, but the severity thresholds live in
  `InputValidator.MIN/MAX_QUESTION_LENGTH` and can be adjusted in code if needed.
- Enable Prometheus with `ENABLE_PROMETHEUS=1` and expose `PROMETHEUS_PORT` to your monitoring stack.

## Operational runbook
1. **Triaging alerts** - check application logs for `ValidationError` entries; they include the
   user_id (if available) and pattern name. Follow up with the user if it looks like a false
   positive, otherwise keep the warning for auditing.
2. **Blocking abusive users** - use the admin commands (see `src/core/admin_modules`) to suspend
   access or adjust quotas. Rate limits are enforced by `core/bot_app/ratelimit.py` and
   `AccessService`.
3. **Instrumented checks** - periodically run `poetry run pytest tests/test_security_monitoring.py`
   to ensure validators catch the latest attack payloads.
4. **CI gate** - `scripts/run_tests.py` already runs pytest + linting. Add `poetry run python
   scripts/validate_project.py` to your pipeline to ensure `.env` values (e.g., `TELEGRAM_BOT_TOKEN`)
   are present and to detect accidental exposure of secret names in commits.
5. **Background verification** - the health checker (`src/core/health.py`) reports the status of the
   DB, OpenAI, session store, and rate limiter. If any component is unhealthy, `system_status` flips
   to `degraded`, alerting your monitoring stack.

## Log locations
- **Stdout/Compose logs** - aiogram dispatcher messages, validation warnings, payment errors.
- **`logs/` mount** - optional file sink if you configure logging handlers to write to disk.
- **Prometheus** - export `/metrics` and build Grafana dashboards with the counters above.

## Hardening tips
- Keep `InputValidator.SUSPICIOUS_PATTERNS` up to date; add more regexes when you encounter new
  payloads.
- Use Telegram’s anti-spam settings (privacy mode, domain restrictions) in BotFather for another
  layer of defence.
- Rotate OpenAI and payment keys regularly; `AppSettings` converts blank strings to `None`, so
  missing keys are caught during startup.
- If you run behind a proxy, set `TELEGRAM_PROXY_URL/USER/PASS` and monitor for repeated 429s, which
  could indicate network-level abuse.

Security monitoring is effective only when coupled with fast feedback. Keep alerts actionable, link
to this runbook, and capture follow-up actions in your incident tracker.
