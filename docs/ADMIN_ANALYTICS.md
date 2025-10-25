# Admin Analytics Toolkit

Product and support teams can inspect retention, cohorts, and PMF metrics directly from Telegram via
the admin modules under `src/core/admin_modules`. This document describes the moving parts so you can
extend them or hand off instructions to non-engineers.

## Modules at a glance
- `retention_analytics.py` — builds rich `RetainedUserProfile` objects with spend, lifetime, and
  feature usage data sourced from `DatabaseAdvanced`.
- `cohort_analytics.py` — calculates retention curves per cohort (registration date, plan, marketing
  channel) and correlates engagement with feature flags.
- `pmf_metrics.py` — aggregates product-market-fit KPIs (NPS-style buckets, activation funnels,
  retention lift) that power the admin dashboards.
- `admin_retention_commands.py` — exposes Telegram commands (prefixed with `/retention`) that call
  the analytics modules and render inline keyboards for deep dives.
- `admin_modules/admin_commands.py` — central registry hooked up from `startup.py` so only
  `ADMIN_IDS` can trigger the commands.

## Available commands
The exact labels can be customized, but by default you get:
- `/retention` — summary card with trial conversion, day-3/day-7 retention, and a quick breakdown by
  acquisition source. Defined in `cmd_retention`.
- `/retention_deep_dive` (button behind `/retention`) — runs `cmd_deep_dive_user`, asks for a user id
  and prints their `RetainedUserProfile` with payment history, feature mix, and streaks.
- Cohort explorer buttons — wired via `cohort_analytics.py` to list cohorts with the worst drop-off,
  top-performing features, and correlation hints (for example, “users who enabled voice mode retain
  12% better”).
- PMF snapshots — embedded in the admin menus so founders can see the latest survey deltas without
  leaving Telegram.

Review `src/core/admin_modules/admin_retention_commands.py` to tweak copywriting or add new inline
actions. All commands receive `(message, db, admin_ids)` so helpers can access the current database
pool and enforce access control.

## Data requirements
- The analytics modules depend on `DatabaseAdvanced` migrations (users, payments, usage telemetry).
  Make sure migrations have been applied before running the commands against a new environment.
- Some metrics use denormalized tables (for example, `feature_usage_stats`). The background tasks and
  user-behaviour tracker populate them automatically; avoid truncating those tables unless you know
  how to rebuild them.
- Excel exports (see `docs/EXCEL_EXPORT.md`) complement the admin flows; handlers inside
  `src/core/bot_app/documents.py` can be wired to send them to admins as well.

## Extending the toolkit
1. Add a method to `RetentionAnalytics` or `CohortAnalytics` that returns the data you need (keep
   heavy queries async).
2. Create a new command function next to `cmd_retention`, register it in `admin_commands.py`, and
   wire an inline button that calls it.
3. Reuse the formatting helpers from `src/core/bot_app/menus.py` or `ui_components.py` so the admin UI
   stays consistent with the rest of the bot.
4. When exporting attachments, ensure that only `ADMIN_IDS` receive them to avoid leaking sensitive
   churn data.

By keeping the analytics layer in Python you can iterate fast: every new question from marketing can
be answered by a focused query and exposed through Telegram without deploying separate dashboards.
