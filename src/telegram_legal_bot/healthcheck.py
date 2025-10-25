from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from src.core.app_context import get_settings
from src.core.settings import AppSettings

PLACEHOLDER_PREFIXES = ("__REQUIRED", "changeme", "<REPLACE", "REPLACE_ME")


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True
    trimmed = value.strip()
    if not trimmed:
        return True
    for prefix in PLACEHOLDER_PREFIXES:
        if trimmed.upper().startswith(prefix.upper()):
            return True
    return False


def _check_required_settings(settings: AppSettings) -> dict[str, Any]:
    missing: list[str] = []
    if _is_missing(settings.telegram_bot_token):
        missing.append("TELEGRAM_BOT_TOKEN")
    if _is_missing(settings.openai_api_key):
        missing.append("OPENAI_API_KEY")

    return {
        "name": "critical_env",
        "status": "pass" if not missing else "fail",
        "missing": missing,
        "description": "Ensures critical secrets are wired for runtime.",
    }


def _check_db_path(settings: AppSettings) -> dict[str, Any]:
    db_path = Path(settings.db_path)
    resolved = db_path if db_path.is_absolute() else Path(os.getcwd()) / db_path
    parent = resolved.parent

    status = "pass"
    issues: list[str] = []

    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            status = "fail"
            issues.append(f"Cannot create database directory '{parent}': {exc}")
    if parent.exists() and not os.access(parent, os.W_OK):
        status = "fail"
        issues.append(f"Database directory '{parent}' is not writable")

    details: dict[str, Any] = {
        "name": "database_path",
        "status": status,
        "path": str(resolved),
    }
    if issues:
        details["issues"] = issues
    return details


def _check_optional_services(settings: AppSettings) -> dict[str, Any]:
    status = "pass"
    notes: list[str] = []
    if settings.redis_url:
        notes.append("Redis URL configured")
    else:
        status = "warn"
        notes.append("Redis URL is not configured; rate limits will use in-memory backend.")

    if settings.enable_prometheus and not settings.prometheus_port:
        status = "warn"
        notes.append("Prometheus enabled but PROMETHEUS_PORT not provided.")

    return {
        "name": "optional_services",
        "status": status,
        "notes": notes,
    }


def run_checks() -> tuple[int, dict[str, Any]]:
    try:
        settings = get_settings(force_reload=True)
    except Exception as exc:  # noqa: BLE001
        payload: dict[str, Any] = {
            "status": "fail",
            "error": f"Failed to load settings: {exc}",
        }
        return 1, payload

    checks = [
        _check_required_settings(settings),
        _check_db_path(settings),
        _check_optional_services(settings),
    ]

    status = "pass"
    for entry in checks:
        entry_status = entry.get("status", "pass")
        if entry_status == "fail":
            status = "fail"
            break
        if entry_status == "warn" and status != "fail":
            status = "warn"

    result_payload: dict[str, Any] = {
        "status": status,
        "checks": checks,
    }
    exit_code = 0 if status == "pass" else 1
    return exit_code, result_payload


def main() -> None:
    exit_code, payload = run_checks()
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
