from __future__ import annotations

import logging

from src.core.app_context import get_settings
from src.core.launcher import main as _launcher_main


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


def _ensure_required_env() -> None:
    """Fail fast if critical environment variables are missing."""
    settings = get_settings(force_reload=True)
    missing = []
    if _is_missing(settings.telegram_bot_token):
        missing.append("TELEGRAM_BOT_TOKEN")
    if _is_missing(settings.openai_api_key):
        missing.append("OPENAI_API_KEY")

    if missing:
        joined = ", ".join(missing)
        logging.getLogger("ai-ivan.simple").critical(
            "Missing required environment variables: %s", joined
        )
        raise SystemExit(f"Missing required environment variables: {joined}")


def main() -> None:
    """CLI entrypoint used by the production runner and console script."""
    _ensure_required_env()
    _launcher_main()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
