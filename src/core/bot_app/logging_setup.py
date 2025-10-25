from __future__ import annotations

import json
import logging
import sys

from src.core.settings import AppSettings

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def setup_logging(settings: AppSettings) -> None:
    level = _LEVELS.get(settings.log_level.upper(), logging.INFO)
    log_json = bool(settings.log_json)

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)

    # Reduce noise from verbose third-party libraries that flood DEBUG logs.
    quiet_loggers: dict[str, int] = {
        "aiosqlite": logging.INFO,
        "sqlite3": logging.INFO,
    }
    for logger_name, min_level in quiet_loggers.items():
        logging.getLogger(logger_name).setLevel(max(level, min_level))

    if log_json:

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
                payload = {
                    "lvl": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc_info"] = self.formatException(record.exc_info)
                return json.dumps(payload, ensure_ascii=False)

        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s: %(name)s: %(message)s"))

    root.handlers[:] = [handler]
