# mypy: ignore-errors
import json
import logging
from types import SimpleNamespace

from core.bot_app.logging_setup import setup_logging


def _reset_logger_state(
    logger: logging.Logger, original_level: int, original_handlers: list[logging.Handler]
) -> None:
    logger.handlers = original_handlers
    logger.setLevel(original_level)


def _restore_levels(levels: dict[str, int | None]) -> None:
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level if level is not None else logging.NOTSET)


def test_setup_logging_with_json_formatter() -> None:
    settings = SimpleNamespace(log_level="debug", log_json=True)
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    original_child_levels = {
        "aiosqlite": logging.getLogger("aiosqlite").level or None,
        "sqlite3": logging.getLogger("sqlite3").level or None,
    }

    try:
        setup_logging(settings)

        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        record = logging.LogRecord(
            "test", logging.INFO, __file__, 1, "hello", args=(), exc_info=None
        )
        payload = json.loads(handler.formatter.format(record))
        assert payload["msg"] == "hello"
        assert payload["lvl"] == "INFO"
        assert logging.getLogger("aiosqlite").level == logging.INFO
    finally:
        _reset_logger_state(root, original_level, original_handlers)
        _restore_levels(original_child_levels)


def test_setup_logging_with_plain_formatter() -> None:
    settings = SimpleNamespace(log_level="warning", log_json=False)
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    original_child_levels = {
        "aiosqlite": logging.getLogger("aiosqlite").level or None,
        "sqlite3": logging.getLogger("sqlite3").level or None,
    }

    try:
        setup_logging(settings)

        assert root.level == logging.WARNING
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        formatter = handler.formatter
        assert isinstance(formatter, logging.Formatter)
        assert formatter._style._fmt == "%(levelname)s: %(name)s: %(message)s"
        assert logging.getLogger("sqlite3").level == logging.WARNING
    finally:
        _reset_logger_state(root, original_level, original_handlers)
        _restore_levels(original_child_levels)
