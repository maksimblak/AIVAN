from __future__ import annotations

import asyncio
import logging
import sys

from src.bot.logging_setup import setup_logging
from src.core.app_context import get_settings, set_settings
from src.core.bootstrap import build_runtime
from src.core.simple_bot.app import refresh_runtime_globals, run_bot, set_runtime


async def _run_async() -> None:
    settings = get_settings()
    setup_logging(settings)
    logger = logging.getLogger("ai-ivan.simple")

    set_settings(settings)
    runtime, _ = build_runtime(settings, logger=logger)
    set_runtime(runtime)
    refresh_runtime_globals()

    await run_bot()


def main() -> None:
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            logging.getLogger("ai-ivan.simple").info("Using Windows Proactor event loop")
        else:
            try:
                import uvloop  # type: ignore

                uvloop.install()
                logging.getLogger("ai-ivan.simple").info("uvloop installed for improved performance")
            except ImportError:
                logging.getLogger("ai-ivan.simple").info("uvloop not available, using default event loop")

        asyncio.run(_run_async())
    except KeyboardInterrupt:
        logging.getLogger("ai-ivan.simple").info("AI-Ivan stopped by user")
    except Exception as exc:  # pragma: no cover - top level guard
        logging.getLogger("ai-ivan.simple").exception("Fatal launcher error: %s", exc)
        raise


if __name__ == "__main__":
    main()

