from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("telegram-legal-bot")
except metadata.PackageNotFoundError:  # pragma: no cover - package metadata missing during local dev
    __version__ = "0.1.0"

__all__ = ["__version__"]
