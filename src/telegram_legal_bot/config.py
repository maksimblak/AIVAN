from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv

from src.core.settings import AppSettings

_cached_config: Optional[AppSettings] = None


def load_config() -> AppSettings:
    """Load and cache application configuration."""
    global _cached_config
    if _cached_config is None:
        load_dotenv()
        _cached_config = AppSettings.load()
    return _cached_config
