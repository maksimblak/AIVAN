from __future__ import annotations

from typing import Optional

from src.core.app_context import get_settings, set_settings
from src.core.settings import AppSettings

_cached_config: Optional[AppSettings] = None


def load_config() -> AppSettings:
    """Load and cache application configuration."""
    global _cached_config
    if _cached_config is None:
        _cached_config = get_settings()
        set_settings(_cached_config)
    return _cached_config
