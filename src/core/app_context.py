from __future__ import annotations

from typing import Mapping, Optional

from dotenv import load_dotenv

from src.core.settings import AppSettings


_settings_cache: Optional[AppSettings] = None


def set_settings(settings: AppSettings) -> None:
    """Сохраняет экземпляр настроек для повторного использования."""
    global _settings_cache
    _settings_cache = settings


def get_settings(env: Mapping[str, str] | None = None, *, force_reload: bool = False) -> AppSettings:
    """Возвращает экземпляр настроек, предпочитая кеш либо загружая из окружения."""
    global _settings_cache

    if env is None and not force_reload and _settings_cache is not None:
        return _settings_cache

    if env is None:
        load_dotenv()

    settings = AppSettings.load(env)

    if env is None:
        _settings_cache = settings

    return settings
