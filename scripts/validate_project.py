#!/usr/bin/env python3
"""Bootstrap validation script for the AIVAN project.

The checks here mirror the expectations of our CI pipeline and should be used
before every release or infrastructure change.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import tempfile
import time
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.db_advanced import DatabaseAdvanced  # noqa: E402
from src.core.di_container import create_container  # noqa: E402
from src.core.settings import AppSettings  # noqa: E402


async def validate_dependencies() -> bool:
    print("\n>> Checking mandatory dependencies")
    try:
        import aiogram  # noqa: F401
        import openai  # noqa: F401
        import aiosqlite  # noqa: F401
        import httpx  # noqa: F401
        import pytest  # noqa: F401
    except ImportError as exc:
        print(f"FAILED: missing required dependency -> {exc}")
        return False

    optional = {
        "redis": "Redis client (rate limiting and caching)",
        "prometheus_client": "Prometheus exporter",
        "psutil": "System metrics collector",
    }
    for module, description in optional.items():
        try:
            __import__(module)
            print(f"OK: {description}")
        except ImportError:
            print(f"INFO: {description} is not installed (optional)")
    return True


async def validate_database() -> bool:
    print("\n>> Checking SQLite bootstrap")
    tmp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_file.close()
    try:
        db = DatabaseAdvanced(tmp_file.name, max_connections=2)
        await db.init()

        user = await db.ensure_user(user_id=123, default_trial=10, is_admin=False)
        assert user.user_id == 123

        trial_ok = await db.decrement_trial(123)
        assert trial_ok

        await db.close()
        print("OK: database bootstrap and basic queries succeeded")
        return True
    except Exception as exc:
        print(f"FAILED: database validation error -> {exc}")
        return False
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_file.name)


async def validate_di_container() -> bool:
    print("\n>> Checking dependency injection container")
    tmp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_file.close()

    env = {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "OPENAI_API_KEY": "test-key",
        "DB_PATH": tmp_file.name,
    }

    try:
        settings = AppSettings.load(env)
        container = create_container(settings)

        resolved_settings = container.get(AppSettings)
        assert resolved_settings is settings

        db = container.get(DatabaseAdvanced)
        await db.init()
        await db.close()

        print("OK: dependency injection container resolved core services")
        return True
    except ValidationError as exc:
        print(f"FAILED: invalid settings -> {exc}")
        return False
    except Exception as exc:
        print(f"FAILED: DI container error -> {exc}")
        return False
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_file.name)


async def validate_cache_and_metrics() -> bool:
    print("\n>> Checking cache and timing helpers")
    try:
        from src.core.cache import CacheEntry, InMemoryCacheBackend
        from src.core.metrics import MetricsCollector
    except ImportError as exc:
        print(f"FAILED: unable to import cache/metrics utilities -> {exc}")
        return False

    try:
        cache = InMemoryCacheBackend(max_size=4, cleanup_interval=0)
        entry = CacheEntry(
            key="demo",
            value="value",
            created_at=time.time(),
            ttl_seconds=60,
        )
        await cache.set("demo", entry)
        stored = await cache.get("demo")
        assert stored is not None and stored.value == "value"

        metrics = MetricsCollector(enable_prometheus=False)
        async with metrics.time_openai_request(model="demo"):
            await asyncio.sleep(0.01)

        print("OK: cache and metrics utilities executed successfully")
        return True
    except Exception as exc:
        print(f"FAILED: cache or timing check error -> {exc}")
        return False


async def validate_configuration() -> bool:
    print("\n>> Checking repository configuration files")

    required_files = [
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "Dockerfile",
        PROJECT_ROOT / "docker-compose.yml",
        PROJECT_ROOT / "README.md",
    ]

    missing = [path.name for path in required_files if not path.exists()]
    if missing:
        print(f"FAILED: missing files -> {', '.join(missing)}")
        return False

    print("OK: core configuration files are present")
    return True


async def validate_tests_present() -> bool:
    print("\n>> Checking tests directory")
    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.exists():
        print("FAILED: tests directory is missing")
        return False

    test_files = list(tests_dir.rglob("test_*.py"))
    if not test_files:
        print("FAILED: no pytest-style files were discovered")
        return False

    print(f"OK: discovered {len(test_files)} test files")
    return True


async def main() -> int:
    print("AIVAN Project Validation")
    print("=" * 72)

    validators = [
        ("Dependencies", validate_dependencies),
        ("Configuration files", validate_configuration),
        ("Database", validate_database),
        ("Dependency injection", validate_di_container),
        ("Cache and metrics", validate_cache_and_metrics),
        ("Tests", validate_tests_present),
    ]

    results: list[bool] = []
    for name, validator in validators:
        try:
            success = await validator()
            results.append(success)
        except Exception as exc:
            print(f"FAILED: unexpected error in {name} -> {exc}")
            results.append(False)

    passed = sum(1 for result in results if result)
    total = len(results)

    print("\n" + "=" * 72)
    print(f"Validation summary: {passed}/{total} checks passed")
    if passed == total:
        print("All validation checks succeeded.")
        return 0

    print("One or more validation checks failed. Review the logs above.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
