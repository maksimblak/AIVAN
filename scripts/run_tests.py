#!/usr/bin/env python3
"""Unified developer test runner for the AIVAN project."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))


def _sanitize(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def run_command(command: str, description: str) -> bool:
    """Execute a shell command and show a compact status report."""
    print(f"\n>> {description}")
    print("-" * 72)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=PROJECT_ROOT,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"ERROR: command failed to execute: {exc}")
        return False

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode == 0:
        print("OK")
        if stdout.strip():
            print(_sanitize(stdout.strip()))
        return True

    print("FAILED")
    if stderr.strip():
        print(_sanitize(stderr.strip()))
    if stdout.strip():
        print(_sanitize(stdout.strip()))
    return False


def main() -> int:
    """Run the standard set of quality gates used by CI."""
    print("AIVAN Test Suite")
    print("=" * 72)

    commands: list[tuple[str, str]] = []
    success_count = 0
    total_count = 0

    if (PROJECT_ROOT / "pyproject.toml").exists():
        commands.append(("poetry check", "Validate pyproject metadata"))

    coverage_supported = True
    try:
        import pytest_cov  # type: ignore  # noqa: F401
    except ImportError:
        coverage_supported = False

    if (PROJECT_ROOT / "tests").exists():
        commands.append(("poetry run pytest tests -v", "Run pytest test suite"))
        if coverage_supported:
            commands.append(
                (
                    "poetry run pytest --cov=src --cov-report=term-missing",
                    "Collect coverage report",
                )
            )
        else:
            print("INFO: pytest-cov is not installed; skipping coverage step.")

    if (PROJECT_ROOT / "src").exists():
        commands.append(("poetry run ruff check src tests", "Run Ruff linting"))
        commands.append(("poetry run black --check src tests", "Run Black formatting check"))
        commands.append(("poetry run mypy src tests", "Run mypy type checks"))

    if (PROJECT_ROOT / "scripts" / "validate_project.py").exists():
        commands.append(
            (
                "poetry run python scripts/validate_project.py",
                "Run project validation script",
            )
        )

    for command, description in commands:
        total_count += 1
        if run_command(command, description):
            success_count += 1

    print("\n" + "=" * 72)
    print(f"Completed checks: {success_count}/{total_count}")

    if success_count == total_count:
        print("All checks passed.")
        return 0

    print("Some checks failed. See logs above.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
