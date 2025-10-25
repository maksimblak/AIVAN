# mypy: ignore-errors
import io
import sys
import types
from types import SimpleNamespace

if "src.core.launcher" not in sys.modules:
    launcher_stub = types.ModuleType("src.core.launcher")

    def _stub_main() -> None:
        return None

    launcher_stub.main = _stub_main
    sys.modules["src.core.launcher"] = launcher_stub

import pytest

from src.telegram_legal_bot import healthcheck, main as main_module


def test_main_is_missing_detects_gaps() -> None:
    assert main_module._is_missing(None) is True
    assert main_module._is_missing("   ") is True
    assert main_module._is_missing("__REQUIRED_TOKEN") is True
    assert main_module._is_missing("valid") is False


def test_main_ensure_required_env_success(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(telegram_bot_token="token", openai_api_key="key")
    monkeypatch.setattr(main_module, "get_settings", lambda force_reload=True: settings)
    main_module._ensure_required_env()


def test_main_ensure_required_env_failure(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    settings = SimpleNamespace(telegram_bot_token=None, openai_api_key=" ")
    monkeypatch.setattr(main_module, "get_settings", lambda force_reload=True: settings)

    with caplog.at_level("CRITICAL", logger="ai-ivan.simple"):
        with pytest.raises(SystemExit) as exc:
            main_module._ensure_required_env()

    assert "Missing required environment variables" in str(exc.value)
    assert "Missing required environment variables" in caplog.text


def test_main_invokes_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(main_module, "_ensure_required_env", lambda: calls.append("ensure"))
    monkeypatch.setattr(main_module, "_launcher_main", lambda: calls.append("launch"))

    main_module.main()

    assert calls == ["ensure", "launch"]


def test_healthcheck_is_missing() -> None:
    assert healthcheck._is_missing("") is True
    assert healthcheck._is_missing("replace_me value") is True
    assert healthcheck._is_missing("configured") is False


def test_healthcheck_check_required_settings_reports_missing() -> None:
    settings = SimpleNamespace(telegram_bot_token=None, openai_api_key="")
    result = healthcheck._check_required_settings(settings)
    assert result["status"] == "fail"
    assert {"TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY"} == set(result["missing"])


def test_healthcheck_check_required_settings_passes_when_populated() -> None:
    settings = SimpleNamespace(telegram_bot_token="token", openai_api_key="key")
    result = healthcheck._check_required_settings(settings)
    assert result["status"] == "pass"
    assert not result["missing"]


def test_healthcheck_check_db_path_creates_directory(tmp_path) -> None:
    target = tmp_path / "nested" / "bot.sqlite"
    settings = SimpleNamespace(db_path=str(target))

    result = healthcheck._check_db_path(settings)

    assert result["status"] == "pass"
    assert target.parent.exists()
    assert result["path"] == str(target)


def test_healthcheck_check_db_path_reports_permission_issue(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "db.sqlite"
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(healthcheck.os, "access", lambda _path, _mode: False)
    settings = SimpleNamespace(db_path=str(target))

    result = healthcheck._check_db_path(settings)

    assert result["status"] == "fail"
    assert any("not writable" in issue for issue in result["issues"])


def test_healthcheck_check_db_path_handles_creation_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "subdir" / "db.sqlite"

    def _failing_mkdir(self, parents=False, exist_ok=False):
        raise PermissionError("denied")

    monkeypatch.setattr(healthcheck.Path, "mkdir", _failing_mkdir)
    settings = SimpleNamespace(db_path=str(target))

    result = healthcheck._check_db_path(settings)

    assert result["status"] == "fail"
    assert any("Cannot create database directory" in issue for issue in result.get("issues", []))


def test_healthcheck_optional_services_statuses() -> None:
    configured = SimpleNamespace(
        redis_url="redis://localhost", enable_prometheus=False, prometheus_port=None
    )
    result_ok = healthcheck._check_optional_services(configured)
    assert result_ok["status"] == "pass"

    minimal = SimpleNamespace(redis_url="", enable_prometheus=True, prometheus_port=None)
    result_warn = healthcheck._check_optional_services(minimal)
    assert result_warn["status"] == "warn"
    assert "Prometheus enabled" in " ".join(result_warn["notes"])


def test_healthcheck_run_checks_success(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        telegram_bot_token="token",
        openai_api_key="key",
        db_path=str(tmp_path / "bot.sqlite"),
        redis_url="redis://localhost",
        enable_prometheus=False,
        prometheus_port=None,
    )
    monkeypatch.setattr(healthcheck, "get_settings", lambda force_reload=True: settings)

    exit_code, payload = healthcheck.run_checks()

    assert exit_code == 0
    assert payload["status"] == "pass"
    assert len(payload["checks"]) == 3


def test_healthcheck_run_checks_warns(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(
        telegram_bot_token="token",
        openai_api_key="key",
        db_path=str(tmp_path / "bot.sqlite"),
        redis_url="",
        enable_prometheus=True,
        prometheus_port=None,
    )
    monkeypatch.setattr(healthcheck, "get_settings", lambda force_reload=True: settings)

    exit_code, payload = healthcheck.run_checks()

    assert exit_code == 1
    assert payload["status"] == "warn"
    statuses = {entry["status"] for entry in payload["checks"]}
    assert "warn" in statuses


def test_healthcheck_run_checks_handles_settings_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        healthcheck,
        "get_settings",
        lambda force_reload=True: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    exit_code, payload = healthcheck.run_checks()

    assert exit_code == 1
    assert payload["status"] == "fail"
    assert "Failed to load settings" in payload["error"]


def test_healthcheck_main_outputs_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(healthcheck, "run_checks", lambda: (1, {"status": "fail"}))
    buffer = io.StringIO()
    monkeypatch.setattr(healthcheck.sys, "stdout", buffer)

    exit_codes: list[int] = []

    def _fake_exit(code: int) -> None:
        exit_codes.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(healthcheck.sys, "exit", _fake_exit)

    with pytest.raises(SystemExit):
        healthcheck.main()

    assert exit_codes == [1]
    buffer.seek(0)
    payload = buffer.read()
    assert '"status": "fail"' in payload
