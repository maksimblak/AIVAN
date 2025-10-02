import asyncio

import pytest

from src.core.health import HealthCheck, HealthCheckResult, HealthStatus


class FailingHealthCheck(HealthCheck):
    def __init__(self) -> None:
        super().__init__("failing")

    async def check(self) -> HealthCheckResult:
        raise RuntimeError("boom")


class SlowHealthCheck(HealthCheck):
    def __init__(self, timeout: float = 0.01) -> None:
        super().__init__("slow", timeout=timeout)

    async def check(self) -> HealthCheckResult:
        await asyncio.sleep(0.05)
        return HealthCheckResult(status=HealthStatus.HEALTHY)


@pytest.mark.asyncio
async def test_health_check_execute_handles_exception() -> None:
    check = FailingHealthCheck()
    result = await check.execute()
    assert result.status is HealthStatus.UNHEALTHY
    assert "boom" in result.message
    assert check.failed_checks == 1


@pytest.mark.asyncio
async def test_health_check_execute_timeout() -> None:
    check = SlowHealthCheck(timeout=0.01)
    result = await check.execute()
    assert result.status is HealthStatus.UNHEALTHY
    assert "timeout" in result.message.lower()
