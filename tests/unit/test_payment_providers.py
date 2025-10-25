import hashlib
from dataclasses import dataclass

import pytest

import src.core.payments as payments_module
from src.core.db_advanced import TransactionStatus
from src.core.payments import RoboKassaProvider, YooKassaProvider
from src.core.settings import AppSettings


@dataclass
class DummyResponse:
    text: str | None = None
    json_payload: dict | None = None

    def raise_for_status(self) -> None:  # pragma: no cover - no error scenario in tests
        return None

    def json(self) -> dict:
        if self.json_payload is None:
            raise AssertionError("JSON payload not provided")
        return self.json_payload


class DummyAsyncClient:
    def __init__(
        self,
        *,
        post_response: DummyResponse | None = None,
        get_response: DummyResponse | None = None,
    ):
        self._post_response = post_response
        self._get_response = get_response

    async def __aenter__(self) -> "DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, *args, **kwargs) -> DummyResponse:
        if self._post_response is None:
            raise AssertionError("Unexpected POST request in test")
        return self._post_response

    async def get(self, *args, **kwargs) -> DummyResponse:
        if self._get_response is None:
            raise AssertionError("Unexpected GET request in test")
        return self._get_response


@pytest.fixture()
def base_env() -> dict[str, str]:
    return {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "OPENAI_API_KEY": "sk-test",
    }


@pytest.mark.asyncio()
async def test_robokassa_create_invoice_signature(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    env = {
        **base_env,
        "ROBOKASSA_MERCHANT_LOGIN": "demo",
        "ROBOKASSA_PASSWORD1": "pass1",
        "ROBOKASSA_PASSWORD2": "pass2",
    }
    settings = AppSettings.load(env)
    provider = RoboKassaProvider(settings=settings)

    invoice_id = 12345
    amount_rub = 1499
    result = await provider.create_invoice(
        amount_rub=amount_rub,
        description="Подписка",
        payload="sub:base_1m:robokassa:42:1700000000",
        invoice_id=invoice_id,
    )

    assert result.ok
    assert result.payment_id == str(invoice_id)
    assert result.url is not None

    expected_amount = f"{amount_rub:.2f}"
    expected_signature = hashlib.md5(
        f"demo:{expected_amount}:{invoice_id}:pass1".encode("utf-8")
    ).hexdigest()
    assert expected_signature in result.url
    assert f"InvId={invoice_id}" in result.url


@pytest.mark.asyncio()
async def test_robokassa_poll_payment_success(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    env = {
        **base_env,
        "ROBOKASSA_MERCHANT_LOGIN": "demo",
        "ROBOKASSA_PASSWORD1": "pass1",
        "ROBOKASSA_PASSWORD2": "pass2",
    }
    settings = AppSettings.load(env)
    provider = RoboKassaProvider(settings=settings)

    xml_response = """<?xml version=\"1.0\" encoding=\"utf-8\"?>
    <OperationStateResponse>
        <StateCode>100</StateCode>
        <OutSum>1499.00</OutSum>
    </OperationStateResponse>
    """

    dummy_client = DummyAsyncClient(get_response=DummyResponse(text=xml_response))
    monkeypatch.setattr(payments_module.httpx, "AsyncClient", lambda **kwargs: dummy_client)

    result = await provider.poll_payment("12345")
    assert result.status == TransactionStatus.COMPLETED
    assert result.paid_amount == 149900


@pytest.mark.asyncio()
async def test_yookassa_create_invoice(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    env = {
        **base_env,
        "YOOKASSA_SHOP_ID": "123456",
        "YOOKASSA_SECRET_KEY": "secret",
        "YOOKASSA_RETURN_URL": "https://example.com/return",
    }
    settings = AppSettings.load(env)
    provider = YooKassaProvider(settings=settings)

    payload = DummyResponse(
        json_payload={
            "id": "pay_123",
            "status": "pending",
            "confirmation": {"confirmation_url": "https://pay.example/redirect"},
        }
    )
    dummy_client = DummyAsyncClient(post_response=payload)
    monkeypatch.setattr(payments_module.httpx, "AsyncClient", lambda **kwargs: dummy_client)

    result = await provider.create_invoice(
        amount_rub=999,
        description="Подписка",
        payload="sub:base_1m:yookassa:42:1700000000",
        metadata={"test": "value"},
    )

    assert result.ok
    assert result.payment_id == "pay_123"
    assert result.url == "https://pay.example/redirect"


@pytest.mark.asyncio()
async def test_yookassa_poll_payment_success(
    monkeypatch: pytest.MonkeyPatch, base_env: dict[str, str]
) -> None:
    env = {
        **base_env,
        "YOOKASSA_SHOP_ID": "123456",
        "YOOKASSA_SECRET_KEY": "secret",
        "YOOKASSA_RETURN_URL": "https://example.com/return",
    }
    settings = AppSettings.load(env)
    provider = YooKassaProvider(settings=settings)

    payload = DummyResponse(
        json_payload={
            "id": "pay_123",
            "status": "succeeded",
            "amount": {"value": "999.00", "currency": "RUB"},
        }
    )
    dummy_client = DummyAsyncClient(get_response=payload)
    monkeypatch.setattr(payments_module.httpx, "AsyncClient", lambda **kwargs: dummy_client)

    result = await provider.poll_payment("pay_123")
    assert result.status == TransactionStatus.COMPLETED
    assert result.paid_amount == 99900
