import math

import pytest

from src.core.payments import CryptoPayProvider, convert_rub_to_xtr


def test_convert_rub_to_xtr_with_rate() -> None:
    assert convert_rub_to_xtr(299, rub_per_xtr=3.5, default_xtr=None) == math.ceil(299 / 3.5)


def test_convert_rub_to_xtr_with_default() -> None:
    assert convert_rub_to_xtr(120, rub_per_xtr=None, default_xtr=3500) == 3500


def test_convert_rub_to_xtr_identity_fallback() -> None:
    assert convert_rub_to_xtr(123.4, rub_per_xtr=None, default_xtr=None) == math.ceil(123.4)


@pytest.mark.asyncio
async def test_crypto_pay_provider_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_create_crypto_invoice_async(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True, "invoice_id": "test"}

    monkeypatch.setattr(
        "src.core.payments.create_crypto_invoice_async",
        fake_create_crypto_invoice_async,
    )

    provider = CryptoPayProvider(asset="TON")
    response = await provider.create_invoice(
        amount_rub=510.75,
        description="subscription",
        payload="user-1",
    )

    assert response["ok"] is True
    assert captured == {
        "amount": 510.75,
        "asset": "TON",
        "description": "subscription",
        "payload": "user-1",
    }
