from __future__ import annotations

import hashlib
import secrets
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any
from urllib.parse import urlencode

import httpx

from src.core.app_context import get_settings
from src.core.db_advanced import TransactionStatus
from src.core.settings import AppSettings

from .crypto_pay import create_crypto_invoice_async


@dataclass(slots=True)
class PaymentCreationResult:
    ok: bool
    payment_id: str | None = None
    url: str | None = None
    raw: Any | None = None
    error: str | None = None


@dataclass(slots=True)
class PaymentStatusResult:
    status: TransactionStatus
    raw: Any | None = None
    description: str | None = None
    paid_amount: int | None = None


class CryptoPayProvider:
    """Provider wrapper for CryptoBot payments."""

    def __init__(self, *, asset: str | None = None, settings: AppSettings | None = None):
        if settings is None:
            settings = get_settings()
        self._settings = settings
        self.asset = asset or settings.crypto_asset

    async def create_invoice(
        self, *, amount_rub: float, description: str, payload: str
    ) -> dict[str, Any]:
        # Validate inputs
        if amount_rub <= 0:
            raise ValueError(f"Invoice amount must be positive: {amount_rub}")
        if not description or not description.strip():
            raise ValueError("Invoice description cannot be empty")
        if not payload or not payload.strip():
            raise ValueError("Invoice payload cannot be empty")

        return await create_crypto_invoice_async(
            amount=float(amount_rub),
            asset=self.asset,
            description=description.strip(),
            payload=payload.strip(),
            settings=self._settings,
        )


class RoboKassaProvider:
    """Оплата через RoboKassa (оплата картами и быстрыми платежами)."""

    _MERCHANT_URL = "https://auth.robokassa.ru/Merchant/Index.aspx"
    _OPSTATE_URL = "https://auth.robokassa.ru/Merchant/WebService/Service.asmx/OpState"

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self.merchant_login = self._settings.robokassa_merchant_login.strip()
        self.password1 = self._settings.robokassa_password1.strip()
        self.password2 = self._settings.robokassa_password2.strip()
        self.is_test_mode = bool(self._settings.robokassa_is_test)
        self.result_url = self._settings.robokassa_result_url
        self.success_url = self._settings.robokassa_success_url
        self.fail_url = self._settings.robokassa_fail_url

    @property
    def is_available(self) -> bool:
        return bool(self.merchant_login and self.password1 and self.password2)

    def _format_amount(self, amount_rub: float) -> str:
        decimal_amount = Decimal(str(amount_rub)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return format(decimal_amount, "f")

    def _generate_invoice_id(self) -> int:
        return secrets.randbelow(900_000_000) + 100_000_000

    def _signature(self, *, amount: str, invoice_id: int, password: str) -> str:
        source = f"{self.merchant_login}:{amount}:{invoice_id}:{password}"
        return hashlib.md5(source.encode("utf-8")).hexdigest()

    async def create_invoice(
        self,
        *,
        amount_rub: float,
        description: str,
        payload: str,
        invoice_id: int | None = None,
    ) -> PaymentCreationResult:
        if not self.is_available:
            return PaymentCreationResult(ok=False, error="RoboKassa provider disabled")
        if amount_rub <= 0:
            return PaymentCreationResult(ok=False, error="Invoice amount must be positive")
        if not description.strip():
            return PaymentCreationResult(ok=False, error="Description cannot be empty")
        if not payload.strip():
            return PaymentCreationResult(ok=False, error="Payload cannot be empty")

        inv_id = invoice_id or self._generate_invoice_id()
        out_sum = self._format_amount(amount_rub)
        signature = self._signature(amount=out_sum, invoice_id=inv_id, password=self.password1)

        query_params: dict[str, str | int] = {
            "MerchantLogin": self.merchant_login,
            "OutSum": out_sum,
            "InvId": inv_id,
            "Description": description,
            "SignatureValue": signature,
            "Encoding": "utf-8",
            "Culture": "ru",
        }
        if self.is_test_mode:
            query_params["IsTest"] = 1
        if self.result_url:
            query_params["ResultURL"] = self.result_url
        if self.success_url:
            query_params["SuccessURL"] = self.success_url
        if self.fail_url:
            query_params["FailURL"] = self.fail_url

        payment_url = f"{self._MERCHANT_URL}?{urlencode(query_params, doseq=False, encoding="utf-8")}"  # type: ignore[arg-type]
        raw = {
            "inv_id": inv_id,
            "payload": payload,
            "amount": out_sum,
            "signature": signature,
        }
        return PaymentCreationResult(ok=True, payment_id=str(inv_id), url=payment_url, raw=raw)

    async def poll_payment(self, invoice_id: str) -> PaymentStatusResult:
        if not self.is_available:
            return PaymentStatusResult(
                status=TransactionStatus.FAILED,
                description="RoboKassa provider disabled",
            )
        if not invoice_id:
            return PaymentStatusResult(
                status=TransactionStatus.FAILED,
                description="Invoice ID is required",
            )

        signature_src = f"{self.merchant_login}:{invoice_id}:{self.password2}"
        signature = hashlib.md5(signature_src.encode("utf-8")).hexdigest()
        params = {
            "MerchantLogin": self.merchant_login,
            "InvoiceID": invoice_id,
            "Signature": signature,
        }
        timeout = httpx.Timeout(15.0, connect=10.0, read=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(self._OPSTATE_URL, params=params)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                return PaymentStatusResult(
                    status=TransactionStatus.PENDING,
                    description=f"HTTP error: {exc}",
                )

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:  # pragma: no cover - unexpected provider response
            return PaymentStatusResult(
                status=TransactionStatus.PENDING,
                description=f"Failed to parse response: {exc}",
                raw=response.text,
            )

        state_code = root.findtext(".//StateCode")
        if state_code == "100":
            amount_text = root.findtext(".//OutSum")
            amount_minor = None
            if amount_text:
                try:
                    decimal_amount = Decimal(amount_text).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    amount_minor = int(decimal_amount * 100)
                except Exception:  # noqa: BLE001
                    amount_minor = None
            return PaymentStatusResult(
                status=TransactionStatus.COMPLETED,
                raw=response.text,
                paid_amount=amount_minor,
            )
        if state_code in {"50", "90"}:
            return PaymentStatusResult(
                status=TransactionStatus.CANCELLED,
                raw=response.text,
                description=f"RoboKassa state {state_code}",
            )
        return PaymentStatusResult(
            status=TransactionStatus.PENDING,
            raw=response.text,
            description=f"RoboKassa state {state_code}",
        )


class YooKassaProvider:
    """Оплата через YooKassa (карты, СБП)."""

    _API_URL = "https://api.yookassa.ru/v3/payments"

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self.shop_id = self._settings.yookassa_shop_id.strip()
        self.secret_key = self._settings.yookassa_secret_key.strip()
        self.return_url = self._settings.yookassa_return_url

    @property
    def is_available(self) -> bool:
        return bool(self.shop_id and self.secret_key and self.return_url)

    def _format_amount(self, amount_rub: float) -> str:
        decimal_amount = Decimal(str(amount_rub)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return format(decimal_amount, "f")

    async def create_invoice(
        self,
        *,
        amount_rub: float,
        description: str,
        payload: str,
        metadata: dict[str, Any] | None = None,
    ) -> PaymentCreationResult:
        if not self.is_available:
            return PaymentCreationResult(ok=False, error="YooKassa provider disabled")
        if amount_rub <= 0:
            return PaymentCreationResult(ok=False, error="Invoice amount must be positive")
        if not description.strip():
            return PaymentCreationResult(ok=False, error="Description cannot be empty")
        if not payload.strip():
            return PaymentCreationResult(ok=False, error="Payload cannot be empty")

        amount_value = self._format_amount(amount_rub)
        request_body: dict[str, Any] = {
            "amount": {"value": amount_value, "currency": "RUB"},
            "capture": True,
            "confirmation": {
                "type": "redirect",
                "return_url": self.return_url,
            },
            "description": description[:128],
            "metadata": {"payload": payload} | (metadata or {}),
        }

        headers = {"Idempotence-Key": str(uuid.uuid4())}
        timeout = httpx.Timeout(30.0, connect=10.0, read=20.0)
        async with httpx.AsyncClient(
            auth=(self.shop_id, self.secret_key), timeout=timeout
        ) as client:
            try:
                response = await client.post(self._API_URL, json=request_body, headers=headers)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                return PaymentCreationResult(ok=False, error=f"HTTP error: {exc}")

        data = response.json()
        payment_id = data.get("id")
        confirmation = data.get("confirmation") or {}
        confirmation_url = confirmation.get("confirmation_url")
        if not payment_id or not confirmation_url:
            return PaymentCreationResult(
                ok=False,
                error="Invalid YooKassa response",
                raw=data,
            )
        return PaymentCreationResult(
            ok=True,
            payment_id=str(payment_id),
            url=str(confirmation_url),
            raw=data,
        )

    async def poll_payment(self, payment_id: str) -> PaymentStatusResult:
        if not self.is_available:
            return PaymentStatusResult(
                status=TransactionStatus.FAILED,
                description="YooKassa provider disabled",
            )
        if not payment_id:
            return PaymentStatusResult(
                status=TransactionStatus.FAILED,
                description="Payment ID is required",
            )

        url = f"{self._API_URL}/{payment_id}"
        timeout = httpx.Timeout(30.0, connect=10.0, read=20.0)
        async with httpx.AsyncClient(
            auth=(self.shop_id, self.secret_key), timeout=timeout
        ) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                return PaymentStatusResult(
                    status=TransactionStatus.PENDING,
                    description=f"HTTP error: {exc}",
                )

        data = response.json()
        status = str(data.get("status", "")).lower()
        paid_amount_minor = None
        if "amount" in data:
            try:
                value = data["amount"].get("value")
                if value is not None:
                    decimal_amount = Decimal(str(value)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    paid_amount_minor = int(decimal_amount * 100)
            except Exception:  # noqa: BLE001
                paid_amount_minor = None

        if status == "succeeded":
            return PaymentStatusResult(
                status=TransactionStatus.COMPLETED,
                raw=data,
                paid_amount=paid_amount_minor,
            )
        if status in {"canceled", "cancelling"}:
            return PaymentStatusResult(
                status=TransactionStatus.CANCELLED,
                raw=data,
                description=data.get("cancellation_details", {}).get("reason"),
            )
        return PaymentStatusResult(
            status=TransactionStatus.PENDING,
            raw=data,
            description=status,
        )


def convert_rub_to_xtr(
    amount_rub: float, *, rub_per_xtr: float | None, default_xtr: int | None = None
) -> int:
    """Convert RUB to XTR (Telegram Stars) using configurable rate.

    - If rub_per_xtr provided: XTR = ceil(amount_rub / rub_per_xtr)
    - Else if default_xtr provided: return it
    - Else: fallback 1 XTR == 1 RUB (identity)

    Args:
        amount_rub: Amount in RUB (must be >= 0)
        rub_per_xtr: Exchange rate (RUB per 1 XTR)
        default_xtr: Default XTR amount if rate not provided

    Returns:
        Amount in XTR (integer)

    Raises:
        ValueError: If amount_rub is negative
    """
    import math

    # Validate input
    if amount_rub < 0:
        raise ValueError(f"Amount cannot be negative: {amount_rub}")

    if amount_rub == 0:
        return 0

    if rub_per_xtr and rub_per_xtr > 0:
        return int(math.ceil(amount_rub / rub_per_xtr))
    if default_xtr is not None:
        return int(max(0, default_xtr))
    return int(math.ceil(amount_rub))
