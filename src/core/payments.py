from __future__ import annotations

from src.core.app_context import get_settings
from src.core.settings import AppSettings

from typing import Any

from .crypto_pay import create_crypto_invoice_async


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
