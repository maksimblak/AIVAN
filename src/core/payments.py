from __future__ import annotations

import os
from typing import Any, Optional

from .crypto_pay import create_crypto_invoice_async


class CryptoPayProvider:
    """Provider wrapper for CryptoBot payments."""

    def __init__(self, *, asset: Optional[str] = None):
        self.asset = asset or os.getenv("CRYPTO_ASSET", "USDT")

    async def create_invoice(self, *, amount_rub: float, description: str, payload: str) -> dict[str, Any]:
        return await create_crypto_invoice_async(
            amount=float(amount_rub),
            asset=self.asset,
            description=description,
            payload=payload,
        )


def convert_rub_to_xtr(amount_rub: float, *, rub_per_xtr: Optional[float], default_xtr: Optional[int] = None) -> int:
    """Convert RUB to XTR (Telegram Stars) using configurable rate.

    - If rub_per_xtr provided: XTR = ceil(amount_rub / rub_per_xtr)
    - Else if default_xtr provided: return it
    - Else: fallback 1 XTR == 1 RUB (identity)
    """
    import math

    if rub_per_xtr and rub_per_xtr > 0:
        return int(math.ceil(amount_rub / rub_per_xtr))
    if default_xtr is not None:
        return int(default_xtr)
    return int(math.ceil(amount_rub))


