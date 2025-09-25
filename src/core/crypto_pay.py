from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx


async def create_crypto_invoice_async(
    *,
    amount: float,
    asset: str,
    description: str,
    payload: str,
    expires_in: int = 3600,
    retries: int = 3,
) -> dict[str, Any]:
    """Create a crypto invoice via Crypto Pay API (CryptoBot) asynchronously.

    Requires CRYPTO_PAY_TOKEN in env. Returns { ok, url?, error? }.
    Docs: https://help.crypt.bot/crypto-pay-api
    """
    token = os.getenv("CRYPTO_PAY_TOKEN", "").strip()
    if not token:
        return {"ok": False, "error": "CRYPTO_PAY_TOKEN is not set"}

    url = "https://pay.crypt.bot/api/createInvoice"
    headers = {
        "Content-Type": "application/json",
        "X-Token": token,
    }
    data = {
        "amount": amount,
        "asset": asset,
        "description": description,
        "payload": payload,
        "expires_in": expires_in,
    }

    timeout = httpx.Timeout(connect=10.0, read=20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        last_err: str | None = None
        for attempt in range(retries):
            try:
                resp = await client.post(url, headers=headers, content=json.dumps(data))
                resp.raise_for_status()
                j = resp.json()
                if not j.get("ok"):
                    return {"ok": False, "error": str(j)}
                result = j.get("result") or {}
                pay_url = (
                    result.get("pay_url")
                    or result.get("bot_invoice_url")
                    or result.get("invoice_url")
                )
                if not pay_url:
                    return {"ok": False, "error": "no_pay_url"}
                return {"ok": True, "url": pay_url, "raw": result}
            except Exception as e:  # noqa: PERF203
                last_err = str(e)
                await asyncio.sleep(min(1.0 * (attempt + 1), 3.0))
        return {"ok": False, "error": last_err or "unknown_error"}
