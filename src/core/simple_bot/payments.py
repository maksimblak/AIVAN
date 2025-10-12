from __future__ import annotations

from aiogram import Dispatcher

__all__ = ["register_payment_handlers"]


def register_payment_handlers(dp: Dispatcher) -> None:
    """Register payment-related handlers."""
    raise NotImplementedError("payments.register_payment_handlers is not yet implemented")
