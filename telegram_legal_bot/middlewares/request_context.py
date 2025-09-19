# telegram_legal_bot/middlewares/request_context.py
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, Awaitable

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Message, CallbackQuery

from telegram_legal_bot.utils.pro_logging import log_context

def _make_cid() -> str:
    return f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"

class RequestContextMiddleware(BaseMiddleware):
    """Вставляет cid/uid/chat_id в контекст логов для каждого апдейта."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
    ) -> Any:
        uid = None
        chat_id = None

        if isinstance(event, Message):
            uid = event.from_user.id if event.from_user else None
            chat_id = event.chat.id
        elif isinstance(event, CallbackQuery):
            uid = event.from_user.id if event.from_user else None
            chat_id = event.message.chat.id if event.message else None

        with log_context(cid=_make_cid(), uid=uid, chat_id=chat_id):
            return await handler(event, data)
