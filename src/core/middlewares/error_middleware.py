from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.enums import ParseMode
from aiogram.types import CallbackQuery, Message, Update

from src.core.exceptions import BaseCustomException, ErrorContext, ErrorHandler
from core.bot_app.ui_components import sanitize_telegram_html

DEFAULT_ERROR_MESSAGE = (
    "\u26a0\ufe0f <b>Произошла ошибка обработки запроса</b>\n\n"
    "Попробуйте ещё раз позднее или обратитесь в поддержку."
)


class ErrorHandlingMiddleware(BaseMiddleware):
    """Перехватывает исключения aiogram-хендлеров и делегирует их ErrorHandler."""

    def __init__(self, error_handler: ErrorHandler, logger: logging.Logger | None = None) -> None:
        self._error_handler = error_handler
        self._logger = logger or logging.getLogger(__name__)

    async def __call__(
        self,
        handler: Callable[[Update, Dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: Dict[str, Any],
    ) -> Any:
        try:
            return await handler(event, data)
        except Exception as exc:  # noqa: BLE001
            context = self._build_context(event, data)
            user_message = DEFAULT_ERROR_MESSAGE
            custom_exc: BaseCustomException | None = None

            if self._error_handler:
                try:
                    custom_exc = await self._error_handler.handle_exception(exc, context)
                    user_message = getattr(custom_exc, "user_message", DEFAULT_ERROR_MESSAGE)
                except Exception as handler_error:  # noqa: BLE001
                    self._logger.exception("Error handler failed to process aiogram exception", extra={"handler_error": str(handler_error)})
            else:
                self._logger.warning("Error handler is not configured; falling back to default response")

            self._log_exception(exc, context, custom_exc)
            await self._reply_to_event(event, user_message)
            return None

    def _log_exception(
        self,
        exc: Exception,
        context: ErrorContext | None,
        custom_exc: BaseCustomException | None = None,
    ) -> None:
        if not self._logger:
            return

        log_extra: dict[str, Any] = {"exception_type": type(exc).__name__}

        if isinstance(exc, BaseCustomException) and custom_exc is None:
            custom_exc = exc

        if custom_exc is not None:
            log_extra["error_type"] = custom_exc.error_type.value
            if custom_exc.context and custom_exc.context is not context:
                context = custom_exc.context

        if context is not None:
            if context.user_id is not None:
                log_extra["user_id"] = context.user_id
            if context.chat_id is not None:
                log_extra["chat_id"] = context.chat_id
            if context.function_name:
                log_extra["handler"] = context.function_name
            additional = context.additional_data or {}
            payload = additional.get("payload")
            if payload:
                trimmed = payload if len(payload) <= 1000 else payload[:1000] + '...'
                log_extra["payload"] = trimmed
            event_type = additional.get("event_type")
            if event_type:
                log_extra["event_type"] = event_type

        self._logger.exception('Unhandled exception in aiogram handler', extra=log_extra)


    def _build_context(self, event: Update, data: Dict[str, Any]) -> ErrorContext:
        user_id = None
        chat_id = None
        message_text = None

        if isinstance(event, Message):
            user_id = event.from_user.id if event.from_user else None
            chat_id = event.chat.id if event.chat else None
            message_text = event.text or event.caption
        elif isinstance(event, CallbackQuery):
            if event.from_user:
                user_id = event.from_user.id
            if event.message and event.message.chat:
                chat_id = event.message.chat.id
            message_text = event.data or (
                event.message.text if event.message else None
            )

        handler_obj = data.get("handler")
        handler_name = getattr(handler_obj, "__qualname__", None) or repr(handler_obj)

        additional: Dict[str, Any] = {
            "event_type": type(event).__name__,
        }
        if message_text:
            additional["payload"] = message_text[:5000]

        return ErrorContext(
            user_id=user_id,
            chat_id=chat_id,
            function_name=handler_name,
            additional_data=additional,
        )

    async def _reply_to_event(self, event: Update, user_message: str) -> None:
        if not user_message:
            user_message = DEFAULT_ERROR_MESSAGE

        safe_message = sanitize_telegram_html(user_message)

        if isinstance(event, Message):
            with suppress(Exception):
                await event.answer(
                    safe_message,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
        elif isinstance(event, CallbackQuery):
            if event.message:
                with suppress(Exception):
                    await event.message.answer(
                        safe_message,
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                    )
            with suppress(Exception):
                await event.answer()
