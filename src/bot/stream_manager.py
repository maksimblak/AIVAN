"""
Менеджер для streaming ответов в Telegram
"""

import asyncio
import logging
import re
import time
from contextlib import suppress

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message

from .ui_components import render_legal_html

logger = logging.getLogger(__name__)

SAFE_LIMIT = 3900  # безопасный лимит для HTML


class StreamManager:
    """Управляет streaming-ответами в Telegram с буферизацией и редактированием сообщения."""

    def __init__(
        self,
        bot: Bot,
        chat_id: int,
        update_interval: float = 1.5,
        buffer_size: int = 50,
        max_retries: int = 3,
    ):
        self.bot = bot
        self.chat_id = chat_id
        self.update_interval = update_interval
        self.buffer_size = buffer_size
        self.max_retries = max_retries

        # Состояние
        self.message: Message | None = None
        self.last_update_time = 0.0
        self.last_sent_text = ""   # последний отправленный (HTML)
        self.last_raw_text = ""    # последний сырой (plain)
        self.pending_text = ""     # текущий сырой буфер
        self.is_final = False
        self._stopped = False
        self.update_task: asyncio.Task | None = None

    async def start_streaming(self, initial_text: str = "🤔 Обдумываю ответ...") -> Message:
        """Отправляет начальное сообщение и запускает цикл обновлений."""
        self.message = await self.bot.send_message(
            chat_id=self.chat_id,
            text=initial_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        self.last_sent_text = initial_text
        self.last_raw_text = ""
        self.last_update_time = time.time()
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("Started streaming for chat %s, msg %s", self.chat_id, self.message.message_id)
        return self.message

    async def update_text(self, new_text: str, is_final: bool = False):
        """Обновляет сырой буфер (plain)."""
        self.pending_text = new_text or ""
        self.is_final = is_final
        if is_final:
            await self._force_update()

    async def _update_loop(self):
        while not self.is_final:
            try:
                await asyncio.sleep(self.update_interval)
                if self._should_update():
                    await self._do_update()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Error in update loop: %s", e)

    def _should_update(self) -> bool:
        if not self.pending_text or not self.message:
            return False
        if self.pending_text == self.last_raw_text:
            return False
        if (time.time() - self.last_update_time) >= self.update_interval:
            return True
        if len(self.pending_text) - len(self.last_raw_text) >= self.buffer_size:
            return True
        return False

    async def _force_update(self):
        if self.pending_text and self.message:
            await self._do_update()

    async def _do_update(self):
        if not self.message or not self.pending_text:
            return

        formatted = render_legal_html(self.pending_text)  # универсальный красивый рендер
        if len(formatted) > SAFE_LIMIT:
            formatted = formatted[: SAFE_LIMIT - 10] + "…"

        if formatted == self.last_sent_text:
            self.last_raw_text = self.pending_text
            return

        for retry in range(self.max_retries):
            try:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self.message.message_id,
                    text=formatted,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                self.last_sent_text = formatted
                self.last_raw_text = self.pending_text
                self.last_update_time = time.time()
                break

            except TelegramRetryAfter as e:
                logger.warning("Rate limit hit, waiting %s s", e.retry_after)
                await asyncio.sleep(e.retry_after)

            except TelegramBadRequest as e:
                low = str(e).lower()
                # Если Telegram не смог распарсить HTML — фолбэк в plain text
                if "can't parse entities" in low or "parse entities" in low:
                    safe_plain = re.sub(r"<[^>]+>", "", formatted)
                    if len(safe_plain) > SAFE_LIMIT:
                        safe_plain = safe_plain[: SAFE_LIMIT - 10] + "…"
                    await self.bot.edit_message_text(
                        chat_id=self.chat_id,
                        message_id=self.message.message_id,
                        text=safe_plain or " ",
                        disable_web_page_preview=True,
                    )
                    self.last_sent_text = safe_plain
                    self.last_raw_text = self.pending_text
                    self.last_update_time = time.time()
                    break
                if "message is not modified" in low:
                    self.last_raw_text = self.pending_text
                    break
                if "message to edit not found" in low:
                    logger.error("Message to edit not found, stopping stream")
                    await self.stop()
                    break
                logger.warning("TelegramBadRequest on retry %d: %s", retry, e)
                if retry == self.max_retries - 1:
                    logger.error("Failed to update message after %d retries", self.max_retries)

            except Exception as e:
                logger.warning("Failed to update on retry %d: %s", retry, e)
                if retry == self.max_retries - 1:
                    logger.error("Failed to update message after %d retries", self.max_retries)

    async def finalize(self, final_text: str | None = None):
        """Закрывает стрим и один раз красиво форматирует полный текст."""
        if self._stopped:
            return
        self._stopped = True
        self.is_final = True

        if self.update_task:
            self.update_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.update_task

        if not self.message:
            return

        full_text = (final_text if isinstance(final_text, str) else None) or (self.pending_text or "")
        try:
            formatted = render_legal_html(full_text) if full_text else "—"
            if len(formatted) > SAFE_LIMIT:
                formatted = formatted[: SAFE_LIMIT - 10] + "…"

            await self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=self.message.message_id,
                text=formatted,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            self.last_sent_text = formatted
            self.last_raw_text = full_text

        except Exception:
            # Фолбэк: plain text
            safe_plain = re.sub(r"<[^>]+>", "", full_text or "")
            if len(safe_plain) > SAFE_LIMIT:
                safe_plain = safe_plain[: SAFE_LIMIT - 10] + "…"
            await self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=self.message.message_id,
                text=safe_plain or " ",
                disable_web_page_preview=True,
            )
            self.last_sent_text = safe_plain
            self.last_raw_text = full_text

    async def stop(self):
        """Останавливает streaming (без принудительного редактирования)."""
        self.is_final = True
        self._stopped = True
        if self.update_task:
            self.update_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.update_task
        logger.info("Stopped streaming for message %s", getattr(self.message, "message_id", "unknown"))


class StreamingCallback:
    """Callback для интеграции с OpenAI streaming."""

    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
        self.total_calls = 0

    async def __call__(self, partial_text: str, is_final: bool):
        self.total_calls += 1
        try:
            if is_final:
                await self.stream_manager.finalize(partial_text)
            else:
                await self.stream_manager.update_text(partial_text, is_final=False)
        except Exception as e:
            logger.error("Error in streaming callback: %s", e)
            if is_final:
                with suppress(Exception):
                    await self.stream_manager.stop()
