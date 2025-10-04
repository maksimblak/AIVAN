# src/bot/stream_manager.py
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Optional

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.types import Message

from src.core.safe_telegram import (
    format_safe_html,
    split_html_for_telegram,
    tg_edit_html,
    tg_send_html,
)

from src.bot.ui_components import Emoji

logger = logging.getLogger(__name__)


class StreamManager:
    """Управляет streaming-ответами в Telegram: буферизация, редактирование, финализация."""

    def __init__(
        self,
        bot: Bot,
        chat_id: int,
        update_interval: float = 1.5,   # как часто редактируем сообщение
        buffer_size: int = 50,          # минимальный прирост текста до апдейта
        max_retries: int = 3,
    ):
        self.bot = bot
        self.chat_id = chat_id
        self.update_interval = update_interval
        self.buffer_size = buffer_size
        self.max_retries = max_retries

        self.message: Optional[Message] = None
        self.last_update_time = 0.0
        self.last_sent_text = ""            # что реально отправлено (уже отформатированное)
        self.pending_text = ""              # сырая сборка из LLM
        self.is_final = False
        self._stopped = False
        self.update_task: Optional[asyncio.Task] = None

    async def start_streaming(self, initial_text: str | None = None) -> Message:
        """Начинаем стрим: отправляем первое сообщение и запускаем фоновые апдейты."""
        # Первое сообщение — краткое, чтобы его можно было редактировать
        if initial_text is None:
            initial_text = f"{Emoji.ROBOT} 🤔 Обдумываю ответ..."
        self.message = await self.bot.send_message(
            chat_id=self.chat_id, text=initial_text, parse_mode=ParseMode.HTML
        )
        self.last_sent_text = initial_text
        self.last_update_time = time.time()

        self.update_task = asyncio.create_task(self._update_loop())
        logger.info(
            "Started streaming for chat %s, message %s", self.chat_id, self.message.message_id
        )
        return self.message

    async def update_text(self, new_text: str, is_final: bool = False):
        """Вызывается коллбэком стрима OpenAI — обновляет буфер и (если final) форсит апдейт."""
        self.pending_text = new_text or ""
        if is_final:
            self.is_final = True
            # Останавливаем update_loop перед финальным обновлением
            if self.update_task and not self.update_task.done():
                self.update_task.cancel()
            await self._force_update()
        else:
            self.is_final = False

    def _should_update(self) -> bool:
        """Решаем, обновлять ли сообщение прямо сейчас."""
        if not self.pending_text or not self.message:
            return False
        if self.pending_text == self.last_sent_text:
            return False

        time_passed = time.time() - self.last_update_time
        if time_passed >= self.update_interval:
            return True

        text_diff = len(self.pending_text) - len(self.last_sent_text)
        return text_diff >= self.buffer_size

    async def _force_update(self):
        if self.pending_text and self.message:
            await self._do_update()

    async def _update_loop(self):
        """Фоновая задача периодического редактирования «одного» сообщения во время стрима."""
        while not self.is_final and not self._stopped:
            try:
                await asyncio.sleep(self.update_interval)
                if self._should_update():
                    await self._do_update()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Error in update loop: %s", e)

    async def _do_update(self):
        """Редактирование сообщения текущим буфером. Ограничиваем длину, чтобы укладываться в 4096."""
        if not self.message:
            return
        # Форматируем + санитизируем
        formatted = format_safe_html(self.pending_text)

        # Во время стрима редактируем одно сообщение → держим ≤ 3900
        # Но обрезаем умно, чтобы не ломать HTML-теги
        if len(formatted) > 3900:
            # Ищем безопасное место для обрезки (после закрывающего тега или пробела)
            cutoff = 3890
            safe_cut = formatted.rfind('>', 0, cutoff)
            if safe_cut == -1:
                safe_cut = formatted.rfind(' ', 0, cutoff)
            if safe_cut == -1:
                safe_cut = cutoff
            formatted = formatted[:safe_cut] + "…"

        # Если реально нет изменений — выходим
        if formatted == self.last_sent_text:
            return

        # Надёжное редактирование
        await tg_edit_html(
            bot=self.bot,
            chat_id=self.chat_id,
            message_id=self.message.message_id,
            html=formatted,
            max_retries=self.max_retries,
        )
        self.last_sent_text = formatted
        self.last_update_time = time.time()

    async def finalize(self, final_text: Optional[str] = None):
        """
        Закрываем стрим:
        1) форматируем полный текст,
        2) режем на куски,
        3) правим исходное сообщение первым куском,
        4) хвосты отправляем отдельными сообщениями.
        """
        if self._stopped:
            return
        self._stopped = True
        self.is_final = True

        # останавливаем фоновую задачу
        if self.update_task:
            self.update_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.update_task

        # полный текст
        full_text = (final_text if isinstance(final_text, str) else None) or (self.pending_text or "")
        formatted = format_safe_html(full_text) or "—"

        try:
            chunks = split_html_for_telegram(formatted, hard_limit=3900)
        except Exception as e:
            logger.warning("split_html_for_telegram failed: %s", e)
            chunks = [formatted[:3900] or "—"]

        if not self.message:
            # если по какой-то причине старт не состоялся — просто отправим все куски
            for idx, chunk in enumerate(chunks):
                await tg_send_html(self.bot, self.chat_id, chunk)
            return

        # 1) правим первое сообщение
        try:
            await tg_edit_html(self.bot, self.chat_id, self.message.message_id, chunks[0])
        except Exception as e:
            logger.warning("edit_message_text failed on finalize, fallback to new message: %s", e)
            # хвостовой фолбэк — отправляем новый блок с заголовком/контекстом
            sent = await tg_send_html(self.bot, self.chat_id, chunks[0])
            self.message = sent
        self.last_sent_text = chunks[0]
        self.last_update_time = time.time()

        # 2) хвост — отдельными сообщениями
        for tail in chunks[1:]:
            await tg_send_html(self.bot, self.chat_id, tail)

    async def stop(self):
        """Принудительная остановка стрима (без финализации)."""
        self._stopped = True
        self.is_final = True
        if self.update_task:
            self.update_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.update_task
        logger.info(
            "Stopped streaming for message %s", self.message.message_id if self.message else "unknown"
        )


class StreamingCallback:
    """Адаптер под OpenAI streaming: вызывает update/finalize у StreamManager."""

    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager

    async def __call__(self, partial_text: str, is_final: bool):
        try:
            await self.stream_manager.update_text(partial_text, is_final=is_final)
        except Exception as e:
            logger.error("Error in streaming callback: %s", e)
            if is_final:
                with suppress(Exception):
                    await self.stream_manager.stop()
