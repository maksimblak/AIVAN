"""
Typing indicator для улучшения UX в Telegram боте.
Показывает пользователю визуальную обратную связь о том, что бот обрабатывает запрос.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Literal

from aiogram import Bot

logger = logging.getLogger(__name__)

# Типы действий в Telegram
ChatAction = Literal[
    "typing",  # Печатает текст
    "upload_photo",  # Отправляет фото
    "record_video",  # Записывает видео
    "upload_video",  # Отправляет видео
    "record_voice",  # Записывает голосовое
    "upload_voice",  # Отправляет голосовое
    "upload_document",  # Отправляет документ
    "choose_sticker",  # Выбирает стикер
    "find_location",  # Находит локацию
    "record_video_note",  # Записывает видеосообщение
    "upload_video_note",  # Отправляет видеосообщение
]


class TypingIndicator:
    """
    Класс для управления typing индикатором в Telegram.

    Показывает пользователю визуальную обратную связь во время обработки запроса.
    Индикатор автоматически обновляется каждые 4 секунды, чтобы не исчезнуть.
    """

    def __init__(
        self,
        bot: Bot,
        chat_id: int,
        action: ChatAction = "typing",
        update_interval: float = 4.0,
    ):
        """
        Инициализация typing индикатора.

        Args:
            bot: Экземпляр aiogram Bot
            chat_id: ID чата, где показывать индикатор
            action: Тип действия (typing, upload_document и т.д.)
            update_interval: Интервал обновления в секундах (по умолчанию 4с)
        """
        self.bot = bot
        self.chat_id = chat_id
        self.action = action
        self.update_interval = update_interval

        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Запустить индикатор"""
        if self._running:
            return

        self._running = True

        # Отправляем первый индикатор сразу
        try:
            await self.bot.send_chat_action(self.chat_id, self.action)
        except Exception as e:
            logger.warning(f"Failed to send initial chat action: {e}")

        # Запускаем фоновую задачу для поддержания индикатора
        self._task = asyncio.create_task(self._keep_alive())

    async def stop(self) -> None:
        """Остановить индикатор"""
        self._running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _keep_alive(self) -> None:
        """
        Фоновая задача для поддержания индикатора активным.
        Telegram автоматически скрывает индикатор через 5 секунд,
        поэтому мы обновляем его каждые 4 секунды.
        """
        while self._running:
            try:
                await asyncio.sleep(self.update_interval)
                if self._running:
                    await self.bot.send_chat_action(self.chat_id, self.action)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in typing indicator keep_alive: {e}")

    async def __aenter__(self):
        """Поддержка async context manager"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Автоматическая остановка при выходе из контекста"""
        await self.stop()


@asynccontextmanager
async def typing_action(
    bot: Bot,
    chat_id: int,
    action: ChatAction = "typing",
):
    """
    Context manager для удобного использования typing индикатора.

    Использование:
        async with typing_action(bot, chat_id, "typing"):
            # Индикатор "печатает" показывается пользователю
            answer = await generate_long_response()
            # Индикатор автоматически исчезнет после выхода из блока

    Args:
        bot: Экземпляр aiogram Bot
        chat_id: ID чата
        action: Тип действия (typing, upload_document и т.д.)
    """
    indicator = TypingIndicator(bot, chat_id, action)
    await indicator.start()
    try:
        yield indicator
    finally:
        await indicator.stop()


async def send_typing_once(bot: Bot, chat_id: int, action: ChatAction = "typing") -> None:
    """
    Отправить typing индикатор один раз (без автоматического обновления).

    Используйте для очень быстрых операций (< 5 секунд).
    Для длительных операций используйте typing_action() или TypingIndicator.

    Args:
        bot: Экземпляр aiogram Bot
        chat_id: ID чата
        action: Тип действия
    """
    try:
        await bot.send_chat_action(chat_id, action)
    except Exception as e:
        logger.warning(f"Failed to send chat action: {e}")
