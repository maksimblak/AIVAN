"""
Менеджер для streaming ответов в Telegram
"""

import asyncio
import logging
import time

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message

from .ui_components import render_legal_html

logger = logging.getLogger(__name__)


class StreamManager:
    """Управляет streaming ответами в Telegram с буферизацией и редактированием сообщений"""

    def __init__(
        self,
        bot: Bot,
        chat_id: int,
        update_interval: float = 1.5,  # Интервал обновления в секундах
        buffer_size: int = 50,  # Минимальный размер буфера для обновления
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
        self.last_sent_text = ""
        self.pending_text = ""
        self.is_final = False
        self.update_task: asyncio.Task | None = None

    async def start_streaming(self, initial_text: str = "🤔 Обдумываю ответ...") -> Message:
        """Начинает streaming, отправляет начальное сообщение"""
        try:
            self.message = await self.bot.send_message(
                chat_id=self.chat_id, text=initial_text, parse_mode=ParseMode.HTML
            )
            self.last_sent_text = initial_text
            self.last_update_time = time.time()

            # Запускаем фоновую задачу обновления
            self.update_task = asyncio.create_task(self._update_loop())

            logger.info(
                f"Started streaming for chat {self.chat_id}, message {self.message.message_id}"
            )
            return self.message

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            raise

    async def update_text(self, new_text: str, is_final: bool = False):
        """Обновляет текст для streaming"""
        self.pending_text = new_text
        self.is_final = is_final

        if is_final:
            # Для финального обновления отправляем сразу
            await self._force_update()

    async def _update_loop(self):
        """Фоновая задача для периодического обновления сообщения"""
        while not self.is_final:
            try:
                await asyncio.sleep(self.update_interval)

                # Проверяем нужно ли обновление
                if self._should_update():
                    await self._do_update()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in update loop: {e}")

    def _should_update(self) -> bool:
        """Определяет нужно ли обновлять сообщение"""
        if not self.pending_text or not self.message:
            return False

        # Если текст не изменился
        if self.pending_text == self.last_sent_text:
            return False

        # Если прошло достаточно времени
        time_passed = time.time() - self.last_update_time
        if time_passed >= self.update_interval:
            return True

        # Если накопилось достаточно нового текста
        text_diff = len(self.pending_text) - len(self.last_sent_text)
        if text_diff >= self.buffer_size:
            return True

        return False

    async def _force_update(self):
        """Принудительное обновление сообщения"""
        if self.pending_text and self.message:
            await self._do_update()

    async def _do_update(self):
        """Выполняет обновление сообщения"""
        if not self.message or not self.pending_text:
            return

        # Форматируем текст
        formatted_text = render_legal_html(self.pending_text)

        # Обрезаем если слишком длинный для Telegram
        if len(formatted_text) > 4000:
            formatted_text = formatted_text[:3990] + "..."

        # Пропускаем если текст не изменился
        if formatted_text == self.last_sent_text:
            return

        for retry in range(self.max_retries):
            try:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self.message.message_id,
                    text=formatted_text,
                    parse_mode=ParseMode.HTML,
                )

                self.last_sent_text = formatted_text
                self.last_update_time = time.time()

                logger.debug(
                    f"Updated message {self.message.message_id}, length: {len(formatted_text)}"
                )
                break

            except TelegramRetryAfter as e:
                logger.warning(f"Rate limit hit, waiting {e.retry_after} seconds")
                await asyncio.sleep(e.retry_after)

            except TelegramBadRequest as e:
                if "message is not modified" in str(e).lower():
                    # Сообщение не изменилось, это нормально
                    break
                elif "message to edit not found" in str(e).lower():
                    logger.error("Message to edit not found, stopping stream")
                    await self.stop()
                    break
                else:
                    logger.warning(f"Telegram bad request on retry {retry}: {e}")
                    if retry == self.max_retries - 1:
                        logger.error(f"Failed to update message after {self.max_retries} retries")

            except Exception as e:
                logger.warning(f"Failed to update message on retry {retry}: {e}")
                if retry == self.max_retries - 1:
                    logger.error(f"Failed to update message after {self.max_retries} retries")

    async def finalize(self, final_text: str):
        """Завершает streaming с финальным текстом"""
        self.is_final = True
        self.pending_text = final_text

        # Останавливаем фоновую задачу
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass

        # Отправляем финальное обновление
        await self._force_update()

        logger.info(
            f"Finalized streaming for message {self.message.message_id if self.message else 'unknown'}"
        )

    async def stop(self):
        """Останавливает streaming"""
        self.is_final = True

        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"Stopped streaming for message {self.message.message_id if self.message else 'unknown'}"
        )


class StreamingCallback:
    """Callback класс для интеграции с OpenAI streaming"""

    def __init__(self, stream_manager: StreamManager):
        self.stream_manager = stream_manager
        self.total_calls = 0

    async def __call__(self, partial_text: str, is_final: bool):
        """Callback функция для OpenAI streaming"""
        self.total_calls += 1

        try:
            if is_final:
                await self.stream_manager.finalize(partial_text)
            else:
                await self.stream_manager.update_text(partial_text, is_final=False)

        except Exception as e:
            logger.error(f"Error in streaming callback: {e}")
            # В случае ошибки все равно пытаемся завершить stream
            if is_final:
                try:
                    await self.stream_manager.stop()
                except:
                    pass
