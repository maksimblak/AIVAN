"""
Менеджер статусов и прогресса для красивого UX
Отвечает за анимацию загрузки, прогресс-бары и статусные сообщения
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.types import Message

from .ui_components import Emoji


class ProgressStatus:
    """Класс для управления статусом обработки запроса"""

    def __init__(self, bot: Bot, chat_id: int):
        self.bot = bot
        self.chat_id = chat_id
        self.status_message: Message | None = None
        self.current_stage = 0
        self.total_stages = 4

    async def start(self, initial_message: str = None) -> Message:
        """Запускает показ статуса"""
        if not initial_message:
            initial_message = (
                f"{Emoji.LOADING} **Обрабатываю ваш запрос\\.\\.\\.** \n\n_Пожалуйста, подождите_"
            )

        self.status_message = await self.bot.send_message(
            self.chat_id, initial_message, parse_mode=ParseMode.MARKDOWN_V2
        )
        return self.status_message

    async def update_stage(self, stage: int, message: str):
        """Обновляет текущую стадию обработки"""
        if not self.status_message:
            return

        self.current_stage = stage
        progress_bar = self._create_progress_bar(stage, self.total_stages)
        percentage = int((stage / self.total_stages) * 100)

        status_text = f"{message}\n\n`{progress_bar}` {percentage}%"

        try:
            await self.status_message.edit_text(status_text, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            # Игнорируем ошибки редактирования (слишком частые обновления)
            pass

    async def complete(self):
        """Завершает показ статуса"""
        if self.status_message:
            try:
                await self.status_message.delete()
            except Exception:
                pass
            self.status_message = None

    def _create_progress_bar(self, current: int, total: int, width: int = 10) -> str:
        """Создает ASCII прогресс-бар"""
        filled = int((current / total) * width)
        empty = width - filled
        return "▓" * filled + "░" * empty


class AnimatedStatus:
    """Класс для анимированного статуса загрузки"""

    LOADING_FRAMES = [
        f"{Emoji.LOADING} Думаю\\.\\.\\.",
        f"{Emoji.LOADING} Думаю\\.\\.\\.",
        f"{Emoji.LOADING} Анализирую\\.\\.\\.",
        f"{Emoji.LOADING} Анализирую\\.\\.\\.",
        f"{Emoji.SEARCH} Ищу практику\\.\\.\\.",
        f"{Emoji.SEARCH} Ищу практику\\.\\.\\.",
        f"{Emoji.DOCUMENT} Составляю ответ\\.\\.\\.",
        f"{Emoji.DOCUMENT} Составляю ответ\\.\\.\\.",
    ]

    def __init__(self, bot: Bot, chat_id: int):
        self.bot = bot
        self.chat_id = chat_id
        self.status_message: Message | None = None
        self.is_running = False
        self.current_frame = 0

    async def start(self) -> Message:
        """Запускает анимацию"""
        self.is_running = True
        self.current_frame = 0

        self.status_message = await self.bot.send_message(
            self.chat_id, self.LOADING_FRAMES[0], parse_mode=ParseMode.MARKDOWN_V2
        )

        # Запускаем анимацию в фоне
        asyncio.create_task(self._animate())

        return self.status_message

    async def stop(self):
        """Останавливает анимацию"""
        self.is_running = False
        if self.status_message:
            try:
                await self.status_message.delete()
            except Exception:
                pass
            self.status_message = None

    async def _animate(self):
        """Анимирует статусное сообщение"""
        while self.is_running and self.status_message:
            try:
                await asyncio.sleep(0.8)  # Интервал анимации
                if not self.is_running:
                    break

                self.current_frame = (self.current_frame + 1) % len(self.LOADING_FRAMES)

                await self.status_message.edit_text(
                    self.LOADING_FRAMES[self.current_frame], parse_mode=ParseMode.MARKDOWN_V2
                )
            except Exception:
                # Останавливаем анимацию при ошибке
                break


class StatusMessages:
    """Готовые статусные сообщения"""

    STAGES = [
        f"{Emoji.SEARCH} Анализирую ваш вопрос\\.\\.\\.",
        f"{Emoji.LOADING} Ищу релевантную судебную практику\\.\\.\\.",
        f"{Emoji.DOCUMENT} Формирую структурированный ответ\\.\\.\\.",
        f"{Emoji.MAGIC} Финализирую рекомендации\\.\\.\\.",
    ]

    @staticmethod
    def get_stage_message(stage: int) -> str:
        """Получить сообщение для стадии"""
        if 0 <= stage < len(StatusMessages.STAGES):
            return StatusMessages.STAGES[stage]
        return StatusMessages.STAGES[-1]

    @staticmethod
    def get_completion_message() -> str:
        """Сообщение о завершении"""
        return f"{Emoji.SUCCESS} **Готово\\!**"


class TypingSimulator:
    """Имитирует печатание для более живого интерфейса"""

    def __init__(self, bot: Bot, chat_id: int):
        self.bot = bot
        self.chat_id = chat_id
        self._typing_task: asyncio.Task | None = None
        self._is_typing = False

    async def start_typing(self):
        """Начинает показ индикатора печатания"""
        if self._is_typing:
            return

        self._is_typing = True
        self._typing_task = asyncio.create_task(self._typing_loop())

    async def stop_typing(self):
        """Останавливает показ индикатора печатания"""
        self._is_typing = False
        if self._typing_task:
            self._typing_task.cancel()
            try:
                await self._typing_task
            except asyncio.CancelledError:
                pass

    async def _typing_loop(self):
        """Цикл отправки индикатора печатания"""
        try:
            while self._is_typing:
                await self.bot.send_chat_action(self.chat_id, "typing")
                await asyncio.sleep(4.5)  # Telegram показывает typing 5 сек
        except asyncio.CancelledError:
            pass


class QuickStatus:
    """Быстрые статусные методы"""

    @staticmethod
    async def send_processing(bot: Bot, chat_id: int) -> Message:
        """Отправляет сообщение об обработке"""
        return await bot.send_message(
            chat_id,
            f"{Emoji.LOADING} **Обрабатываю\\.\\.\\.**\n\n_Это займет несколько секунд_",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    @staticmethod
    async def send_searching(bot: Bot, chat_id: int) -> Message:
        """Отправляет сообщение о поиске"""
        return await bot.send_message(
            chat_id,
            f"{Emoji.SEARCH} **Ищу информацию\\.\\.\\.**\n\n_Проверяю базы данных судебной практики_",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    @staticmethod
    async def send_analyzing(bot: Bot, chat_id: int) -> Message:
        """Отправляет сообщение об анализе"""
        return await bot.send_message(
            chat_id,
            f"{Emoji.DOCUMENT} **Анализирую найденную информацию\\.\\.\\.**\n\n_Формирую правовую позицию_",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    @staticmethod
    async def send_error(bot: Bot, chat_id: int, error_text: str = "") -> Message:
        """Отправляет сообщение об ошибке"""
        message = (
            f"{Emoji.ERROR} **Произошла ошибка**\n\n_Попробуйте позже или переформулируйте вопрос_"
        )

        if error_text:
            message += f"\n\n`{error_text[:100]}`"

        return await bot.send_message(chat_id, message, parse_mode=ParseMode.MARKDOWN_V2)


# ============ КОНТЕКСТНЫЕ МЕНЕДЖЕРЫ ============


class StatusContext:
    """Контекстный менеджер для статусов"""

    def __init__(self, bot: Bot, chat_id: int, use_animation: bool = False):
        self.bot = bot
        self.chat_id = chat_id
        self.use_animation = use_animation
        self.status = None

    async def __aenter__(self):
        if self.use_animation:
            self.status = AnimatedStatus(self.bot, self.chat_id)
        else:
            self.status = ProgressStatus(self.bot, self.chat_id)

        await self.status.start()
        return self.status

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.status:
            if hasattr(self.status, "complete"):
                await self.status.complete()
            else:
                await self.status.stop()


class TypingContext:
    """Контекстный менеджер для индикатора печатания"""

    def __init__(self, bot: Bot, chat_id: int):
        self.typing = TypingSimulator(bot, chat_id)

    async def __aenter__(self):
        await self.typing.start_typing()
        return self.typing

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.typing.stop_typing()


# ============ ДЕКОРАТОРЫ ============


def with_status(use_animation: bool = False):
    """Декоратор для автоматического управления статусом"""

    def decorator(func):
        async def wrapper(message: Message, *args, **kwargs):
            bot = message.bot
            chat_id = message.chat.id

            async with StatusContext(bot, chat_id, use_animation) as status:
                return await func(message, status, *args, **kwargs)

        return wrapper

    return decorator


def with_typing(func):
    """Декоратор для автоматического показа индикатора печатания"""

    async def wrapper(message: Message, *args, **kwargs):
        bot = message.bot
        chat_id = message.chat.id

        async with TypingContext(bot, chat_id):
            return await func(message, *args, **kwargs)

    return wrapper


# ============ СТАТИСТИКА ВРЕМЕНИ ============


class ResponseTimer:
    """Таймер для измерения времени ответа"""

    def __init__(self):
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def start(self):
        """Начать измерение"""
        self.start_time = datetime.now()

    def stop(self):
        """Закончить измерение"""
        self.end_time = datetime.now()

    @property
    def duration(self) -> float:
        """Длительность в секундах"""
        if not self.start_time or not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def get_duration_text(self) -> str:
        """Получить текст с длительностью"""
        duration = self.duration
        if duration < 1:
            return f"{int(duration * 1000)}мс"
        elif duration < 60:
            return f"{duration:.1f}с"
        else:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            return f"{minutes}м {seconds}с"
