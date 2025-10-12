from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from contextlib import suppress
from html import escape as html_escape
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Sequence, TYPE_CHECKING

from aiogram import Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile, Message

from src.bot.typing_indicator import typing_action
from src.bot.ui_components import Emoji
from src.core.simple_bot import context as simple_context

if TYPE_CHECKING:
    from src.core.audio_service import AudioService

__all__ = ["register_voice_handlers"]

logger = logging.getLogger("ai-ivan.simple.voice")


class ProcessQuestionHandler(Protocol):
    async def __call__(
        self,
        message: Message,
        *,
        text_override: str | None = None,
        attachments: Sequence[Any] | None = None,
    ) -> str | None:
        ...


VOICE_REPLY_CAPTION = (
    f"{Emoji.MICROPHONE} <b>Голосовой ответ готов</b>"
    f"\n{Emoji.INFO} Нажмите, чтобы прослушать."
)


def register_voice_handlers(
    dp: Dispatcher,
    process_question: ProcessQuestionHandler,
) -> None:
    """Register voice-related handlers."""
    dp.message.register(_build_voice_handler(process_question), F.voice)


def _build_voice_handler(
    process_question: ProcessQuestionHandler,
) -> Callable[[Message], Awaitable[None]]:
    async def handle_voice_message(message: Message) -> None:
        if not message.voice:
            return

        audio_service = simple_context.audio_service
        try:
            voice_enabled = simple_context.settings().voice_mode_enabled
        except RuntimeError:
            voice_enabled = False

        if audio_service is None or not voice_enabled:
            await message.answer("Voice mode is currently unavailable. Please send text.")
            return

        if not message.bot:
            await message.answer("Unable to access bot context for processing the voice message.")
            return

        temp_voice_path: Path | None = None
        tts_paths: list[Path] = []

        try:
            await audio_service.ensure_short_enough(message.voice.duration)

            async with typing_action(message.bot, message.chat.id, "record_voice"):
                temp_voice_path = await _download_voice_to_temp(message)
                transcript = await audio_service.transcribe(temp_voice_path)

            preview = html_escape(transcript[:500])
            if len(transcript) > 500:
                preview += "..."
            await message.answer(
                f"{Emoji.ROBOT} Recognized: <i>{preview}</i>",
                parse_mode=ParseMode.HTML,
            )

            response_text = await process_question(message, text_override=transcript)
            if not response_text:
                return

            async with typing_action(message.bot, message.chat.id, "upload_voice"):
                try:
                    tts_paths = await audio_service.synthesize(response_text, prefer_male=True)
                except Exception as tts_error:
                    logger.warning("Text-to-speech failed: %s", tts_error)
                    return

                if not tts_paths:
                    logger.warning("Text-to-speech returned no audio chunks")
                    return

                for idx, generated_path in enumerate(tts_paths):
                    caption = VOICE_REPLY_CAPTION if idx == 0 else None
                    await message.answer_voice(
                        FSInputFile(generated_path),
                        caption=caption,
                        parse_mode=ParseMode.HTML if caption else None,
                    )

        except ValueError as duration_error:
            logger.warning("Voice message duration exceeded: %s", duration_error)
            limit = (
                audio_service.max_duration_seconds
                if audio_service is not None
                else simple_context.settings().voice_max_duration_seconds
            )
            await message.answer(
                f"{Emoji.WARNING} Voice message is too long. Maximum duration is {limit} seconds.",
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:
            logger.exception("Failed to process voice message: %s", exc)
            await message.answer(
                f"{Emoji.ERROR} Could not process the voice message. Please try again later.",
                parse_mode=ParseMode.HTML,
            )
        finally:
            with suppress(Exception):
                if temp_voice_path:
                    temp_voice_path.unlink()
            with suppress(Exception):
                for generated_path in tts_paths:
                    generated_path.unlink()

    return handle_voice_message


async def _download_voice_to_temp(message: Message) -> Path:
    """Download Telegram voice message into a temporary file."""
    if not message.bot:
        raise RuntimeError("Bot instance is not available for voice download")
    if not message.voice:
        raise RuntimeError("Voice payload is missing")

    file_info = await message.bot.get_file(message.voice.file_id)
    file_path = file_info.file_path
    if not file_path:
        raise RuntimeError("Telegram did not return a file path for the voice message")

    temp_path = await asyncio.to_thread(_create_temp_file_path, ".ogg")

    file_stream = await message.bot.download_file(file_path)
    try:
        await asyncio.to_thread(_write_stream_to_path, file_stream, temp_path)
    finally:
        close_method = getattr(file_stream, "close", None)
        if callable(close_method):
            close_method()

    return temp_path


def _create_temp_file_path(suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        return Path(tmp.name)
    finally:
        tmp.close()


def _write_stream_to_path(stream: Any, target: Path) -> None:
    with target.open("wb") as destination:
        shutil.copyfileobj(stream, destination, length=128 * 1024)
