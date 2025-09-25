from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

try:
    from src.bot.openai_gateway import _make_async_client as _gateway_make_async_client  # type: ignore
except Exception:  # noqa: BLE001
    _gateway_make_async_client = None

logger = logging.getLogger(__name__)


async def _acquire_client() -> AsyncOpenAI:
    """Reuse gateway client factory when available to keep proxy/timeouts consistent."""
    if _gateway_make_async_client is not None:
        return await _gateway_make_async_client()
    return AsyncOpenAI()


class AudioService:
    """Handles speech-to-text and text-to-speech interactions."""

    def __init__(
        self,
        stt_model: str,
        tts_model: str,
        tts_voice: str,
        tts_format: str = "ogg",
        *,
        max_duration_seconds: int = 120,
    ) -> None:
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.tts_format = tts_format
        self.max_duration_seconds = max_duration_seconds

    async def transcribe(self, file_path: Path, *, language: Optional[str] = None) -> str:
        """Convert voice message to text using configured STT model."""
        client = await _acquire_client()
        async with client as oai:
            with file_path.open("rb") as audio_file:
                response = await oai.audio.transcriptions.create(
                    model=self.stt_model,
                    file=audio_file,
                    response_format="text",
                    language=language,
                )
        if isinstance(response, str):
            text = response.strip()
        else:
            text = getattr(response, "text", "") or ""
            text = text.strip()
        if not text:
            raise RuntimeError("Speech recognition returned empty text")
        return text

    async def synthesize(self, text: str) -> Path:
        """Generate speech file from text and return path to audio file."""
        client = await _acquire_client()
        tmp = tempfile.NamedTemporaryFile(
            suffix=f".{self.tts_format}", delete=False
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        async with client as oai:
            try:
                speech = await oai.audio.speech.create(
                    model=self.tts_model,
                    voice=self.tts_voice,
                    input=text,
                    format=self.tts_format,
                )
            except TypeError:
                # Older client versions use response_format instead of format
                speech = await oai.audio.speech.create(
                    model=self.tts_model,
                    voice=self.tts_voice,
                    input=text,
                    response_format=self.tts_format,
                )
            if hasattr(speech, "stream_to_file"):
                await speech.stream_to_file(tmp_path)
            else:
                data = None
                if hasattr(speech, "read"):
                    maybe = speech.read()  # type: ignore[attr-defined]
                    if asyncio.iscoroutine(maybe):
                        data = await maybe
                    else:
                        data = maybe  # type: ignore[assignment]
                elif isinstance(speech, (bytes, bytearray)):
                    data = bytes(speech)
                if data is None:
                    raise RuntimeError("Unsupported speech response payload")
                tmp_path.write_bytes(data)
        return tmp_path

    async def ensure_short_enough(self, duration_seconds: Optional[int]) -> None:
        """Raise if voice message is too long."""
        if duration_seconds is None:
            return
        if duration_seconds > self.max_duration_seconds:
            raise ValueError(
                f"Voice message longer than allowed {self.max_duration_seconds} seconds"
            )


async def transcribe_voice(
    audio_service: AudioService,
    file_path: Path,
    *,
    language: Optional[str] = None,
) -> str:
    return await audio_service.transcribe(file_path, language=language)


async def synthesize_voice(audio_service: AudioService, text: str) -> Path:
    return await audio_service.synthesize(text)
