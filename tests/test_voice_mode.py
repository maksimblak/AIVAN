# mypy: ignore-errors
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiogram.enums import ParseMode
from src.core.bot_app import voice as voice_mod
from src.core.bot_app import context as ctx


class DummyAudioService:
    def __init__(self, transcript: str, output_path: Path):
        self.transcript = transcript
        self.output_path = output_path
        self.ensure_called = False
        self.transcribe_path: Path | None = None

    async def ensure_short_enough(self, duration: int | None) -> None:
        assert duration is not None
        self.ensure_called = True

    async def transcribe(self, file_path: Path) -> str:
        self.transcribe_path = Path(file_path)
        assert self.transcribe_path.exists()
        return self.transcript

    async def synthesize(
        self,
        text: str,
        *,
        prefer_male: bool = False,
        voice_override: str | None = None,
    ) -> list[Path]:
        assert text
        self.output_path.write_bytes(b"fake-voice")
        return [self.output_path]


class DummyVoice:
    def __init__(self, duration: int, file_id: str):
        self.duration = duration
        self.file_id = file_id


class DummyBot:
    async def get_file(self, file_id: str):
        assert file_id == "voice-id"
        return SimpleNamespace(file_path="voice.ogg")

    async def download_file(self, file_path: str):
        assert file_path == "voice.ogg"
        return io.BytesIO(b"voice-bytes")


class DummyMessage:
    def __init__(self) -> None:
        self.voice = DummyVoice(duration=5, file_id="voice-id")
        self.bot = DummyBot()
        self._answers = []
        self._voice_answers = []

    async def answer(self, text: str, *, parse_mode=None):
        self._answers.append((text, parse_mode))

    async def answer_voice(self, fs_input_file, *, caption=None, parse_mode=None):
        self._voice_answers.append((fs_input_file, caption, parse_mode))


@pytest.mark.asyncio
async def test_process_voice_message_happy_path(monkeypatch, tmp_path):
    original_audio_service = getattr(voice_mod.simple_context, "audio_service", None)

    tts_file = tmp_path / "response.ogg"
    dummy_service = DummyAudioService("Распознанный текст", tts_file)

    captured = {}

    async def fake_process_question(message, *, text_override=None):
        captured["message"] = message
        captured["text_override"] = text_override
        return "Готовый ответ"

    try:
        # Подменяем контекст и настройку голосового режима
        voice_mod.simple_context.audio_service = dummy_service
        monkeypatch.setattr(
            voice_mod.simple_context,
            "settings",
            lambda: SimpleNamespace(voice_mode_enabled=True),
            raising=True,
        )
        handler = voice_mod._build_voice_handler(fake_process_question)

        message = DummyMessage()
        await handler(message)

        # Проверяем, что распознанный текст был отправлен пользователю
        assert message._answers, "expected at least one text answer"
        preview_text, parse_mode = message._answers[0]
        assert "Распознанный текст" in preview_text
        assert parse_mode == "HTML"

        # Проверяем, что в process_question пришел распознанный текст
        assert captured["text_override"] == "Распознанный текст"

        # Проверяем, что ответ озвучен
        assert message._voice_answers, "expected voice reply"
        fs_input, caption, parse_mode = message._voice_answers[0]
        assert caption == voice_mod.VOICE_REPLY_CAPTION
        assert parse_mode == ParseMode.HTML
        assert Path(fs_input.path) == tts_file

        # Исходный voice временный файл должен быть удален
        assert dummy_service.transcribe_path is not None
        assert not dummy_service.transcribe_path.exists()

        assert dummy_service.ensure_called
    finally:
        voice_mod.simple_context.audio_service = original_audio_service

