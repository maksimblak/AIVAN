# mypy: ignore-errors
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core import main_simple as ms


class DummyUser:
    def __init__(self, user_id: int, first_name: str = "Test") -> None:
        self.id = user_id
        self.first_name = first_name


class DummyChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class DummyMessage:
    def __init__(self) -> None:
        self.from_user = DummyUser(42, "Maks")
        self.chat = DummyChat(99)
        self.message_id = 1
        self._photos: list[Path] = []
        self._videos: list[Path] = []
        self._texts: list[str] = []

    async def answer_photo(self, fs_input_file, *, caption=None):
        self._photos.append(Path(fs_input_file.path))

    async def answer_video(self, fs_input_file, *, caption=None):
        self._videos.append(Path(fs_input_file.path))

    async def answer(self, text: str, **kwargs):
        self._texts.append(text)


def _prepare_session_store():
    return SimpleNamespace(get_or_create=lambda *_: SimpleNamespace(pending_feedback_request_id=None))


@pytest.mark.asyncio
async def test_cmd_start_sends_welcome_photo(tmp_path):
    original_session_store = ms.session_store
    original_media = ms.WELCOME_MEDIA

    ms.session_store = _prepare_session_store()

    card = tmp_path / "welcome.png"
    card.write_bytes(b"fake-image")
    ms.WELCOME_MEDIA = ms.WelcomeMedia(card, "photo")

    message = DummyMessage()

    try:
        await ms.cmd_start(message)
    finally:
        ms.session_store = original_session_store
        ms.WELCOME_MEDIA = original_media

    assert message._photos == [card]
    assert not message._videos
    assert message._texts, "welcome text should follow the media"


@pytest.mark.asyncio
async def test_cmd_start_sends_welcome_video(tmp_path):
    original_session_store = ms.session_store
    original_media = ms.WELCOME_MEDIA

    ms.session_store = _prepare_session_store()

    clip = tmp_path / "welcome.mp4"
    clip.write_bytes(b"fake-video")
    ms.WELCOME_MEDIA = ms.WelcomeMedia(clip, "video")

    message = DummyMessage()

    try:
        await ms.cmd_start(message)
    finally:
        ms.session_store = original_session_store
        ms.WELCOME_MEDIA = original_media

    assert message._videos == [clip]
    assert not message._photos
    assert message._texts, "welcome text should follow the media"


