# mypy: ignore-errors
import asyncio
import sys
import types

if "aiogram" not in sys.modules:
    aiogram_module = types.ModuleType("aiogram")
    aiogram_module.Bot = object
    sys.modules["aiogram"] = aiogram_module

    enums_module = types.ModuleType("aiogram.enums")
    enums_module.ParseMode = types.SimpleNamespace(HTML="HTML", MARKDOWN_V2="MARKDOWN_V2")
    sys.modules["aiogram.enums"] = enums_module

    exceptions_module = types.ModuleType("aiogram.exceptions")

    class FakeTelegramBadRequest(Exception):
        pass

    exceptions_module.TelegramBadRequest = FakeTelegramBadRequest
    sys.modules["aiogram.exceptions"] = exceptions_module

    types_module = types.ModuleType("aiogram.types")

    class _FakeMessage:
        pass

    types_module.Message = _FakeMessage
    sys.modules["aiogram.types"] = types_module

from core.bot_app.status_manager import ProgressStatus


class DummyMessage:
    def __init__(self, chat_id: int, text: str, parse_mode=None) -> None:
        self.chat_id = chat_id
        self.text = text
        self.parse_mode = parse_mode
        self.edits: list[tuple[str, object]] = []
        self.deleted = False

    async def edit_text(self, text: str, parse_mode=None) -> None:
        self.text = text
        self.edits.append((text, parse_mode))

    async def delete(self) -> None:
        self.deleted = True


class DummyBot:
    def __init__(self) -> None:
        self.messages: list[DummyMessage] = []

    async def send_message(self, chat_id: int, text: str, parse_mode=None) -> DummyMessage:
        message = DummyMessage(chat_id, text, parse_mode=parse_mode)
        self.messages.append(message)
        return message


def test_progress_status_cycles_message_with_auto_updates():
    asyncio.run(_assert_auto_cycle())


def test_progress_status_manual_update_adjusts_flow():
    asyncio.run(_assert_manual_update_advances_flow())


async def _assert_auto_cycle() -> None:
    bot = DummyBot()
    status = ProgressStatus(bot, chat_id=123, auto_interval=0.05, manual_hold=0.01)

    message = await status.start(auto_cycle=True, interval=0.05)
    try:
        await asyncio.sleep(0.18)
        assert message.edits, "Auto cycle should edit status message at least once"
        assert any("<code>" in text for text, _ in message.edits)
    finally:
        await status.complete()

    assert message.deleted


async def _assert_manual_update_advances_flow() -> None:
    bot = DummyBot()
    status = ProgressStatus(bot, chat_id=555, auto_interval=0.05, manual_hold=0.05)

    message = await status.start(auto_cycle=True, interval=0.05)
    try:
        await asyncio.sleep(0.12)
        await status.update_stage(5, "Custom stage")
        assert status.flow_index == 4

        edits_before = len(message.edits)
        await asyncio.sleep(0.12)
        assert len(message.edits) > edits_before
        latest_text = message.edits[-1][0]
        assert "Custom stage" in latest_text or "Working..." in latest_text
    finally:
        await status.complete()

    assert message.deleted
