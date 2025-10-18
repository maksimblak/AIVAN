import re
import sys
import types
from unittest.mock import AsyncMock

import pytest

if "aiogram" not in sys.modules:
    aiogram_module = types.ModuleType("aiogram")

    class _Bot:
        pass

    aiogram_module.Bot = _Bot
    sys.modules["aiogram"] = aiogram_module

    enums_module = types.ModuleType("aiogram.enums")

    class _ParseMode:
        HTML = "HTML"

    enums_module.ParseMode = _ParseMode
    sys.modules["aiogram.enums"] = enums_module

    exceptions_module = types.ModuleType("aiogram.exceptions")

    class _TelegramBadRequest(Exception):
        pass

    class _TelegramRetryAfter(Exception):
        def __init__(self, retry_after: float) -> None:
            self.retry_after = retry_after

    exceptions_module.TelegramBadRequest = _TelegramBadRequest
    exceptions_module.TelegramRetryAfter = _TelegramRetryAfter
    sys.modules["aiogram.exceptions"] = exceptions_module

import src.core.safe_telegram as safe_telegram
from src.bot.ui_components import sanitize_telegram_html
from src.core.safe_telegram import (
    _plain,
    format_safe_html,
    send_html_text,
    split_html_for_telegram,
    tg_edit_html,
    tg_send_html,
)


def _is_balanced(html: str) -> bool:
    stack: list[str] = []
    for match in re.finditer(r"<(/?)([a-zA-Z0-9-]+)(?:[^>]*)>", html):
        name = match.group(2).lower()
        if name == "br":
            continue
        is_closing = bool(match.group(1))
        if is_closing:
            if not stack or stack[-1] != name:
                return False
            stack.pop()
        else:
            stack.append(name)
    return not stack


def test_invalid_anchor_becomes_plain_text() -> None:
    raw = '<a href="javascript:alert(1)">Click me</a>'
    sanitized = sanitize_telegram_html(raw)
    assert "&lt;a" in sanitized
    assert "&lt;/a>" in sanitized
    assert "<a " not in sanitized


def test_split_html_balances_tags_for_long_anchor() -> None:
    long_text = '<a href="https://example.com/">' + ("hello " * 200) + "</a>"
    formatted = format_safe_html(long_text)
    chunks = split_html_for_telegram(formatted, hard_limit=120)
    assert len(chunks) > 1
    for chunk in chunks:
        assert _is_balanced(chunk)


def test_unbalanced_markup_is_normalised() -> None:
    raw = "<b><i>text</b></i>"
    sanitized = sanitize_telegram_html(raw)
    assert sanitized == "<b><i>text</i></b>"


def test_format_safe_html_converts_markdown_and_newlines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safe_telegram, "sanitize_telegram_html", lambda text: text)
    raw = "Header\\n\\nParagraph\n[Link](https://example.com)\nC:\\new_case"
    formatted = format_safe_html(raw)
    assert '<a href="https://example.com">Link</a>' in formatted
    assert "Header\n\nParagraph" in formatted
    assert "C:\\new_case" in formatted


def test_format_safe_html_handles_sanitize_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(_: str) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(safe_telegram, "sanitize_telegram_html", _explode)
    assert format_safe_html("Hello") == "Hello"
    assert format_safe_html("") == "-"


def test_split_html_for_telegram_empty_input_returns_placeholder() -> None:
    assert split_html_for_telegram("") == ["\u2014"]


def test_plain_removes_markup_and_normalises_breaks() -> None:
    html = "<b>Bold</b><br/>Line<br><i>More</i>"
    assert _plain(html) == "Bold\nLine\nMore"


@pytest.mark.asyncio
async def test_tg_edit_html_retries_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch, mock_telegram_bot
) -> None:
    class FakeRetryAfter(Exception):
        def __init__(self, retry_after: float) -> None:
            self.retry_after = retry_after

    monkeypatch.setattr(safe_telegram, "TelegramRetryAfter", FakeRetryAfter)
    sleep_mock = AsyncMock()
    monkeypatch.setattr(safe_telegram.asyncio, "sleep", sleep_mock)
    mock_telegram_bot.edit_message_text.side_effect = [FakeRetryAfter(0.01), None]

    await tg_edit_html(mock_telegram_bot, chat_id=1, message_id=2, html="<b>hi</b>")

    assert mock_telegram_bot.edit_message_text.await_count == 2
    sleep_mock.assert_awaited_once_with(0.01)


@pytest.mark.asyncio
async def test_tg_edit_html_falls_back_to_plain_text(
    monkeypatch: pytest.MonkeyPatch, mock_telegram_bot
) -> None:
    class FakeBadRequest(Exception):
        def __init__(self, message: str) -> None:
            self._message = message

        def __str__(self) -> str:
            return self._message

    monkeypatch.setattr(safe_telegram, "TelegramBadRequest", FakeBadRequest)
    html = "<b>bad</b>"
    plain = _plain(html)[:3900] or " "
    mock_telegram_bot.edit_message_text.side_effect = [FakeBadRequest("can't parse entities"), None]

    await tg_edit_html(mock_telegram_bot, chat_id=1, message_id=2, html=html)

    assert mock_telegram_bot.edit_message_text.await_count == 2
    first_call = mock_telegram_bot.edit_message_text.await_args_list[0]
    assert first_call.kwargs["parse_mode"] == safe_telegram.ParseMode.HTML
    fallback_call = mock_telegram_bot.edit_message_text.await_args_list[1]
    assert fallback_call.kwargs["text"] == plain
    assert "parse_mode" not in fallback_call.kwargs
    assert fallback_call.kwargs["disable_web_page_preview"] is True


@pytest.mark.asyncio
async def test_tg_send_html_handles_parse_error(
    monkeypatch: pytest.MonkeyPatch, mock_telegram_bot
) -> None:
    class FakeBadRequest(Exception):
        def __init__(self, message: str) -> None:
            self._message = message

        def __str__(self) -> str:
            return self._message

    monkeypatch.setattr(safe_telegram, "TelegramBadRequest", FakeBadRequest)
    html = "<i>broken</i>"
    plain = _plain(html)[:3900] or " "
    mock_telegram_bot.send_message.side_effect = [FakeBadRequest("Can't parse entities"), None]

    await tg_send_html(mock_telegram_bot, chat_id=5, html=html, reply_to_message_id=42)

    assert mock_telegram_bot.send_message.await_count == 2
    first_call = mock_telegram_bot.send_message.await_args_list[0]
    assert first_call.kwargs["parse_mode"] == safe_telegram.ParseMode.HTML
    fallback_call = mock_telegram_bot.send_message.await_args_list[1]
    assert fallback_call.kwargs["text"] == plain
    assert "parse_mode" not in fallback_call.kwargs
    assert fallback_call.kwargs["reply_to_message_id"] == 42


@pytest.mark.asyncio
async def test_send_html_text_sends_all_chunks(
    monkeypatch: pytest.MonkeyPatch, mock_telegram_bot
) -> None:
    chunks = ["chunk1", "chunk2", "chunk3"]
    monkeypatch.setattr(safe_telegram, "format_safe_html", lambda raw: raw.upper())
    monkeypatch.setattr(safe_telegram, "split_html_for_telegram", lambda html, hard_limit=3900: chunks)
    send_mock = AsyncMock()
    monkeypatch.setattr(safe_telegram, "tg_send_html", send_mock)

    await send_html_text(mock_telegram_bot, chat_id=7, raw_text="text", reply_to_message_id=100)

    assert send_mock.await_count == len(chunks)
    first_call = send_mock.await_args_list[0]
    _, first_kwargs = first_call
    assert first_kwargs["reply_to_message_id"] == 100
    for _, kwargs in send_mock.await_args_list[1:]:
        assert kwargs["reply_to_message_id"] is None


@pytest.mark.asyncio
async def test_send_html_text_handles_split_failure(
    monkeypatch: pytest.MonkeyPatch, mock_telegram_bot
) -> None:
    def _failing_split(*_args, **_kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(safe_telegram, "split_html_for_telegram", _failing_split)
    send_mock = AsyncMock()
    monkeypatch.setattr(safe_telegram, "tg_send_html", send_mock)
    monkeypatch.setattr(safe_telegram, "format_safe_html", lambda raw: raw)

    await send_html_text(mock_telegram_bot, chat_id=11, raw_text="hello world", reply_to_message_id=None)

    send_mock.assert_awaited_once()
    _, kwargs = send_mock.await_args
    assert kwargs["html"] == "hello world"
