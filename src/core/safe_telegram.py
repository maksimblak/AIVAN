# src/core/safe_telegram.py
from __future__ import annotations

import asyncio
import logging
import re
from contextlib import suppress
from html import escape as html_escape
from typing import List, Optional

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter

from core.bot_app.ui_components import sanitize_telegram_html

logger = logging.getLogger(__name__)

_HTML_TOKEN_RE = re.compile(r"<[^>]+>|[^<]+")
_HTML_TAG_RE = re.compile(r"<(/?)([a-zA-Z0-9-]+)([^>]*)>")
_SELF_CLOSING_TAGS = frozenset({"br"})


def format_safe_html(raw_text: str) -> str:
    """Sanitize text for Telegram while preserving simple markup and line breaks."""
    normalized = (raw_text or "").replace("\r\n", "\n")
    # Convert double-escaped newlines (`\\n\\n`) eagerly, but treat isolated
    # `\n` sequences conservatively so that file paths like `C:\new_case`
    # stay intact. Single `\n` shifts only when followed by whitespace,
    # bullet markers or punctuation typically used to start a new block.
    normalized = normalized.replace("\\n\\n", "\n\n")
    normalized = re.sub(
        r"\\n(?=(?:\s|$|[-–—•*]|[0-9]+[.)]|<[A-Za-z]|\\n))",
        "\n",
        normalized,
    )
    normalized = normalized.replace(" ", " ").replace("‑", "-")
    normalized = re.sub(r"<br\s*/?>", "\n", normalized, flags=re.IGNORECASE)

    def _convert_markdown_link(match: re.Match[str]) -> str:
        text_part = html_escape(match.group(1).strip())
        href = html_escape(match.group(2).strip(), quote=True)
        return f'<a href="{href}">{text_part}</a>'

    normalized = re.sub(
        r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)",
        _convert_markdown_link,
        normalized,
    )
    try:
        safe_html = sanitize_telegram_html(normalized)
        safe_html = re.sub(r"<blockquote(?:[^>]*)>", "<i>", safe_html, flags=re.IGNORECASE)
        safe_html = re.sub(r"</blockquote>", "</i>", safe_html, flags=re.IGNORECASE)
    except Exception as e:
        logger.warning("sanitize_telegram_html failed: %s", e)
        safe_html = normalized or "-"
    return safe_html


def split_html_for_telegram(html: str, hard_limit: int = 3900) -> List[str]:
    """Split HTML into Telegram-sized chunks preserving balanced tags."""
    if not html:
        return ["—"]

    chunks: list[str] = []
    open_stack: list[tuple[str, str, str]] = []
    current_parts: list[str] = []
    current_len = 0
    has_visible_text = False

    def append_token(token: str, *, visible: bool = False) -> None:
        nonlocal current_len, has_visible_text
        current_parts.append(token)
        current_len += len(token)
        if visible and not has_visible_text and token.strip():
            has_visible_text = True

    def reopen_prefix() -> str:
        return "".join(info[1] for info in open_stack)

    def append_closings() -> str:
        return "".join(info[2] for info in reversed(open_stack))

    def flush_chunk() -> None:
        nonlocal current_parts, current_len, has_visible_text
        if not current_parts and not open_stack:
            return
        chunk_body = "".join(current_parts)
        closings = append_closings()
        chunk_full = chunk_body + closings

        # Skip chunks that reopen tags but do not contain visible text.
        if not has_visible_text:
            prefix = reopen_prefix()
            current_parts = [prefix] if prefix else []
            current_len = len(prefix)
            has_visible_text = False
            return

        chunks.append(chunk_full if chunk_full.strip() else "—")
        prefix = reopen_prefix()
        current_parts = [prefix] if prefix else []
        current_len = len(prefix)
        has_visible_text = False

    for match in _HTML_TOKEN_RE.finditer(html):
        token = match.group(0)
        tag_match = _HTML_TAG_RE.fullmatch(token)
        if tag_match:
            is_closing = bool(tag_match.group(1))
            tag_name_raw = tag_match.group(2)
            tag_name = tag_name_raw.lower()

            if not is_closing:
                if tag_name in _SELF_CLOSING_TAGS:
                    if current_len + len(token) > hard_limit:
                        flush_chunk()
                    append_token(token)
                    continue

                if current_len + len(token) > hard_limit:
                    flush_chunk()

                close_token = f"</{tag_name_raw}>"
                open_stack.append((tag_name, token, close_token))
                append_token(token)
            else:
                if current_len + len(token) > hard_limit:
                    flush_chunk()
                append_token(token)
                for idx in range(len(open_stack) - 1, -1, -1):
                    if open_stack[idx][0] == tag_name:
                        open_stack.pop(idx)
                        break
        else:
            text_chunk = token
            while text_chunk:
                space_left = hard_limit - current_len
                if space_left <= 0:
                    flush_chunk()
                    space_left = hard_limit - current_len
                if len(text_chunk) <= space_left:
                    append_token(text_chunk, visible=bool(text_chunk.strip()))
                    text_chunk = ""
                else:
                    slice_len = space_left
                    slice_part = text_chunk[:slice_len]
                    split_at = slice_part.rfind(" ")
                    if 0 < split_at < slice_len:
                        slice_len = split_at + 1
                        slice_part = text_chunk[:slice_len]
                    elif slice_len < len(text_chunk):
                        # Avoid cutting an HTML entity like &nbsp; or &#128512; in half.
                        last_amp = slice_part.rfind("&")
                        if last_amp != -1:
                            entity_candidate = slice_part[last_amp:]
                            if ";" not in entity_candidate:
                                slice_len = last_amp
                                slice_part = text_chunk[:slice_len]
                    if slice_len == 0:
                        # Fall back to a hard cut to guarantee progress.
                        slice_len = min(space_left, len(text_chunk))
                        slice_part = text_chunk[:slice_len]
                    append_token(slice_part, visible=bool(slice_part.strip()))
                    text_chunk = text_chunk[slice_len:]
                    flush_chunk()

    flush_chunk()

    return chunks or ["—"]


def _plain(text: str) -> str:
    """Убираем HTML, сохраняя переносы."""
    if not text:
        return " "
    normalized = re.sub(r"<br\s*/?>", "\n", text)
    plain = re.sub(r"<[^>]+>", "", normalized)
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain or " "


def _split_plain_text(text: str, limit: int = 3900) -> list[str]:
    """Split plain text into Telegram-safe chunks preserving natural breaks."""
    normalized = (text or "").replace("\r\n", "\n")
    if not normalized:
        return [" "]

    chunks: list[str] = []
    remaining = normalized

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        cutoff = remaining.rfind("\n", 0, limit)
        if cutoff == -1:
            cutoff = remaining.rfind(" ", 0, limit)
        if cutoff == -1 or cutoff < max(0, limit - 200):
            cutoff = limit

        chunk = remaining[:cutoff].rstrip("\n")
        if not chunk:
            chunk = remaining[:limit]
            cutoff = len(chunk)

        chunks.append(chunk)
        remaining = remaining[cutoff:]
        remaining = remaining.lstrip("\n")

    return chunks or [" "]


async def tg_edit_html(
    bot: Bot,
    chat_id: int,
    message_id: int,
    html: str,
    max_retries: int = 3,
) -> None:
    """
    Надёжное редактирование сообщения HTML-текстом.
    Обрабатывает rate limit, parse-ошибки; при entity/parse/too long — падает в plain.
    """
    for attempt in range(max_retries):
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=html,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            return
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except TelegramBadRequest as e:
            low = str(e).lower()
            if "message is not modified" in low:
                return
            if "message to edit not found" in low:
                raise
            if "can't parse entities" in low or "entity" in low or "too long" in low:
                logger.warning(
                    "Telegram rejected HTML chunk (len=%s), falling back to plain text: %s",
                    len(html),
                    e,
                )
                plain_chunks = _split_plain_text(_plain(html))
                try:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=plain_chunks[0],
                        disable_web_page_preview=True,
                    )
                except TelegramBadRequest:
                    raise

                for extra_chunk in plain_chunks[1:]:
                    with suppress(Exception):
                        await bot.send_message(
                            chat_id=chat_id,
                            text=extra_chunk,
                            disable_web_page_preview=True,
                        )
                return
            if attempt == max_retries - 1:
                raise


async def tg_send_html(
    bot: Bot,
    chat_id: int,
    html: str,
    reply_to_message_id: Optional[int] = None,
    max_retries: int = 3,
) -> None:
    """
    Надёжная отправка HTML-сообщения с фолбэком в plain и обработкой rate-limit.
    """
    for attempt in range(max_retries):
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("sending telegram chunk (len=%s): %s", len(html), html[:200])
            await bot.send_message(
                chat_id=chat_id,
                text=html,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
                reply_to_message_id=reply_to_message_id if attempt == 0 else None,
            )
            return
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after)
        except TelegramBadRequest as e:
            low = str(e).lower()
            if "can't parse entities" in low or "entity" in low or "too long" in low:
                logger.warning(
                    "Telegram rejected HTML chunk (len=%s), falling back to plain text: %s",
                    len(html),
                    e,
                )
                plain_chunks = _split_plain_text(_plain(html))
                for idx, chunk in enumerate(plain_chunks):
                    await bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        disable_web_page_preview=True,
                        reply_to_message_id=(
                            reply_to_message_id if idx == 0 and attempt == 0 else None
                        ),
                    )
                return
            if attempt == max_retries - 1:
                raise


async def send_html_text(
    bot: Bot,
    chat_id: int,
    raw_text: str,
    reply_to_message_id: Optional[int] = None,
) -> None:
    """
    Полный цикл для «обычного» ответа: format → sanitize → split → send (по кускам).
    """
    formatted = format_safe_html(raw_text)
    try:
        # Используем более консервативный предел, чтобы исключить edge-case ошибки Telegram
        chunks = split_html_for_telegram(formatted, hard_limit=3000)
    except Exception as e:
        logger.warning("split_html_for_telegram failed: %s", e)
        chunks = [(formatted or "")[:3900] or "—"]

    for idx, chunk in enumerate(chunks):
        await tg_send_html(
            bot=bot,
            chat_id=chat_id,
            html=chunk,
            reply_to_message_id=reply_to_message_id if idx == 0 else None,
        )
