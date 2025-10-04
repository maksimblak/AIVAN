# src/core/safe_telegram.py
from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Optional

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter

from src.bot.ui_components import sanitize_telegram_html

logger = logging.getLogger(__name__)



def format_safe_html(raw_text: str) -> str:
    """Sanitize text for Telegram while preserving simple markup and line breaks."""
    normalized = (raw_text or "").replace("\r\n", "\n")
    try:
        safe_html = sanitize_telegram_html(normalized)
    except Exception as e:
        logger.warning("sanitize_telegram_html failed: %s", e)
        safe_html = normalized or "-"
    return safe_html.replace("\n", "<br>")







def split_html_for_telegram(html: str, hard_limit: int = 3900) -> List[str]:
    """Split HTML into Telegram-sized chunks preserving balanced tags."""
    if not html:
        return ["—"]

    token_re = re.compile(r"<[^>]+>|[^<]+")
    tag_re = re.compile(r"<(/?)([a-zA-Z0-9-]+)([^>]*)>")
    self_closing = {"br"}

    chunks: list[str] = []
    open_stack: list[tuple[str, str, str]] = []
    current_parts: list[str] = []
    current_len = 0

    def append_token(token: str) -> None:
        nonlocal current_len
        current_parts.append(token)
        current_len += len(token)

    def reopen_prefix() -> str:
        return "".join(info[1] for info in open_stack)

    def append_closings() -> str:
        return "".join(info[2] for info in reversed(open_stack))

    def flush_chunk() -> None:
        nonlocal current_parts, current_len
        if not current_parts and not open_stack:
            return
        chunk_body = "".join(current_parts)
        chunk_body += append_closings()
        chunks.append(chunk_body if chunk_body.strip() else "—")
        prefix = reopen_prefix()
        current_parts = [prefix] if prefix else []
        current_len = len(prefix)

    for match in token_re.finditer(html):
        token = match.group(0)
        tag_match = tag_re.fullmatch(token)
        if tag_match:
            is_closing = bool(tag_match.group(1))
            tag_name_raw = tag_match.group(2)
            tag_name = tag_name_raw.lower()

            if not is_closing:
                if tag_name in self_closing:
                    if current_len + len(token) > hard_limit:
                        flush_chunk()
                    append_token(token)
                    continue

                close_token = f"</{tag_name_raw}>"
                open_stack.append((tag_name, token, close_token))
                if current_len + len(token) > hard_limit:
                    flush_chunk()
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
                    append_token(text_chunk)
                    text_chunk = ""
                else:
                    slice_len = space_left
                    slice_part = text_chunk[:slice_len]
                    split_at = slice_part.rfind(" ")
                    if 0 < split_at < slice_len:
                        slice_len = split_at + 1
                        slice_part = text_chunk[:slice_len]
                    append_token(slice_part)
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
                # plain fallback (всё равно ограничение по длине применяет Telegram)
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=_plain(html)[:3900] or " ",
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
                await bot.send_message(
                    chat_id=chat_id,
                    text=_plain(html)[:3900] or " ",
                    disable_web_page_preview=True,
                    reply_to_message_id=reply_to_message_id if attempt == 0 else None,
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
        chunks = split_html_for_telegram(formatted, hard_limit=3900)
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
