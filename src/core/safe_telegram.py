# src/core/safe_telegram.py
from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Optional

from aiogram import Bot
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter

from src.bot.ui_components import render_legal_html, sanitize_telegram_html

logger = logging.getLogger(__name__)



def format_safe_html(raw_text: str) -> str:
    """
    Форматируем сырой текст в наш «красивый» HTML и санитизируем под Telegram.
    Гарантируем, что на выходе безопасный для отправки HTML (или plain в фолбэке).
    """
    try:
        html = render_legal_html(raw_text or "")
    except Exception as e:
        logger.warning("render_legal_html failed: %s", e)
        html = (raw_text or "").strip()
    try:
        safe_html = sanitize_telegram_html(html)
    except Exception as e:
        logger.warning("sanitize_telegram_html failed: %s", e)
        # безопасный фолбэк — экранируем все HTML
        from html import escape
        safe_html = escape(html)
    return safe_html


def split_html_for_telegram(html: str, hard_limit: int = 3900) -> List[str]:
    """
    Режем уже безопасный HTML на части ≤ hard_limit.
    Приоритет: абзацы (<br><br>) → строки (<br>) → предложения (.?! ) → жёсткая нарезка.
    """
    if not html:
        return ["—"]

    # нормализуем переносы
    text = re.sub(r"<br\s*/?>", "<br>", html, flags=re.IGNORECASE)
    chunks: list[str] = []

    def pack(parts: list[str], sep: str) -> list[str]:
        out, cur, ln = [], [], 0
        for p in parts:
            add = p
            sep_len = len(sep) if cur else 0
            if ln + sep_len + len(add) <= hard_limit:
                if cur:
                    cur.append(sep)
                cur.append(add)
                ln += sep_len + len(add)
            else:
                if cur:
                    out.append("".join(cur))
                cur, ln = [add], len(add)
        if cur:
            out.append("".join(cur))
        return out

    # 1) по абзацам
    paras = re.split(r"(?:<br>\s*){2,}", text)
    stage1 = pack(paras, "<br><br>")

    # 2) по строкам
    stage2: list[str] = []
    for block in stage1:
        if len(block) <= hard_limit:
            stage2.append(block)
            continue
        lines = block.split("<br>")
        stage2.extend(pack(lines, "<br>"))

    # 3) по предложениям
    final: list[str] = []
    sent_re = re.compile(r"(?<=[\.\!\?])\s+")
    for block in stage2:
        if len(block) <= hard_limit:
            final.append(block)
            continue
        sentences = sent_re.split(block)
        if len(sentences) > 1:
            final.extend(pack(sentences, " "))
        else:
            # 4) жёсткая нарезка
            for i in range(0, len(block), hard_limit):
                final.append(block[i : i + hard_limit])

    return [b.strip() for b in final if b.strip()] or ["—"]


def _plain(text: str) -> str:
    """Убираем любые теги — plain-версия."""
    return re.sub(r"<[^>]+>", "", text) or " "


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
