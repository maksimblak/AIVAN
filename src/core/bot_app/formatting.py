from __future__ import annotations

import re
from datetime import datetime
from html import escape as html_escape
from typing import Any, Iterable

DEFAULT_TEXT_LIMIT = 3900
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[\.\!\?])\s+")


def chunk_text(text: str, max_length: int | None = None, *, default_limit: int = DEFAULT_TEXT_LIMIT) -> list[str]:
    """Split long Telegram messages into chunks respecting limits."""
    limit = max_length or default_limit
    if len(text) <= limit:
        return [text]

    separator = "\n\n"
    separator_len = len(separator)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        if not current_parts:
            return
        joined = separator.join(current_parts).strip()
        chunks.append(joined)
        current_parts = []
        current_len = 0

    for paragraph in text.split("\n\n"):
        part = paragraph
        while True:
            separator_overhead = separator_len if current_parts else 0
            candidate_len = current_len + separator_overhead + len(part)
            if candidate_len <= limit:
                if current_parts:
                    current_len += separator_len
                current_parts.append(part)
                current_len += len(part)
                break

            if current_parts:
                flush_current()
                continue

            if len(part) <= limit:
                current_parts.append(part)
                current_len = len(part)
                break

            chunks.append(part[:limit])
            part = part[limit:]
            if not part:
                current_parts = []
                current_len = 0
                break

    flush_current()

    return chunks


def _split_plain_text(text: str, limit: int = DEFAULT_TEXT_LIMIT) -> list[str]:
    if not text:
        return []

    if len(text) <= limit:
        return [text]

    line_sep = chr(10)
    paragraph_sep = line_sep * 2
    chunks: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            chunks.append(paragraph_sep.join(current))
            current.clear()

    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines():
        if line.strip():
            buffer.append(line)
        else:
            if buffer:
                paragraphs.append(line_sep.join(buffer))
                buffer.clear()
    if buffer:
        paragraphs.append(line_sep.join(buffer))

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        joined = paragraph_sep.join(current + [paragraph])
        if len(joined) <= limit:
            current.append(paragraph)
            continue
        flush_current()
        if len(paragraph) > limit:
            for i in range(0, len(paragraph), limit):
                chunks.append(paragraph[i : i + limit])
        else:
            current.append(paragraph)
    flush_current()

    return chunks


def _split_html_safely(html: str, hard_limit: int = DEFAULT_TEXT_LIMIT) -> list[str]:
    """
    Аккуратно делим HTML сообщение, стараясь сохранять структуру и границы предложений.
    """
    if not html:
        return []

    cleaned = re.sub(r"<br\s*/?>", "<br>", html, flags=re.IGNORECASE)

    def _pack(parts: Iterable[str], sep: str) -> list[str]:
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

    paragraphs = re.split(r"(?:<br>\s*){2,}", cleaned)
    tmp = _pack(paragraphs, "<br><br>")

    next_stage: list[str] = []
    for block in tmp:
        if len(block) <= hard_limit:
            next_stage.append(block)
            continue
        lines = block.split("<br>")
        next_stage.extend(_pack(lines, "<br>"))

    final: list[str] = []
    for block in next_stage:
        block = block.strip()
        if not block:
            continue
        if len(block) <= hard_limit:
            final.append(block)
            continue
        sentences = _SENTENCE_BOUNDARY_RE.split(block)
        if len(sentences) > 1:
            final.extend(_pack(sentences, " "))
        else:
            for i in range(0, len(block), hard_limit):
                final.append(block[i : i + hard_limit])

    return [b.strip() for b in final if b.strip()]


def _format_datetime(ts: int | None, *, default: str = "неизвестно") -> str:
    if not ts or ts <= 0:
        return default
    try:
        return datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return default


def _format_response_time(ms: int) -> str:
    if ms <= 0:
        return "-"
    if ms < 1000:
        return f"{ms} мс"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes = int(seconds // 60)
    rem_seconds = int(seconds % 60)
    return f"{minutes} мин {rem_seconds:02d} с"


def _format_number(value: int | float | None) -> str:
    if value is None:
        return "0"
    try:
        return f"{float(value):,.0f}".replace(",", " ")
    except (ValueError, TypeError):
        return "0"


def _format_trend_value(current: int, previous: int) -> str:
    diff = current - previous
    arrow = "↗️" if diff > 0 else "↘️" if diff < 0 else "➡️"
    return f"{_format_number(current)} {arrow} ({diff:+})"


def _format_stat_row(label: str, value: str) -> str:
    return f"<b>{label}</b> — {value}"


def _format_hour_label(hour: str) -> str:
    if not hour:
        return hour
    try:
        hour_int = int(hour)
        return f"{hour_int:02d}:00"
    except ValueError:
        return hour


def _format_currency(amount_minor: int | None, currency: str) -> str:
    if amount_minor is None:
        return f"0 {currency.upper()}"
    if currency.upper() == "RUB":
        value = amount_minor / 100
        return f"{value:,.2f} ₽".replace(",", " ")
    return f"{amount_minor} {currency.upper()}"


def _format_risk_count(count: int) -> str:
    count = int(count)
    suffix = "рисков"
    if count % 10 == 1 and count % 100 != 11:
        suffix = "риск"
    elif count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        suffix = "риска"
    return f"Найдено {count} {suffix}"


def _format_progress_extras(update: dict[str, Any]) -> str:
    parts: list[str] = []
    if update.get("violations") is not None:
        parts.append(f"⚠️ Нарушений: {int(update['violations'])}")
    if update.get("chunks_total") and update.get("chunk_index"):
        parts.append(f"📦 Чанк {int(update['chunk_index'])}/{int(update['chunks_total'])}")
    elif update.get("chunks_total") is not None:
        parts.append(f"📦 Чанков: {int(update['chunks_total'])}")
    if update.get("language_pair"):
        parts.append(f"🌐 {html_escape(str(update['language_pair']))}")
    if update.get("pages_total") is not None:
        done = int(update.get("pages_done") or 0)
        total = int(update["pages_total"])
        parts.append(f"📄 Страницы: {done}/{total}")
    if update.get("confidence") is not None:
        parts.append(f"🔍 Уверенность: {float(update['confidence']):.1f}%")
    if update.get("note"):
        parts.append(f"🗒️ {html_escape(str(update['note']))}")
    return " | ".join(parts)


__all__ = [
    "chunk_text",
    "_split_plain_text",
    "_split_html_safely",
    "_format_datetime",
    "_format_response_time",
    "_format_number",
    "_format_trend_value",
    "_format_stat_row",
    "_format_hour_label",
    "_format_currency",
    "_format_risk_count",
    "_format_progress_extras",
    "DEFAULT_TEXT_LIMIT",
]
