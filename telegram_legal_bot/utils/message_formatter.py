from __future__ import annotations

import re
from typing import Iterable, Sequence

from telegram.helpers import escape_markdown

TG_MAX_LEN = 4096

_url_re = re.compile(r"https?://\S+", re.IGNORECASE)


def md2(text: str) -> str:
    """Экранирует текст под Telegram MarkdownV2."""
    return escape_markdown(text, version=2)


def format_laws(laws: Iterable[str] | None) -> str:
    """
    Форматирует список норм/ссылок:
    - Название/номер нормы даём моноширинно
    - URL рендерим как обычный (кликабельный) текст после тире
    Пример: • `ст. 10 ГК РФ` — https://.../article/10
    """
    if not laws:
        return "Нормы права не найдены."

    lines: list[str] = []
    for raw in laws:
        if not raw:
            continue
        raw = raw.strip()
        # Если строка содержит URL — отделяем подпись и ссылку
        m = _url_re.search(raw)
        if m:
            url = m.group(0)
            label = raw.replace(url, "").strip(" -—:") or "Норма"
            lines.append(f"• `{md2(label)}` — {url}")
        else:
            lines.append(f"• `{md2(raw)}`")
    return "\n".join(lines)


def build_legal_reply(summary: str, details: str, laws: Sequence[str] | None) -> str:
    """
    Собирает красивый ответ в MarkdownV2 по требуемой структуре.
    """
    return (
        f"⚖️ *{md2('ЮРИДИЧЕСКАЯ КОНСУЛЬТАЦИЯ')}*\n\n"
        f"📋 *{md2('Краткий ответ:')}*\n"
        f"{md2(summary)}\n\n"
        f"📄 *{md2('Подробное разъяснение:')}*\n"
        f"{md2(details)}\n\n"
        f"📚 *{md2('Применимые нормы права:')}*\n"
        f"{format_laws(laws)}\n\n"
        f"⚠️ *{md2('Важно:')}*\n"
        f"{md2('Данная консультация носит информационный характер и не заменяет профессиональную юридическую помощь.')}"
    )


def chunk_markdown_v2(text: str, limit: int = TG_MAX_LEN) -> list[str]:
    """
    Аккуратно режем длинное сообщение для Telegram.
    Правила:
      1) Пытаемся резать по двойным переносам.
      2) Если блок очень длинный — дорезаем по одиночным переносам.
      3) В крайнем случае — просто по символам.
    """
    parts: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) <= limit:
            parts.append(block)
            continue

        # Ищем ближайший перенос строки
        cur = block
        while len(cur) > limit:
            cut = cur.rfind("\n", 0, limit)
            if cut == -1:
                cut = limit
            parts.append(cur[:cut])
            cur = cur[cut:].lstrip()
        if cur:
            parts.append(cur)
    return parts
