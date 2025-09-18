from __future__ import annotations

import re
from typing import Iterable, Sequence

TG_MAX_LEN = 4096
_url_re = re.compile(r"https?://\S+", re.IGNORECASE)

_MD2_NEED_ESCAPE = re.compile(r'([_*\[\]()~`>#+\-=|{}.!])')


def escape_md2(text: str) -> str:
    """
    Минималистичный экранировщик под Telegram MarkdownV2.
    Экранируем спецсимволы из доки Telegram:
    _ * [ ] ( ) ~ ` > # + - = | { } . !
    и сначала удваиваем обратные слеши.
    """
    if not text:
        return ""
    text = text.replace("\\", "\\\\")
    return _MD2_NEED_ESCAPE.sub(r"\\\1", text)


def md2(text: str) -> str:
    """Сахарная обёртка — используем в проекте везде вместо внешних хелперов."""
    return escape_md2(text)


def format_laws(laws: Iterable[str] | None) -> str:
    """
    Форматирует список норм/ссылок:
    - Текст нормы — моноширинный,
    - URL — кликабельный (не прячем в моноширинный блок).
    Пример: • `ст. 10 ГК РФ` — https://.../article/10
    """
    if not laws:
        return "Нормы права не найдены."
    lines: list[str] = []
    for raw in laws:
        if not raw:
            continue
        raw = raw.strip()
        m = _url_re.search(raw)
        if m:
            url = m.group(0)
            label = raw.replace(url, "").strip(" -—:") or "Норма"
            lines.append(f"• `{md2(label)}` — {url}")
        else:
            lines.append(f"• `{md2(raw)}`")
    return "\n".join(lines)


def build_legal_reply(summary: str, details: str, laws: Sequence[str] | None) -> str:
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
    Режем длинный MarkdownV2-текст на части, стараясь попадать в переносы.
    """
    parts: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) <= limit:
            parts.append(block)
            continue
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

# back-compat alias (если где-то осталось старое имя)
format_legal_response = build_legal_reply
