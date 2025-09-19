from __future__ import annotations

import re
from typing import Iterable, List

# ── Экранирование для Telegram MarkdownV2 ─────────────────────────────────────
_MD2_NEED_ESCAPE = r"_*[]()~`>#+-=|{}.!"
_MD2_RE = re.compile(f"[{re.escape(_MD2_NEED_ESCAPE)}]")


def md2(text: str) -> str:
    """
    Экранирует произвольный текст под Telegram MarkdownV2.
    """
    if not text:
        return ""
    return _MD2_RE.sub(lambda m: "\\" + m.group(0), text)


def _escape_md2_url(url: str) -> str:
    """
    Для MarkdownV2 в URL критичны только круглые скобки — экранируем.
    """
    return url.replace("(", r"\(").replace(")", r"\)")


_url_re = re.compile(r"https?://\S+", re.IGNORECASE)


def format_laws(laws: Iterable[str] | None) -> str:
    """
    Форматирует список «норм права» дружелюбно к MarkdownV2.
    Если внутри есть URL — делаем кликабельную метку.
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
            url = _escape_md2_url(m.group(0))
            label = raw.replace(m.group(0), "").strip(" -—:") or "Ссылка"
            lines.append(f"• `{md2(label)}` — [{md2('открыть')}]({url})")
        else:
            lines.append(f"• `{md2(raw)}`")
    return "\n".join(lines)


def build_legal_answer_message(
    answer: str,
    laws: Iterable[str] | None = None,
    intro: str | None = None,
) -> str:
    """
    Собирает финальный ответ: вступление (если есть), текст ответа, блок «Нормы права».
    """
    parts: list[str] = []
    if intro:
        parts.append(md2(intro.strip()))
    if answer:
        parts.append(md2(answer.strip()))
    if laws is not None:
        parts.append("")
        parts.append("*Нормы права:*")
        parts.append(format_laws(laws))
    return "\n".join(parts).strip()


def chunk_for_telegram(text: str, limit: int = 4096) -> List[str]:
    """
    Дробит длинное сообщение на куски ≤ limit.
    Старается резать по пустым строкам, чтобы не рвать абзацы.
    """
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        cut = text.rfind("\n\n", start, end)
        if cut == -1 or cut <= start:
            cut = end
        chunks.append(text[start:cut])
        start = cut
    return [c for c in chunks if c]


# ── Алиасы под ожидаемые имена (чтобы импорты не падали) ─────────────────────
def build_legal_reply(answer: str, laws: Iterable[str] | None = None, intro: str | None = None) -> str:
    """Синоним build_legal_answer_message (для обратной совместимости)."""
    return build_legal_answer_message(answer=answer, laws=laws, intro=intro)


def chunk_markdown_v2(text: str, limit: int = 4096) -> List[str]:
    """Синоним chunk_for_telegram (для обратной совместимости)."""
    return chunk_for_telegram(text=text, limit=limit)


__all__ = [
    "md2",
    "format_laws",
    "build_legal_answer_message",
    "chunk_for_telegram",
    # алиасы:
    "build_legal_reply",
    "chunk_markdown_v2",
]
