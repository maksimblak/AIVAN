# telegram_legal_bot/utils/message_formatter.py
from __future__ import annotations

import re
from typing import Iterable, List

# ── Экранирование для Telegram MarkdownV2 ─────────────────────────────────────
_MD2_NEED_ESCAPE = r"_*[]()~`>#+-=|{}.!\\"
_MD2_RE = re.compile(f"[{re.escape(_MD2_NEED_ESCAPE)}]")

def md2(text: str) -> str:
    """
    Экранирует произвольный текст под Telegram MarkdownV2.
    """
    if not text:
        return ""
    return _MD2_RE.sub(lambda m: "\\" + m.group(0), text)

def strip_md2_escapes(text: str) -> str:
    """
    Убирает обратные слеши перед спецсимволами Telegram MarkdownV2.
    Нужен для фоллбек-отправки plain-текста, если форматирование упало.
    """
    if not text:
        return ""
    # снимаем экранирование только с допустимых символов
    return re.sub(r"\\([_*[\]()~`>#+\-=|{}.!\\])", r"\1", text)

def _escape_md2_url(url: str) -> str:
    """
    Для MarkdownV2 в URL критичны только круглые скобки — экранируем.
    """
    return url.replace("(", r"\(").replace(")", r"\)")

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

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
        m = _URL_RE.search(raw)
        if m:
            url = _escape_md2_url(m.group(0))
            label = raw.replace(m.group(0), "").strip(" -—:") or "Ссылка"
            lines.append(f"• `{md2(label)}` — [{md2('открыть')}]({url})")
        else:
            lines.append(f"• `{md2(raw)}`")
    return "\n".join(lines)

def build_legal_reply(
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

def chunk_markdown_v2(text: str, limit: int = 4096) -> List[str]:
    """
    Дробит длинное сообщение на куски ≤ limit.
    Старается резать по пустым строкам, затем по строкам, затем жёстко.
    """
    if not text:
        return [""]
    out: List[str] = []
    buf: List[str] = []

    def flush():
        if not buf:
            return
        chunk = "\n".join(buf)
        if len(chunk) <= limit:
            out.append(chunk)
        else:
            # жёстко режем, если всё равно перебор
            s = chunk
            while len(s) > limit:
                out.append(s[:limit])
                s = s[limit:]
            if s:
                out.append(s)
        buf.clear()

    # пробуем резать по параграфам
    for block in text.split("\n\n"):
        if not buf:
            buf.append(block)
        else:
            candidate = "\n\n".join([ "\n".join(buf), block ])
            if len(candidate) <= limit:
                buf.append("")  # восстановим двойной перевод строки
                buf.append(block)
            else:
                flush()
                buf.append(block)
    flush()
    return out
