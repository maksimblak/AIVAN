from __future__ import annotations

import re
from typing import Iterable, List

from telegram_legal_bot.constants import (
    MD_ESCAPE_CHARS,
    TELEGRAM_MESSAGE_MAX_LENGTH,
    MAX_CHUNK_BACKOFF_CHARS,
)

_MD2_RE = re.compile(f"[{re.escape(MD_ESCAPE_CHARS)}]")

_URL_RE = re.compile(r"(https?://[^\s)]+)", re.IGNORECASE)


def md2(text: str) -> str:
    if not text:
        return ""
    return _MD2_RE.sub(lambda m: "\\" + m.group(0), text)


def _escape_md2_url(url: str) -> str:
    # Для MarkdownV2 в URL скобки нужно экранировать
    return url.replace(")", r"\)").replace("(", r"\(")


def strip_md2_escapes(text: str) -> str:
    return text.replace("\\", "")


def _backoff_incomplete_link(chunk: str) -> str:
    # Если кусок обрывает конструкцию ](http..., закроем скобки на всякий случай
    if chunk.count("(") > chunk.count(")"):
        return chunk + ")"
    return chunk


def chunk_markdown_v2(text: str, limit: int = TELEGRAM_MESSAGE_MAX_LENGTH) -> List[str]:
    """
    Безопасная разбивка MarkdownV2 по лимиту Telegram с защитой от
    обрыва ссылок/форматирования.
    """
    if not text:
        return [""]
    if len(text) <= limit:
        return [text]

    parts: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            parts.append(_backoff_incomplete_link("\n".join(cur)))
            cur = []
            cur_len = 0

    for ln in text.splitlines():
        ln = ln.rstrip()
        if cur_len + len(ln) + 1 <= limit:
            cur.append(ln)
            cur_len += len(ln) + 1
            continue

        # разбиваем строку по словам
        if len(ln) > limit:
            words = ln.split(" ")
            wbuf: List[str] = []
            wlen = 0
            for w in words:
                if wlen + len(w) + 1 <= limit - MAX_CHUNK_BACKOFF_CHARS:
                    wbuf.append(w)
                    wlen += len(w) + 1
                else:
                    seg = " ".join(wbuf).rstrip()
                    parts.append(_backoff_incomplete_link(seg))
                    wbuf = [w]
                    wlen = len(w)
            if wbuf:
                parts.append(_backoff_incomplete_link(" ".join(wbuf).rstrip()))
        else:
            flush()
            cur = [ln]
            cur_len = len(ln)

    flush()
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# ФОЛБЭК-РЕНДЕР: аккуратный ответ без «Нормы права не найдены.»
# ─────────────────────────────────────────────────────────────────────────────
def build_legal_reply(*, answer: str, laws: List[str]) -> str:
    """
    Человекочитаемый ответ в MarkdownV2.
    Особенности:
      • если laws пуст — блок «Нормы права» не печатается вовсе;
      • блок «Ответ» есть всегда (если пусто — даём вежливую подсказку);
      • URL в законах распознаются и корректно экранируются.
    """
    answer = (answer or "").strip()
    laws = laws or []

    lines: List[str] = []

    # 1) Ответ
    if answer:
        lines.append(f"*Ответ:* {md2(answer)}")
    else:
        lines.append(
            "*Ответ:* Не удалось автоматически сформировать развёрнутый ответ. "
            "Пожалуйста, уточните детали (работодатель/регион, даты и суммы, переписка/приказы, и т. п.)."
        )

    # 2) Нормы права — только если есть хотя бы одна
    law_lines: List[str] = []
    for raw in laws:
        raw = (raw or "").strip()
        if not raw:
            continue
        m = _URL_RE.search(raw)
        if m:
            url = _escape_md2_url(m.group(0))
            label = raw.replace(m.group(0), "").strip(" -—:") or "Ссылка"
            law_lines.append(f"• `{md2(label)}` — [{md2('открыть')}]({url})")
        else:
            law_lines.append("• " + md2(raw))

    if law_lines:
        lines.append("\n*Нормы права:*")
        lines.extend(law_lines)

    return "\n".join(lines).strip() or "*Ответ:* Нужны уточнения к вашему вопросу."
