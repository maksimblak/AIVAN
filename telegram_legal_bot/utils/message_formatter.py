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
    Для MarkdownV2 в URL критичны круглые скобки — экранируем их.
    Остальные символы оставляем как есть (Telegram сам корректно обрабатывает).
    """
    return url.replace("(", r"\(").replace(")", r"\)")


# ── Форматирование правовых норм / источников ────────────────────────────────
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


# ── Безопасная разбивка MarkdownV2-сообщений ─────────────────────────────────
_TELEGRAM_HARD_LIMIT = 4096
_FENCE_LINE = re.compile(r"^\s*```")            # строка с тройными бэктиками
_BULLET = re.compile(r"^\s*(?:[-*•]|—|\d+\.)\s+")  # признаки списка для мягкого реза


def chunk_markdown_v2(text: str, limit: int = _TELEGRAM_HARD_LIMIT) -> List[str]:
    """
    Дробит длинное MarkdownV2-сообщение на куски ≤ limit.
    Стратегия:
      1) нормализуем переносы, режем по пустым строкам (параграфы);
      2) собираем чанки, стараясь не превышать limit;
      3) для слишком больших параграфов — режем по строкам/словам;
    Защиты:
      • если чанк заканчивается открытым ``` — закрываем его и при необходимости
        открываем заново в начале следующего чанка;
      • лёгкий «бэкофф» от незакрытых '[' или '(' в конце чанка (чтобы не ломать ссылки).
    """
    if not text:
        return []

    limit = max(1, min(int(limit), _TELEGRAM_HARD_LIMIT))
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = re.split(r"\n{2,}", text)

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    code_open = False      # открыт ли сейчас ``` внутри буфера
    reopen_next = False    # нужно ли открыть ``` в начале следующего чанка

    def fence_toggles_in_block(block: str) -> int:
        return sum(1 for ln in block.split("\n") if _FENCE_LINE.match(ln))

    def _backoff_incomplete_link(s: str) -> str:
        tail = s[-200:] if len(s) > 200 else s
        last_open = max(tail.rfind("["), tail.rfind("("))
        last_close = max(tail.rfind("]"), tail.rfind(")"))
        if last_open > last_close >= 0:
            cut = len(s) - (len(tail) - last_open)
            return s[:cut].rstrip()
        return s

    def push_buf():
        nonlocal buf, buf_len, code_open, reopen_next
        if not buf:
            return
        out = "\n\n".join(buf)
        if code_open:
            out += "\n```"   # закрываем незакрытый блок в конце чанка
            reopen_next = True
            code_open = False
        out = _backoff_incomplete_link(out)
        chunks.append(out)
        buf = []
        buf_len = 0

    def try_reopen():
        nonlocal buf, buf_len, reopen_next, code_open
        if reopen_next:
            # открыть код-блок в начале нового чанка
            if buf_len:
                push_buf()
            buf.append("```")
            buf_len = len(buf[0])
            code_open = True
            reopen_next = False

    def append_block(block: str) -> bool:
        nonlocal buf, buf_len, code_open
        # +2 — это возможные \n\n между абзацами
        add_len = (2 if buf_len else 0) + len(block)
        if buf_len + add_len <= limit:
            if buf_len:
                buf.append("")  # восстанавливаем пустую строку между абзацами
            buf.append(block)
            buf_len += add_len
            if fence_toggles_in_block(block) % 2 == 1:
                code_open = not code_open
            return True
        return False

    for para in paragraphs:
        if not para:
            continue

        # если параграф сам по себе > limit — режем его мелко
        if len(para) > limit:
            push_buf()
            try_reopen()
            for sub in _split_long_block(para, limit):
                if not append_block(sub):
                    push_buf()
                    try_reopen()
                    append_block(sub)
            continue

        if not append_block(para):
            push_buf()
            try_reopen()
            append_block(para)

    push_buf()
    return chunks


def _split_long_block(block: str, limit: int) -> List[str]:
    """
    Режем ОДИН очень длинный абзац:
      • сначала по строкам,
      • затем (при необходимости) по словам.
    Следим за ``` внутри кусков и не оставляем незакрытым на конце части.
    """
    lines = block.split("\n")
    parts: List[str] = []
    cur: List[str] = []
    cur_len = 0
    local_code_open = False

    def _backoff_incomplete_link(s: str) -> str:
        tail = s[-200:] if len(s) > 200 else s
        last_open = max(tail.rfind("["), tail.rfind("("))
        last_close = max(tail.rfind("]"), tail.rfind(")"))
        if last_open > last_close >= 0:
            cut = len(s) - (len(tail) - last_open)
            return s[:cut].rstrip()
        return s

    def flush():
        nonlocal cur, cur_len, local_code_open
        if not cur:
            return
        out = "\n".join(cur).rstrip()
        if local_code_open:
            out += "\n```"
            local_code_open = False
        parts.append(_backoff_incomplete_link(out))
        cur = []
        cur_len = 0

    for ln in lines:
        if _FENCE_LINE.match(ln):
            local_code_open = not local_code_open

        ln_len = (1 if cur else 0) + len(ln)
        if cur_len + ln_len <= limit:
            cur.append(ln)
            cur_len += ln_len
            continue

        # если строка сама длиннее limit — режем по словам
        if len(ln) > limit:
            flush()
            words = ln.split(" ")
            wbuf: List[str] = []
            wlen = 0
            for w in words:
                wl = (1 if wbuf else 0) + len(w)
                if wlen + wl <= limit:
                    wbuf.append(w)
                    wlen += wl
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
