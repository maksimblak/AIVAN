# telegram_legal_bot/utils/pro_logging.py
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

# ── контекстные поля для логов ────────────────────────────────────────────────
_cid = contextvars.ContextVar("cid", default=None)       # correlation id
_uid = contextvars.ContextVar("uid", default=None)       # user id
_chat = contextvars.ContextVar("chat_id", default=None)  # chat id

def get_cid() -> Optional[str]: return _cid.get()
def get_uid() -> Optional[int]: return _uid.get()
def get_chat_id() -> Optional[int]: return _chat.get()

@contextlib.contextmanager
def log_context(*, cid: Optional[str] = None, uid: Optional[int] = None, chat_id: Optional[int] = None):
    t1 = _cid.set(cid if cid is not None else _cid.get())
    t2 = _uid.set(uid if uid is not None else _uid.get())
    t3 = _chat.set(chat_id if chat_id is not None else _chat.get())
    try:
        yield
    finally:
        _cid.reset(t1); _uid.reset(t2); _chat.reset(t3)

# ── санитайзинг ───────────────────────────────────────────────────────────────
_SECRET_PATTERNS = [
    (re.compile(r'sk-[A-Za-z0-9_\-]{12,}'), "sk-***MASKED***"),
    (re.compile(r'([Bb]earer)\s+[A-Za-z0-9_\-]{12,}'), r'\1 ***MASKED***'),
    (re.compile(r'://([^:\s]+):([^@/]+)@'), r'://\1:***@'),
]

def redact_text(s: Any) -> str:
    text = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, default=str)
    for pat, repl in _SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text

def shorten(s: str, max_len: int = 1200) -> str:
    if len(s) <= max_len: return s
    head = s[: max_len - 20]
    tail = s[-10:]
    return f"{head}…{tail} (len={len(s)})"

def dump_payload(obj: Any, max_chars: int = 2000, *, redacted: bool = True) -> str:
    try:
        raw = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        raw = str(obj)
    if redacted:
        raw = redact_text(raw)
    return shorten(raw, max_chars)

# ── форматтер JSON с контекстом ───────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "t": int(time.time()),
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "name": record.name,
        }
        cid = get_cid(); uid = get_uid(); chat = get_chat_id()
        if cid is not None: payload["cid"] = cid
        if uid is not None: payload["uid"] = uid
        if chat is not None: payload["chat"] = chat
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def setup_pro_logging(json_mode: bool = True, level: str = "INFO") -> None:
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(lvl)

    handler = logging.StreamHandler()
    if json_mode:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [cid=%(cid)s uid=%(uid)s chat=%(chat)s]: %(message)s"))

    # фильтр, который добавляет контекст в record для текстового форматтера
    class CtxFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.cid = get_cid()
            record.uid = get_uid()
            record.chat = get_chat_id()
            return True

    handler.addFilter(CtxFilter())
    root.addHandler(handler)

# ── флаги логирования промптов (через ENV или Settings) ───────────────────────
def should_log_prompts(settings: Any) -> bool:
    env = os.getenv("LOG_PROMPT_DEBUG", "")
    if env:
        return env.strip() not in {"0", "false", "False", "no"}
    return bool(getattr(settings, "log_prompt_debug", False))

def prompt_maxlen(settings: Any, default: int = 2000) -> int:
    val = os.getenv("LOG_PROMPT_MAXLEN")
    if val and val.isdigit():
        return max(200, int(val))
    return int(getattr(settings, "log_prompt_maxlen", default) or default)

def include_system_prompt(settings: Any) -> bool:
    env = os.getenv("LOG_PROMPT_INCLUDE_SYSTEM", "")
    if env:
        return env.strip() not in {"0", "false", "False", "no"}
    return bool(getattr(settings, "log_prompt_include_system", False))
