# progress_ui.py  — версия со стримингом текста
from __future__ import annotations
import asyncio, time, re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from aiogram.enums import ParseMode
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, Message
from aiogram import F, Router

ICON_IDLE = "○"
ICON_RUN  = "⏳"
ICON_DONE = "✅"

def _bar(done: int, total: int, width: int = 12) -> str:
    if total <= 0: return "—"
    frac = max(0.0, min(1.0, done / total))
    full = int(frac * width)
    return "▓" * full + "░" * (width - full)

def _html_escape(s: str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))

def _to_html_chunk(raw: str) -> str:
    # грубый, но быстрый рендер: переносы -> <br>, ссылки -> <a>
    t = _html_escape(raw)
    t = re.sub(r"(https?://\S+)", r'<a href="\1">\1</a>', t)
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return t

@dataclass
class ChecklistProgress:
    bot: any
    chat_id: int
    stages: List[str]
    session_store: Optional[any] = None
    context_questions: Optional[List[str]] = None
    throttle_sec: float = 0.8
    max_message_len: int = 3900   # безопасный лимит для всего сообщения

    message_id: Optional[int] = field(default=None, init=False)
    start_ts: float = field(default=0, init=False)
    cur: int = field(default=-1, init=False)
    _last_edit: float = field(default=0, init=False)
    _ticker_task: Optional[asyncio.Task] = field(default=None, init=False)
    show_ctx: bool = field(default=False, init=False)
    _done: bool = field(default=False, init=False)

    # --- стриминговый блок ---
    stream_title: str = field(default="Формирую ответ", init=False)
    _stream_raw: List[str] = field(default_factory=list, init=False)
    _stream_enabled: bool = field(default=False, init=False)

    async def start(self, show_ctx_default: bool = False):
        self.show_ctx = show_ctx_default
        self.start_ts = time.time()
        msg = await self.bot.send_message(
            self.chat_id, self._render(), parse_mode=ParseMode.HTML,
            reply_markup=self._kb(), disable_web_page_preview=True
        )
        self.message_id = msg.message_id
        self._save_state()
        self._ticker_task = asyncio.create_task(self._ticker())

    async def to_stage(self, idx: int):
        idx = max(-1, min(idx, len(self.stages) - 1))
        if idx == self.cur:
            return
        self.cur = idx
        await self._safe_edit()

    async def next(self): await self.to_stage(self.cur + 1)

    async def complete(self):
        self.cur = len(self.stages) - 1
        self._done = True
        await self._safe_edit(force=True)
        if self._ticker_task: self._ticker_task.cancel()

    # ---------- стрим ----------
    async def start_stream(self, title: str = "Формирую ответ"):
        self.stream_title = title
        self._stream_raw.clear()
        self._stream_enabled = True
        await self._safe_edit(force=True)

    async def push_stream_delta(self, delta: str):
        if not self._stream_enabled or not delta:
            return
        self._stream_raw.append(delta)
        await self._safe_edit()

    async def finish_stream(self):
        self._stream_enabled = False
        await self._safe_edit(force=True)

    # ---------- служебные ----------
    async def _ticker(self):
        try:
            while not self._done:
                await asyncio.sleep(1.0)
                await self._safe_edit(force=True)
        except asyncio.CancelledError:
            pass

    def _elapsed(self) -> str:
        s = int(time.time() - self.start_ts)
        return f"{s//60:02d}:{s%60:02d}"

    def _render(self) -> str:
        done = max(0, self.cur + 1)
        header = (
            f"<b>AI-юрист — прогресс</b>\n"
            f"Действие выполнено за: <code>{self._elapsed()}</code>\n"
            f"{_bar(done, len(self.stages))}  {done}/{len(self.stages)}\n\n"
        )
        rows = []
        for i, title in enumerate(self.stages):
            icon = ICON_DONE if i < self.cur else (ICON_RUN if i == self.cur else ICON_IDLE)
            rows.append(f"{icon} {title}")
        body = "\n".join(rows)

        ctx = ""
        if self.context_questions is not None:
            state = "вкл" if self.show_ctx else "выкл"
            ctx = f"\n\n<i>Контекстные вопросы: {state}</i>"
            if self.show_ctx and self.context_questions:
                ctx_list = "\n".join(f"• {q}" for q in self.context_questions[:6])
                ctx += f"\n{ctx_list}"

        stream_html = ""
        if self._stream_enabled or self._stream_raw:
            raw = "".join(self._stream_raw)
            rendered = _to_html_chunk(raw)
            stream_header = f"\n\n<b>{_html_escape(self.stream_title)}</b>\n"
            # обрежем хвост, чтобы не выйти за лимит
            base_len = len(header) + len(body) + len(ctx) + len(stream_header)
            room = max(0, self.max_message_len - base_len)
            if len(rendered) > room:
                rendered = "… " + rendered[-room:]
            stream_html = stream_header + rendered

        return header + body + ctx + stream_html

    def _kb(self) -> InlineKeyboardMarkup:
        btn = InlineKeyboardButton(
            text=("Скрыть контекст" if self.show_ctx else "Показать контекст"),
            callback_data=f"progress:ctx:{int(self.show_ctx)}:{self.message_id or 0}"
        )
        return InlineKeyboardMarkup(inline_keyboard=[[btn]])

    async def _safe_edit(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_edit < self.throttle_sec:
            return
        self._last_edit = now
        if self.message_id is None: return
        try:
            await self.bot.edit_message_text(
                chat_id=self.chat_id, message_id=self.message_id,
                text=self._render(), parse_mode=ParseMode.HTML,
                reply_markup=self._kb(), disable_web_page_preview=True
            )
        except Exception:
            pass
        self._save_state()

    def _save_state(self):
        if not self.session_store or self.message_id is None:
            return
        key = f"progress:{self.message_id}"
        self.session_store.set(key, {
            "chat_id": self.chat_id,
            "cur": self.cur,
            "stages": self.stages,
            "start_ts": self.start_ts,
            "show_ctx": self.show_ctx,
        })

def setup_progress_router(router: Router, session_store):
    @router.callback_query(F.data.startswith("progress:ctx:"))
    async def on_toggle(call: CallbackQuery):
        try:
            _, _, state, mid = call.data.split(":")
            mid = int(mid)
        except Exception:
            await call.answer(); return

        key = f"progress:{mid}"
        data: Dict = await session_store.get(key) if hasattr(session_store, "get") else None
        if not data:
            await call.answer("Истёк контекст"); return

        show_ctx = not bool(int(state))
        data["show_ctx"] = show_ctx
        await session_store.set(key, data)
        # быстрый «перерисовать»: просто дергаем edit через тот же текст
        try:
            txt = "<i>Обновляю…</i>"
            await call.message.edit_text(txt, parse_mode=ParseMode.HTML)
        except Exception:
            pass
        await call.answer()
